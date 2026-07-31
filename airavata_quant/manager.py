"""Model loading, generation and benchmarking."""

from __future__ import annotations

import contextlib
import gc
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .benchmark import DEFAULT_PROMPTS, LatencySample, compare, summarize
from .config import Settings
from .quantization import (
    DYNAMIC_QUANT,
    ORIGINAL,
    VARIANTS,
    Variant,
    get_variant,
    resolve_preload,
    supports_device,
    variants_for_device,
)
from .schemas import GenerationRequest

logger = logging.getLogger(__name__)

BYTES_PER_MB = 1024 * 1024


class VariantUnavailableError(RuntimeError):
    """Raised when a variant exists but cannot run on the current device."""


class ModelLoadError(RuntimeError):
    """Raised when a variant is supported but failed to load."""


class GenerationResult(Dict[str, Any]):
    """Plain dict subclass so results can be splatted into a response model."""


def _resolve_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"{name!r} is not a torch dtype")
    return dtype


def build_bnb_config(variant: Variant) -> Optional[BitsAndBytesConfig]:
    """Turn a variant's declarative preset into a ``BitsAndBytesConfig``."""
    if not variant.bnb_kwargs:
        return None
    kwargs: Dict[str, Any] = {}
    for key, value in variant.bnb_kwargs.items():
        kwargs[key] = _resolve_dtype(value) if key.endswith("compute_dtype") else value
    return BitsAndBytesConfig(**kwargs)


def _packed_parameter_bytes(model: Any) -> int:
    """Bytes held by quantized modules that hide their weights from ``parameters()``.

    ``torch.ao.quantization.quantize_dynamic`` replaces ``nn.Linear`` with a
    module whose weight lives in a ``LinearPackedParams`` script object. It is
    neither a parameter nor a buffer, so a naive walk reports a dynamically
    quantized model as occupying *zero* bytes.
    """
    total = 0
    for module in model.modules():
        packed = getattr(module, "_packed_params", None)
        # The script object one level down has no such accessor; skip it.
        weight_bias = getattr(packed, "_weight_bias", None)
        if weight_bias is None:
            continue
        try:
            tensors = weight_bias()
        except (RuntimeError, AttributeError, TypeError):  # pragma: no cover
            logger.debug("could not read packed params of %r", module, exc_info=True)
            continue
        for tensor in tensors:
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return total


def model_memory_bytes(model: Any) -> int:
    """Bytes of weights held by ``model``.

    This is the number a quantization benchmark actually needs: process RSS and
    ``torch.cuda.memory_allocated`` both include every *other* variant loaded in
    the same process, so they cannot be compared across variants.
    ``nn.Module.parameters()`` already de-duplicates tied weights.
    """
    total = 0
    for tensor in model.parameters():
        total += tensor.numel() * tensor.element_size()
    for tensor in model.buffers():
        total += tensor.numel() * tensor.element_size()
    return total + _packed_parameter_bytes(model)


class ModelManager:
    """Owns the tokenizer and every loaded model variant.

    Loading is lazy and guarded by a *per-variant* lock so two concurrent
    requests for the same variant do not both pull a multi-gigabyte checkpoint
    into memory, while a request for an already-resident variant is never made
    to queue behind an unrelated download.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.models: Dict[str, Any] = {}
        self.tokenizer = None
        self.device = self._resolve_device(self.settings.device)
        #: Guards the registry itself (``models``/``tokenizer``/``_variant_locks``).
        #: Never held across a load.
        self._lock = threading.RLock()
        self._variant_locks: Dict[str, threading.RLock] = {}
        #: Serialises generations that pin the global torch RNG; see ``_seeded``.
        self._seed_lock = threading.Lock()
        self._executor = None

    # ------------------------------------------------------------------
    # device / lifecycle
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_device(preference: str) -> torch.device:
        if preference == "cpu":
            return torch.device("cpu")
        if preference == "cuda":
            if not torch.cuda.is_available():
                raise VariantUnavailableError(
                    "device=cuda was requested but torch reports no CUDA device"
                )
            return torch.device("cuda")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def executor(self):
        """Thread pool used to run blocking inference off the event loop."""
        if self._executor is None:
            from concurrent.futures import ThreadPoolExecutor

            self._executor = ThreadPoolExecutor(
                max_workers=self.settings.max_workers,
                thread_name_prefix="airavata",
            )
        return self._executor

    def available_variants(self) -> List[str]:
        return variants_for_device(self.device.type)

    def supports(self, name: str) -> bool:
        return supports_device(name, self.device.type)

    def shutdown(self) -> None:
        with self._lock:
            self.models.clear()
            self.tokenizer = None
            self._variant_locks.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload(self, name: str) -> bool:
        with self._lock:
            if name not in self.models:
                return False
            del self.models[name]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def _hub_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "cache_dir": str(self.settings.cache_dir),
            "trust_remote_code": self.settings.trust_remote_code,
        }
        if self.settings.hf_token:
            kwargs["token"] = self.settings.hf_token
        return kwargs

    def load_tokenizer(self):
        with self._lock:
            if self.tokenizer is None:
                logger.info("loading tokenizer for %s", self.settings.model_name)
                tokenizer = AutoTokenizer.from_pretrained(
                    self.settings.model_name, **self._hub_kwargs()
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                # Decoder-only models must be left padded or generation
                # continues from the padding instead of from the prompt.
                tokenizer.padding_side = "left"
                self.tokenizer = tokenizer
            return self.tokenizer

    def _variant_lock(self, name: str) -> threading.RLock:
        with self._lock:
            lock = self._variant_locks.get(name)
            if lock is None:
                lock = threading.RLock()
                self._variant_locks[name] = lock
            return lock

    def ensure_loaded(self, name: str) -> Any:
        """Return a loaded model, loading it on first use.

        Raises :class:`VariantUnavailableError` for a variant the current device
        cannot run, and :class:`ModelLoadError` when loading itself fails.
        """
        variant = get_variant(name)
        if not self.supports(name):
            raise VariantUnavailableError(
                f"variant {name!r} requires a {variant.requires_device} device; "
                f"this host is running on {self.device.type}"
            )

        # Fast path: dict lookups are atomic under the GIL, so a resident model
        # is returned without touching any lock.
        model = self.models.get(name)
        if model is not None:
            return model

        with self._variant_lock(name):
            model = self.models.get(name)
            if model is not None:
                return model

            self.load_tokenizer()
            try:
                model = self._load_variant(variant)
            except (ModelLoadError, VariantUnavailableError):
                # Already actionable (e.g. raised by a dependency load); do not
                # bury it under a second layer of wrapping.
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced as ModelLoadError
                logger.exception("failed to load variant %s", name)
                raise ModelLoadError(f"could not load {name!r}: {exc}") from exc

            model.eval()
            with self._lock:
                self.models[name] = model
            return model

    def _load_variant(self, variant: Variant) -> Any:
        if variant.name == ORIGINAL:
            return self._load_original()
        if variant.name == DYNAMIC_QUANT:
            return self._load_dynamic_quant()
        return self._load_bnb(variant)

    def _load_original(self) -> Any:
        logger.info("loading %s (original)", self.settings.model_name)
        on_cuda = self.device.type == "cuda"
        dtype = torch.float16 if on_cuda else torch.float32

        if on_cuda:
            try:
                return AutoModelForCausalLM.from_pretrained(
                    self.settings.model_name,
                    torch_dtype=dtype,
                    device_map="auto",
                    **self._hub_kwargs(),
                )
            except ImportError as exc:
                # device_map="auto" needs accelerate, but a single-GPU host does
                # not: load onto the CPU and move the whole model across.
                logger.warning(
                    "accelerate is unavailable (%s); falling back to a "
                    "single-device load. Install accelerate for multi-GPU "
                    "sharding and CPU offload.",
                    exc,
                )

        model = AutoModelForCausalLM.from_pretrained(
            self.settings.model_name, torch_dtype=dtype, **self._hub_kwargs()
        )
        return model.to(self.device)

    def _load_bnb(self, variant: Variant) -> Any:
        logger.info("loading %s (%s)", self.settings.model_name, variant.name)
        try:
            return AutoModelForCausalLM.from_pretrained(
                self.settings.model_name,
                quantization_config=build_bnb_config(variant),
                device_map="auto",
                **self._hub_kwargs(),
            )
        except ImportError as exc:
            # Unlike the FP baseline there is no fallback here: bitsandbytes
            # quantization is applied during the accelerate-driven load.
            raise ModelLoadError(
                f"{variant.name} needs accelerate and bitsandbytes: "
                f"pip install 'airavata-quant[gpu]' ({exc})"
            ) from exc

    def _load_dynamic_quant(self) -> Any:
        logger.info("applying dynamic quantization on CPU")
        # Route through ensure_loaded so the FP baseline is loaded exactly once
        # even when a request for `original` is in flight on another thread, and
        # so it stays resident for /benchmark/all to compare against.
        base = self.ensure_loaded(ORIGINAL)
        quantized = torch.ao.quantization.quantize_dynamic(
            base, {nn.Linear}, dtype=torch.qint8
        )

        converted = sum(
            isinstance(module, torch.ao.nn.quantized.dynamic.Linear)
            for module in quantized.modules()
        )
        before = model_memory_bytes(base)
        after = model_memory_bytes(quantized)
        if after >= before:
            # Found by benchmarking a real GPT-2 checkpoint. torch only
            # quantizes nn.Linear; GPT-2's attention and MLP are
            # `transformers.pytorch_utils.Conv1D`, so quantize_dynamic returns
            # an almost untouched copy and says nothing about it. The variant
            # would then be served, benchmarked and compared as if it had been
            # quantized - the measured footprint even grew by 4%, because
            # quantizing the tied lm_head un-ties it from the embedding.
            raise ModelLoadError(
                f"dynamic quantization did not shrink {self.settings.model_name!r} "
                f"({before / BYTES_PER_MB:.1f} MB -> {after / BYTES_PER_MB:.1f} MB "
                f"with {converted} nn.Linear layer(s) converted): torch's dynamic "
                "quantization only covers nn.Linear, and GPT-2 family models keep "
                "their weights in Conv1D. Use the int8/int4 GPU variants for this "
                "architecture"
            )
        logger.info(
            "dynamic quantization converted %d layers: %.1f MB -> %.1f MB",
            converted,
            before / BYTES_PER_MB,
            after / BYTES_PER_MB,
        )
        return quantized

    def preload(self, requested: Optional[Sequence[str]] = None) -> Dict[str, str]:
        """Load the configured variants up front.

        Returns a mapping of variant name to error message for the ones that
        failed; a failure never aborts startup because the remaining variants
        are still useful.
        """
        names = resolve_preload(
            list(requested if requested is not None else self.settings.preload),
            self.device.type,
        )
        errors: Dict[str, str] = {}
        if not names:
            return errors

        self.load_tokenizer()
        for name in names:
            try:
                self.ensure_loaded(name)
            except (VariantUnavailableError, ModelLoadError) as exc:
                errors[name] = str(exc)
                logger.warning("preload of %s failed: %s", name, exc)
        return errors

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    def _autocast(self, name: str):
        """Mixed precision only helps the FP baseline on CUDA.

        bitsandbytes variants already fix their own compute dtype, and
        autocasting them produces dtype mismatches inside the custom kernels.
        """
        if self.device.type == "cuda" and self.settings.use_amp and name == ORIGINAL:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _seeded(self, seed: Optional[int]):
        """Apply ``seed`` to the torch RNG without leaking it out of the call.

        ``torch.manual_seed`` mutates process-global state. Without this, a
        seeded request silently makes every *concurrent* unseeded request
        deterministic, and two overlapping seeded requests interleave their
        draws so neither is reproducible - the exact opposite of what asking
        for a seed means. Seeded generations are therefore serialised against
        each other; unseeded ones are untouched and stay fully parallel.
        """
        if seed is None:
            yield
            return
        devices = [self.device] if self.device.type == "cuda" else []
        with self._seed_lock:
            with torch.random.fork_rng(devices=devices, enabled=True):
                torch.manual_seed(seed)
                yield

    def _encode(self, prompt: str) -> Dict[str, torch.Tensor]:
        tokenizer = self.load_tokenizer()
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.settings.max_input_tokens,
        )
        inputs = {key: value.to(self.device) for key, value in encoded.items()}
        input_ids = inputs.get("input_ids")
        if input_ids is None or input_ids.numel() == 0:
            raise ValueError(
                "the prompt tokenized to zero tokens; provide text the "
                "tokenizer can encode"
            )
        return inputs

    def generate(self, model_type: str, request: GenerationRequest) -> GenerationResult:
        """Run generation and report timing measured around the model call."""
        model = self.ensure_loaded(model_type)
        tokenizer = self.load_tokenizer()

        if request.max_length > self.settings.max_new_tokens:
            raise ValueError(
                f"max_length {request.max_length} exceeds the configured limit of "
                f"{self.settings.max_new_tokens}"
            )
        if request.num_return_sequences > self.settings.max_return_sequences:
            raise ValueError(
                f"num_return_sequences {request.num_return_sequences} exceeds the "
                f"configured limit of {self.settings.max_return_sequences}"
            )

        inputs = self._encode(request.prompt)
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        # temperature == 0 means greedy; sampling with 0 raises in transformers.
        sampling = request.temperature > 0.0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": request.max_length,
            "num_return_sequences": request.num_return_sequences,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": sampling,
        }
        if sampling:
            gen_kwargs["temperature"] = request.temperature
            gen_kwargs["top_p"] = request.top_p

        start = time.perf_counter()
        with torch.inference_mode():
            with self._seeded(request.seed), self._autocast(model_type):
                outputs = model.generate(**inputs, **gen_kwargs)
        if self.device.type == "cuda":
            # generate() is synchronous, but be explicit so timings are not
            # skewed by queued kernels.
            torch.cuda.synchronize()
        inference_time = time.perf_counter() - start

        texts, generated_tokens = self._decode(outputs, prompt_tokens, tokenizer)
        tokens_per_second = generated_tokens / inference_time if inference_time > 0 else 0.0

        return GenerationResult(
            generated_text=texts,
            inference_time=inference_time,
            tokens_per_second=tokens_per_second,
            model_type=model_type,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            device=str(self.device),
        )

    @staticmethod
    def _count_generated(
        continuation: torch.Tensor, pad_id: Optional[int], eos_id: Optional[int]
    ) -> int:
        """Number of tokens the model actually produced in ``continuation``.

        Only the *trailing* run of padding is discarded. Counting every
        non-pad token instead (the obvious version) is wrong twice over: it
        drops pad-valued tokens the model legitimately emitted mid-sequence,
        and because ``pad_token`` is set to ``eos_token`` for the many
        checkpoints that ship without one, it also drops the terminating EOS
        the model spent a forward pass computing.
        """
        length = int(continuation.numel())
        if pad_id is None or length == 0:
            return length

        values = continuation.tolist()
        end = length
        while end > 0 and values[end - 1] == pad_id:
            end -= 1
        if end < length and eos_id == pad_id:
            # The first token of the trailing run was a real generated EOS.
            end += 1
        return end

    def _decode(
        self, outputs: torch.Tensor, prompt_tokens: int, tokenizer
    ) -> Tuple[List[str], int]:
        """Decode only the continuation.

        Slicing the *token* sequence rather than the decoded string avoids the
        classic off-by-a-few bug where special tokens or normalisation make the
        decoded prompt a different length than the prompt that went in.
        """
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id
        texts: List[str] = []
        generated_tokens = 0
        for sequence in outputs:
            continuation = sequence[prompt_tokens:]
            texts.append(
                tokenizer.decode(continuation, skip_special_tokens=True).strip()
            )
            generated_tokens += self._count_generated(continuation, pad_id, eos_id)
        return texts, generated_tokens

    # ------------------------------------------------------------------
    # benchmarking
    # ------------------------------------------------------------------
    def benchmark(
        self,
        model_type: str,
        iterations: int = 10,
        prompts: Optional[Iterable[str]] = None,
        max_new_tokens: int = 50,
        warmup: int = 1,
    ) -> Dict[str, Any]:
        """Time ``iterations`` passes over ``prompts`` and summarise them.

        A warmup pass is discarded: the first call pays for CUDA graph capture,
        kernel autotuning and lazy weight materialisation, which would otherwise
        dominate the average on short runs.
        """
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        model = self.ensure_loaded(model_type)
        prompt_list = list(prompts) if prompts is not None else list(DEFAULT_PROMPTS)
        if not prompt_list:
            raise ValueError("benchmark requires at least one prompt")

        def run(prompt: str) -> GenerationResult:
            return self.generate(
                model_type,
                GenerationRequest(
                    prompt=prompt, max_length=max_new_tokens, temperature=0.0
                ),
            )

        for _ in range(max(0, warmup)):
            run(prompt_list[0])

        on_cuda = self.device.type == "cuda"
        if on_cuda:
            # Peak allocation is only meaningful when measured over the run
            # itself; otherwise it reports whatever an earlier variant touched.
            torch.cuda.reset_peak_memory_stats()

        samples: List[LatencySample] = []
        for _ in range(iterations):
            for prompt in prompt_list:
                result = run(prompt)
                samples.append(
                    LatencySample(
                        latency=result["inference_time"],
                        tokens=result["generated_tokens"],
                    )
                )

        from . import hardware  # local import keeps psutil off the import path of pure modules

        stats = summarize(samples)
        return {
            "model_type": model_type,
            "iterations": iterations,
            **stats,
            # Weight footprint of *this* variant, unlike memory_usage below
            # which is whole-process and therefore not comparable across
            # variants loaded side by side.
            "model_memory_mb": model_memory_bytes(model) / BYTES_PER_MB,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated() / BYTES_PER_MB if on_cuda else None
            ),
            "memory_usage": hardware.memory_usage(self.device.type),
            "hardware_info": hardware.hardware_info(str(self.device)),
        }

    def benchmark_all(
        self,
        variants: Optional[Sequence[str]] = None,
        iterations: int = 3,
        prompts: Optional[Iterable[str]] = None,
        max_new_tokens: int = 50,
        free_after: bool = True,
    ) -> Dict[str, Any]:
        """Benchmark several variants, releasing the ones the sweep loaded.

        Without ``free_after`` the sweep holds every variant resident at once,
        so on a card that fits exactly one copy of the weights - the normal
        case for the model this serves - the second variant OOMs and the
        comparison the endpoint exists to produce never materialises.

        Variants that were already loaded before the sweep are left alone: they
        were somebody else's decision. A variant another pending variant is
        derived from is also kept, so the sweep does not pay to load it twice.
        """
        names = list(variants) if variants is not None else self.available_variants()
        prompt_list = list(prompts) if prompts is not None else None
        resident_before = set(self.models)

        results: Dict[str, Dict[str, Any]] = {}
        errors: Dict[str, str] = {}

        for position, name in enumerate(names):
            try:
                results[name] = self.benchmark(
                    name,
                    iterations=iterations,
                    prompts=prompt_list,
                    max_new_tokens=max_new_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - one variant failing is fine
                errors[name] = str(exc)
                logger.warning("benchmark of %s failed: %s", name, exc)

            if free_after:
                self._free_after_sweep(name, names[position + 1 :], resident_before)

        return {"results": results, "errors": errors, "comparison": compare(results)}

    def _free_after_sweep(
        self, name: str, pending: Sequence[str], resident_before: set
    ) -> None:
        """Unload variants the sweep itself brought in and nothing else needs."""
        needed_by_pending = {
            ORIGINAL
            for other in pending
            if other in VARIANTS and get_variant(other).derived_from_original
        }
        for candidate in list(self.models):
            if candidate in resident_before:
                continue  # somebody else loaded it; not ours to evict
            if candidate in pending or candidate in needed_by_pending:
                continue
            logger.debug("releasing %s after benchmarking %s", candidate, name)
            self.unload(candidate)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, model_type: str, destination: Optional[Path] = None) -> Dict[str, Any]:
        """Persist a loaded variant under ``quantized_model_path``.

        This is what the previously unused ``quantized_models/`` directory was
        for: exporting a quantized checkpoint so it can be reloaded without
        re-running quantization.
        """
        variant = get_variant(model_type)
        model = self.ensure_loaded(model_type)
        target = Path(destination or self.settings.quantized_model_path / model_type)
        target.mkdir(parents=True, exist_ok=True)

        if variant.serializable:
            model.save_pretrained(str(target))
            tokenizer = self.load_tokenizer()
            tokenizer.save_pretrained(str(target))
        else:
            # Dynamically quantized modules cannot round-trip through
            # save_pretrained, so persist the state dict instead.
            torch.save(model.state_dict(), target / "pytorch_model_quantized.pt")

        size_bytes = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        return {
            "model_type": model_type,
            "path": str(target.resolve()),
            "size_mb": size_bytes / BYTES_PER_MB,
        }
