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

from .benchmark import DEFAULT_PROMPTS, LatencySample, summarize
from .config import Settings
from .quantization import (
    DYNAMIC_QUANT,
    ORIGINAL,
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


class ModelManager:
    """Owns the tokenizer and every loaded model variant.

    Loading is lazy and guarded by a lock so two concurrent requests for the
    same variant do not both pull a multi-gigabyte checkpoint into memory.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or Settings()
        self.models: Dict[str, Any] = {}
        self.tokenizer = None
        self.device = self._resolve_device(self.settings.device)
        self._lock = threading.RLock()
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

        with self._lock:
            model = self.models.get(name)
            if model is not None:
                return model

            self.load_tokenizer()
            try:
                model = self._load_variant(variant)
            except Exception as exc:  # noqa: BLE001 - surfaced as ModelLoadError
                logger.exception("failed to load variant %s", name)
                raise ModelLoadError(f"could not load {name!r}: {exc}") from exc

            model.eval()
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
        base = self.models.get(ORIGINAL)
        if base is None:
            base = self._load_original()
            base.eval()
            # Keep the FP baseline around so /benchmark/all can compare the
            # two; it was already paid for.
            self.models[ORIGINAL] = base
        return torch.ao.quantization.quantize_dynamic(
            base, {nn.Linear}, dtype=torch.qint8
        )

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

    def _encode(self, prompt: str) -> Dict[str, torch.Tensor]:
        tokenizer = self.load_tokenizer()
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.settings.max_input_tokens,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

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

        if request.seed is not None:
            torch.manual_seed(request.seed)

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
            with self._autocast(model_type):
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

    def _decode(
        self, outputs: torch.Tensor, prompt_tokens: int, tokenizer
    ) -> Tuple[List[str], int]:
        """Decode only the continuation.

        Slicing the *token* sequence rather than the decoded string avoids the
        classic off-by-a-few bug where special tokens or normalisation make the
        decoded prompt a different length than the prompt that went in.
        """
        pad_id = tokenizer.pad_token_id
        texts: List[str] = []
        generated_tokens = 0
        for sequence in outputs:
            continuation = sequence[prompt_tokens:]
            texts.append(
                tokenizer.decode(continuation, skip_special_tokens=True).strip()
            )
            if pad_id is None:
                generated_tokens += int(continuation.numel())
            else:
                generated_tokens += int((continuation != pad_id).sum().item())
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

        self.ensure_loaded(model_type)
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
            "memory_usage": hardware.memory_usage(self.device.type),
            "hardware_info": hardware.hardware_info(str(self.device)),
        }

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
