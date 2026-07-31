import pytest
import torch
import torch.nn as nn

from airavata_quant.config import Settings
from airavata_quant.manager import (
    ModelLoadError,
    ModelManager,
    VariantUnavailableError,
    build_bnb_config,
    model_memory_bytes,
)
from airavata_quant.quantization import INT4, INT8, UnknownVariantError, get_variant
from airavata_quant.schemas import GenerationRequest

from .conftest import EOS_ID, PAD_ID, PROMPT_IDS, FakeModel, FakeTokenizer, StubManager


def test_bnb_config_resolves_string_dtypes_to_torch_dtypes():
    config = build_bnb_config(get_variant(INT4))
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True
    assert config.bnb_4bit_compute_dtype is torch.float16

    config8 = build_bnb_config(get_variant(INT8))
    assert config8.load_in_8bit is True


def test_original_variant_has_no_bnb_config():
    assert build_bnb_config(get_variant("original")) is None


def test_lazy_loading_happens_once_per_variant(manager):
    first = manager.ensure_loaded("original")
    second = manager.ensure_loaded("original")
    assert first is second
    assert manager.loaded_variants == ["original"]


def test_unknown_variant_raises(manager):
    with pytest.raises(UnknownVariantError):
        manager.ensure_loaded("int2")


def test_gpu_only_variant_is_rejected_on_cpu(manager):
    with pytest.raises(VariantUnavailableError) as excinfo:
        manager.ensure_loaded("int8")
    assert "cpu" in str(excinfo.value)


def test_load_failures_are_wrapped(settings):
    class Broken(StubManager):
        def _load_variant(self, variant):
            raise OSError("no checkpoint")

    broken = Broken(settings)
    with pytest.raises(ModelLoadError) as excinfo:
        broken.ensure_loaded("original")
    assert "no checkpoint" in str(excinfo.value)
    broken.shutdown()


def test_requesting_cuda_without_a_device_fails_fast(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(VariantUnavailableError):
        ModelManager(Settings(device="cuda"))


def test_generate_returns_only_the_continuation(manager):
    result = manager.generate(
        "original", GenerationRequest(prompt="hello", max_length=4, temperature=0.0)
    )
    assert result["prompt_tokens"] == len(PROMPT_IDS)
    assert result["generated_tokens"] == 4
    # The fake model emits tokens 10..13 after echoing the prompt; the prompt
    # tokens must not leak into the decoded text.
    assert result["generated_text"] == ["10 11 12 13"]
    assert result["inference_time"] > 0
    assert result["tokens_per_second"] > 0
    assert result["model_type"] == "original"


def test_generate_excludes_padding_from_the_token_count(settings):
    padded = StubManager(settings, pad_tail=2)
    result = padded.generate(
        "original", GenerationRequest(prompt="hello", max_length=5, temperature=0.0)
    )
    assert result["generated_tokens"] == 3
    assert str(PAD_ID) not in result["generated_text"][0].split()
    padded.shutdown()


def test_a_trailing_eos_counts_as_a_generated_token_when_pad_is_eos(manager):
    """Checkpoints without a pad token reuse EOS as pad.

    Counting "every token that is not the pad id" therefore threw away the
    terminating EOS the model really did generate, under-reporting throughput
    by one token per sequence.
    """
    count = ModelManager._count_generated(
        torch.tensor([10, 11, EOS_ID, EOS_ID, EOS_ID]), pad_id=EOS_ID, eos_id=EOS_ID
    )
    assert count == 3


def test_pad_valued_tokens_inside_the_continuation_are_counted(manager):
    count = ModelManager._count_generated(
        torch.tensor([10, PAD_ID, 11, PAD_ID, PAD_ID]), pad_id=PAD_ID, eos_id=EOS_ID
    )
    assert count == 3


def test_token_count_without_a_pad_token_uses_the_full_length():
    assert (
        ModelManager._count_generated(
            torch.tensor([1, 2, 3]), pad_id=None, eos_id=None
        )
        == 3
    )
    assert ModelManager._count_generated(torch.tensor([], dtype=torch.long), 0, 0) == 0


def test_generate_rejects_a_prompt_that_tokenizes_to_nothing(settings):
    class EmptyTokenizer(FakeTokenizer):
        def __call__(self, prompt, **kwargs):
            ids = torch.zeros((1, 0), dtype=torch.long)
            return {"input_ids": ids, "attention_mask": ids}

    class EmptyManager(StubManager):
        def load_tokenizer(self):
            if self.tokenizer is None:
                self.tokenizer = EmptyTokenizer()
            return self.tokenizer

    instance = EmptyManager(settings)
    with pytest.raises(ValueError, match="zero tokens"):
        instance.generate("original", GenerationRequest(prompt="​", max_length=2))
    instance.shutdown()


class _RandomModel(FakeModel):
    """Consumes the global RNG so seeding is observable."""

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.draws.append(float(torch.rand(1)))
        return super().generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def __init__(self, pad_tail: int = 0) -> None:
        super().__init__(pad_tail=pad_tail)
        self.draws = []


def _random_manager(settings):
    class RandomManager(StubManager):
        def _load_variant(self, variant):
            self.loaded_variants.append(variant.name)
            return _RandomModel()

    return RandomManager(settings)


def test_the_same_seed_reproduces_the_same_draws(settings):
    instance = _random_manager(settings)
    for _ in range(2):
        instance.generate(
            "original", GenerationRequest(prompt="x", max_length=2, seed=1234)
        )
    draws = instance.models["original"].draws
    assert draws[0] == draws[1]
    instance.shutdown()


def test_a_seeded_request_does_not_hijack_the_global_rng(settings):
    """A seed must scope to its own request.

    ``torch.manual_seed`` is process-global: without isolation one seeded
    request silently pins every concurrent unseeded request to the same
    stream, which is both wrong and a reproducibility trap.
    """
    instance = _random_manager(settings)
    instance.ensure_loaded("original")  # module init also draws; get it out of the way

    torch.manual_seed(999)
    before = float(torch.rand(1))

    torch.manual_seed(999)
    instance.generate("original", GenerationRequest(prompt="x", max_length=2, seed=7))
    after = float(torch.rand(1))

    assert before == after
    instance.shutdown()


def test_greedy_decoding_omits_sampling_parameters(manager):
    manager.generate(
        "original", GenerationRequest(prompt="x", max_length=2, temperature=0.0)
    )
    kwargs = manager.models["original"].generate_calls[-1]
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    assert kwargs["max_new_tokens"] == 2


def test_sampling_parameters_are_forwarded(manager):
    manager.generate(
        "original",
        GenerationRequest(prompt="x", max_length=2, temperature=0.8, top_p=0.5),
    )
    kwargs = manager.models["original"].generate_calls[-1]
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == pytest.approx(0.8)
    assert kwargs["top_p"] == pytest.approx(0.5)


def test_multiple_return_sequences(manager):
    result = manager.generate(
        "original",
        GenerationRequest(
            prompt="x", max_length=3, temperature=0.0, num_return_sequences=2
        ),
    )
    assert len(result["generated_text"]) == 2
    assert result["generated_tokens"] == 6


def test_generate_enforces_the_configured_token_ceiling(settings):
    settings.max_new_tokens = 8
    strict = StubManager(settings)
    with pytest.raises(ValueError, match="exceeds the configured limit"):
        strict.generate(
            "original", GenerationRequest(prompt="x", max_length=50, temperature=0.0)
        )
    strict.shutdown()


def test_generate_enforces_the_sequence_ceiling(settings):
    settings.max_return_sequences = 1
    strict = StubManager(settings)
    with pytest.raises(ValueError, match="num_return_sequences"):
        strict.generate(
            "original",
            GenerationRequest(prompt="x", max_length=2, num_return_sequences=3),
        )
    strict.shutdown()


def test_benchmark_summarises_all_prompts(manager):
    stats = manager.benchmark(
        "original", iterations=2, prompts=["a", "b"], max_new_tokens=4, warmup=0
    )
    assert stats["model_type"] == "original"
    assert stats["iterations"] == 2
    assert stats["samples"] == 4
    assert stats["total_tokens"] == 16
    assert stats["avg_latency"] > 0
    assert stats["p95_latency"] >= stats["p50_latency"]
    assert "ram_total_gb" in stats["memory_usage"]
    assert stats["hardware_info"]["device"] == "cpu"


def test_benchmark_warmup_calls_are_not_counted(manager):
    manager.benchmark(
        "original", iterations=1, prompts=["a"], max_new_tokens=2, warmup=2
    )
    # 2 warmups + 1 measured call all reach the model, but only one is scored.
    assert len(manager.models["original"].generate_calls) == 3


def test_benchmark_rejects_bad_arguments(manager):
    with pytest.raises(ValueError):
        manager.benchmark("original", iterations=0)
    with pytest.raises(ValueError):
        manager.benchmark("original", prompts=[])


def test_model_memory_counts_dynamically_quantized_weights():
    """A qint8 model must not be reported as occupying nothing.

    ``quantize_dynamic`` moves the weight into a packed script object, so it
    appears in neither ``parameters()`` nor ``buffers()``; a naive walk answers
    0 bytes and the whole point of the benchmark - showing the memory saving -
    silently evaporates.
    """
    fp32 = nn.Sequential(nn.Linear(128, 128), nn.Linear(128, 64))
    quantized = torch.ao.quantization.quantize_dynamic(
        fp32, {nn.Linear}, dtype=torch.qint8
    )

    fp32_bytes = model_memory_bytes(fp32)
    quantized_bytes = model_memory_bytes(quantized)

    assert fp32_bytes == pytest.approx((128 * 128 + 128 * 64) * 4 + (128 + 64) * 4)
    assert quantized_bytes > 0
    # int8 weights with fp32 biases: roughly a quarter of the fp32 footprint.
    assert 0.2 < quantized_bytes / fp32_bytes < 0.4


def test_benchmark_reports_the_variant_weight_footprint(manager):
    stats = manager.benchmark(
        "original", iterations=1, prompts=["a"], max_new_tokens=2, warmup=0
    )
    expected = model_memory_bytes(manager.models["original"]) / (1024 * 1024)
    assert stats["model_memory_mb"] == pytest.approx(expected)
    assert stats["model_memory_mb"] > 0
    assert stats["peak_gpu_memory_mb"] is None  # CPU run


def test_dynamic_quant_reuses_the_already_loaded_baseline(settings):
    """Exercises the real quantization path, not the stubbed loader."""

    class RealQuantManager(ModelManager):
        def __init__(self, s):
            super().__init__(s)
            self.original_loads = 0

        def load_tokenizer(self):
            if self.tokenizer is None:
                self.tokenizer = FakeTokenizer()
            return self.tokenizer

        def _load_original(self):
            self.original_loads += 1
            return nn.Sequential(nn.Linear(32, 32))

    instance = RealQuantManager(settings)
    quantized = instance.ensure_loaded("dynamic_quant")

    assert instance.original_loads == 1
    assert "original" in instance.models  # the FP baseline stays for comparison
    assert isinstance(quantized[0], torch.ao.nn.quantized.dynamic.Linear)
    assert model_memory_bytes(quantized) < model_memory_bytes(instance.models["original"])

    # A second variant request must not re-load the baseline.
    instance.ensure_loaded("original")
    assert instance.original_loads == 1
    instance.shutdown()


class _Conv1DLike(nn.Module):
    """Weights in a bare Parameter, like transformers' Conv1D blocks."""

    def __init__(self, size: int = 64):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size, size))


def _manager_over(settings, module_factory):
    class Fixed(ModelManager):
        def load_tokenizer(self):
            if self.tokenizer is None:
                self.tokenizer = FakeTokenizer()
            return self.tokenizer

        def _load_original(self):
            return module_factory()

    return Fixed(settings)


def test_dynamic_quant_refuses_an_architecture_it_cannot_shrink(settings):
    """A silent no-op is worse than an error.

    ``quantize_dynamic`` only converts ``nn.Linear``. GPT-2 style models keep
    their weights in ``Conv1D``, so it hands back an all-but-untouched copy
    and says nothing - and the variant is then served, benchmarked and
    compared as though it had been quantized.
    """
    instance = _manager_over(settings, lambda: nn.Sequential(_Conv1DLike()))
    with pytest.raises(ModelLoadError) as excinfo:
        instance.ensure_loaded("dynamic_quant")
    message = str(excinfo.value)
    assert "did not shrink" in message
    assert "0 nn.Linear layer(s) converted" in message
    instance.shutdown()


def test_dynamic_quant_accepts_an_architecture_it_does_shrink(settings):
    instance = _manager_over(settings, lambda: nn.Sequential(nn.Linear(64, 64)))
    quantized = instance.ensure_loaded("dynamic_quant")
    assert model_memory_bytes(quantized) < model_memory_bytes(
        instance.models["original"]
    )
    instance.shutdown()


def test_sweep_releases_the_variants_it_loaded(manager):
    """The sweep must fit on a device that holds one copy of the weights."""
    sweep = manager.benchmark_all(iterations=1, prompts=["a"], max_new_tokens=2)

    assert set(sweep["results"]) == {"original", "dynamic_quant"}
    assert sweep["errors"] == {}
    assert sweep["comparison"]["original"]["latency_speedup"] == pytest.approx(1.0)
    assert manager.models == {}, "the sweep left variants resident"


def test_sweep_keeps_variants_that_were_already_loaded(manager):
    manager.ensure_loaded("original")
    manager.benchmark_all(iterations=1, prompts=["a"], max_new_tokens=2)
    # `original` was somebody else's decision; only sweep-loaded ones go.
    assert set(manager.models) == {"original"}


def test_sweep_keeps_the_baseline_while_a_derived_variant_is_pending(manager):
    manager.benchmark_all(
        variants=["original", "dynamic_quant"],
        iterations=1,
        prompts=["a"],
        max_new_tokens=2,
    )
    # dynamic_quant is derived from original, so the baseline is loaded once.
    assert manager.loaded_variants == ["original", "dynamic_quant"]


def test_sweep_can_be_asked_to_keep_everything(manager):
    manager.benchmark_all(
        iterations=1, prompts=["a"], max_new_tokens=2, free_after=False
    )
    assert set(manager.models) == {"original", "dynamic_quant"}


def test_sweep_records_a_failing_variant_without_aborting(settings):
    class HalfBroken(StubManager):
        def _load_variant(self, variant):
            if variant.name == "dynamic_quant":
                raise RuntimeError("out of memory")
            return super()._load_variant(variant)

    instance = HalfBroken(settings)
    sweep = instance.benchmark_all(iterations=1, prompts=["a"], max_new_tokens=2)
    assert set(sweep["results"]) == {"original"}
    assert "out of memory" in sweep["errors"]["dynamic_quant"]
    instance.shutdown()


def test_sweep_rejects_an_unknown_variant_as_an_error_entry(manager):
    sweep = manager.benchmark_all(
        variants=["original", "int2"], iterations=1, prompts=["a"], max_new_tokens=2
    )
    assert "int2" in sweep["errors"]
    assert "original" in sweep["results"]


def test_preload_reports_failures_without_raising(settings):
    class PartiallyBroken(StubManager):
        def _load_variant(self, variant):
            if variant.name == "dynamic_quant":
                raise RuntimeError("quantization backend missing")
            return super()._load_variant(variant)

    partial = PartiallyBroken(settings)
    errors = partial.preload(["auto"])
    assert "original" in partial.models
    assert "dynamic_quant" in errors
    assert "quantization backend missing" in errors["dynamic_quant"]
    partial.shutdown()


def test_preload_ignores_variants_the_device_cannot_run(manager):
    assert manager.preload(["int8", "int4"]) == {}
    assert manager.models == {}


def test_save_writes_a_serializable_variant_to_disk(manager, settings):
    result = manager.save("original")
    target = settings.quantized_model_path / "original"
    assert target.is_dir()
    assert (target / "model.bin").exists()
    assert result["size_mb"] > 0
    assert result["path"].endswith("original")


def test_save_uses_a_state_dict_for_dynamic_quantization(manager, settings):
    result = manager.save("dynamic_quant")
    target = settings.quantized_model_path / "dynamic_quant"
    assert (target / "pytorch_model_quantized.pt").exists()
    assert result["model_type"] == "dynamic_quant"


def test_unload_frees_a_variant(manager):
    manager.ensure_loaded("original")
    assert manager.unload("original") is True
    assert manager.unload("original") is False
    assert "original" not in manager.models


def test_shutdown_clears_state(manager):
    manager.ensure_loaded("original")
    manager.shutdown()
    assert manager.models == {}
    assert manager.tokenizer is None


class _NoDeviceModel:
    """Stand-in for a model loaded without accelerate's device_map."""

    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        return self


def _fake_auto_class(fail_on_device_map: bool):
    class FakeAuto:
        calls = []

        @staticmethod
        def from_pretrained(name, **kwargs):
            FakeAuto.calls.append(kwargs)
            if fail_on_device_map and "device_map" in kwargs:
                raise ImportError(
                    "Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate"
                )
            return _NoDeviceModel()

    return FakeAuto


def _cuda_manager(monkeypatch, settings, fake_auto):
    import airavata_quant.manager as manager_module

    from .conftest import FakeTokenizer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(manager_module, "AutoModelForCausalLM", fake_auto)
    settings.device = "cuda"
    instance = ModelManager(settings)
    monkeypatch.setattr(instance, "load_tokenizer", lambda: FakeTokenizer())
    return instance


def test_original_falls_back_to_a_single_device_load_without_accelerate(
    monkeypatch, settings
):
    fake_auto = _fake_auto_class(fail_on_device_map=True)
    instance = _cuda_manager(monkeypatch, settings, fake_auto)

    model = instance.ensure_loaded("original")

    assert len(fake_auto.calls) == 2
    assert fake_auto.calls[0]["device_map"] == "auto"
    assert "device_map" not in fake_auto.calls[1]
    assert model.moved_to.type == "cuda"


def test_original_prefers_device_map_when_accelerate_is_present(monkeypatch, settings):
    fake_auto = _fake_auto_class(fail_on_device_map=False)
    instance = _cuda_manager(monkeypatch, settings, fake_auto)

    instance.ensure_loaded("original")

    assert len(fake_auto.calls) == 1
    assert fake_auto.calls[0]["device_map"] == "auto"


def test_quantized_variants_report_an_actionable_missing_dependency_error(
    monkeypatch, settings
):
    fake_auto = _fake_auto_class(fail_on_device_map=True)
    instance = _cuda_manager(monkeypatch, settings, fake_auto)

    with pytest.raises(ModelLoadError) as excinfo:
        instance.ensure_loaded("int4")
    assert "airavata-quant[gpu]" in str(excinfo.value)
