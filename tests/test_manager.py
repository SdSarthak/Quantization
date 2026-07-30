import pytest
import torch

from airavata_quant.config import Settings
from airavata_quant.manager import (
    ModelLoadError,
    ModelManager,
    VariantUnavailableError,
    build_bnb_config,
)
from airavata_quant.quantization import INT4, INT8, UnknownVariantError, get_variant
from airavata_quant.schemas import GenerationRequest

from .conftest import PAD_ID, PROMPT_IDS, StubManager


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
