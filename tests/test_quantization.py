import pytest

from airavata_quant.quantization import (
    DYNAMIC_QUANT,
    INT4,
    INT8,
    ORIGINAL,
    UnknownVariantError,
    get_variant,
    memory_ratio,
    resolve_preload,
    supports_device,
    variants_for_device,
)


def test_unknown_variant_reports_the_available_names():
    with pytest.raises(UnknownVariantError) as excinfo:
        get_variant("int2")
    assert excinfo.value.name == "int2"
    assert "int4" in str(excinfo.value)


def test_device_requirements():
    assert supports_device(ORIGINAL, "cpu")
    assert supports_device(ORIGINAL, "cuda")
    assert supports_device(INT8, "cuda")
    assert not supports_device(INT8, "cpu")
    assert supports_device(DYNAMIC_QUANT, "cpu")
    assert not supports_device(DYNAMIC_QUANT, "cuda")


def test_variants_for_device_are_device_appropriate():
    assert variants_for_device("cpu") == [ORIGINAL, DYNAMIC_QUANT]
    assert variants_for_device("cuda") == [ORIGINAL, INT8, INT4]


def test_resolve_preload_expands_auto():
    assert resolve_preload(["auto"], "cuda") == [ORIGINAL, INT8, INT4]
    assert resolve_preload(["all"], "cpu") == [ORIGINAL, DYNAMIC_QUANT]


def test_resolve_preload_drops_unsupported_and_unknown_entries():
    # A single config shared between a CPU box and a GPU box must still start.
    assert resolve_preload(["original", "int4", "nonsense"], "cpu") == [ORIGINAL]


def test_resolve_preload_deduplicates_and_preserves_order():
    assert resolve_preload(["int4", "original", "int4"], "cuda") == [INT4, ORIGINAL]


def test_resolve_preload_empty_means_lazy():
    assert resolve_preload([], "cuda") == []


def test_memory_ratio_matches_bit_width():
    assert memory_ratio(ORIGINAL) == 1.0
    assert memory_ratio(INT8) == 0.5
    assert memory_ratio(INT4) == 0.25


def test_memory_ratio_uses_the_baseline_of_the_device_the_variant_runs_on():
    """The CPU baseline is FP32, not FP16.

    Reporting ``dynamic_quant`` as 0.5 understated the saving by 2x: int8
    against an FP32 baseline is a quarter of the weights, which is what the
    measured footprint in the benchmark shows.
    """
    assert memory_ratio(DYNAMIC_QUANT, "cpu") == 0.25
    assert memory_ratio(DYNAMIC_QUANT) == 0.25  # cpu-only, so device-independent

    # GPU-only variants always compare against FP16, even when inspected from
    # a CPU host where they cannot run.
    assert memory_ratio(INT4, "cpu") == 0.25
    assert memory_ratio(INT8, "cuda") == 0.5

    assert memory_ratio(ORIGINAL, "cpu") == 1.0
    assert memory_ratio(ORIGINAL, "cuda") == 1.0


def test_dynamic_quant_is_flagged_as_not_serializable():
    assert get_variant(DYNAMIC_QUANT).serializable is False
    assert get_variant(DYNAMIC_QUANT).derived_from_original is True
    assert get_variant(INT4).serializable is True
