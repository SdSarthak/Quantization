"""Quantization variant registry.

Deliberately free of torch/transformers imports: the presets are plain data so
they can be inspected, validated and unit tested without a GPU or a model
download. ``ModelManager`` turns them into real ``BitsAndBytesConfig`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

ORIGINAL = "original"
INT8 = "int8"
INT4 = "int4"
DYNAMIC_QUANT = "dynamic_quant"


@dataclass(frozen=True)
class Variant:
    """Metadata describing one way of loading the model."""

    name: str
    description: str
    #: ``"cuda"``, ``"cpu"`` or ``None`` when the variant runs anywhere.
    requires_device: Optional[str] = None
    #: Keyword arguments for ``BitsAndBytesConfig``. Values that name a torch
    #: dtype are written as strings (e.g. ``"float16"``) and resolved later.
    bnb_kwargs: Dict[str, Any] = field(default_factory=dict)
    #: Approximate bits per weight, used to report expected memory savings.
    bits: int = 16
    #: True when the variant is produced from an already loaded FP model.
    derived_from_original: bool = False
    #: True when ``save_pretrained`` can persist the variant to disk.
    serializable: bool = True


VARIANTS: Dict[str, Variant] = {
    ORIGINAL: Variant(
        name=ORIGINAL,
        description="Baseline model: FP16 on GPU, FP32 on CPU.",
        bits=16,
    ),
    INT8: Variant(
        name=INT8,
        description="8-bit weights via bitsandbytes LLM.int8(). GPU only.",
        requires_device="cuda",
        bnb_kwargs={
            "load_in_8bit": True,
            "bnb_8bit_compute_dtype": "float16",
        },
        bits=8,
    ),
    INT4: Variant(
        name=INT4,
        description="4-bit NF4 weights with double quantization. GPU only.",
        requires_device="cuda",
        bnb_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        },
        bits=4,
    ),
    DYNAMIC_QUANT: Variant(
        name=DYNAMIC_QUANT,
        description="torch dynamic quantization of nn.Linear to qint8. CPU only.",
        requires_device="cpu",
        bits=8,
        derived_from_original=True,
        # Dynamically quantized modules cannot be round-tripped through
        # save_pretrained; only the state dict is portable.
        serializable=False,
    ),
}

#: Variant names in the order they should be preloaded.
VARIANT_ORDER: List[str] = [ORIGINAL, INT8, INT4, DYNAMIC_QUANT]


class UnknownVariantError(KeyError):
    """Raised when a caller asks for a variant that does not exist."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"unknown variant {self.name!r}; available: {sorted(VARIANTS)}"


def get_variant(name: str) -> Variant:
    try:
        return VARIANTS[name]
    except KeyError:
        raise UnknownVariantError(name) from None


def supports_device(name: str, device_type: str) -> bool:
    """True when ``name`` can run on ``device_type`` (``"cpu"``/``"cuda"``)."""
    variant = get_variant(name)
    return variant.requires_device in (None, device_type)


def variants_for_device(device_type: str) -> List[str]:
    """Variant names that are usable on the given device, in load order."""
    return [name for name in VARIANT_ORDER if supports_device(name, device_type)]


def resolve_preload(requested: List[str], device_type: str) -> List[str]:
    """Expand a preload list into concrete, device-compatible variant names.

    ``"auto"`` and ``"all"`` expand to every variant the device supports.
    Unsupported or duplicated entries are dropped rather than raising, so a
    config shared between a CPU box and a GPU box still starts on both.
    """
    if not requested:
        return []

    expanded: List[str] = []
    for entry in requested:
        entry = entry.strip().lower()
        if not entry:
            continue
        if entry in ("auto", "all"):
            expanded.extend(variants_for_device(device_type))
        else:
            expanded.append(entry)

    resolved: List[str] = []
    for name in expanded:
        if name in VARIANTS and supports_device(name, device_type) and name not in resolved:
            resolved.append(name)
    return resolved


def memory_ratio(name: str) -> float:
    """Weight-memory footprint of a variant relative to the FP16 baseline."""
    return get_variant(name).bits / VARIANTS[ORIGINAL].bits
