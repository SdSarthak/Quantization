"""Host and accelerator telemetry.

Every probe degrades to ``None`` rather than raising: monitoring endpoints must
never take the service down, and ``psutil``/``GPUtil`` behave differently across
Windows, containers and virtualised hosts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import psutil
import torch

logger = logging.getLogger(__name__)

try:  # pragma: no cover - depends on host packages
    import GPUtil

    GPUTIL_AVAILABLE = True
except Exception:  # noqa: BLE001 - GPUtil raises non-ImportError on odd hosts
    GPUtil = None
    GPUTIL_AVAILABLE = False

BYTES_PER_MB = 1024 * 1024
BYTES_PER_GB = 1024**3


def cpu_frequency_mhz() -> Optional[float]:
    try:
        freq = psutil.cpu_freq()
    except Exception:  # noqa: BLE001 - unsupported on some VMs/containers
        return None
    return float(freq.current) if freq else None


_cpu_percent_primed = False


def cpu_info(sample_interval: float = 0.0) -> Dict[str, Optional[float]]:
    """CPU counters. ``sample_interval`` of 0 returns a non-blocking reading.

    ``psutil.cpu_percent(interval=None)`` is differential: the *first* call in a
    process has no previous sample to diff against and always answers ``0.0``.
    Priming it once means the first ``/system/info`` request reports real load
    instead of a permanently idle CPU.
    """
    global _cpu_percent_primed
    if not _cpu_percent_primed and not sample_interval:
        psutil.cpu_percent(interval=None)
        _cpu_percent_primed = True
        usage = psutil.cpu_percent(interval=0.05)
    else:
        usage = psutil.cpu_percent(interval=sample_interval or None)
    return {
        "count": psutil.cpu_count(),
        "usage_percent": usage,
        "frequency_mhz": cpu_frequency_mhz(),
    }


def memory_info() -> Dict[str, float]:
    virtual = psutil.virtual_memory()
    return {
        "total_gb": virtual.total / BYTES_PER_GB,
        "available_gb": virtual.available / BYTES_PER_GB,
        "used_gb": virtual.used / BYTES_PER_GB,
        "used_percent": float(virtual.percent),
    }


def _gputil_stats() -> Dict[str, Optional[float]]:
    if not GPUTIL_AVAILABLE:
        return {}
    try:
        gpus = GPUtil.getGPUs()
    except Exception:  # noqa: BLE001 - nvidia-smi missing or unparseable
        logger.debug("GPUtil probe failed", exc_info=True)
        return {}
    if not gpus:
        return {}
    gpu = gpus[0]
    # `if gpu.temperature` would discard a genuine 0 C reading, and `gpu.load`
    # is None rather than 0.0 when the driver does not report utilisation.
    return {
        "memory_used_mb": float(gpu.memoryUsed),
        "memory_total_mb": float(gpu.memoryTotal),
        "utilization_percent": (
            float(gpu.load) * 100.0 if gpu.load is not None else None
        ),
        "temperature_c": (
            float(gpu.temperature) if gpu.temperature is not None else None
        ),
    }


def gpu_info(index: int = 0) -> Optional[Dict[str, Optional[float]]]:
    """GPU counters, or ``None`` when no CUDA device is present."""
    if not torch.cuda.is_available():
        return None

    try:
        properties = torch.cuda.get_device_properties(index)
        total_mb = properties.total_memory / BYTES_PER_MB
    except Exception:  # noqa: BLE001 - driver hiccup
        total_mb = None

    stats: Dict[str, Optional[float]] = {
        "memory_used_mb": torch.cuda.memory_allocated(index) / BYTES_PER_MB,
        "memory_reserved_mb": torch.cuda.memory_reserved(index) / BYTES_PER_MB,
        "memory_total_mb": total_mb,
        "utilization_percent": None,
        "temperature_c": None,
    }
    # GPUtil reports process-wide usage which is more useful than the
    # torch-allocator view, so it wins when available.
    stats.update({k: v for k, v in _gputil_stats().items() if v is not None})
    return stats


def gpu_name(index: int = 0) -> Optional[str]:
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_name(index)
    except Exception:  # noqa: BLE001
        return None


def memory_usage(device_type: str) -> Dict[str, Any]:
    """Memory payload shaped for :class:`~airavata_quant.schemas.MemoryUsage`."""
    memory = memory_info()
    payload: Dict[str, Any] = {
        "ram_used_gb": memory["used_gb"],
        "ram_total_gb": memory["total_gb"],
        "ram_used_percent": memory["used_percent"],
    }
    if device_type == "cuda":
        gpu = gpu_info() or {}
        payload.update(
            {
                "gpu_memory_used_mb": gpu.get("memory_used_mb"),
                "gpu_memory_total_mb": gpu.get("memory_total_mb"),
                "gpu_utilization_percent": gpu.get("utilization_percent"),
                "gpu_temperature_c": gpu.get("temperature_c"),
            }
        )
    return payload


def hardware_info(device: str) -> Dict[str, Any]:
    """Hardware payload shaped for :class:`~airavata_quant.schemas.HardwareInfo`."""
    payload: Dict[str, Any] = {
        "device": device,
        "torch_version": torch.__version__,
        "cpu_count": psutil.cpu_count(),
        "cpu_freq_mhz": cpu_frequency_mhz(),
    }
    if device.startswith("cuda"):
        payload["gpu_name"] = gpu_name()
        payload["cuda_version"] = torch.version.cuda
    return payload
