"""Telemetry probes.

The point of these is that a probe must never raise and must never invent a
number: monitoring endpoints run on hosts where psutil, nvidia-smi or the CUDA
driver behave in surprising ways.
"""

from airavata_quant import hardware


class _FakeGpu:
    def __init__(self, load=0.5, temperature=61.0):
        self.memoryUsed = 512.0
        self.memoryTotal = 4096.0
        self.load = load
        self.temperature = temperature


def _use_fake_gputil(monkeypatch, gpus):
    class FakeGPUtil:
        @staticmethod
        def getGPUs():
            if isinstance(gpus, Exception):
                raise gpus
            return gpus

    monkeypatch.setattr(hardware, "GPUtil", FakeGPUtil)
    monkeypatch.setattr(hardware, "GPUTIL_AVAILABLE", True)


def test_a_zero_reading_is_reported_not_discarded(monkeypatch):
    """``if gpu.temperature`` treated a genuine 0 C / idle GPU as "unknown"."""
    _use_fake_gputil(monkeypatch, [_FakeGpu(load=0.0, temperature=0.0)])
    stats = hardware._gputil_stats()
    assert stats["temperature_c"] == 0.0
    assert stats["utilization_percent"] == 0.0


def test_a_missing_load_reading_stays_none(monkeypatch):
    _use_fake_gputil(monkeypatch, [_FakeGpu(load=None, temperature=None)])
    stats = hardware._gputil_stats()
    assert stats["utilization_percent"] is None
    assert stats["temperature_c"] is None
    assert stats["memory_total_mb"] == 4096.0


def test_a_failing_gpu_probe_degrades_to_empty(monkeypatch):
    _use_fake_gputil(monkeypatch, RuntimeError("nvidia-smi not found"))
    assert hardware._gputil_stats() == {}


def test_no_gputil_means_no_stats(monkeypatch):
    monkeypatch.setattr(hardware, "GPUTIL_AVAILABLE", False)
    assert hardware._gputil_stats() == {}


def test_cpu_percent_is_primed_so_the_first_reading_is_real(monkeypatch):
    """``cpu_percent(interval=None)`` always answers 0.0 on its first call."""
    calls = []

    def fake_cpu_percent(interval=None):
        calls.append(interval)
        return 0.0 if len(calls) == 1 else 17.5

    monkeypatch.setattr(hardware, "_cpu_percent_primed", False)
    monkeypatch.setattr(hardware.psutil, "cpu_percent", fake_cpu_percent)

    first = hardware.cpu_info()
    assert first["usage_percent"] == 17.5
    assert calls[0] is None and calls[1] > 0

    second = hardware.cpu_info()
    assert len(calls) == 3  # no re-priming
    assert second["usage_percent"] == 17.5


def test_cpu_frequency_degrades_to_none_when_unsupported(monkeypatch):
    def boom():
        raise NotImplementedError("cpu_freq is unsupported in this container")

    monkeypatch.setattr(hardware.psutil, "cpu_freq", boom)
    assert hardware.cpu_frequency_mhz() is None


def test_memory_usage_omits_gpu_fields_on_cpu():
    payload = hardware.memory_usage("cpu")
    assert payload["ram_total_gb"] > 0
    assert "gpu_memory_used_mb" not in payload


def test_gpu_info_is_none_without_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert hardware.gpu_info() is None
    assert hardware.gpu_name() is None


def test_hardware_info_shape_on_cpu():
    payload = hardware.hardware_info("cpu")
    assert payload["device"] == "cpu"
    assert payload["torch_version"]
    assert "gpu_name" not in payload
