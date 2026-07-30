import pytest

from airavata_quant.benchmark import LatencySample, compare, percentile, summarize


def test_percentile_endpoints_and_interpolation():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 1.0) == 4.0
    assert percentile(values, 0.5) == pytest.approx(2.5)


def test_percentile_of_single_value():
    assert percentile([7.0], 0.95) == 7.0


def test_percentile_rejects_empty_input_and_bad_fraction():
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile([1.0], 1.5)


def test_summarize_uses_aggregate_throughput():
    samples = [
        LatencySample(latency=1.0, tokens=10),
        LatencySample(latency=3.0, tokens=30),
    ]
    stats = summarize(samples)
    assert stats["samples"] == 2
    assert stats["avg_latency"] == pytest.approx(2.0)
    assert stats["total_tokens"] == 40
    # 40 tokens over 4 seconds, not the mean of the two per-request rates.
    assert stats["throughput"] == pytest.approx(10.0)


def test_summarize_handles_zero_duration_without_dividing_by_zero():
    stats = summarize([LatencySample(latency=0.0, tokens=5)])
    assert stats["throughput"] == 0.0


def test_summarize_rejects_empty_samples():
    with pytest.raises(ValueError):
        summarize([])


def test_compare_reports_speedup_against_the_baseline():
    results = {
        "original": {"avg_latency": 1.0, "throughput": 100.0},
        "int4": {"avg_latency": 0.25, "throughput": 400.0},
    }
    comparison = compare(results)
    assert comparison["int4"]["latency_speedup"] == pytest.approx(4.0)
    assert comparison["int4"]["throughput_ratio"] == pytest.approx(4.0)
    assert comparison["original"]["latency_speedup"] == pytest.approx(1.0)


def test_compare_returns_empty_when_the_baseline_is_missing():
    assert compare({"int4": {"avg_latency": 0.5, "throughput": 1.0}}) == {}
