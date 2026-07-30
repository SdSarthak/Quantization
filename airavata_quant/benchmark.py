"""Benchmark sampling and statistics.

Pure Python (no numpy, no torch) so the aggregation logic is directly testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

DEFAULT_PROMPTS: List[str] = [
    "The future of artificial intelligence is",
    "In the context of machine learning, transformer models have revolutionized",
    (
        "Climate change is one of the most pressing issues of our time, requiring "
        "immediate action from governments and individuals alike to mitigate its "
        "effects on"
    ),
]


@dataclass(frozen=True)
class LatencySample:
    """One timed generation call."""

    latency: float
    tokens: int


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile. ``fraction`` is in ``[0, 1]``."""
    if not values:
        raise ValueError("percentile() requires at least one value")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")

    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])

    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize(samples: Sequence[LatencySample]) -> Dict[str, float]:
    """Aggregate raw samples into the numbers reported by ``/benchmark``.

    Throughput is total generated tokens over total wall time, which is the
    honest figure for a serial benchmark; averaging per-request token rates
    over-weights short requests.
    """
    if not samples:
        raise ValueError("summarize() requires at least one sample")

    latencies = [s.latency for s in samples]
    total_time = sum(latencies)
    total_tokens = sum(s.tokens for s in samples)

    return {
        "samples": len(samples),
        "avg_latency": total_time / len(samples),
        "p50_latency": percentile(latencies, 0.50),
        "p95_latency": percentile(latencies, 0.95),
        "throughput": (total_tokens / total_time) if total_time > 0 else 0.0,
        "total_tokens": total_tokens,
    }


def compare(results: Dict[str, Dict[str, float]], baseline: str = "original") -> Dict[str, Dict[str, float]]:
    """Speedup of every benchmarked variant relative to ``baseline``.

    Variants missing from ``results`` are skipped; if the baseline itself is
    absent an empty mapping is returned rather than raising, so ``/benchmark/all``
    still works on a box where the FP model did not fit.
    """
    base = results.get(baseline)
    if not base or not base.get("avg_latency"):
        return {}

    comparison: Dict[str, Dict[str, float]] = {}
    for name, stats in results.items():
        avg = stats.get("avg_latency") or 0.0
        throughput = stats.get("throughput") or 0.0
        comparison[name] = {
            "latency_speedup": (base["avg_latency"] / avg) if avg > 0 else 0.0,
            "throughput_ratio": (
                throughput / base["throughput"] if base.get("throughput") else 0.0
            ),
        }
    return comparison
