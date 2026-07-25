"""
app/metrics/timer.py

Latency statistics.

Per-call latency is measured inside each agent and the baseline (using
time.perf_counter). This module provides descriptive statistics over a
collected list of latency measurements (in milliseconds).
"""

import statistics


def summarise_latency(latency_values: list) -> dict:
    """Compute descriptive statistics for a list of latency values (ms)."""
    values = [v for v in latency_values if v is not None]
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": 0.0, "median": 0.0, "stdev": 0.0,
                "min": 0.0, "max": 0.0}
    return {
        "n": n,
        "mean": round(statistics.mean(values), 1),
        "median": round(statistics.median(values), 1),
        "stdev": round(statistics.stdev(values), 1) if n > 1 else 0.0,
        "min": round(min(values), 1),
        "max": round(max(values), 1),
    }
