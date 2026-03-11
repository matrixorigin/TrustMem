"""Memory metrics — plain class, no singleton. Pass instances via DI."""

from __future__ import annotations

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MetricStats:
    """Statistics for a single metric."""

    count: int = 0
    total: float = 0.0
    min_val: float = float("inf")
    max_val: float = float("-inf")

    def record(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "total": self.total,
            "avg": self.avg,
            "min": self.min_val if self.count > 0 else 0,
            "max": self.max_val if self.count > 0 else 0,
        }


class MemoryMetrics:
    """Thread-safe metrics collector for memory operations. No singleton."""

    def __init__(self) -> None:
        self._metrics: dict[str, MetricStats] = defaultdict(MetricStats)
        self._counters: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record_latency(self, operation: str, latency_ms: float) -> None:
        with self._lock:
            self._metrics[f"{operation}_latency_ms"].record(latency_ms)

    def increment(self, counter: str, value: int = 1) -> None:
        with self._lock:
            self._counters[counter] += value

    def get_stats(self) -> dict:
        with self._lock:
            return {
                "latencies": {k: v.to_dict() for k, v in self._metrics.items()},
                "counters": dict(self._counters),
            }

    def reset(self) -> None:
        with self._lock:
            self._metrics.clear()
            self._counters.clear()


class Timer:
    """Context manager for timing operations."""

    def __init__(self, operation: str, metrics: MemoryMetrics):
        self.operation = operation
        self.metrics = metrics
        self.start_time: float = 0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        self.metrics.record_latency(self.operation, elapsed_ms)
