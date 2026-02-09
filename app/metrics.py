"""
Metrics Collection for Monitoring and Observability.

Provides comprehensive metrics collection for:
- API latencies
- Rate limiter utilization
- Queue depths
- Memory usage
- Error rates
- Cache performance
"""

import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
from loguru import logger


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """Thread-safe counter metric."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0
        self._lock = threading.Lock()

    def increment(self, value: int = 1) -> int:
        """Increment the counter."""
        with self._lock:
            self._value += value
            return self._value

    def get(self) -> int:
        """Get current value."""
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reset to zero."""
        with self._lock:
            self._value = 0


class Gauge:
    """Thread-safe gauge metric."""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        """Set the gauge value."""
        with self._lock:
            self._value = value

    def increment(self, value: float = 1.0) -> float:
        """Increment the gauge."""
        with self._lock:
            self._value += value
            return self._value

    def decrement(self, value: float = 1.0) -> float:
        """Decrement the gauge."""
        with self._lock:
            self._value -= value
            return self._value

    def get(self) -> float:
        """Get current value."""
        with self._lock:
            return self._value


class Histogram:
    """Thread-safe histogram for tracking distributions."""

    def __init__(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None,
        max_samples: int = 1000,
        retention_seconds: float = 300.0
    ):
        self.name = name
        self.description = description
        self.buckets = buckets or [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        self.max_samples = max_samples
        self.retention_seconds = retention_seconds

        self._samples: List[MetricPoint] = []
        self._lock = threading.Lock()
        self._count = 0
        self._sum = 0.0

    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record an observation."""
        now = time.time()

        with self._lock:
            self._samples.append(MetricPoint(
                timestamp=now,
                value=value,
                labels=labels or {}
            ))
            self._count += 1
            self._sum += value

            # Cleanup old samples
            self._cleanup()

    def _cleanup(self) -> None:
        """Remove old samples."""
        cutoff = time.time() - self.retention_seconds

        # Remove by time
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.pop(0)

        # Remove by count
        while len(self._samples) > self.max_samples:
            self._samples.pop(0)

    def get_stats(self) -> Dict[str, Any]:
        """Get histogram statistics."""
        with self._lock:
            if not self._samples:
                return {
                    "count": 0,
                    "sum": 0,
                    "avg": 0,
                    "min": 0,
                    "max": 0,
                    "p50": 0,
                    "p90": 0,
                    "p95": 0,
                    "p99": 0
                }

            values = sorted(p.value for p in self._samples)
            count = len(values)

            return {
                "count": count,
                "sum": sum(values),
                "avg": sum(values) / count,
                "min": values[0],
                "max": values[-1],
                "p50": self._percentile(values, 0.50),
                "p90": self._percentile(values, 0.90),
                "p95": self._percentile(values, 0.95),
                "p99": self._percentile(values, 0.99)
            }

    def _percentile(self, sorted_values: List[float], p: float) -> float:
        """Calculate percentile from sorted values."""
        if not sorted_values:
            return 0
        idx = int(len(sorted_values) * p)
        return sorted_values[min(idx, len(sorted_values) - 1)]

    @contextmanager
    def time(self, labels: Optional[Dict[str, str]] = None):
        """Context manager to time an operation."""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.observe(duration, labels)


class MetricsCollector:
    """
    Central metrics collector for the application.

    Collects and aggregates metrics for:
    - API latencies
    - Rate limiter states
    - Queue depths
    - Memory usage
    - Error rates
    - Cache hit rates
    """

    def __init__(self, retention_seconds: float = 300.0):
        """
        Initialize metrics collector.

        Args:
            retention_seconds: How long to retain histogram samples
        """
        self.retention_seconds = retention_seconds
        self._lock = threading.Lock()

        # Pre-defined metrics
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}

        # Initialize common metrics
        self._init_common_metrics()

    def _init_common_metrics(self) -> None:
        """Initialize commonly used metrics."""
        # Counters
        self.counter("api_requests_total", "Total API requests")
        self.counter("api_errors_total", "Total API errors")
        self.counter("embeddings_generated_total", "Total embeddings generated")
        self.counter("documents_processed_total", "Total documents processed")
        self.counter("chunks_created_total", "Total chunks created")
        self.counter("cache_hits_total", "Total cache hits")
        self.counter("cache_misses_total", "Total cache misses")

        # Gauges
        self.gauge("active_jobs", "Number of active jobs")
        self.gauge("queue_depth", "Number of pending jobs in queue")
        self.gauge("active_workers", "Number of active workers")
        self.gauge("memory_usage_percent", "System memory usage percent")
        self.gauge("rate_limiter_rpm", "Current RPM usage")
        self.gauge("rate_limiter_tpm", "Current TPM usage")

        # Histograms
        self.histogram(
            "api_latency_seconds",
            "API request latency",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
        )
        self.histogram(
            "embedding_latency_seconds",
            "Embedding generation latency",
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
        )
        self.histogram(
            "document_processing_seconds",
            "Document processing time",
            buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
        )
        self.histogram(
            "chunk_size_tokens",
            "Chunk size in tokens",
            buckets=[100, 250, 500, 750, 1000, 1500, 2000]
        )

    def counter(self, name: str, description: str = "") -> Counter:
        """Get or create a counter."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description)
            return self._counters[name]

    def gauge(self, name: str, description: str = "") -> Gauge:
        """Get or create a gauge."""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description)
            return self._gauges[name]

    def histogram(
        self,
        name: str,
        description: str = "",
        buckets: Optional[List[float]] = None
    ) -> Histogram:
        """Get or create a histogram."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(
                    name, description, buckets,
                    retention_seconds=self.retention_seconds
                )
            return self._histograms[name]

    def increment(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter (convenience method)."""
        key = self._make_key(name, labels)
        self.counter(key).increment(value)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge value (convenience method)."""
        key = self._make_key(name, labels)
        self.gauge(key).set(value)

    def observe(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a histogram value (convenience method)."""
        key = self._make_key(name, labels)
        self.histogram(key).observe(value)

    def _make_key(self, name: str, labels: Optional[Dict[str, str]] = None) -> str:
        """Create a key from name and labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    @contextmanager
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Context manager to time an operation."""
        key = self._make_key(name, labels)
        hist = self.histogram(key)
        with hist.time(labels):
            yield

    def get_all(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            return {
                "counters": {
                    name: counter.get()
                    for name, counter in self._counters.items()
                },
                "gauges": {
                    name: gauge.get()
                    for name, gauge in self._gauges.items()
                },
                "histograms": {
                    name: hist.get_stats()
                    for name, hist in self._histograms.items()
                }
            }

    def get_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []

        with self._lock:
            # Counters
            for name, counter in self._counters.items():
                lines.append(f"# HELP {name} {counter.description}")
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {counter.get()}")

            # Gauges
            for name, gauge in self._gauges.items():
                lines.append(f"# HELP {name} {gauge.description}")
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {gauge.get()}")

            # Histograms
            for name, hist in self._histograms.items():
                stats = hist.get_stats()
                lines.append(f"# HELP {name} {hist.description}")
                lines.append(f"# TYPE {name} histogram")
                lines.append(f"{name}_count {stats['count']}")
                lines.append(f"{name}_sum {stats['sum']}")
                lines.append(f"{name}_avg {stats['avg']:.6f}")
                lines.append(f"{name}_p50 {stats['p50']:.6f}")
                lines.append(f"{name}_p90 {stats['p90']:.6f}")
                lines.append(f"{name}_p95 {stats['p95']:.6f}")
                lines.append(f"{name}_p99 {stats['p99']:.6f}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            for counter in self._counters.values():
                counter.reset()
            for gauge in self._gauges.values():
                gauge.set(0)
            self._histograms.clear()
            self._init_common_metrics()

        logger.info("Metrics reset")


# Global metrics collector instance
metrics = MetricsCollector(retention_seconds=300.0)


# Convenience functions
def record_api_request(
    endpoint: str,
    method: str,
    status_code: int,
    duration_seconds: float
) -> None:
    """Record an API request."""
    labels = {"endpoint": endpoint, "method": method, "status": str(status_code)}

    metrics.increment("api_requests_total", labels=labels)

    if status_code >= 400:
        metrics.increment("api_errors_total", labels=labels)

    metrics.observe("api_latency_seconds", duration_seconds, labels=labels)


def record_embedding_batch(
    tokens: int,
    chunks: int,
    duration_seconds: float,
    success: bool
) -> None:
    """Record an embedding batch generation."""
    if success:
        metrics.counter("embeddings_generated_total").increment(chunks)

    metrics.observe("embedding_latency_seconds", duration_seconds)


def record_document_processed(
    file_type: str,
    chunks: int,
    duration_seconds: float
) -> None:
    """Record a processed document."""
    labels = {"file_type": file_type}
    metrics.increment("documents_processed_total", labels=labels)
    metrics.increment("chunks_created_total", value=chunks)
    metrics.observe("document_processing_seconds", duration_seconds, labels=labels)


def record_cache_access(hit: bool, cache_type: str = "general") -> None:
    """Record a cache access."""
    labels = {"cache_type": cache_type}
    if hit:
        metrics.increment("cache_hits_total", labels=labels)
    else:
        metrics.increment("cache_misses_total", labels=labels)


def update_system_metrics(
    active_jobs: int,
    queue_depth: int,
    active_workers: int,
    memory_percent: float
) -> None:
    """Update system-level metrics."""
    metrics.set_gauge("active_jobs", active_jobs)
    metrics.set_gauge("queue_depth", queue_depth)
    metrics.set_gauge("active_workers", active_workers)
    metrics.set_gauge("memory_usage_percent", memory_percent)


def update_rate_limiter_metrics(rpm: int, tpm: int) -> None:
    """Update rate limiter metrics."""
    metrics.set_gauge("rate_limiter_rpm", rpm)
    metrics.set_gauge("rate_limiter_tpm", tpm)
