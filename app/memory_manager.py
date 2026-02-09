"""
Memory Management for High-Throughput Processing.

Provides memory-aware processing to prevent OOM errors during
concurrent document processing and embedding generation.
"""

import gc
import os
import threading
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
from loguru import logger

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory management will be limited")


@dataclass
class MemoryAllocation:
    """Tracks a memory allocation."""
    task_id: str
    size_mb: float
    allocated_at: float
    description: str = ""


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    total_allocations: int = 0
    total_releases: int = 0
    peak_allocated_mb: float = 0.0
    gc_collections: int = 0
    allocation_denials: int = 0


class MemoryManager:
    """
    Manages memory allocation for concurrent processing.

    Features:
    - Track allocations per task
    - Automatic GC triggering at threshold
    - Block new allocations when memory is tight
    - Memory-aware batch sizing
    - Process-level monitoring
    """

    def __init__(
        self,
        max_memory_percent: float = 80.0,
        gc_threshold_percent: float = 70.0,
        min_free_mb: int = 500,
        check_interval_seconds: float = 5.0
    ):
        """
        Initialize memory manager.

        Args:
            max_memory_percent: Maximum system memory usage before blocking
            gc_threshold_percent: Trigger GC when this threshold is reached
            min_free_mb: Minimum free memory required (MB)
            check_interval_seconds: Interval for background monitoring
        """
        self.max_memory_percent = max_memory_percent
        self.gc_threshold_percent = gc_threshold_percent
        self.min_free_mb = min_free_mb
        self.check_interval_seconds = check_interval_seconds

        # Allocation tracking
        self._allocations: Dict[str, MemoryAllocation] = {}
        self._lock = threading.RLock()

        # Metrics
        self.metrics = MemoryMetrics()

        # Background monitoring
        self._stop_monitoring = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None

        logger.info(
            f"MemoryManager initialized: max={max_memory_percent}%, "
            f"gc_threshold={gc_threshold_percent}%, min_free={min_free_mb}MB"
        )

    def start_monitoring(self) -> None:
        """Start background memory monitoring thread."""
        if self._monitor_thread is not None:
            return

        self._stop_monitoring.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MemoryMonitor"
        )
        self._monitor_thread.start()
        logger.info("Memory monitoring started")

    def stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        if self._monitor_thread is None:
            return

        self._stop_monitoring.set()
        self._monitor_thread.join(timeout=5.0)
        self._monitor_thread = None
        logger.info("Memory monitoring stopped")

    def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while not self._stop_monitoring.is_set():
            try:
                status = self.get_status()

                # Trigger GC if needed
                if status["system_percent"] >= self.gc_threshold_percent:
                    self._trigger_gc("threshold")

                # Log warning if memory is high
                if status["system_percent"] >= self.max_memory_percent - 5:
                    logger.warning(
                        f"Memory usage high: {status['system_percent']:.1f}% "
                        f"({status['system_available_mb']:.0f}MB available)"
                    )

            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")

            self._stop_monitoring.wait(self.check_interval_seconds)

    def _get_system_memory(self) -> Dict[str, float]:
        """Get system memory statistics."""
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            return {
                "total_mb": memory.total / (1024 * 1024),
                "available_mb": memory.available / (1024 * 1024),
                "used_mb": memory.used / (1024 * 1024),
                "percent": memory.percent
            }
        else:
            # Fallback for systems without psutil
            return {
                "total_mb": 8192,  # Assume 8GB
                "available_mb": 4096,  # Assume 4GB available
                "used_mb": 4096,
                "percent": 50.0
            }

    def _get_process_memory(self) -> float:
        """Get current process memory usage in MB."""
        if PSUTIL_AVAILABLE:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        else:
            return 0.0

    def can_allocate(self, size_mb: float) -> bool:
        """
        Check if an allocation is safe.

        Args:
            size_mb: Requested allocation size in MB

        Returns:
            True if allocation can proceed
        """
        memory = self._get_system_memory()

        # Check percent used
        if memory["percent"] >= self.max_memory_percent:
            return False

        # Check absolute free memory
        if memory["available_mb"] - size_mb < self.min_free_mb:
            return False

        return True

    def allocate(
        self,
        task_id: str,
        size_mb: float,
        description: str = "",
        wait: bool = False,
        timeout: float = 60.0
    ) -> bool:
        """
        Register a memory allocation.

        Args:
            task_id: Unique identifier for the task
            size_mb: Estimated memory usage in MB
            description: Human-readable description
            wait: If True, wait for memory to become available
            timeout: Maximum wait time in seconds

        Returns:
            True if allocation was successful
        """
        start_time = time.time()

        while True:
            with self._lock:
                if self.can_allocate(size_mb):
                    self._allocations[task_id] = MemoryAllocation(
                        task_id=task_id,
                        size_mb=size_mb,
                        allocated_at=time.time(),
                        description=description
                    )

                    # Update metrics
                    self.metrics.total_allocations += 1
                    current_allocated = sum(a.size_mb for a in self._allocations.values())
                    self.metrics.peak_allocated_mb = max(
                        self.metrics.peak_allocated_mb,
                        current_allocated
                    )

                    logger.debug(
                        f"Memory allocated: {task_id} ({size_mb:.1f}MB) - {description}"
                    )
                    return True

                # Try GC first
                self._trigger_gc("allocation_attempt")

                if self.can_allocate(size_mb):
                    continue  # Retry after GC

                if not wait:
                    self.metrics.allocation_denials += 1
                    logger.warning(
                        f"Memory allocation denied: {task_id} ({size_mb:.1f}MB)"
                    )
                    return False

            # Wait and retry
            if time.time() - start_time > timeout:
                self.metrics.allocation_denials += 1
                logger.warning(
                    f"Memory allocation timeout: {task_id} ({size_mb:.1f}MB)"
                )
                return False

            time.sleep(1.0)

    def release(self, task_id: str) -> bool:
        """
        Release a memory allocation.

        Args:
            task_id: Task identifier to release

        Returns:
            True if allocation was found and released
        """
        with self._lock:
            if task_id in self._allocations:
                allocation = self._allocations.pop(task_id)
                self.metrics.total_releases += 1

                logger.debug(
                    f"Memory released: {task_id} ({allocation.size_mb:.1f}MB)"
                )

                # Trigger GC if memory is tight
                memory = self._get_system_memory()
                if memory["percent"] >= self.gc_threshold_percent:
                    self._trigger_gc("post_release")

                return True

            return False

    @contextmanager
    def allocation_context(
        self,
        task_id: str,
        size_mb: float,
        description: str = "",
        wait: bool = True,
        timeout: float = 60.0
    ):
        """
        Context manager for memory allocation.

        Usage:
            with memory_manager.allocation_context("task1", 100):
                # Do memory-intensive work
                pass
        """
        if not self.allocate(task_id, size_mb, description, wait, timeout):
            raise MemoryError(
                f"Could not allocate {size_mb}MB for task {task_id}"
            )

        try:
            yield
        finally:
            self.release(task_id)

    def _trigger_gc(self, reason: str) -> int:
        """
        Trigger garbage collection.

        Args:
            reason: Reason for triggering GC

        Returns:
            Number of objects collected
        """
        collected = gc.collect()
        self.metrics.gc_collections += 1

        logger.debug(f"GC triggered ({reason}): collected {collected} objects")
        return collected

    def get_optimal_batch_size(
        self,
        item_size_mb: float,
        max_batch: int = 1000,
        target_memory_mb: float = 500
    ) -> int:
        """
        Calculate optimal batch size based on available memory.

        Args:
            item_size_mb: Estimated size per item in MB
            max_batch: Maximum batch size
            target_memory_mb: Target memory usage for batch

        Returns:
            Optimal batch size
        """
        memory = self._get_system_memory()

        # Calculate available memory for batching
        available_for_batch = min(
            memory["available_mb"] * 0.5,  # Use max 50% of available
            target_memory_mb
        )

        # Calculate batch size
        if item_size_mb <= 0:
            return max_batch

        optimal = int(available_for_batch / item_size_mb)
        return max(1, min(optimal, max_batch))

    def get_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        memory = self._get_system_memory()
        process_mb = self._get_process_memory()

        with self._lock:
            allocated_mb = sum(a.size_mb for a in self._allocations.values())
            active_allocations = len(self._allocations)

        return {
            "system_total_mb": memory["total_mb"],
            "system_available_mb": memory["available_mb"],
            "system_used_mb": memory["used_mb"],
            "system_percent": memory["percent"],
            "process_mb": process_mb,
            "tracked_allocated_mb": allocated_mb,
            "active_allocations": active_allocations,
            "max_memory_percent": self.max_memory_percent,
            "gc_threshold_percent": self.gc_threshold_percent,
            "can_allocate_100mb": self.can_allocate(100),
            "can_allocate_500mb": self.can_allocate(500)
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get memory manager metrics."""
        status = self.get_status()

        return {
            **status,
            "total_allocations": self.metrics.total_allocations,
            "total_releases": self.metrics.total_releases,
            "peak_allocated_mb": self.metrics.peak_allocated_mb,
            "gc_collections": self.metrics.gc_collections,
            "allocation_denials": self.metrics.allocation_denials
        }

    def get_active_allocations(self) -> Dict[str, Dict[str, Any]]:
        """Get details of active allocations."""
        with self._lock:
            return {
                task_id: {
                    "size_mb": alloc.size_mb,
                    "allocated_at": alloc.allocated_at,
                    "description": alloc.description,
                    "duration_seconds": time.time() - alloc.allocated_at
                }
                for task_id, alloc in self._allocations.items()
            }

    def force_gc(self) -> int:
        """Force garbage collection."""
        return self._trigger_gc("manual")


# Global memory manager instance
memory_manager = MemoryManager(
    max_memory_percent=80.0,
    gc_threshold_percent=70.0,
    min_free_mb=500
)


def check_memory_available(required_mb: float = 500) -> bool:
    """
    Quick check if sufficient memory is available.

    Args:
        required_mb: Required memory in MB

    Returns:
        True if sufficient memory is available
    """
    return memory_manager.can_allocate(required_mb)


def get_memory_status() -> Dict[str, Any]:
    """Get current memory status."""
    return memory_manager.get_status()
