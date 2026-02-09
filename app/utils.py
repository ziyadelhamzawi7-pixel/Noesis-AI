"""
Utility functions for system monitoring and resource management.
"""

import psutil
from pathlib import Path
from typing import Optional
from loguru import logger

from app.config import settings


def check_memory_available(required_mb: float = 500) -> bool:
    """
    Check if sufficient memory is available for processing.

    Args:
        required_mb: Minimum required memory in MB (default 500MB)

    Returns:
        True if sufficient memory is available
    """
    available = psutil.virtual_memory().available / (1024 * 1024)
    return available > required_mb


def get_available_memory_mb() -> float:
    """Get available system memory in MB."""
    return psutil.virtual_memory().available / (1024 * 1024)


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def estimate_file_memory(file_size_bytes: int) -> float:
    """
    Estimate memory needed to process a file.

    Heuristic: Files typically expand 3-5x when parsed (text extraction,
    chunking, embeddings). We use 5x as a conservative estimate.

    Args:
        file_size_bytes: Size of file in bytes

    Returns:
        Estimated memory requirement in MB
    """
    # PDF/Office files expand ~5x when fully processed
    # Add ~3KB per chunk for embeddings (1536 dims * 4 bytes * safety factor)
    expansion_factor = 5
    return (file_size_bytes * expansion_factor) / (1024 * 1024)


def can_process_file(file_path: str, buffer_mb: float = 200) -> tuple[bool, Optional[str]]:
    """
    Check if system has enough memory to process a file.

    Args:
        file_path: Path to the file to process
        buffer_mb: Additional buffer memory to maintain (default 200MB)

    Returns:
        Tuple of (can_process: bool, error_message: Optional[str])
    """
    try:
        file_size = Path(file_path).stat().st_size
        estimated_memory = estimate_file_memory(file_size)
        available_memory = get_available_memory_mb()
        required_memory = estimated_memory + buffer_mb

        if available_memory < required_memory:
            return False, (
                f"Insufficient memory to process file. "
                f"Available: {available_memory:.0f}MB, "
                f"Required: {required_memory:.0f}MB (estimated {estimated_memory:.0f}MB for file + {buffer_mb:.0f}MB buffer)"
            )

        return True, None

    except FileNotFoundError:
        return False, f"File not found: {file_path}"
    except Exception as e:
        return False, f"Error checking file: {str(e)}"


def get_system_status() -> dict:
    """
    Get comprehensive system status for monitoring.

    Returns:
        Dict with memory, CPU, and disk metrics
    """
    process = psutil.Process()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(Path(settings.database_path).parent)

    return {
        "process": {
            "memory_mb": round(process.memory_info().rss / (1024 * 1024), 1),
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads()
        },
        "system": {
            "memory_total_mb": round(memory.total / (1024 * 1024), 1),
            "memory_available_mb": round(memory.available / (1024 * 1024), 1),
            "memory_percent": memory.percent,
            "disk_free_gb": round(disk.free / (1024 * 1024 * 1024), 1),
            "disk_percent": disk.percent
        }
    }


class RateLimiter:
    """
    Simple rate limiter using a sliding window algorithm.

    Useful for limiting API calls or processing rates.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in the window
            window_seconds: Time window in seconds
        """
        from collections import deque
        import time

        self.max_requests = max_requests
        self.window = window_seconds
        self.timestamps: deque = deque()
        self._time = time

    def acquire(self, timeout: float = 60.0) -> bool:
        """
        Acquire a slot in the rate limiter.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if slot acquired, False if timeout
        """
        start = self._time.time()

        while self._time.time() - start < timeout:
            now = self._time.time()

            # Remove timestamps outside the window
            while self.timestamps and now - self.timestamps[0] > self.window:
                self.timestamps.popleft()

            # Check if we can proceed
            if len(self.timestamps) < self.max_requests:
                self.timestamps.append(now)
                return True

            # Wait a bit and retry
            self._time.sleep(0.1)

        return False

    def can_proceed(self) -> bool:
        """
        Check if a request can proceed without waiting.

        Returns:
            True if under the rate limit
        """
        now = self._time.time()

        # Remove old timestamps
        while self.timestamps and now - self.timestamps[0] > self.window:
            self.timestamps.popleft()

        return len(self.timestamps) < self.max_requests

    @property
    def current_count(self) -> int:
        """Get current request count in the window."""
        now = self._time.time()
        while self.timestamps and now - self.timestamps[0] > self.window:
            self.timestamps.popleft()
        return len(self.timestamps)
