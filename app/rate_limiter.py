"""
Adaptive Rate Limiter for OpenAI API Tier 3.

Implements dual-bucket rate limiting for both RPM (requests per minute)
and TPM (tokens per minute) to maximize API utilization without hitting limits.
"""

import asyncio
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Deque, Tuple
from loguru import logger


@dataclass
class RateLimitMetrics:
    """Metrics for rate limiter monitoring."""
    total_requests: int = 0
    total_tokens: int = 0
    requests_delayed: int = 0
    total_delay_seconds: float = 0.0
    rate_limit_hits: int = 0
    current_rpm: int = 0
    current_tpm: int = 0
    last_reset: float = field(default_factory=time.time)


class AdaptiveRateLimiter:
    """
    Dual-bucket rate limiter for OpenAI API Tier 3.

    Features:
    - Tracks both RPM and TPM limits simultaneously
    - Sliding window algorithm for accurate rate tracking
    - Automatic backoff on rate limit errors (429)
    - Thread-safe and async-safe
    - Metrics collection for monitoring

    OpenAI Tier 3 Limits:
    - Embeddings: 5,000 RPM, 5,000,000 TPM
    """

    def __init__(
        self,
        rpm_limit: int = 5000,
        tpm_limit: int = 5_000_000,
        safety_margin: float = 0.90,
        window_seconds: float = 60.0,
        backoff_multiplier: float = 1.5,
        max_backoff_seconds: float = 60.0
    ):
        """
        Initialize the rate limiter.

        Args:
            rpm_limit: Maximum requests per minute
            tpm_limit: Maximum tokens per minute
            safety_margin: Use this fraction of limits (0.90 = 90%)
            window_seconds: Sliding window duration
            backoff_multiplier: Multiply wait time on each retry
            max_backoff_seconds: Maximum backoff duration
        """
        # Apply safety margin to limits
        self.rpm_limit = int(rpm_limit * safety_margin)
        self.tpm_limit = int(tpm_limit * safety_margin)
        self.window_seconds = window_seconds
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff_seconds = max_backoff_seconds

        # Sliding window tracking
        # Each entry is (timestamp, token_count)
        self._request_log: Deque[Tuple[float, int]] = deque()

        # Thread safety - use Condition for safe waiting without deadlocks
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._async_lock: Optional[asyncio.Lock] = None

        # Backoff state
        self._backoff_until: float = 0.0
        self._consecutive_rate_limits: int = 0

        # Metrics
        self.metrics = RateLimitMetrics()

        logger.info(
            f"AdaptiveRateLimiter initialized: "
            f"RPM={self.rpm_limit}, TPM={self.tpm_limit}, "
            f"safety_margin={safety_margin}"
        )

    def _get_async_lock(self) -> asyncio.Lock:
        """Lazily create async lock."""
        if self._async_lock is None:
            self._async_lock = asyncio.Lock()
        return self._async_lock

    def _cleanup_old_entries(self, now: float) -> None:
        """Remove entries outside the sliding window."""
        cutoff = now - self.window_seconds
        while self._request_log and self._request_log[0][0] < cutoff:
            self._request_log.popleft()

    def _get_current_usage(self) -> Tuple[int, int]:
        """Get current RPM and TPM usage within the window."""
        now = time.time()
        self._cleanup_old_entries(now)

        rpm = len(self._request_log)
        tpm = sum(tokens for _, tokens in self._request_log)

        return rpm, tpm

    def _calculate_wait_time(self, estimated_tokens: int) -> float:
        """
        Calculate how long to wait before making a request.

        Args:
            estimated_tokens: Estimated tokens for the upcoming request

        Returns:
            Wait time in seconds (0 if immediate request is allowed)
        """
        now = time.time()

        # Check if we're in backoff period
        if now < self._backoff_until:
            return self._backoff_until - now

        self._cleanup_old_entries(now)

        current_rpm, current_tpm = self._get_current_usage()

        # Check RPM limit
        if current_rpm >= self.rpm_limit:
            # Find when the oldest request will expire
            oldest_time = self._request_log[0][0] if self._request_log else now
            rpm_wait = (oldest_time + self.window_seconds) - now + 0.1
        else:
            rpm_wait = 0.0

        # Check TPM limit
        if current_tpm + estimated_tokens > self.tpm_limit:
            # Calculate how long until we have enough token budget
            tokens_needed = (current_tpm + estimated_tokens) - self.tpm_limit

            # Find when enough tokens will be freed
            tpm_wait = 0.0
            freed_tokens = 0
            cutoff_time = now - self.window_seconds

            for timestamp, tokens in self._request_log:
                if freed_tokens >= tokens_needed:
                    break
                freed_tokens += tokens
                tpm_wait = max(tpm_wait, (timestamp + self.window_seconds) - now + 0.1)
        else:
            tpm_wait = 0.0

        return max(rpm_wait, tpm_wait)

    def acquire(self, estimated_tokens: int, timeout: float = 300.0) -> float:
        """
        Synchronously acquire permission to make a request.

        Args:
            estimated_tokens: Estimated tokens for the request
            timeout: Maximum time to wait

        Returns:
            Actual wait time that was spent

        Raises:
            TimeoutError: If timeout exceeded
        """
        start_time = time.time()
        total_wait = 0.0

        with self._condition:
            while True:
                wait_time = self._calculate_wait_time(estimated_tokens)

                if wait_time <= 0:
                    # Record the request
                    now = time.time()
                    self._request_log.append((now, estimated_tokens))

                    # Update metrics
                    self.metrics.total_requests += 1
                    self.metrics.total_tokens += estimated_tokens
                    self.metrics.current_rpm, self.metrics.current_tpm = self._get_current_usage()

                    if total_wait > 0:
                        self.metrics.requests_delayed += 1
                        self.metrics.total_delay_seconds += total_wait

                    # Notify other waiting threads that state changed
                    self._condition.notify_all()
                    return total_wait

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed + wait_time > timeout:
                    raise TimeoutError(
                        f"Rate limiter timeout: waited {elapsed:.1f}s, "
                        f"need {wait_time:.1f}s more"
                    )

                # Wait on condition - this safely releases lock while sleeping
                # and re-acquires it atomically when waking up
                sleep_time = min(wait_time, 1.0)
                self._condition.wait(timeout=sleep_time)
                total_wait += sleep_time

    async def acquire_async(self, estimated_tokens: int, timeout: float = 300.0) -> float:
        """
        Asynchronously acquire permission to make a request.

        Args:
            estimated_tokens: Estimated tokens for the request
            timeout: Maximum time to wait

        Returns:
            Actual wait time that was spent

        Raises:
            TimeoutError: If timeout exceeded
        """
        start_time = time.time()
        total_wait = 0.0
        async_lock = self._get_async_lock()

        while True:
            async with async_lock:
                with self._lock:
                    wait_time = self._calculate_wait_time(estimated_tokens)

                    if wait_time <= 0:
                        # Record the request
                        now = time.time()
                        self._request_log.append((now, estimated_tokens))

                        # Update metrics
                        self.metrics.total_requests += 1
                        self.metrics.total_tokens += estimated_tokens
                        self.metrics.current_rpm, self.metrics.current_tpm = self._get_current_usage()

                        if total_wait > 0:
                            self.metrics.requests_delayed += 1
                            self.metrics.total_delay_seconds += total_wait

                        return total_wait

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed + wait_time > timeout:
                raise TimeoutError(
                    f"Rate limiter timeout: waited {elapsed:.1f}s, "
                    f"need {wait_time:.1f}s more"
                )

            # Sleep outside the lock
            sleep_time = min(wait_time, 1.0)
            await asyncio.sleep(sleep_time)
            total_wait += sleep_time

    def report_success(self) -> None:
        """Report a successful API call - resets backoff state."""
        with self._condition:
            self._consecutive_rate_limits = 0
            # Notify waiting threads that backoff may have cleared
            self._condition.notify_all()

    def report_rate_limit(self, retry_after: Optional[float] = None) -> float:
        """
        Report a rate limit error (429).

        Args:
            retry_after: Retry-After header value from API response

        Returns:
            Recommended wait time before retrying
        """
        with self._lock:
            self._consecutive_rate_limits += 1
            self.metrics.rate_limit_hits += 1

            # Calculate backoff time
            if retry_after:
                backoff = retry_after
            else:
                # Exponential backoff
                backoff = min(
                    2 ** self._consecutive_rate_limits * self.backoff_multiplier,
                    self.max_backoff_seconds
                )

            self._backoff_until = time.time() + backoff

            logger.warning(
                f"Rate limit hit #{self._consecutive_rate_limits}, "
                f"backing off for {backoff:.1f}s"
            )

            return backoff

    def can_proceed(self, estimated_tokens: int) -> bool:
        """
        Check if a request can proceed immediately without waiting.

        Args:
            estimated_tokens: Estimated tokens for the request

        Returns:
            True if request can proceed immediately
        """
        with self._lock:
            return self._calculate_wait_time(estimated_tokens) <= 0

    def get_capacity(self) -> Tuple[float, float]:
        """
        Get remaining capacity as fractions.

        Returns:
            Tuple of (rpm_remaining_fraction, tpm_remaining_fraction)
        """
        with self._lock:
            current_rpm, current_tpm = self._get_current_usage()

            rpm_remaining = max(0, self.rpm_limit - current_rpm) / self.rpm_limit
            tpm_remaining = max(0, self.tpm_limit - current_tpm) / self.tpm_limit

            return rpm_remaining, tpm_remaining

    def get_metrics(self) -> dict:
        """Get rate limiter metrics for monitoring."""
        with self._lock:
            current_rpm, current_tpm = self._get_current_usage()

            return {
                "rpm_limit": self.rpm_limit,
                "tpm_limit": self.tpm_limit,
                "current_rpm": current_rpm,
                "current_tpm": current_tpm,
                "rpm_utilization": current_rpm / self.rpm_limit if self.rpm_limit > 0 else 0,
                "tpm_utilization": current_tpm / self.tpm_limit if self.tpm_limit > 0 else 0,
                "total_requests": self.metrics.total_requests,
                "total_tokens": self.metrics.total_tokens,
                "requests_delayed": self.metrics.requests_delayed,
                "total_delay_seconds": self.metrics.total_delay_seconds,
                "rate_limit_hits": self.metrics.rate_limit_hits,
                "consecutive_rate_limits": self._consecutive_rate_limits,
                "in_backoff": time.time() < self._backoff_until
            }

    def reset(self) -> None:
        """Reset the rate limiter state."""
        with self._lock:
            self._request_log.clear()
            self._backoff_until = 0.0
            self._consecutive_rate_limits = 0
            self.metrics = RateLimitMetrics()
            logger.info("Rate limiter reset")


class MultiModelRateLimiter:
    """
    Manages rate limiters for multiple OpenAI models/endpoints.

    Different endpoints have different rate limits:
    - Embeddings: 5,000 RPM, 5,000,000 TPM (Tier 3)
    - Chat completions: varies by model
    """

    def __init__(self):
        self._limiters: dict[str, AdaptiveRateLimiter] = {}
        self._lock = threading.Lock()

    def get_limiter(
        self,
        model: str,
        rpm_limit: int = 5000,
        tpm_limit: int = 5_000_000,
        safety_margin: float = 0.90
    ) -> AdaptiveRateLimiter:
        """
        Get or create a rate limiter for a specific model.

        Args:
            model: Model name (e.g., "text-embedding-3-small")
            rpm_limit: Requests per minute limit
            tpm_limit: Tokens per minute limit
            safety_margin: Safety margin fraction

        Returns:
            AdaptiveRateLimiter instance for the model
        """
        with self._lock:
            if model not in self._limiters:
                self._limiters[model] = AdaptiveRateLimiter(
                    rpm_limit=rpm_limit,
                    tpm_limit=tpm_limit,
                    safety_margin=safety_margin
                )
                logger.info(f"Created rate limiter for model: {model}")

            return self._limiters[model]

    def get_all_metrics(self) -> dict:
        """Get metrics for all rate limiters."""
        with self._lock:
            return {
                model: limiter.get_metrics()
                for model, limiter in self._limiters.items()
            }


# Global rate limiter instance for embeddings (uses config values)
try:
    from app.config import settings
    embedding_rate_limiter = AdaptiveRateLimiter(
        rpm_limit=settings.embedding_rpm_limit,
        tpm_limit=settings.embedding_tpm_limit,
        safety_margin=settings.embedding_safety_margin
    )
except ImportError:
    # Fallback if config not available
    embedding_rate_limiter = AdaptiveRateLimiter(
        rpm_limit=5000,
        tpm_limit=5_000_000,
        safety_margin=0.95
    )

# Multi-model rate limiter for different endpoints
model_rate_limiters = MultiModelRateLimiter()
