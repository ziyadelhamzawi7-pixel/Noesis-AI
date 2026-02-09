"""
Circuit Breaker Pattern for External API Resilience.

Implements the circuit breaker pattern to handle failures gracefully
and prevent cascading failures when external services are unavailable.
"""

import asyncio
import time
import threading
from enum import Enum
from typing import Callable, Any, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from functools import wraps
from loguru import logger

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation - requests flow through
    OPEN = "open"           # Failing - requests are rejected immediately
    HALF_OPEN = "half_open" # Testing recovery - limited requests allowed


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker monitoring."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_changes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    time_in_open_state: float = 0.0
    recovery_attempts: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, circuit_name: str, message: str = "Circuit breaker is open"):
        self.circuit_name = circuit_name
        super().__init__(f"[{circuit_name}] {message}")


class CircuitBreaker:
    """
    Circuit breaker for external API calls.

    States:
    - CLOSED: Normal operation, requests flow through
    - OPEN: Too many failures, requests are rejected
    - HALF_OPEN: Testing if service recovered, limited requests allowed

    Transitions:
    - CLOSED -> OPEN: When failure_threshold consecutive failures occur
    - OPEN -> HALF_OPEN: After recovery_timeout seconds
    - HALF_OPEN -> CLOSED: After half_open_successes consecutive successes
    - HALF_OPEN -> OPEN: On any failure
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_successes: int = 3,
        failure_rate_threshold: float = 0.5,
        min_calls_for_rate: int = 10,
        excluded_exceptions: tuple = ()
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Name for logging and identification
            failure_threshold: Consecutive failures to trip circuit
            recovery_timeout: Seconds before attempting recovery
            half_open_successes: Successes needed to close circuit
            failure_rate_threshold: Failure rate to trip (alternative to consecutive)
            min_calls_for_rate: Minimum calls before rate-based tripping
            excluded_exceptions: Exceptions that don't count as failures
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_successes = half_open_successes
        self.failure_rate_threshold = failure_rate_threshold
        self.min_calls_for_rate = min_calls_for_rate
        self.excluded_exceptions = excluded_exceptions

        # State
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_success_count = 0
        self._last_failure_time: float = 0
        self._open_start_time: float = 0

        # Thread safety
        self._lock = threading.RLock()

        # Metrics
        self.metrics = CircuitBreakerMetrics()

        # Recent calls for rate-based calculation
        self._recent_calls: list[bool] = []  # True = success, False = failure
        self._recent_calls_max = 100

        logger.info(
            f"CircuitBreaker '{name}' initialized: "
            f"threshold={failure_threshold}, recovery={recovery_timeout}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for automatic transitions."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._open_start_time >= self.recovery_timeout:
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self.metrics.state_changes += 1

        if new_state == CircuitState.OPEN:
            self._open_start_time = time.time()
            self.metrics.recovery_attempts = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_success_count = 0
            self.metrics.recovery_attempts += 1
        elif new_state == CircuitState.CLOSED:
            if old_state == CircuitState.OPEN:
                self.metrics.time_in_open_state += time.time() - self._open_start_time
            self._failure_count = 0
            self._recent_calls.clear()

        logger.info(f"CircuitBreaker '{self.name}': {old_state.value} -> {new_state.value}")

    def _record_success(self) -> None:
        """Record a successful call."""
        self.metrics.total_calls += 1
        self.metrics.successful_calls += 1
        self.metrics.last_success_time = time.time()
        self._success_count += 1
        self._failure_count = 0

        # Track for rate calculation
        self._recent_calls.append(True)
        if len(self._recent_calls) > self._recent_calls_max:
            self._recent_calls.pop(0)

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_success_count += 1
            if self._half_open_success_count >= self.half_open_successes:
                self._transition_to(CircuitState.CLOSED)

    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        self.metrics.total_calls += 1
        self.metrics.failed_calls += 1
        self.metrics.last_failure_time = time.time()
        self._last_failure_time = time.time()
        self._failure_count += 1
        self._success_count = 0

        # Track for rate calculation
        self._recent_calls.append(False)
        if len(self._recent_calls) > self._recent_calls_max:
            self._recent_calls.pop(0)

        logger.warning(
            f"CircuitBreaker '{self.name}' recorded failure "
            f"({self._failure_count}/{self.failure_threshold}): {exception}"
        )

        # Check if we should trip the circuit
        should_trip = False

        # Consecutive failure threshold
        if self._failure_count >= self.failure_threshold:
            should_trip = True
            logger.warning(
                f"CircuitBreaker '{self.name}': Consecutive failure threshold reached"
            )

        # Rate-based threshold
        if len(self._recent_calls) >= self.min_calls_for_rate:
            failure_rate = self._recent_calls.count(False) / len(self._recent_calls)
            if failure_rate >= self.failure_rate_threshold:
                should_trip = True
                logger.warning(
                    f"CircuitBreaker '{self.name}': Failure rate {failure_rate:.1%} "
                    f"exceeds threshold {self.failure_rate_threshold:.1%}"
                )

        if should_trip:
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                self._transition_to(CircuitState.OPEN)

    def _is_excluded_exception(self, exception: Exception) -> bool:
        """Check if exception should be excluded from failure count."""
        return isinstance(exception, self.excluded_exceptions)

    def call(
        self,
        func: Callable[..., T],
        *args,
        fallback: Optional[Callable[..., T]] = None,
        **kwargs
    ) -> T:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            fallback: Optional fallback function if circuit is open
            **kwargs: Keyword arguments for func

        Returns:
            Result of func or fallback

        Raises:
            CircuitBreakerError: If circuit is open and no fallback provided
        """
        with self._lock:
            current_state = self.state  # Triggers auto-transition check

            if current_state == CircuitState.OPEN:
                self.metrics.rejected_calls += 1
                if fallback:
                    logger.debug(f"CircuitBreaker '{self.name}': Using fallback")
                    return fallback(*args, **kwargs)
                raise CircuitBreakerError(
                    self.name,
                    f"Circuit is OPEN, recovery in "
                    f"{self.recovery_timeout - (time.time() - self._open_start_time):.1f}s"
                )

        # Execute the function (outside lock to allow concurrent calls)
        try:
            result = func(*args, **kwargs)
            with self._lock:
                self._record_success()
            return result

        except Exception as e:
            with self._lock:
                if not self._is_excluded_exception(e):
                    self._record_failure(e)
            raise

    async def call_async(
        self,
        func: Callable[..., T],
        *args,
        fallback: Optional[Callable[..., T]] = None,
        **kwargs
    ) -> T:
        """
        Execute an async function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            fallback: Optional fallback function if circuit is open
            **kwargs: Keyword arguments for func

        Returns:
            Result of func or fallback

        Raises:
            CircuitBreakerError: If circuit is open and no fallback provided
        """
        with self._lock:
            current_state = self.state

            if current_state == CircuitState.OPEN:
                self.metrics.rejected_calls += 1
                if fallback:
                    logger.debug(f"CircuitBreaker '{self.name}': Using fallback")
                    if asyncio.iscoroutinefunction(fallback):
                        return await fallback(*args, **kwargs)
                    return fallback(*args, **kwargs)
                raise CircuitBreakerError(
                    self.name,
                    f"Circuit is OPEN, recovery in "
                    f"{self.recovery_timeout - (time.time() - self._open_start_time):.1f}s"
                )

        try:
            result = await func(*args, **kwargs)
            with self._lock:
                self._record_success()
            return result

        except Exception as e:
            with self._lock:
                if not self._is_excluded_exception(e):
                    self._record_failure(e)
            raise

    def is_available(self) -> bool:
        """Check if the circuit breaker allows requests."""
        return self.state != CircuitState.OPEN

    def force_open(self) -> None:
        """Manually force the circuit open."""
        with self._lock:
            self._transition_to(CircuitState.OPEN)
            logger.warning(f"CircuitBreaker '{self.name}': Manually forced OPEN")

    def force_close(self) -> None:
        """Manually force the circuit closed."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)
            logger.info(f"CircuitBreaker '{self.name}': Manually forced CLOSED")

    def reset(self) -> None:
        """Reset the circuit breaker to initial state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_success_count = 0
            self._recent_calls.clear()
            self.metrics = CircuitBreakerMetrics()
            logger.info(f"CircuitBreaker '{self.name}': Reset to initial state")

    def get_metrics(self) -> dict:
        """Get circuit breaker metrics for monitoring."""
        with self._lock:
            failure_rate = 0.0
            if self._recent_calls:
                failure_rate = self._recent_calls.count(False) / len(self._recent_calls)

            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "failure_rate": failure_rate,
                "total_calls": self.metrics.total_calls,
                "successful_calls": self.metrics.successful_calls,
                "failed_calls": self.metrics.failed_calls,
                "rejected_calls": self.metrics.rejected_calls,
                "state_changes": self.metrics.state_changes,
                "recovery_attempts": self.metrics.recovery_attempts,
                "time_in_open_state": self.metrics.time_in_open_state,
                "last_failure_time": self.metrics.last_failure_time,
                "last_success_time": self.metrics.last_success_time
            }


def circuit_breaker_decorator(
    circuit: CircuitBreaker,
    fallback: Optional[Callable] = None
):
    """
    Decorator to wrap a function with circuit breaker protection.

    Usage:
        @circuit_breaker_decorator(openai_circuit)
        def call_openai_api():
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                return await circuit.call_async(func, *args, fallback=fallback, **kwargs)
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                return circuit.call(func, *args, fallback=fallback, **kwargs)
            return sync_wrapper
    return decorator


# Global circuit breakers for external services
openai_circuit = CircuitBreaker(
    name="openai_api",
    failure_threshold=5,
    recovery_timeout=60.0,
    half_open_successes=3
)

chromadb_circuit = CircuitBreaker(
    name="chromadb",
    failure_threshold=3,
    recovery_timeout=30.0,
    half_open_successes=2
)


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._circuits: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def register(self, circuit: CircuitBreaker) -> None:
        """Register a circuit breaker."""
        with self._lock:
            self._circuits[circuit.name] = circuit

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        with self._lock:
            return self._circuits.get(name)

    def get_all_metrics(self) -> dict:
        """Get metrics for all registered circuit breakers."""
        with self._lock:
            return {
                name: circuit.get_metrics()
                for name, circuit in self._circuits.items()
            }

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for circuit in self._circuits.values():
                circuit.reset()


# Global registry
circuit_registry = CircuitBreakerRegistry()
circuit_registry.register(openai_circuit)
circuit_registry.register(chromadb_circuit)
