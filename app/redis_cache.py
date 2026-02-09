"""
Redis Cache Layer for High-Throughput Operations.

Provides distributed caching for:
- Embedding results (reduce duplicate API calls)
- Query results (faster repeated queries)
- Document hashes (skip re-processing unchanged files)
- Rate limiter state (distributed rate limiting)
"""

import asyncio
import hashlib
import json
import pickle
import threading
import time
from typing import Any, Dict, List, Optional, TypeVar, Generic, Callable
from dataclasses import dataclass
from contextlib import contextmanager
from loguru import logger

try:
    import redis
    from redis import asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - falling back to in-memory caching")


T = TypeVar('T')


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class InMemoryFallback:
    """In-memory cache fallback when Redis is unavailable."""

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry)
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if expiry is None or time.time() < expiry:
                    return value
                del self._cache[key]
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size:
                # Remove oldest entries
                oldest = sorted(
                    self._cache.items(),
                    key=lambda x: x[1][1] or float('inf')
                )[:self.max_size // 10]
                for k, _ in oldest:
                    del self._cache[k]

            expiry = time.time() + (ttl or self.default_ttl) if ttl != 0 else None
            self._cache[key] = (value, expiry)
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def keys(self, pattern: str = "*") -> List[str]:
        """Simple pattern matching (only supports * at end)."""
        with self._lock:
            if pattern == "*":
                return list(self._cache.keys())
            prefix = pattern.rstrip("*")
            return [k for k in self._cache.keys() if k.startswith(prefix)]


class RedisCache:
    """
    Redis-based cache with fallback to in-memory.

    Features:
    - Automatic connection management
    - Serialization/deserialization
    - TTL support
    - Key prefixing for namespace isolation
    - Statistics tracking
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "noesis:",
        default_ttl: int = 3600,
        max_retries: int = 3,
        retry_delay: float = 0.5
    ):
        """
        Initialize Redis cache.

        Args:
            url: Redis connection URL
            prefix: Key prefix for namespace isolation
            default_ttl: Default TTL in seconds
            max_retries: Maximum connection retries
            retry_delay: Delay between retries
        """
        self.url = url
        self.prefix = prefix
        self.default_ttl = default_ttl
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Connection
        self._client: Optional[redis.Redis] = None
        self._async_client: Optional[aioredis.Redis] = None
        self._lock = threading.Lock()

        # Fallback
        self._fallback = InMemoryFallback(default_ttl=default_ttl)
        self._using_fallback = not REDIS_AVAILABLE

        # Stats
        self.stats = CacheStats()

        if REDIS_AVAILABLE:
            self._connect()

    def _connect(self) -> bool:
        """Establish Redis connection."""
        if not REDIS_AVAILABLE:
            return False

        try:
            self._client = redis.from_url(
                self.url,
                decode_responses=False,
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )
            # Test connection
            self._client.ping()
            self._using_fallback = False
            logger.info(f"Redis connected: {self.url}")
            return True

        except Exception as e:
            logger.warning(f"Redis connection failed, using fallback: {e}")
            self._using_fallback = True
            return False

    async def _connect_async(self) -> bool:
        """Establish async Redis connection."""
        if not REDIS_AVAILABLE:
            return False

        try:
            self._async_client = await aioredis.from_url(
                self.url,
                decode_responses=False,
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )
            await self._async_client.ping()
            self._using_fallback = False
            logger.info(f"Redis async connected: {self.url}")
            return True

        except Exception as e:
            logger.warning(f"Redis async connection failed, using fallback: {e}")
            self._using_fallback = True
            return False

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        return pickle.loads(data)

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        full_key = self._make_key(key)

        if self._using_fallback:
            value = self._fallback.get(full_key)
            if value is not None:
                self.stats.hits += 1
            else:
                self.stats.misses += 1
            return value

        try:
            data = self._client.get(full_key)
            if data is not None:
                self.stats.hits += 1
                return self._deserialize(data)
            self.stats.misses += 1
            return None

        except Exception as e:
            self.stats.errors += 1
            logger.warning(f"Redis get error: {e}")
            # Fallback
            return self._fallback.get(full_key)

    async def get_async(self, key: str) -> Optional[Any]:
        """Async get value from cache."""
        full_key = self._make_key(key)

        if self._using_fallback or self._async_client is None:
            value = self._fallback.get(full_key)
            if value is not None:
                self.stats.hits += 1
            else:
                self.stats.misses += 1
            return value

        try:
            data = await self._async_client.get(full_key)
            if data is not None:
                self.stats.hits += 1
                return self._deserialize(data)
            self.stats.misses += 1
            return None

        except Exception as e:
            self.stats.errors += 1
            logger.warning(f"Redis async get error: {e}")
            return self._fallback.get(full_key)

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None uses default)

        Returns:
            True if successful
        """
        full_key = self._make_key(key)
        ttl = ttl if ttl is not None else self.default_ttl

        if self._using_fallback:
            self._fallback.set(full_key, value, ttl)
            self.stats.sets += 1
            return True

        try:
            data = self._serialize(value)
            if ttl > 0:
                self._client.setex(full_key, ttl, data)
            else:
                self._client.set(full_key, data)
            self.stats.sets += 1
            return True

        except Exception as e:
            self.stats.errors += 1
            logger.warning(f"Redis set error: {e}")
            self._fallback.set(full_key, value, ttl)
            return False

    async def set_async(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Async set value in cache."""
        full_key = self._make_key(key)
        ttl = ttl if ttl is not None else self.default_ttl

        if self._using_fallback or self._async_client is None:
            self._fallback.set(full_key, value, ttl)
            self.stats.sets += 1
            return True

        try:
            data = self._serialize(value)
            if ttl > 0:
                await self._async_client.setex(full_key, ttl, data)
            else:
                await self._async_client.set(full_key, data)
            self.stats.sets += 1
            return True

        except Exception as e:
            self.stats.errors += 1
            logger.warning(f"Redis async set error: {e}")
            self._fallback.set(full_key, value, ttl)
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        full_key = self._make_key(key)

        if self._using_fallback:
            self._fallback.delete(full_key)
            self.stats.deletes += 1
            return True

        try:
            self._client.delete(full_key)
            self.stats.deletes += 1
            return True

        except Exception as e:
            self.stats.errors += 1
            logger.warning(f"Redis delete error: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        full_key = self._make_key(key)

        if self._using_fallback:
            return self._fallback.exists(full_key)

        try:
            return bool(self._client.exists(full_key))
        except Exception as e:
            self.stats.errors += 1
            return self._fallback.exists(full_key)

    def get_or_set(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: Optional[int] = None
    ) -> T:
        """
        Get value or compute and cache it.

        Args:
            key: Cache key
            factory: Function to compute value if not cached
            ttl: Time-to-live in seconds

        Returns:
            Cached or computed value
        """
        value = self.get(key)
        if value is not None:
            return value

        value = factory()
        self.set(key, value, ttl)
        return value

    async def get_or_set_async(
        self,
        key: str,
        factory: Callable[[], T],
        ttl: Optional[int] = None
    ) -> T:
        """Async get value or compute and cache it."""
        value = await self.get_async(key)
        if value is not None:
            return value

        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        await self.set_async(key, value, ttl)
        return value

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "hit_rate": self.stats.hit_rate,
            "sets": self.stats.sets,
            "deletes": self.stats.deletes,
            "errors": self.stats.errors,
            "using_fallback": self._using_fallback,
            "connected": not self._using_fallback
        }

    def clear_namespace(self, namespace: str = "") -> int:
        """Clear all keys in a namespace."""
        pattern = self._make_key(f"{namespace}*")
        count = 0

        if self._using_fallback:
            keys = self._fallback.keys(pattern)
            for key in keys:
                self._fallback.delete(key)
                count += 1
        else:
            try:
                keys = self._client.keys(pattern)
                if keys:
                    count = self._client.delete(*keys)
            except Exception as e:
                self.stats.errors += 1
                logger.warning(f"Redis clear error: {e}")

        logger.info(f"Cleared {count} keys matching {pattern}")
        return count


class EmbeddingCache:
    """
    Specialized cache for embedding vectors.

    Uses content hashing to avoid recomputing embeddings
    for identical text.
    """

    def __init__(self, redis_cache: RedisCache, ttl: int = 86400):
        """
        Initialize embedding cache.

        Args:
            redis_cache: RedisCache instance
            ttl: Time-to-live in seconds (default 24 hours)
        """
        self.cache = redis_cache
        self.ttl = ttl
        self.namespace = "embedding:"

    def _hash_text(self, text: str) -> str:
        """Create hash of text content."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text."""
        key = f"{self.namespace}{self._hash_text(text)}"
        return self.cache.get(key)

    async def get_async(self, text: str) -> Optional[List[float]]:
        """Async get cached embedding."""
        key = f"{self.namespace}{self._hash_text(text)}"
        return await self.cache.get_async(key)

    def set(self, text: str, embedding: List[float]) -> bool:
        """Cache embedding for text."""
        key = f"{self.namespace}{self._hash_text(text)}"
        return self.cache.set(key, embedding, self.ttl)

    async def set_async(self, text: str, embedding: List[float]) -> bool:
        """Async cache embedding."""
        key = f"{self.namespace}{self._hash_text(text)}"
        return await self.cache.set_async(key, embedding, self.ttl)

    def get_many(self, texts: List[str]) -> Dict[str, Optional[List[float]]]:
        """Get cached embeddings for multiple texts."""
        return {text: self.get(text) for text in texts}

    def set_many(self, embeddings: Dict[str, List[float]]) -> int:
        """Cache multiple embeddings."""
        count = 0
        for text, embedding in embeddings.items():
            if self.set(text, embedding):
                count += 1
        return count


class DocumentCache:
    """
    Cache for document processing results.

    Caches parsed documents to avoid re-parsing unchanged files.
    """

    def __init__(self, redis_cache: RedisCache, ttl: int = 3600):
        self.cache = redis_cache
        self.ttl = ttl
        self.namespace = "document:"

    def _hash_file(self, file_path: str, file_size: int, modified_time: float) -> str:
        """Create hash from file metadata."""
        content = f"{file_path}:{file_size}:{modified_time}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(
        self,
        file_path: str,
        file_size: int,
        modified_time: float
    ) -> Optional[Dict[str, Any]]:
        """Get cached document parse result."""
        key = f"{self.namespace}{self._hash_file(file_path, file_size, modified_time)}"
        return self.cache.get(key)

    def set(
        self,
        file_path: str,
        file_size: int,
        modified_time: float,
        result: Dict[str, Any]
    ) -> bool:
        """Cache document parse result."""
        key = f"{self.namespace}{self._hash_file(file_path, file_size, modified_time)}"
        return self.cache.set(key, result, self.ttl)


class QueryCache:
    """
    Cache for query results.

    Caches semantic search results for repeated queries.
    """

    def __init__(self, redis_cache: RedisCache, ttl: int = 300):
        self.cache = redis_cache
        self.ttl = ttl
        self.namespace = "query:"

    def _hash_query(self, data_room_id: str, query: str, top_k: int) -> str:
        """Create hash from query parameters."""
        content = f"{data_room_id}:{query}:{top_k}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(
        self,
        data_room_id: str,
        query: str,
        top_k: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached query results."""
        key = f"{self.namespace}{self._hash_query(data_room_id, query, top_k)}"
        return self.cache.get(key)

    def set(
        self,
        data_room_id: str,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 10
    ) -> bool:
        """Cache query results."""
        key = f"{self.namespace}{self._hash_query(data_room_id, query, top_k)}"
        return self.cache.set(key, results, self.ttl)

    def invalidate(self, data_room_id: str) -> int:
        """Invalidate all cached queries for a data room."""
        return self.cache.clear_namespace(f"{self.namespace}{data_room_id[:8]}")


# Global cache instances
import os
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_cache = RedisCache(url=redis_url, prefix="noesis:")
embedding_cache = EmbeddingCache(redis_cache)
document_cache = DocumentCache(redis_cache)
query_cache = QueryCache(redis_cache)
