"""
Embedding generation tool using OpenAI API.
Batches requests, implements rate limiting, and caches results.
Supports parallel processing for faster embedding generation.

Tier 3 optimized with:
- Adaptive rate limiting (5,000 RPM, 5M TPM)
- Circuit breaker for resilience
- Redis caching for duplicate text detection
- High concurrency (100+ concurrent requests)
"""

import sys
import os
import re
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
from loguru import logger

try:
    from openai import OpenAI, AsyncOpenAI
    from dotenv import load_dotenv
except ImportError:
    logger.error("Required packages not installed. Run: pip install openai python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Import rate limiter and circuit breaker
try:
    from app.rate_limiter import embedding_rate_limiter, AdaptiveRateLimiter
    from app.circuit_breaker import openai_circuit, CircuitBreakerError
    from app.metrics import metrics, record_embedding_batch
    TIER3_ENABLED = True
except ImportError:
    TIER3_ENABLED = False
    logger.warning("Tier 3 components not available - using basic rate limiting")

# Import Redis cache if available
try:
    from app.redis_cache import embedding_cache
    REDIS_CACHE_ENABLED = True
except ImportError:
    REDIS_CACHE_ENABLED = False


class EmbeddingGenerator:
    """
    OpenAI embedding generator with batching, rate limiting, and parallel processing.

    Tier 3 Features:
    - Adaptive rate limiting for 5,000 RPM / 5M TPM
    - Circuit breaker for API resilience
    - Redis caching to avoid duplicate embeddings
    - High concurrency support (100+ requests)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        batch_size: int = 2048,
        max_retries: int = 5,
        max_concurrent: int = 100,
        use_rate_limiter: bool = True,
        use_circuit_breaker: bool = True,
        use_cache: bool = True
    ):
        """
        Initialize embedding generator.

        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            model: Embedding model to use
            batch_size: Number of texts to embed in each API call
            max_retries: Maximum number of retries on failure
            max_concurrent: Maximum concurrent API requests (Tier 3: 100)
            use_rate_limiter: Enable adaptive rate limiting
            use_circuit_breaker: Enable circuit breaker pattern
            use_cache: Enable Redis caching for embeddings
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)
        self._async_client = None  # Created lazily inside async context
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self.max_tokens_per_request = 280_000  # 20K margin under OpenAI's 300K limit

        # Tier 3 features
        self.use_rate_limiter = use_rate_limiter and TIER3_ENABLED
        self.use_circuit_breaker = use_circuit_breaker and TIER3_ENABLED
        self.use_cache = use_cache and REDIS_CACHE_ENABLED

        # Cost tracking
        self.total_tokens = 0
        self.total_cost = 0.0
        self.last_error = None
        self._lock = None  # Created lazily inside async context
        self._sync_lock = __import__('threading').Lock()

        # Cache stats
        self.cache_hits = 0
        self.cache_misses = 0

        # Model pricing (per 1M tokens)
        self.pricing = {
            "text-embedding-3-large": 0.13,
            "text-embedding-3-small": 0.02,
            "text-embedding-ada-002": 0.10
        }

        logger.info(
            f"Initialized EmbeddingGenerator: model={model}, "
            f"max_concurrent={max_concurrent}, "
            f"rate_limiter={self.use_rate_limiter}, "
            f"circuit_breaker={self.use_circuit_breaker}, "
            f"cache={self.use_cache}"
        )

    def generate(self, chunks: List[Dict[str, Any]], parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Generate embeddings for chunks.

        Args:
            chunks: List of chunk dictionaries with 'chunk_text' field
            parallel: Use parallel processing (default: True, 3-5x faster)

        Returns:
            Chunks with 'embedding' field added
        """
        if not chunks:
            logger.warning("No chunks provided")
            return []

        if parallel:
            # Check if we're already in an async context (e.g., FastAPI background task)
            try:
                loop = asyncio.get_running_loop()
                # We're inside an async context - run async in a background thread
                # This avoids the event loop conflict while keeping async performance
                from concurrent.futures import ThreadPoolExecutor
                logger.info(f"Running async embeddings in background thread (max_concurrent={self.max_concurrent})")
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(asyncio.run, self.generate_async(chunks))
                    return future.result()
            except RuntimeError:
                # No event loop running, safe to use asyncio.run()
                return asyncio.run(self.generate_async(chunks))

        # Sequential fallback
        return self._generate_sequential(chunks)

    def _generate_sequential(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Embedding generation with concurrent batch processing."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        texts = [chunk['chunk_text'] for chunk in chunks]
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        workers = min(self.max_concurrent, total_batches)
        logger.info(f"Generating embeddings for {len(chunks)} chunks ({total_batches} batches, {workers} workers)")

        # Prepare indexed batches to preserve ordering
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batches.append((i // self.batch_size, texts[i:i + self.batch_size]))

        all_embeddings = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="embed") as executor:
            future_to_batch = {
                executor.submit(self._generate_batch, batch_texts): (batch_idx, batch_texts)
                for batch_idx, batch_texts in batches
            }

            for future in as_completed(future_to_batch):
                batch_idx, batch_texts = future_to_batch[future]
                batch_num = batch_idx + 1
                start_pos = batch_idx * self.batch_size

                try:
                    batch_embeddings = future.result()
                    if batch_embeddings:
                        for j, emb in enumerate(batch_embeddings):
                            all_embeddings[start_pos + j] = emb
                        logger.info(f"Batch {batch_num}/{total_batches} complete ({len(batch_texts)} texts)")
                    else:
                        logger.error(f"Failed to generate embeddings for batch {batch_num}")
                        for j in range(len(batch_texts)):
                            all_embeddings[start_pos + j] = None
                except Exception as e:
                    logger.error(f"Batch {batch_num} failed: {e}")
                    for j in range(len(batch_texts)):
                        all_embeddings[start_pos + j] = None

        for chunk, embedding in zip(chunks, all_embeddings):
            chunk['embedding'] = embedding

        logger.success(f"Generated {len(all_embeddings)} embeddings")
        logger.info(f"Total tokens: {self.total_tokens:,}")
        logger.info(f"Estimated cost: ${self.total_cost:.4f}")

        return chunks

    async def generate_async(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for chunks using parallel async requests.

        Args:
            chunks: List of chunk dictionaries with 'chunk_text' field

        Returns:
            Chunks with 'embedding' field added
        """
        # Create async resources lazily (inside the event loop context)
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self.api_key)
        if self._lock is None:
            self._lock = asyncio.Lock()

        logger.info(f"Generating embeddings in parallel for {len(chunks)} chunks (max {self.max_concurrent} concurrent)")

        texts = [chunk['chunk_text'] for chunk in chunks]

        # Create batches with their indices
        batches: List[Tuple[int, List[str]]] = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batches.append((i, batch_texts))

        total_batches = len(batches)
        logger.info(f"Split into {total_batches} batches of up to {self.batch_size} texts each")

        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def process_batch(batch_idx: int, batch_texts: List[str], batch_num: int) -> Tuple[int, List[List[float]]]:
            async with semaphore:
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
                embeddings = await self._generate_batch_async(batch_texts)
                if embeddings is None:
                    logger.error(f"Failed to generate embeddings for batch {batch_num}")
                    embeddings = [None] * len(batch_texts)
                return (batch_idx, embeddings)

        # Process all batches concurrently (limited by semaphore)
        tasks = [
            process_batch(batch_idx, batch_texts, i + 1)
            for i, (batch_idx, batch_texts) in enumerate(batches)
        ]

        results = await asyncio.gather(*tasks)

        # Sort results by original batch index and flatten
        results.sort(key=lambda x: x[0])
        all_embeddings = []
        for _, embeddings in results:
            all_embeddings.extend(embeddings)

        # Add embeddings to chunks
        for chunk, embedding in zip(chunks, all_embeddings):
            chunk['embedding'] = embedding

        logger.success(f"Generated {len(all_embeddings)} embeddings in parallel")
        logger.info(f"Total tokens: {self.total_tokens:,}")
        logger.info(f"Estimated cost: ${self.total_cost:.4f}")

        return chunks

    async def _generate_batch_async(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for a batch of texts asynchronously with retry logic."""
        texts = [self._truncate_text(t) for t in texts]

        # Split into sub-batches to stay under per-request token limit
        sub_batches = self._split_by_token_limit(texts)
        all_embeddings = []

        for sub_batch in sub_batches:
            embeddings = await self._call_embedding_api_async(sub_batch)
            if embeddings is None:
                return None
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def _call_embedding_api_async(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Call the OpenAI embedding API asynchronously for a single sub-batch.

        Tier 3 features:
        - Adaptive rate limiting (RPM/TPM)
        - Circuit breaker protection
        - Smart retry with rate limit handling
        """
        # Estimate tokens for rate limiter
        estimated_tokens = sum(len(t) // 4 for t in texts)

        base_delay = 1.0
        max_delay = 60.0

        for attempt in range(self.max_retries):
            try:
                # Check circuit breaker
                if self.use_circuit_breaker and not openai_circuit.is_available():
                    raise CircuitBreakerError(
                        "openai_api",
                        "Circuit breaker is open - API temporarily unavailable"
                    )

                # Acquire rate limiter permission
                if self.use_rate_limiter:
                    wait_time = await embedding_rate_limiter.acquire_async(estimated_tokens)
                    if wait_time > 0:
                        logger.debug(f"Rate limiter wait: {wait_time:.2f}s")

                # Make API call
                start_time = time.time()
                response = await self._async_client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                duration = time.time() - start_time

                embeddings = [item.embedding for item in response.data]
                tokens_used = response.usage.total_tokens
                cost_per_token = self.pricing.get(self.model, 0.13) / 1_000_000
                batch_cost = tokens_used * cost_per_token

                async with self._lock:
                    self.total_tokens += tokens_used
                    self.total_cost += batch_cost

                # Report success to rate limiter and circuit breaker
                if self.use_rate_limiter:
                    embedding_rate_limiter.report_success()
                if self.use_circuit_breaker:
                    openai_circuit._record_success()

                # Record metrics
                if TIER3_ENABLED:
                    record_embedding_batch(tokens_used, len(texts), duration, True)

                logger.debug(f"Sub-batch: {tokens_used} tokens, ${batch_cost:.4f}, {duration:.2f}s")
                return embeddings

            except CircuitBreakerError as e:
                logger.warning(f"Circuit breaker open: {e}")
                self.last_error = str(e)
                return None

            except Exception as e:
                error_str = str(e).lower()
                self.last_error = str(e)

                # Handle rate limits specially
                if "rate_limit" in error_str or "429" in error_str:
                    retry_after = self._parse_retry_after(e)

                    if self.use_rate_limiter:
                        wait_time = embedding_rate_limiter.report_rate_limit(retry_after)
                    else:
                        wait_time = retry_after or min(base_delay * (2 ** attempt), max_delay)

                    logger.warning(f"Rate limited, waiting {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    continue

                # Handle server errors with backoff
                if any(code in error_str for code in ["500", "502", "503", "504"]):
                    wait_time = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Server error, retrying in {wait_time:.1f}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue

                # Report failure to circuit breaker
                if self.use_circuit_breaker:
                    openai_circuit._record_failure(e)

                logger.warning(f"Async embedding attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    wait_time = min(base_delay * (2 ** attempt), max_delay)
                    logger.info(f"Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to generate embeddings after {self.max_retries} attempts")
                    if TIER3_ENABLED:
                        record_embedding_batch(0, len(texts), 0, False)
                    return None

        return None

    def _parse_retry_after(self, error) -> Optional[float]:
        """Parse Retry-After header from rate limit errors."""
        error_str = str(error)
        match = re.search(r'retry[_-]?after[:\s]+(\d+(?:\.\d+)?)', error_str, re.I)
        if match:
            return float(match.group(1))
        return None

    def _split_by_token_limit(self, texts: List[str]) -> List[List[str]]:
        """Split texts into sub-batches that fit within the per-request token limit."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            token_counts = [len(enc.encode(t)) for t in texts]
        except Exception:
            # Fallback: ~4 chars per token
            token_counts = [len(t) // 4 for t in texts]

        sub_batches = []
        current_batch: List[str] = []
        current_tokens = 0

        for text, tok_count in zip(texts, token_counts):
            if current_batch and current_tokens + tok_count > self.max_tokens_per_request:
                sub_batches.append(current_batch)
                current_batch = []
                current_tokens = 0
            current_batch.append(text)
            current_tokens += tok_count

        if current_batch:
            sub_batches.append(current_batch)

        if len(sub_batches) > 1:
            logger.info(f"Split batch of {len(texts)} texts into {len(sub_batches)} sub-batches by token limit")

        return sub_batches

    def _truncate_text(self, text: str, max_tokens: int = 8000) -> str:
        """Truncate text to stay within embedding model token limits (8191 tokens)."""
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            logger.warning(f"Truncating text from {len(tokens)} to {max_tokens} tokens for embedding")
            return enc.decode(tokens[:max_tokens])
        except Exception:
            # Fallback: conservative char limit (~3 chars per token for dense data)
            max_chars = max_tokens * 3
            if len(text) <= max_chars:
                return text
            logger.warning(f"Truncating text from {len(text)} to {max_chars} chars for embedding (fallback)")
            return text[:max_chars]

    def _generate_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Generate embeddings for a batch of texts with retry logic."""
        # Truncate any texts that exceed token limits
        texts = [self._truncate_text(t) for t in texts]

        # Split into sub-batches to stay under per-request token limit
        sub_batches = self._split_by_token_limit(texts)
        all_embeddings = []

        for sub_batch in sub_batches:
            embeddings = self._call_embedding_api(sub_batch)
            if embeddings is None:
                return None
            all_embeddings.extend(embeddings)

        return all_embeddings

    def _call_embedding_api(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Call the OpenAI embedding API for a single sub-batch with retry logic.

        Tier 3 features:
        - Adaptive rate limiting (RPM/TPM)
        - Circuit breaker protection
        - Smart retry with rate limit handling
        """
        # Estimate tokens for rate limiter
        estimated_tokens = sum(len(t) // 4 for t in texts)

        base_delay = 1.0
        max_delay = 60.0

        for attempt in range(self.max_retries):
            try:
                # Check circuit breaker
                if self.use_circuit_breaker and not openai_circuit.is_available():
                    raise CircuitBreakerError(
                        "openai_api",
                        "Circuit breaker is open - API temporarily unavailable"
                    )

                # Acquire rate limiter permission (sync version)
                if self.use_rate_limiter:
                    wait_time = embedding_rate_limiter.acquire(estimated_tokens)
                    if wait_time > 0:
                        logger.debug(f"Rate limiter wait: {wait_time:.2f}s")

                # Make API call
                start_time = time.time()
                response = self.client.embeddings.create(
                    model=self.model,
                    input=texts
                )
                duration = time.time() - start_time

                embeddings = [item.embedding for item in response.data]
                tokens_used = response.usage.total_tokens

                with self._sync_lock:
                    self.total_tokens += tokens_used
                    cost_per_token = self.pricing.get(self.model, 0.13) / 1_000_000
                    batch_cost = tokens_used * cost_per_token
                    self.total_cost += batch_cost

                # Report success
                if self.use_rate_limiter:
                    embedding_rate_limiter.report_success()
                if self.use_circuit_breaker:
                    openai_circuit._record_success()

                # Record metrics
                if TIER3_ENABLED:
                    record_embedding_batch(tokens_used, len(texts), duration, True)

                logger.debug(f"Sub-batch: {tokens_used} tokens, ${batch_cost:.4f}, {duration:.2f}s")
                return embeddings

            except CircuitBreakerError as e:
                logger.warning(f"Circuit breaker open: {e}")
                self.last_error = str(e)
                return None

            except Exception as e:
                error_str = str(e).lower()
                self.last_error = str(e)

                # Handle rate limits specially
                if "rate_limit" in error_str or "429" in error_str:
                    retry_after = self._parse_retry_after(e)

                    if self.use_rate_limiter:
                        wait_time = embedding_rate_limiter.report_rate_limit(retry_after)
                    else:
                        wait_time = retry_after or min(base_delay * (2 ** attempt), max_delay)

                    logger.warning(f"Rate limited, waiting {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue

                # Handle server errors with backoff
                if any(code in error_str for code in ["500", "502", "503", "504"]):
                    wait_time = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Server error, retrying in {wait_time:.1f}s: {e}")
                    time.sleep(wait_time)
                    continue

                # Report failure to circuit breaker
                if self.use_circuit_breaker:
                    openai_circuit._record_failure(e)

                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    wait_time = min(base_delay * (2 ** attempt), max_delay)
                    logger.info(f"Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {self.max_retries} attempts: {e}")
                    if TIER3_ENABLED:
                        record_embedding_batch(0, len(texts), 0, False)
                    return None

        return None

    def get_cost_estimate(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Estimate cost of generating embeddings.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Cost estimate dictionary
        """
        # Estimate tokens (rough: 1 token ≈ 4 characters)
        total_chars = sum(len(chunk['chunk_text']) for chunk in chunks)
        estimated_tokens = total_chars // 4

        # Calculate cost
        cost_per_token = self.pricing.get(self.model, 0.13) / 1_000_000
        estimated_cost = estimated_tokens * cost_per_token

        return {
            "chunks": len(chunks),
            "estimated_tokens": estimated_tokens,
            "estimated_cost": estimated_cost,
            "model": self.model,
            "pricing_per_1m_tokens": self.pricing.get(self.model, 0.13)
        }


def generate_embeddings(
    chunks: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: str = "text-embedding-3-small",
    batch_size: int = 2048,
    max_concurrent: int = 100,
    parallel: bool = True
) -> List[Dict[str, Any]]:
    """
    Convenience function to generate embeddings.

    Args:
        chunks: List of chunk dictionaries
        api_key: OpenAI API key (optional, uses env var if not provided)
        model: Embedding model to use
        batch_size: Batch size for API calls
        max_concurrent: Maximum concurrent API requests (default: 5)
        parallel: Use parallel processing (default: True, 3-5x faster)

    Returns:
        Chunks with embeddings added

    Example:
        >>> from chunk_documents import chunk_documents
        >>> from parse_pdf import parse_pdf
        >>> parsed = parse_pdf("pitch_deck.pdf")
        >>> chunks = chunk_documents(parsed)
        >>> chunks_with_embeddings = generate_embeddings(chunks)
        >>> print(f"Generated {len(chunks_with_embeddings)} embeddings")
    """
    generator = EmbeddingGenerator(
        api_key=api_key,
        model=model,
        batch_size=batch_size,
        max_concurrent=max_concurrent
    )

    return generator.generate(chunks, parallel=parallel)


def generate_embeddings_streaming(
    chunks: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: str = "text-embedding-3-small",
    batch_size: int = 2048,
    max_concurrent: int = 100
):
    """
    Generate embeddings in streaming mode, yielding batches for immediate processing.

    Uses parallel workers to process multiple API batches concurrently while
    yielding results in order for immediate indexing.

    Args:
        chunks: List of chunk dictionaries with 'chunk_text' field
        api_key: OpenAI API key (optional, uses env var if not provided)
        model: Embedding model to use
        batch_size: Number of chunks per batch
        max_concurrent: Maximum parallel API requests (default: 5)

    Yields:
        List of chunk dicts with embeddings added (one batch at a time)

    Example:
        >>> for batch in generate_embeddings_streaming(chunks):
        ...     index_to_vectordb(data_room_id, batch)
        ...     db.create_chunks(batch)
        ...     del batch  # Free memory
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not chunks:
        logger.warning("No chunks provided for streaming embeddings")
        return

    generator = EmbeddingGenerator(
        api_key=api_key,
        model=model,
        batch_size=batch_size,
        max_concurrent=max_concurrent
    )

    # Split chunks into batches
    all_batches = []
    for i in range(0, len(chunks), batch_size):
        all_batches.append((i, chunks[i:i + batch_size]))

    total_batches = len(all_batches)
    workers = min(max_concurrent, total_batches)
    logger.info(f"Streaming embeddings: {len(chunks)} chunks in {total_batches} batches, {workers} parallel workers")

    # Process all batches with a single persistent executor, yield as each completes
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="embed-stream") as executor:
        future_to_batch = {}
        for batch_idx, (_, batch_chunks) in enumerate(all_batches):
            texts = [chunk['chunk_text'] for chunk in batch_chunks]
            future = executor.submit(generator._generate_batch, texts)
            future_to_batch[future] = (batch_idx, batch_chunks)

        for future in as_completed(future_to_batch):
            batch_idx, batch_chunks = future_to_batch[future]
            batch_num = batch_idx + 1
            try:
                batch_embeddings = future.result()
                if batch_embeddings:
                    for chunk, embedding in zip(batch_chunks, batch_embeddings):
                        chunk['embedding'] = embedding
                    logger.info(f"Batch {batch_num}/{total_batches} complete ({len(batch_chunks)} chunks)")
                else:
                    error_msg = generator.last_error or "Unknown embedding error"
                    logger.error(f"Failed to generate embeddings for batch {batch_num}: {error_msg}")
                    for chunk in batch_chunks:
                        chunk['embedding'] = None
                        chunk['embedding_failed'] = True
                        chunk['embedding_error'] = error_msg
            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                for chunk in batch_chunks:
                    chunk['embedding'] = None
                    chunk['embedding_failed'] = True
                    chunk['embedding_error'] = str(e)
            yield batch_chunks

    logger.success(f"Streaming complete: {generator.total_tokens:,} tokens, ${generator.total_cost:.4f}")


def main():
    """CLI interface for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate embeddings for document chunks")
    parser.add_argument("file", help="Path to chunks JSON file")
    parser.add_argument("--model", default="text-embedding-3-large", help="Embedding model")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--estimate-only", action="store_true", help="Only show cost estimate")

    args = parser.parse_args()

    # Load chunks
    with open(args.file, 'r') as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks")

    # Initialize generator
    generator = EmbeddingGenerator(model=args.model, batch_size=args.batch_size)

    # Show cost estimate
    estimate = generator.get_cost_estimate(chunks)
    print(f"\n{'='*60}")
    print(f"Cost Estimate:")
    print(f"Chunks: {estimate['chunks']}")
    print(f"Estimated Tokens: {estimate['estimated_tokens']:,}")
    print(f"Estimated Cost: ${estimate['estimated_cost']:.4f}")
    print(f"Model: {estimate['model']}")
    print(f"{'='*60}\n")

    if args.estimate_only:
        logger.info("Estimate only mode. Exiting.")
        return

    # Confirm if cost is high
    if estimate['estimated_cost'] > 1.0:
        response = input(f"Estimated cost is ${estimate['estimated_cost']:.2f}. Continue? (y/n): ")
        if response.lower() != 'y':
            logger.info("Cancelled by user")
            return

    # Generate embeddings
    chunks_with_embeddings = generator.generate(chunks)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(chunks_with_embeddings, indent=2))
        logger.success(f"Saved {len(chunks_with_embeddings)} chunks with embeddings to {output_path}")
    else:
        print(f"\nGenerated {len(chunks_with_embeddings)} embeddings")
        print(f"Actual tokens: {generator.total_tokens:,}")
        print(f"Actual cost: ${generator.total_cost:.4f}")
        print(f"\nFirst embedding dimension: {len(chunks_with_embeddings[0]['embedding'])}")


if __name__ == "__main__":
    main()
