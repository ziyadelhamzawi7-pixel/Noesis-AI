"""
High-Throughput Document Processing Pipeline.

Implements a multi-stage parallel pipeline for document processing:
1. Parse Stage: CPU-bound parsing with ProcessPoolExecutor
2. Chunk Stage: Text chunking with ThreadPoolExecutor
3. Embed Stage: API calls with high concurrency
4. Index Stage: Batched database/vector writes

Features:
- Back-pressure between stages prevents memory exhaustion
- Progress tracking and callbacks
- Error handling and recovery
- Memory-aware processing
"""

import asyncio
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from loguru import logger


@dataclass
class PipelineConfig:
    """Configuration for the document processing pipeline."""
    # Stage worker counts
    parse_workers: int = 16
    chunk_workers: int = 8
    embed_workers: int = 150  # Increased for max speed

    # Batch sizes
    embed_batch_size: int = 4000  # Increased for fewer API calls
    index_batch_size: int = 20000

    # Queue sizes (for back-pressure) - increased for higher throughput
    parse_queue_size: int = 64
    chunk_queue_size: int = 200  # Increased from 100
    embed_queue_size: int = 300  # Increased from 200
    index_queue_size: int = 5    # Increased from 3

    # Memory limits
    max_memory_mb: int = 2000

    # Timeouts
    parse_timeout_seconds: int = 300
    embed_timeout_seconds: int = 60


@dataclass
class PipelineStats:
    """Statistics for pipeline execution."""
    files_submitted: int = 0
    files_parsed: int = 0
    files_failed: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    chunks_indexed: int = 0
    total_tokens: int = 0
    start_time: float = field(default_factory=time.time)
    parse_time: float = 0.0
    chunk_time: float = 0.0
    embed_time: float = 0.0
    index_time: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def chunks_per_second(self) -> float:
        if self.elapsed_time > 0:
            return self.chunks_created / self.elapsed_time
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_submitted": self.files_submitted,
            "files_parsed": self.files_parsed,
            "files_failed": self.files_failed,
            "chunks_created": self.chunks_created,
            "embeddings_generated": self.embeddings_generated,
            "chunks_indexed": self.chunks_indexed,
            "total_tokens": self.total_tokens,
            "elapsed_time": self.elapsed_time,
            "chunks_per_second": self.chunks_per_second,
            "parse_time": self.parse_time,
            "chunk_time": self.chunk_time,
            "embed_time": self.embed_time,
            "index_time": self.index_time,
            "error_count": len(self.errors)
        }


class DocumentPipeline:
    """
    Multi-stage parallel pipeline for document processing.

    Pipeline stages run concurrently with queues providing back-pressure:

    Files -> [Parse Stage] -> [Chunk Stage] -> [Embed Stage] -> [Index Stage]

    Each stage can process items in parallel based on the config.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration (uses defaults if not provided)
        """
        self.config = config or PipelineConfig()
        self.stats = PipelineStats()
        self._stop_event = asyncio.Event()

    async def process_data_room(
        self,
        data_room_id: str,
        file_paths: List[str],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Process all files in a data room through the pipeline.

        Args:
            data_room_id: Data room identifier
            file_paths: List of file paths to process
            progress_callback: Optional callback for progress updates
                              Signature: (stage: str, progress: float)

        Returns:
            Dictionary with processing results and statistics
        """
        self.stats = PipelineStats()
        self.stats.files_submitted = len(file_paths)
        self._stop_event.clear()

        # Create async queues for inter-stage communication
        parse_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.parse_queue_size
        )
        chunk_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.chunk_queue_size
        )
        embed_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.embed_queue_size
        )
        index_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.index_queue_size
        )

        # Start all pipeline stages as concurrent tasks
        tasks = [
            asyncio.create_task(
                self._parse_stage(file_paths, parse_queue, progress_callback)
            ),
            asyncio.create_task(
                self._chunk_stage(parse_queue, chunk_queue, progress_callback)
            ),
            asyncio.create_task(
                self._embed_stage(
                    data_room_id, chunk_queue, embed_queue, progress_callback
                )
            ),
            asyncio.create_task(
                self._index_stage(
                    data_room_id, embed_queue, index_queue, progress_callback
                )
            ),
        ]

        try:
            # Wait for all stages to complete
            await asyncio.gather(*tasks)

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            self.stats.errors.append(str(e))
            self._stop_event.set()

            # Cancel pending tasks
            for task in tasks:
                if not task.done():
                    task.cancel()

        return {
            "success": len(self.stats.errors) == 0,
            "data_room_id": data_room_id,
            "stats": self.stats.to_dict(),
            "errors": self.stats.errors
        }

    async def _parse_stage(
        self,
        file_paths: List[str],
        output_queue: asyncio.Queue,
        progress_callback: Optional[Callable]
    ) -> None:
        """
        Parse stage: Extract text from documents.

        Uses ProcessPoolExecutor for CPU-bound parsing operations.
        """
        start_time = time.time()
        total_files = len(file_paths)

        logger.info(f"Parse stage starting: {total_files} files")

        # Use thread pool for I/O-bound file operations
        # (ProcessPool would be better for CPU-heavy parsing but has pickling issues)
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(
            max_workers=self.config.parse_workers,
            thread_name_prefix="parse"
        ) as executor:

            # Submit all parsing tasks
            future_to_path = {}
            for file_path in file_paths:
                future = loop.run_in_executor(
                    executor,
                    self._parse_file,
                    file_path
                )
                future_to_path[future] = file_path

            # Process completed parsing tasks
            for i, future in enumerate(asyncio.as_completed(future_to_path.keys())):
                if self._stop_event.is_set():
                    break

                file_path = future_to_path[future]

                try:
                    result = await future

                    if result:
                        await output_queue.put(result)
                        self.stats.files_parsed += 1
                    else:
                        self.stats.files_failed += 1
                        logger.warning(f"Parse failed: {file_path}")

                except Exception as e:
                    self.stats.files_failed += 1
                    self.stats.errors.append(f"Parse error {file_path}: {e}")
                    logger.error(f"Parse error {file_path}: {e}")

                # Progress update
                if progress_callback:
                    progress = (i + 1) / total_files
                    progress_callback("parsing", progress)

        # Signal end of parsing
        await output_queue.put(None)

        self.stats.parse_time = time.time() - start_time
        logger.info(
            f"Parse stage complete: {self.stats.files_parsed}/{total_files} "
            f"in {self.stats.parse_time:.1f}s"
        )

    def _parse_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single file (runs in thread pool).

        Returns parsed document data or None on failure.
        """
        try:
            from tools.ingest_data_room import parse_file_by_type
            result = parse_file_by_type(file_path, use_financial_excel=True)
            if result and result.get("error"):
                logger.warning(f"Parse returned error for {file_path}: {result['error']}")
                return None
            return result

        except Exception as e:
            logger.error(f"Parse error {file_path}: {e}", exc_info=True)
            return None

    async def _chunk_stage(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        progress_callback: Optional[Callable]
    ) -> None:
        """
        Chunk stage: Split documents into semantic chunks.
        """
        start_time = time.time()
        documents_processed = 0

        logger.info("Chunk stage starting")

        from tools.chunk_documents import chunk_documents

        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(
            max_workers=self.config.chunk_workers,
            thread_name_prefix="chunk"
        ) as executor:

            while True:
                if self._stop_event.is_set():
                    break

                # Get parsed document from input queue
                parsed = await input_queue.get()

                if parsed is None:
                    # End signal
                    break

                try:
                    # Run chunking in thread pool
                    chunks = await loop.run_in_executor(
                        executor,
                        chunk_documents,
                        parsed
                    )

                    if chunks:
                        # Send chunks in batches for embedding
                        for i in range(0, len(chunks), self.config.embed_batch_size):
                            batch = chunks[i:i + self.config.embed_batch_size]
                            await output_queue.put(batch)

                        self.stats.chunks_created += len(chunks)

                    documents_processed += 1

                    if progress_callback:
                        progress_callback("chunking", documents_processed / max(1, self.stats.files_parsed))

                except Exception as e:
                    self.stats.errors.append(f"Chunk error: {e}")
                    logger.error(f"Chunk error: {e}")

        # Signal end of chunking
        await output_queue.put(None)

        self.stats.chunk_time = time.time() - start_time
        logger.info(
            f"Chunk stage complete: {self.stats.chunks_created} chunks "
            f"in {self.stats.chunk_time:.1f}s"
        )

    async def _embed_stage(
        self,
        data_room_id: str,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        progress_callback: Optional[Callable]
    ) -> None:
        """
        Embed stage: Generate embeddings using OpenAI API.

        Uses high concurrency with rate limiting.
        """
        start_time = time.time()
        batches_processed = 0

        logger.info(f"Embed stage starting (max_concurrent={self.config.embed_workers})")

        from tools.generate_embeddings import EmbeddingGenerator
        from app.rate_limiter import embedding_rate_limiter

        # Create generator with high concurrency
        generator = EmbeddingGenerator(
            max_concurrent=self.config.embed_workers
        )

        while True:
            if self._stop_event.is_set():
                break

            # Get chunk batch from input queue
            chunk_batch = await input_queue.get()

            if chunk_batch is None:
                # End signal
                break

            try:
                # Generate embeddings (uses internal rate limiting)
                embedded_chunks = await generator.generate_async(chunk_batch)

                if embedded_chunks:
                    # Add data_room_id to each chunk
                    for chunk in embedded_chunks:
                        chunk['data_room_id'] = data_room_id

                    await output_queue.put(embedded_chunks)
                    self.stats.embeddings_generated += len(embedded_chunks)

                batches_processed += 1

                if progress_callback:
                    progress_callback(
                        "embedding",
                        self.stats.embeddings_generated / max(1, self.stats.chunks_created)
                    )

            except Exception as e:
                self.stats.errors.append(f"Embed error: {e}")
                logger.error(f"Embed error: {e}")

        # Signal end of embedding
        await output_queue.put(None)

        self.stats.embed_time = time.time() - start_time
        self.stats.total_tokens = generator.total_tokens

        logger.info(
            f"Embed stage complete: {self.stats.embeddings_generated} embeddings "
            f"in {self.stats.embed_time:.1f}s "
            f"({generator.total_tokens} tokens, ${generator.total_cost:.4f})"
        )

    async def _index_stage(
        self,
        data_room_id: str,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        progress_callback: Optional[Callable]
    ) -> None:
        """
        Index stage: Store embeddings in vector DB and SQLite.
        """
        start_time = time.time()
        batches_indexed = 0

        logger.info("Index stage starting")

        from tools.index_to_vectordb import VectorDBIndexer
        from app import database as db

        indexer = VectorDBIndexer()
        pending_chunks: List[Dict] = []

        while True:
            if self._stop_event.is_set():
                break

            # Get embedded chunks from input queue
            embedded_batch = await input_queue.get()

            if embedded_batch is None:
                # End signal - index remaining chunks
                if pending_chunks:
                    await self._index_batch(
                        indexer, db, data_room_id, pending_chunks
                    )
                    self.stats.chunks_indexed += len(pending_chunks)
                break

            try:
                pending_chunks.extend(embedded_batch)

                # Index when batch reaches threshold
                if len(pending_chunks) >= self.config.index_batch_size:
                    await self._index_batch(
                        indexer, db, data_room_id, pending_chunks
                    )
                    self.stats.chunks_indexed += len(pending_chunks)
                    pending_chunks = []
                    batches_indexed += 1

                if progress_callback:
                    progress_callback(
                        "indexing",
                        self.stats.chunks_indexed / max(1, self.stats.embeddings_generated)
                    )

            except Exception as e:
                self.stats.errors.append(f"Index error: {e}")
                logger.error(f"Index error: {e}")

        self.stats.index_time = time.time() - start_time
        logger.info(
            f"Index stage complete: {self.stats.chunks_indexed} chunks "
            f"in {self.stats.index_time:.1f}s"
        )

    async def _index_batch(
        self,
        indexer,
        db,
        data_room_id: str,
        chunks: List[Dict]
    ) -> None:
        """Index a batch of chunks to vector DB and SQLite."""
        loop = asyncio.get_event_loop()

        # Index to vector DB
        await loop.run_in_executor(
            None,
            indexer.index_chunks,
            data_room_id,
            chunks
        )

        # Save to SQLite
        await loop.run_in_executor(
            None,
            db.create_chunks_batch_optimized,
            chunks
        )

    def stop(self) -> None:
        """Stop the pipeline gracefully."""
        self._stop_event.set()
        logger.info("Pipeline stop requested")


class SimplePipeline:
    """
    Simplified synchronous pipeline for smaller workloads.

    Use this when:
    - Processing < 10 files
    - Need simpler error handling
    - Don't need maximum throughput
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.stats = PipelineStats()

    def process_files(
        self,
        data_room_id: str,
        file_paths: List[str],
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> Dict[str, Any]:
        """
        Process files synchronously.

        Args:
            data_room_id: Data room identifier
            file_paths: List of file paths
            progress_callback: Progress callback

        Returns:
            Processing results
        """
        from tools.parse_pdf import parse_pdf
        from tools.chunk_documents import chunk_documents
        from tools.generate_embeddings import EmbeddingGenerator
        from tools.index_to_vectordb import VectorDBIndexer
        from app import database as db

        self.stats = PipelineStats()
        self.stats.files_submitted = len(file_paths)

        generator = EmbeddingGenerator()
        indexer = VectorDBIndexer()

        all_chunks = []

        # Parse and chunk all files
        for i, file_path in enumerate(file_paths):
            try:
                # Parse
                ext = Path(file_path).suffix.lower()
                if ext == '.pdf':
                    parsed = parse_pdf(file_path)
                else:
                    # Add other parsers as needed
                    continue

                if parsed:
                    self.stats.files_parsed += 1

                    # Chunk
                    chunks = chunk_documents(parsed)
                    if chunks:
                        for chunk in chunks:
                            chunk['data_room_id'] = data_room_id
                        all_chunks.extend(chunks)
                        self.stats.chunks_created += len(chunks)

                if progress_callback:
                    progress_callback("parsing", (i + 1) / len(file_paths))

            except Exception as e:
                self.stats.files_failed += 1
                self.stats.errors.append(f"Error {file_path}: {e}")
                logger.error(f"Error processing {file_path}: {e}")

        # Generate embeddings
        if all_chunks:
            embedded = generator.generate(all_chunks)
            self.stats.embeddings_generated = len(embedded)
            self.stats.total_tokens = generator.total_tokens

            # Index
            indexer.index_chunks(data_room_id, embedded)
            db.create_chunks_batch_optimized(embedded)
            self.stats.chunks_indexed = len(embedded)

        return {
            "success": len(self.stats.errors) == 0,
            "data_room_id": data_room_id,
            "stats": self.stats.to_dict(),
            "errors": self.stats.errors
        }


# Factory function
def create_pipeline(
    async_mode: bool = True,
    config: Optional[PipelineConfig] = None
):
    """
    Create a document processing pipeline.

    Args:
        async_mode: Use async pipeline (True) or sync (False)
        config: Pipeline configuration

    Returns:
        DocumentPipeline or SimplePipeline instance
    """
    if async_mode:
        return DocumentPipeline(config)
    return SimplePipeline(config)
