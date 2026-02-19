"""
Background worker for processing jobs from the job queue.

Provides reliable job processing with:
- Memory-aware processing
- Graceful shutdown
- Job-specific handlers
"""

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from loguru import logger

from app.config import settings
from app.job_queue import JobQueue, JobType, job_queue
from app.utils import can_process_file, get_memory_usage_mb
from app import database as db


# In-memory progress tracker to avoid repeated COUNT queries per file completion
_folder_progress_lock = threading.Lock()
_folder_progress: Dict[str, Dict[str, int]] = {}  # folder_id → {processed, total}

# In-memory progress tracker for multi-file uploads (same pattern as folder progress)
_upload_progress_lock = threading.Lock()
_upload_progress: Dict[str, Dict[str, int]] = {}  # data_room_id → {processed, total}


class JobWorker:
    """
    Background worker that processes jobs from the queue.

    Features:
    - Runs in a daemon thread
    - Memory-aware processing (checks before heavy jobs)
    - Graceful shutdown support
    - Extensible job handlers
    """

    def __init__(
        self,
        queue: JobQueue,
        worker_id: Optional[str] = None,
        poll_interval: float = 1.0
    ):
        """
        Initialize job worker.

        Args:
            queue: JobQueue instance to pull jobs from
            worker_id: Unique worker identifier
            poll_interval: Seconds between polling for new jobs
        """
        import os
        self.queue = queue
        self.worker_id = worker_id or f"worker_{os.getpid()}_{threading.current_thread().ident}"
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._handlers: Dict[str, Callable] = {}
        self._indexer = None  # Lazy-initialized VectorDBIndexer, reused across files
        self._chunker = None  # Lazy-initialized DocumentChunker, reused across files for token cache

        # Register default handlers
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register built-in job handlers."""
        self.register_handler(JobType.PROCESS_FILE.value, self._handle_process_file)
        self.register_handler(JobType.PROCESS_DATA_ROOM.value, self._handle_process_data_room)
        self.register_handler(JobType.GENERATE_EMBEDDINGS.value, self._handle_generate_embeddings)

    def register_handler(self, job_type: str, handler: Callable[[Dict[str, Any]], None]):
        """
        Register a handler for a job type.

        Args:
            job_type: Job type string
            handler: Function that takes job payload dict
        """
        self._handlers[job_type] = handler
        logger.debug(f"Registered handler for job type: {job_type}")

    def start(self):
        """Start the worker thread."""
        if self._thread and self._thread.is_alive():
            logger.warning(f"Worker {self.worker_id} already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"JobWorker-{self.worker_id}"
        )
        self._thread.start()
        logger.info(f"Started job worker: {self.worker_id}")

    def stop(self, timeout: float = 10.0):
        """
        Stop the worker gracefully.

        Args:
            timeout: Seconds to wait for current job to complete
        """
        logger.info(f"Stopping worker {self.worker_id}...")
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning(f"Worker {self.worker_id} did not stop within timeout")

    @property
    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._thread is not None and self._thread.is_alive()

    def _run(self):
        """Main worker loop."""
        logger.info(f"Worker {self.worker_id} started processing loop")

        while not self._stop_event.is_set():
            try:
                job = self.queue.claim_job(self.worker_id)

                if job:
                    self._process_job(job)
                else:
                    # No jobs available, wait before polling again
                    self._stop_event.wait(self.poll_interval)

            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                time.sleep(1)  # Brief pause on error

        logger.info(f"Worker {self.worker_id} stopped")

    def _process_job(self, job: Dict[str, Any]):
        """
        Process a single job.

        Args:
            job: Job dict from queue
        """
        job_id = job['id']
        job_type = job['job_type']
        payload = job['payload']

        logger.info(f"[{job_id}] Processing job (type={job_type}, attempt={job['attempts']})")
        start_time = time.time()

        try:
            # Get handler for job type
            handler = self._handlers.get(job_type)

            if not handler:
                raise ValueError(f"No handler registered for job type: {job_type}")

            # Execute handler
            handler(payload)

            # Mark complete
            duration_ms = int((time.time() - start_time) * 1000)
            self.queue.complete_job(job_id)
            logger.success(f"[{job_id}] Job completed in {duration_ms}ms")

        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            logger.error(f"[{job_id}] Job failed: {e}")
            self.queue.fail_job(job_id, error_msg, retry=True)

            # Only mark as permanently failed if no retries remain
            job_info = self.queue.get_job(job_id)
            is_permanent_failure = job_info and job_info['status'] == 'failed'

            # Update document parse_status so the sidebar shows failure instead of a spinner
            document_id = payload.get('document_id')
            if document_id and is_permanent_failure:
                db.update_document_status(document_id, 'failed', error_message=str(e)[:500])

            # Update synced_file status if this came from Drive sync
            synced_file_id = payload.get('synced_file_id')
            if synced_file_id and is_permanent_failure:
                db.update_synced_file_status(
                    synced_file_id,
                    sync_status='failed',
                    error_message=str(e)[:500]
                )

            # Increment progress even for permanent failures so the
            # completion condition (processed >= total) can still be met.
            # Without this, a single failed file blocks the entire data room.
            if is_permanent_failure and payload.get('connected_folder_id'):
                self._update_folder_progress(payload, payload.get('data_room_id', ''))

            if is_permanent_failure and payload.get('upload_total_files') and payload.get('data_room_id'):
                self._update_upload_progress(payload['data_room_id'], payload['upload_total_files'])

    # ========================================================================
    # Folder Progress Tracking
    # ========================================================================

    def _update_folder_progress(self, payload: Dict[str, Any], data_room_id: str):
        """
        Increment folder progress counter and check for completion.
        Called from both success and permanent-failure paths so that
        failed files don't prevent the data room from completing.
        """
        connected_folder_id = payload.get('connected_folder_id')
        if not connected_folder_id:
            return

        with _folder_progress_lock:
            if connected_folder_id not in _folder_progress:
                # Initialize from DB — count already-finished files so the counter
                # survives server restarts (in-memory dict is lost on restart).
                # Use total_files from connected_folder (set during discovery phase)
                # as the authoritative total. This avoids a race condition where
                # dynamically counting by status returns a partial count while
                # downloads are still in progress, causing premature completion.
                already_done = db.count_processed_synced_files(connected_folder_id)
                folder_record = db.get_connected_folder(connected_folder_id)
                total = folder_record['total_files'] if folder_record and folder_record.get('total_files') else 0
                _folder_progress[connected_folder_id] = {'processed': already_done, 'total': total}

            _folder_progress[connected_folder_id]['processed'] += 1
            processed = _folder_progress[connected_folder_id]['processed']
            total = _folder_progress[connected_folder_id]['total']

        # Progress: 10-100% range (0-10% reserved for discovery/download phases)
        progress = (10 + (processed / total) * 90) if total > 0 else 10

        db.update_connected_folder_status(
            connected_folder_id,
            sync_status='syncing',
            processed_files=processed
        )

        # Update data room progress (monotonically increasing by design).
        # Skip intermediate 'parsing' update on last file — completion block below
        # sets 'complete' + 100% atomically, avoiding a race where the frontend
        # polls and sees status='parsing' at 100%.
        if data_room_id and processed < total:
            db.update_data_room_status(data_room_id, 'parsing', progress=progress)

        # Complete when all files are processed (including failures)
        if processed >= total and total > 0:
            # Verify against DB to prevent race conditions. The in-memory total
            # may have been initialized before all files were discovered/enqueued.
            # Only trigger completion when:
            # 1. Downloads are finished (sync_stage past 'processing')
            # 2. All synced files are in a terminal state (complete/failed)
            folder_record = db.get_connected_folder(connected_folder_id)
            downloads_done = folder_record.get('sync_stage') in ('queued', 'complete') if folder_record else False
            db_processed = db.count_processed_synced_files(connected_folder_id)
            # Use actual synced_files count from DB as the authoritative total.
            # The total_files field on connected_folders can be stale if discovery
            # overcounted (e.g. ON CONFLICT upserts during resumed discovery).
            db_total = db.count_all_synced_files(connected_folder_id)

            if not downloads_done or db_processed < db_total:
                # Not actually complete — downloads still running or files still processing.
                # Correct the in-memory total so future checks use the right number.
                with _folder_progress_lock:
                    if connected_folder_id in _folder_progress:
                        _folder_progress[connected_folder_id]['total'] = db_total
                if data_room_id:
                    corrected_progress = (10 + (db_processed / db_total) * 90) if db_total > 0 else 10
                    db.update_data_room_status(data_room_id, 'parsing', progress=corrected_progress)
                return

            try:
                db.update_connected_folder_stage(connected_folder_id, sync_stage='complete')
            except Exception as e:
                logger.error(f"Failed to update folder stage to complete: {e}")
            try:
                db.update_connected_folder_status(connected_folder_id, sync_status='active', processed_files=processed)
            except Exception as e:
                logger.error(f"Failed to update folder status to active: {e}")
            try:
                if data_room_id:
                    db.update_data_room_status(data_room_id, 'complete', progress=100)
                    doc_count = len(db.get_documents_by_data_room(data_room_id))
                    with db.get_db_connection() as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "UPDATE data_rooms SET total_documents = %s WHERE id = %s",
                            (doc_count, data_room_id)
                        )
                        conn.commit()
            except Exception as e:
                logger.error(f"Failed to update data room to complete: {e}")
            # Clean up in-memory tracker
            with _folder_progress_lock:
                _folder_progress.pop(connected_folder_id, None)

    # ========================================================================
    # Upload Progress Tracking (multi-file uploads)
    # ========================================================================

    def _update_upload_progress(self, data_room_id: str, total_files: int):
        """
        Increment upload progress counter and check for completion.
        Called after each file finishes processing in a multi-file upload.
        """
        # Use actual document count as the source of truth for total,
        # not the declared total_files from the payload.  Files may be
        # skipped during upload validation (empty, invalid format), so
        # the declared total can exceed the actual enqueued job count,
        # causing the data room to never complete.
        docs = db.get_documents_by_data_room(data_room_id)
        actual_total = len(docs) if docs else total_files

        with _upload_progress_lock:
            if data_room_id not in _upload_progress:
                # Initialize — count already-finished documents so tracker
                # survives server restarts (in-memory dict is lost on restart).
                already_done = sum(1 for d in docs if d.get('parse_status') in ('parsed', 'failed'))
                _upload_progress[data_room_id] = {'processed': already_done, 'total': actual_total}
            else:
                # Update total to reflect actual document count in case
                # more files have been uploaded since initialization.
                _upload_progress[data_room_id]['total'] = actual_total

            _upload_progress[data_room_id]['processed'] += 1
            processed = _upload_progress[data_room_id]['processed']
            total = _upload_progress[data_room_id]['total']

        progress = int(10 + (processed / total) * 90) if total > 0 else 10

        if processed < total:
            db.update_data_room_status(data_room_id, 'parsing', progress=progress)
        else:
            # All files done — check for failures
            failed_names = [d['file_name'] for d in docs if d.get('parse_status') == 'failed']
            if failed_names and len(failed_names) == total:
                error_msg = f"All {total} files failed: {', '.join(failed_names[:5])}"
                db.update_data_room_status(data_room_id, 'failed', error_message=error_msg)
            elif failed_names:
                error_msg = f"{len(failed_names)}/{total} files failed: {', '.join(failed_names[:5])}"
                db.update_data_room_status(data_room_id, 'complete', progress=100, error_message=error_msg)
            else:
                db.update_data_room_status(data_room_id, 'complete', progress=100)

            # Clean up in-memory tracker
            with _upload_progress_lock:
                _upload_progress.pop(data_room_id, None)

            logger.success(f"[{data_room_id}] All {total} files processed ({total - len(failed_names)} succeeded, {len(failed_names)} failed)")

    # ========================================================================
    # Job Handlers
    # ========================================================================

    @staticmethod
    def _is_english_text(text: str) -> bool:
        """Check if text is predominantly English by examining character distribution.

        Uses a simple heuristic: English text is mostly ASCII letters, digits,
        punctuation, and whitespace. Non-Latin scripts (Arabic, CJK, Cyrillic, etc.)
        will have a low ratio of these characters.
        """
        if not text or len(text.strip()) < 50:
            return True  # Too short to judge — assume English

        # Sample up to 2000 chars, strip whitespace-heavy sections
        sample = text[:2000]
        # Count characters that are ASCII letters or common in English text
        english_chars = sum(1 for c in sample if c.isascii())
        ratio = english_chars / len(sample)

        # English text is typically >85% ASCII. Non-Latin scripts drop well below 50%.
        return ratio > 0.6

    def _mark_file_failed(self, payload: Dict[str, Any], data_room_id: str, error_msg: str, page_count: int = 0):
        """Mark a file as failed and update all progress trackers so the data room can still complete."""
        if 'document_id' in payload:
            db.update_document_status(
                payload['document_id'], 'failed',
                error_message=error_msg[:500],
                page_count=page_count,
            )
        synced_file_id = payload.get('synced_file_id')
        if synced_file_id:
            db.update_synced_file_status(synced_file_id, sync_status='failed', error_message=error_msg[:500])
        self._update_folder_progress(payload, data_room_id)
        if not payload.get('connected_folder_id'):
            upload_total = payload.get('upload_total_files')
            if upload_total and upload_total > 1 and data_room_id:
                self._update_upload_progress(data_room_id, upload_total)
            elif data_room_id:
                db.update_data_room_status(data_room_id, 'complete', progress=100)

    def _handle_process_file(self, payload: Dict[str, Any]):
        """
        Process a single file (parse, chunk, embed, index).

        Payload:
            data_room_id: str
            file_path: str
            file_name: str
            document_id: str (optional)
        """
        data_room_id = payload['data_room_id']
        file_path = payload['file_path']
        file_name = payload.get('file_name', Path(file_path).name)

        logger.info(f"[{data_room_id}] Processing file: {file_name}")

        # Signal that a worker has claimed this job (moves progress off 0%)
        # Skip for folder syncs and multi-file uploads (they have their own progress tracking)
        if data_room_id and not payload.get('connected_folder_id') and not (payload.get('upload_total_files', 0) > 1):
            db.update_data_room_status(data_room_id, 'uploading', progress=8)

        # Guard: skip if document is already fully parsed (prevents re-parsing on restart)
        document_id = payload.get('document_id')
        if document_id:
            existing_doc = db.get_document_by_id(document_id)
            if existing_doc and existing_doc.get('parse_status') == 'parsed':
                logger.info(f"[{data_room_id}] Skipping already-parsed document: {file_name}")
                synced_file_id = payload.get('synced_file_id')
                if synced_file_id:
                    db.update_synced_file_status(synced_file_id, sync_status='complete', document_id=document_id)
                self._update_folder_progress(payload, data_room_id)
                if not payload.get('connected_folder_id'):
                    upload_total = payload.get('upload_total_files')
                    if upload_total and upload_total > 1 and data_room_id:
                        self._update_upload_progress(data_room_id, upload_total)
                    elif data_room_id:
                        db.update_data_room_status(data_room_id, 'complete', progress=100)
                return

        # Check memory before processing
        can_process, error = can_process_file(file_path)
        if not can_process:
            raise MemoryError(error)

        # Import processing tools
        from tools.ingest_data_room import parse_file_by_type
        from tools.chunk_documents import chunk_documents, DocumentChunker
        from tools.generate_embeddings import generate_embeddings, generate_embeddings_streaming
        from tools.index_to_vectordb import index_to_vectordb, VectorDBIndexer

        file_path_obj = Path(file_path)
        file_type = file_path_obj.suffix.lower()

        # Parse document with timeout to prevent hanging
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

        # Update data room status to parsing
        # Skip for folder syncs (folder-level progress handles it) and
        # multi-file uploads (proportional progress in _update_upload_progress handles it).
        is_multi_file = payload.get('upload_total_files', 0) > 1
        if data_room_id and not payload.get('connected_folder_id'):
            if is_multi_file:
                db.update_data_room_status(data_room_id, 'parsing', progress=5)
            else:
                db.update_data_room_status(data_room_id, 'parsing', progress=25)

        logger.info(f"[{data_room_id}] Parsing {file_name} (timeout: {settings.parse_timeout_seconds}s)...")
        with ThreadPoolExecutor(max_workers=1) as parse_executor:
            future = parse_executor.submit(
                parse_file_by_type,
                str(file_path),
                True,  # use_ocr
                settings.max_ocr_pages,
                True,  # use_financial_excel
            )
            try:
                parsed = future.result(timeout=settings.parse_timeout_seconds)
            except FuturesTimeoutError:
                raise TimeoutError(f"Parsing timed out after {settings.parse_timeout_seconds}s")

        # Check if parser returned an error (e.g., encrypted PDF, invalid header, empty file)
        parse_error = parsed.get('error')
        if parse_error:
            error_msg = f"Parse error: {parse_error}"
            logger.error(f"[{data_room_id}] {file_name}: {error_msg}")
            self._mark_file_failed(payload, data_room_id, error_msg, page_count=parsed.get('page_count', 0))
            return  # Deterministic error — no point retrying; _mark_file_failed already updated progress

        # Skip non-English documents (spreadsheets are language-agnostic, so skip this check for them)
        if file_type not in ('.xlsx', '.xls', '.csv'):
            text_sample = (parsed.get('text') or '')[:2000]
            if text_sample and not self._is_english_text(text_sample):
                error_msg = f"Skipped: document does not appear to be in English"
                logger.warning(f"[{data_room_id}] {file_name}: {error_msg}")
                self._mark_file_failed(payload, data_room_id, error_msg, page_count=parsed.get('page_count', 0))
                return

        # Chunk document
        logger.info(f"[{data_room_id}] Chunking {file_name}...")
        if self._chunker is None:
            self._chunker = DocumentChunker(chunk_size=settings.max_chunk_size, overlap=settings.chunk_overlap)
        chunks = chunk_documents(parsed, chunker=self._chunker)

        # Chunking complete — intermediate milestone so bar doesn't sit at 25% for the full parse+chunk duration
        if data_room_id and not payload.get('connected_folder_id') and not is_multi_file:
            db.update_data_room_status(data_room_id, 'parsing', progress=35)

        indexed_count = 0
        total_token_count = 0

        if not chunks:
            error_msg = f"No text could be extracted from {file_name}. The document may be image-only, empty, or in an unsupported format."
            logger.error(f"[{data_room_id}] {error_msg}")
            self._mark_file_failed(payload, data_room_id, error_msg, page_count=parsed.get('page_count', 0))
            return  # Deterministic error — no point retrying; _mark_file_failed already updated progress
        else:
            # Update data room status to embedding
            # Skip for folder syncs and multi-file uploads (same reason as parsing above).
            if data_room_id and not payload.get('connected_folder_id') and not is_multi_file:
                db.update_data_room_status(data_room_id, 'indexing', progress=50)

            # Generate embeddings with streaming (embed + index in parallel)
            logger.info(f"[{data_room_id}] Streaming embeddings for {len(chunks)} chunks (batch_size={settings.batch_size}, max_concurrent={settings.embedding_max_concurrent})...")

            document_id = payload.get('document_id')
            if self._indexer is None:
                self._indexer = VectorDBIndexer()
            indexer = self._indexer
            WRITE_TIMEOUT = 120  # seconds — timeout per write future
            MAX_PENDING_WRITES = 4  # backpressure: max in-flight write batches before blocking
            write_pool = ThreadPoolExecutor(max_workers=3)

            try:
                pending_futures = []
                for batch in generate_embeddings_streaming(
                    chunks,
                    batch_size=settings.batch_size,
                    max_concurrent=settings.embedding_max_concurrent
                ):
                    # Add document metadata to batch
                    for chunk in batch:
                        chunk['document_id'] = document_id
                        chunk['data_room_id'] = data_room_id

                    # Track token count incrementally (no need to accumulate all chunks)
                    total_token_count += sum(c.get('token_count', 0) for c in batch)

                    # Pipeline VectorDB and DB writes — don't block between batches
                    vector_future = write_pool.submit(index_to_vectordb, data_room_id, batch, indexer=indexer)
                    db_future = write_pool.submit(db.create_chunks_batch_optimized, batch)
                    pending_futures.append((vector_future, db_future))

                    # Backpressure: drain oldest writes when too many are in-flight.
                    # Prevents unbounded memory growth when embeddings outpace writes.
                    while len(pending_futures) >= MAX_PENDING_WRITES:
                        v_fut, d_fut = pending_futures.pop(0)
                        v_fut.result(timeout=WRITE_TIMEOUT)
                        d_fut.result(timeout=WRITE_TIMEOUT)

                    indexed_count += len(batch)
                    logger.debug(f"[{data_room_id}] Indexed batch: {indexed_count}/{len(chunks)} chunks")

                    # Sub-file progress: interpolate within this file's slice so the
                    # progress bar moves during large-file embedding, not just at file boundaries.
                    if not payload.get('connected_folder_id') and payload.get('upload_total_files', 0) > 1:
                        upload_total = payload['upload_total_files']
                        with _upload_progress_lock:
                            base = _upload_progress.get(data_room_id, {}).get('processed', 0)
                        file_frac = indexed_count / len(chunks)
                        overall = 10 + ((base + file_frac) / upload_total) * 90
                        db.update_data_room_status(data_room_id, 'parsing', progress=int(min(overall, 99)))
                    elif not payload.get('connected_folder_id') and not is_multi_file and data_room_id:
                        # Single-file embedding progress: range 50-95%
                        file_frac = indexed_count / len(chunks)
                        single_progress = 50 + file_frac * 45
                        db.update_data_room_status(data_room_id, 'indexing', progress=int(min(single_progress, 95)))

                # Drain remaining pipelined writes
                for v_fut, d_fut in pending_futures:
                    try:
                        v_fut.result(timeout=WRITE_TIMEOUT)
                    except (TimeoutError, FuturesTimeoutError):
                        raise TimeoutError(f"VectorDB write timed out after {WRITE_TIMEOUT}s for {file_name}")
                    try:
                        d_fut.result(timeout=WRITE_TIMEOUT)
                    except (TimeoutError, FuturesTimeoutError):
                        raise TimeoutError(f"Database write timed out after {WRITE_TIMEOUT}s for {file_name}")
            finally:
                write_pool.shutdown(wait=False)

        # Validate embedding results — fail if all embeddings failed
        failed_embedding_count = sum(1 for c in chunks if c.get('embedding_failed'))
        if failed_embedding_count == len(chunks):
            error_msg = f"All {len(chunks)} chunks failed embedding generation for {file_name}"
            logger.error(f"[{data_room_id}] {error_msg}")
            self._mark_file_failed(payload, data_room_id, error_msg, page_count=parsed.get('page_count', 0))
            return
        elif failed_embedding_count > 0:
            logger.warning(f"[{data_room_id}] {failed_embedding_count}/{len(chunks)} chunks failed embedding for {file_name}")

        # Update document status
        if 'document_id' in payload:
            db.update_document_status(
                payload['document_id'],
                'parsed',
                page_count=parsed.get('page_count', 0),
                token_count=total_token_count
            )

        # Update synced_file status if this came from Drive sync
        synced_file_id = payload.get('synced_file_id')
        if synced_file_id:
            db.update_synced_file_status(
                synced_file_id,
                sync_status='complete',
                document_id=payload.get('document_id')
            )

        # Update folder progress
        self._update_folder_progress(payload, data_room_id)

        if not payload.get('connected_folder_id'):
            upload_total = payload.get('upload_total_files')
            if upload_total and upload_total > 1 and data_room_id:
                # Multi-file upload: track progress and only complete when all files are done
                self._update_upload_progress(data_room_id, upload_total)
            elif data_room_id:
                # Single file or individual Drive file connection — mark complete immediately
                db.update_data_room_status(data_room_id, 'complete', progress=100)

        logger.success(f"[{data_room_id}] File {file_name} processed: {indexed_count} chunks")

        # Auto-trigger financial analysis for Excel files if enabled
        if file_type in ['.xlsx', '.xls'] and settings.enable_auto_financial_analysis:
            try:
                from tools.financial_analysis_agent import analyze_financial_model

                document_id = payload.get('document_id')
                if document_id and file_path_obj.exists():
                    logger.info(f"[{data_room_id}] Auto-triggering financial analysis for {file_name}")

                    result = analyze_financial_model(
                        file_path=str(file_path),
                        data_room_id=data_room_id,
                        document_id=document_id,
                        analysis_model=settings.financial_analysis_model,
                        extraction_model=settings.financial_extraction_model,
                        max_cost=settings.financial_analysis_max_cost
                    )

                    db.save_financial_analysis(
                        analysis_id=result['analysis_id'],
                        data_room_id=data_room_id,
                        document_id=document_id,
                        file_name=result['file_name'],
                        status=result['status'],
                        model_structure=result.get('model_structure'),
                        extracted_metrics=result.get('extracted_metrics'),
                        time_series=result.get('time_series'),
                        missing_metrics=result.get('missing_metrics'),
                        validation_results=result.get('validation_results'),
                        insights=result.get('insights'),
                        follow_up_questions=result.get('follow_up_questions'),
                        key_metrics_summary=result.get('key_metrics_summary'),
                        risk_assessment=result.get('risk_assessment'),
                        investment_thesis_notes=result.get('investment_thesis_notes'),
                        executive_summary=result.get('executive_summary'),
                        analysis_cost=result.get('analysis_cost', 0.0),
                        tokens_used=result.get('tokens_used', 0),
                        processing_time_ms=result.get('processing_time_ms', 0),
                        error_message=result.get('error')
                    )

                    logger.success(f"[{data_room_id}] Financial analysis complete for {file_name}")

            except ImportError as e:
                logger.warning(f"[{data_room_id}] Financial analysis module not available: {e}")
            except Exception as e:
                logger.error(f"[{data_room_id}] Financial analysis failed for {file_name}: {e}")

    def _handle_process_data_room(self, payload: Dict[str, Any]):
        """
        Process all files in a data room IN PARALLEL.

        Payload:
            data_room_id: str
            file_paths: List[str]
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        data_room_id = payload['data_room_id']
        file_paths = payload['file_paths']

        # Process files in parallel (not one-by-one!)
        max_parallel_files = min(len(file_paths), settings.max_parse_workers)
        logger.info(f"[{data_room_id}] Processing {len(file_paths)} files in parallel (max {max_parallel_files} concurrent)")

        succeeded = 0
        failed = 0
        failed_names = []

        with ThreadPoolExecutor(max_workers=max_parallel_files) as executor:
            futures = {
                executor.submit(self._handle_process_file, {
                    'data_room_id': data_room_id,
                    'file_path': file_path,
                    'file_name': Path(file_path).name
                }): file_path
                for file_path in file_paths
            }

            for future in as_completed(futures):
                file_path = futures[future]
                try:
                    future.result()
                    succeeded += 1
                    logger.info(f"[{data_room_id}] Completed {succeeded + failed}/{len(file_paths)}: {Path(file_path).name}")
                except Exception as e:
                    failed += 1
                    failed_names.append(Path(file_path).name)
                    logger.error(f"[{data_room_id}] Failed to process {file_path}: {e}")

        # Determine final status based on results
        if succeeded == 0 and failed > 0:
            error_msg = f"All {failed} files failed processing: {', '.join(failed_names[:5])}"
            if len(failed_names) > 5:
                error_msg += f" (and {len(failed_names) - 5} more)"
            db.update_data_room_status(data_room_id, "failed", error_message=error_msg)
            logger.error(f"[{data_room_id}] Data room processing FAILED: {error_msg}")
        elif failed > 0:
            error_msg = f"{failed}/{len(file_paths)} files failed: {', '.join(failed_names[:5])}"
            if len(failed_names) > 5:
                error_msg += f" (and {len(failed_names) - 5} more)"
            db.update_data_room_status(data_room_id, "complete", progress=100, error_message=error_msg)
            logger.warning(f"[{data_room_id}] Data room processing completed with errors: {error_msg}")
        else:
            db.update_data_room_status(data_room_id, "complete", progress=100)
            logger.success(f"[{data_room_id}] Data room processing complete")

    def _handle_generate_embeddings(self, payload: Dict[str, Any]):
        """
        Generate embeddings for chunks (streaming mode).

        Payload:
            data_room_id: str
            chunks: List[Dict] - list of chunk dicts
        """
        from tools.generate_embeddings import generate_embeddings
        from tools.index_to_vectordb import index_to_vectordb

        data_room_id = payload['data_room_id']
        chunks = payload['chunks']

        logger.info(f"[{data_room_id}] Generating embeddings for {len(chunks)} chunks (batch_size={settings.batch_size}, max_concurrent={settings.embedding_max_concurrent})")

        # Generate embeddings
        chunks_with_embeddings = generate_embeddings(
            chunks,
            batch_size=settings.batch_size,
            max_concurrent=settings.embedding_max_concurrent
        )

        # Run VectorDB and DB writes in parallel
        with ThreadPoolExecutor(max_workers=6) as write_pool:
            vector_future = write_pool.submit(index_to_vectordb, data_room_id, chunks_with_embeddings)
            db_future = write_pool.submit(db.create_chunks_batch_optimized, chunks_with_embeddings)
            vector_future.result()
            db_future.result()

        logger.success(f"[{data_room_id}] Embeddings generated and indexed")


class WorkerPool:
    """
    Auto-scaling pool of job workers for concurrent processing.

    Tier 4 Features:
    - Automatic scaling based on queue depth
    - Memory-aware scaling (pause if memory is tight)
    - Configurable min/max workers
    - Scaling monitor thread
    """

    def __init__(
        self,
        min_workers: int = 5,
        max_workers: int = 20,
        scale_up_threshold: int = 10,
        scale_down_threshold: int = 3,
        check_interval: float = 2.0
    ):
        """
        Initialize auto-scaling worker pool.

        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers
            scale_up_threshold: Queue depth to trigger scale up
            scale_down_threshold: Queue depth to trigger scale down
            check_interval: Seconds between scaling checks
        """
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.check_interval = check_interval

        self.workers: list[JobWorker] = []
        self._scaling_lock = __import__('threading').Lock()
        self._monitor_thread = None
        self._stop_event = __import__('threading').Event()
        self._worker_counter = 0

    def start(self, enable_autoscaling: bool = True):
        """
        Start the worker pool.

        Args:
            enable_autoscaling: Enable automatic scaling
        """
        # Start minimum workers
        for i in range(self.min_workers):
            self._add_worker()

        logger.info(f"Started worker pool with {self.min_workers} workers")

        # Start scaling monitor
        if enable_autoscaling:
            self._stop_event.clear()
            self._monitor_thread = __import__('threading').Thread(
                target=self._scaling_monitor,
                daemon=True,
                name="WorkerScaler"
            )
            self._monitor_thread.start()
            logger.info("Auto-scaling monitor started")

    def _add_worker(self) -> JobWorker:
        """Add a new worker to the pool."""
        self._worker_counter += 1
        worker = JobWorker(
            queue=job_queue,
            worker_id=f"scalable_worker_{self._worker_counter}",
            poll_interval=settings.job_poll_interval_seconds
        )
        worker.start()
        self.workers.append(worker)
        return worker

    def _remove_worker(self) -> bool:
        """Remove the last idle worker from the pool."""
        if len(self.workers) <= self.min_workers:
            return False

        # Find an idle worker (not currently processing)
        for i in range(len(self.workers) - 1, -1, -1):
            worker = self.workers[i]
            if worker.is_running:
                try:
                    worker.stop(timeout=5.0)
                    self.workers.pop(i)
                    return True
                except Exception as e:
                    logger.warning(f"Error stopping worker: {e}")

        return False

    def _scaling_monitor(self):
        """Monitor queue depth, replace dead workers, and scale pool."""
        while not self._stop_event.is_set():
            try:
                # Get queue statistics
                stats = job_queue.get_queue_stats()
                pending = stats.get("pending", 0)
                running = stats.get("running", 0)

                # Replace dead workers — threads can die from segfaults in
                # native libraries (PyMuPDF, etc.) without raising a Python
                # exception.  Without this check, dead workers stay in the
                # list and block scaling.
                with self._scaling_lock:
                    alive = [w for w in self.workers if w.is_running]
                    dead_count = len(self.workers) - len(alive)
                    if dead_count > 0:
                        logger.warning(f"Detected {dead_count} dead workers, replacing them")
                        self.workers = alive
                        for _ in range(dead_count):
                            self._add_worker()

                    current_workers = len(self.workers)

                # Check memory before scaling up
                memory_ok = True
                try:
                    from app.memory_manager import check_memory_available
                    memory_ok = check_memory_available(required_mb=300)
                except ImportError:
                    pass

                with self._scaling_lock:
                    # Scale up if queue is backing up and memory allows
                    if pending > self.scale_up_threshold and memory_ok:
                        if current_workers < self.max_workers:
                            # Burst-scale: add multiple workers at once for large queues
                            if pending > 20:
                                workers_to_add = self.max_workers - current_workers
                            else:
                                workers_to_add = min(3, self.max_workers - current_workers)
                            for _ in range(workers_to_add):
                                self._add_worker()
                            logger.info(
                                f"Scaled up to {len(self.workers)} workers "
                                f"(+{workers_to_add}, queue depth: {pending})"
                            )

                    # Scale down if queue is mostly empty
                    elif pending <= self.scale_down_threshold and running == 0:
                        if current_workers > self.min_workers:
                            self._remove_worker()
                            logger.info(
                                f"Scaled down to {len(self.workers)} workers "
                                f"(queue depth: {pending})"
                            )

            except Exception as e:
                logger.error(f"Scaling monitor error: {e}")

            self._stop_event.wait(self.check_interval)

    def stop(self, timeout: float = 30.0):
        """Stop all workers gracefully."""
        logger.info("Stopping worker pool...")

        # Stop scaling monitor
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

        # Stop all workers
        per_worker_timeout = timeout / max(1, len(self.workers))
        for worker in self.workers:
            try:
                worker.stop(timeout=per_worker_timeout)
            except Exception as e:
                logger.warning(f"Error stopping worker: {e}")

        self.workers.clear()
        logger.info("Worker pool stopped")

    def scale_to(self, num_workers: int):
        """Manually scale to a specific number of workers."""
        num_workers = max(self.min_workers, min(num_workers, self.max_workers))

        with self._scaling_lock:
            current = len(self.workers)

            if num_workers > current:
                for _ in range(num_workers - current):
                    self._add_worker()
                logger.info(f"Scaled up to {num_workers} workers")

            elif num_workers < current:
                for _ in range(current - num_workers):
                    self._remove_worker()
                logger.info(f"Scaled down to {num_workers} workers")

    @property
    def active_count(self) -> int:
        """Get count of active workers."""
        return sum(1 for w in self.workers if w.is_running)

    def get_stats(self) -> dict:
        """Get worker pool statistics."""
        return {
            "total_workers": len(self.workers),
            "active_workers": self.active_count,
            "min_workers": self.min_workers,
            "max_workers": self.max_workers,
            "autoscaling_enabled": self._monitor_thread is not None
        }


# Global worker pool instance with Tier 3 auto-scaling
worker_pool = WorkerPool(
    min_workers=getattr(settings, 'min_workers', 5),
    max_workers=getattr(settings, 'max_workers', 20),
    scale_up_threshold=getattr(settings, 'scale_up_threshold', 10),
    scale_down_threshold=getattr(settings, 'scale_down_threshold', 3)
)
