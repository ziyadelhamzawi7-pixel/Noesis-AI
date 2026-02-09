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
from pathlib import Path
from typing import Dict, Any, Callable, Optional
from loguru import logger

from app.config import settings
from app.job_queue import JobQueue, JobType, job_queue
from app.utils import can_process_file, get_memory_usage_mb
from app import database as db


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

            # Update synced_file status if this came from Drive sync
            synced_file_id = payload.get('synced_file_id')
            if synced_file_id:
                # Only mark as failed if no retries remain
                job_info = self.queue.get_job(job_id)
                if job_info and job_info['status'] == 'failed':
                    db.update_synced_file_status(
                        synced_file_id,
                        sync_status='failed',
                        error_message=str(e)[:500]
                    )

    # ========================================================================
    # Job Handlers
    # ========================================================================

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

        # Check memory before processing
        can_process, error = can_process_file(file_path)
        if not can_process:
            raise MemoryError(error)

        # Import processing tools
        from tools.parse_pdf import parse_pdf
        from tools.parse_excel_financial import parse_excel_financial
        from tools.parse_pptx import parse_pptx
        from tools.chunk_documents import chunk_documents
        from tools.generate_embeddings import generate_embeddings, generate_embeddings_streaming
        from tools.index_to_vectordb import index_to_vectordb, VectorDBIndexer

        file_path_obj = Path(file_path)
        file_type = file_path_obj.suffix.lower()

        # Parse document with timeout to prevent hanging
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

        def _parse_file(fpath, ftype, fname):
            if ftype == '.pdf':
                return parse_pdf(fpath, use_ocr=True, max_ocr_pages=settings.max_ocr_pages)
            elif ftype in ['.xlsx', '.xls']:
                return parse_excel_financial(fpath)
            elif ftype in ['.pptx']:
                return parse_pptx(fpath)
            elif ftype == '.csv':
                import pandas as pd
                df = pd.read_csv(fpath)
                return {
                    'file_name': fname,
                    'file_type': 'csv',
                    'text': df.to_string(),
                    'sheets': [{'sheet_name': 'data', 'data': df.to_dict()}]
                }
            elif ftype in ['.docx', '.doc']:
                from docx import Document
                doc = Document(fpath)
                text_content = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                return {
                    'file_name': fname,
                    'file_type': 'docx',
                    'text': text_content,
                    'page_count': 1,
                    'pages': [{'page_number': 1, 'text': text_content}]
                }
            else:
                raise ValueError(f"Unsupported file type: {ftype}")

        # Update data room status to parsing
        if data_room_id:
            db.update_data_room_status(data_room_id, 'parsing', progress=25)

        logger.info(f"[{data_room_id}] Parsing {file_name} (timeout: {settings.parse_timeout_seconds}s, workers: {settings.max_parse_workers})...")
        with ThreadPoolExecutor(max_workers=settings.max_parse_workers) as parse_executor:
            future = parse_executor.submit(_parse_file, str(file_path), file_type, file_name)
            try:
                parsed = future.result(timeout=settings.parse_timeout_seconds)
            except FuturesTimeoutError:
                raise TimeoutError(f"Parsing timed out after {settings.parse_timeout_seconds}s")

        # Chunk document
        logger.info(f"[{data_room_id}] Chunking {file_name}...")
        chunks = chunk_documents(parsed)

        if not chunks:
            logger.warning(f"[{data_room_id}] No chunks generated for {file_name}")
            return

        # Update data room status to embedding
        if data_room_id:
            db.update_data_room_status(data_room_id, 'indexing', progress=50)

        # Generate embeddings with streaming (embed + index in parallel)
        logger.info(f"[{data_room_id}] Streaming embeddings for {len(chunks)} chunks (batch_size={settings.batch_size}, max_concurrent={settings.embedding_max_concurrent})...")

        document_id = payload.get('document_id')
        indexer = VectorDBIndexer()
        chunks_with_embeddings = []
        indexed_count = 0

        for batch in generate_embeddings_streaming(
            chunks,
            batch_size=settings.batch_size,
            max_concurrent=settings.embedding_max_concurrent
        ):
            # Add document metadata to batch
            for chunk in batch:
                chunk['document_id'] = document_id
                chunk['data_room_id'] = data_room_id

            # Index batch immediately as it completes
            index_to_vectordb(data_room_id, batch, indexer=indexer)

            # Save batch to database
            db.create_chunks(batch)

            chunks_with_embeddings.extend(batch)
            indexed_count += len(batch)
            logger.debug(f"[{data_room_id}] Indexed batch: {indexed_count}/{len(chunks)} chunks")

        # Update document status
        if 'document_id' in payload:
            db.update_document_status(
                payload['document_id'],
                'parsed',
                page_count=parsed.get('page_count', 0),
                token_count=sum(c.get('token_count', 0) for c in chunks_with_embeddings)
            )

        # Update synced_file status if this came from Drive sync
        synced_file_id = payload.get('synced_file_id')
        if synced_file_id:
            db.update_synced_file_status(
                synced_file_id,
                sync_status='complete',
                document_id=payload.get('document_id')
            )

        # Update folder progress (count complete + failed as processed)
        connected_folder_id = payload.get('connected_folder_id')
        if connected_folder_id:
            # This is a folder sync - track progress
            completed = db.count_synced_files_by_status(connected_folder_id, 'complete')
            failed = db.count_synced_files_by_status(connected_folder_id, 'failed')
            processed = completed + failed
            total = db.count_synced_files_by_status(connected_folder_id)

            # Calculate progress percentage for data room
            progress = (processed / total * 100) if total > 0 else 0

            db.update_connected_folder_status(
                connected_folder_id,
                sync_status='syncing',
                processed_files=processed
            )

            # Update data room progress (but NOT to 'complete' yet)
            if data_room_id:
                db.update_data_room_status(data_room_id, 'parsing', progress=progress)

            # Only set to complete when ALL files are processed
            if processed >= total and total > 0:
                db.update_connected_folder_stage(connected_folder_id, sync_stage='complete')
                db.update_connected_folder_status(connected_folder_id, sync_status='active', processed_files=processed)
                # NOW set data room to complete
                if data_room_id:
                    db.update_data_room_status(data_room_id, 'complete', progress=100)
        else:
            # Individual file connection (not from folder) - mark complete immediately
            if data_room_id:
                db.update_data_room_status(data_room_id, 'complete', progress=100)

        logger.success(f"[{data_room_id}] File {file_name} processed: {len(chunks_with_embeddings)} chunks")

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

        with ThreadPoolExecutor(max_workers=max_parallel_files) as executor:
            futures = {
                executor.submit(self._handle_process_file, {
                    'data_room_id': data_room_id,
                    'file_path': file_path,
                    'file_name': Path(file_path).name
                }): file_path
                for file_path in file_paths
            }

            completed = 0
            for future in as_completed(futures):
                file_path = futures[future]
                completed += 1
                try:
                    future.result()
                    logger.info(f"[{data_room_id}] Completed {completed}/{len(file_paths)}: {Path(file_path).name}")
                except Exception as e:
                    logger.error(f"[{data_room_id}] Failed to process {file_path}: {e}")

        # Update data room status
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

        # Index immediately
        index_to_vectordb(data_room_id, chunks_with_embeddings)

        # Save to database
        db.create_chunks(chunks_with_embeddings)

        logger.success(f"[{data_room_id}] Embeddings generated and indexed")


class WorkerPool:
    """
    Auto-scaling pool of job workers for concurrent processing.

    Tier 3 Features:
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
        check_interval: float = 5.0
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
        """Monitor queue depth and scale workers."""
        while not self._stop_event.is_set():
            try:
                # Get queue statistics
                stats = job_queue.get_queue_stats()
                pending = stats.get("pending", 0)
                running = stats.get("running", 0)
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
                            self._add_worker()
                            logger.info(
                                f"Scaled up to {len(self.workers)} workers "
                                f"(queue depth: {pending})"
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
