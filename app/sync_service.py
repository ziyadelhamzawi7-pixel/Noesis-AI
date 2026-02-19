"""
Auto-sync background service for Google Drive connected folders.
Monitors connected folders and processes new/updated files automatically.

Features:
- Rate limiting to prevent overwhelming the system
- Memory-aware processing
- Parallel downloads with backpressure
"""

import asyncio
import collections
import math
import threading
import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback
import requests as requests_lib

from loguru import logger

from app import database as db
from app.google_drive import GoogleDriveService, SUPPORTED_MIME_TYPES
from app.google_oauth import google_oauth_service
from app.config import settings
from app.utils import RateLimiter, check_memory_available, get_available_memory_mb


class SyncService:
    """Background service for syncing Google Drive folders with parallel processing."""

    def __init__(
        self,
        sync_interval_seconds: int = 300,
        max_parallel_downloads: int = settings.drive_sync_max_parallel_downloads,
        max_parallel_processing: int = 5
    ):
        """
        Initialize sync service with parallel processing capabilities.

        Args:
            sync_interval_seconds: Interval between sync checks (default 5 minutes)
            max_parallel_downloads: Max concurrent file downloads
            max_parallel_processing: Max concurrent document processing (embedding/indexing)
        """
        self.sync_interval = sync_interval_seconds
        self.max_parallel_downloads = max_parallel_downloads
        self.max_parallel_processing = max_parallel_processing
        self.is_running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Rate limiters to prevent overwhelming the system
        self._download_rate_limiter = RateLimiter(
            max_requests=settings.drive_sync_max_downloads_per_minute,
            window_seconds=60
        )
        # Per-folder locks to prevent concurrent syncs of the same folder
        self._folder_locks: Dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()
        # Per-user locks to prevent thundering herd of concurrent token refreshes
        self._token_refresh_locks: Dict[str, threading.Lock] = {}

        # Separate executors for different workloads
        self._download_executor = ThreadPoolExecutor(
            max_workers=max_parallel_downloads,
            thread_name_prefix="download_worker"
        )
        self._process_executor = ThreadPoolExecutor(
            max_workers=max_parallel_processing,
            thread_name_prefix="process_worker"
        )

        logger.info(
            f"SyncService initialized: downloads={settings.drive_sync_max_downloads_per_minute}/min, "
            f"parallel_downloads={max_parallel_downloads}"
        )

    def start(self):
        """Start the background sync service."""
        if self.is_running:
            logger.warning("Sync service is already running")
            return

        self.is_running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_sync_loop, daemon=True)
        self._thread.start()
        logger.info(f"Sync service started (interval: {self.sync_interval}s)")

    def stop(self):
        """Stop the background sync service."""
        if not self.is_running:
            return

        logger.info("Stopping sync service...")
        self._stop_event.set()
        self.is_running = False

        if self._thread:
            self._thread.join(timeout=10)

        self._download_executor.shutdown(wait=False)
        self._process_executor.shutdown(wait=False)
        logger.info("Sync service stopped")

    def _run_sync_loop(self):
        """Main sync loop running in background thread."""
        while not self._stop_event.is_set():
            try:
                self._sync_all_folders()
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                logger.error(traceback.format_exc())

            # Wait for interval or stop event
            self._stop_event.wait(timeout=self.sync_interval)

    def _sync_all_folders(self):
        """Sync all active connected folders."""
        try:
            folders = db.get_active_connected_folders()
            logger.info(f"Checking {len(folders)} connected folders for sync")

            for folder in folders:
                if self._stop_event.is_set():
                    break

                try:
                    self._sync_folder(folder)
                except Exception as e:
                    logger.error(f"Failed to sync folder {folder['id']}: {e}")
                    db.update_connected_folder_status(
                        folder['id'],
                        sync_status='error',
                        error_message=str(e)
                    )

        except Exception as e:
            logger.error(f"Failed to get active folders: {e}")

    def _sync_folder(self, folder: Dict[str, Any]):
        """
        Sync a single connected folder using staged approach.

        Stage 1: Discovery - scan all folders and count files
        Stage 2: Processing - download and process files in parallel

        Args:
            folder: Connected folder record with user tokens
        """
        folder_id = folder['id']
        folder_name = folder['folder_name']

        # Acquire per-folder lock (non-blocking) to prevent concurrent syncs
        with self._locks_lock:
            if folder_id not in self._folder_locks:
                self._folder_locks[folder_id] = threading.Lock()
            folder_lock = self._folder_locks[folder_id]

        if not folder_lock.acquire(blocking=False):
            logger.info(f"Sync already in progress for folder: {folder_name}, skipping")
            return

        try:
            # Skip folders that have already completed syncing
            if folder.get('sync_stage') == 'complete':
                logger.debug(f"Folder already complete, skipping: {folder_name}")
                return

            data_room_id = folder.get('data_room_id')
            logger.info(f"Starting staged sync for folder: {folder_name} (data_room={data_room_id})")

            # Update status to syncing
            db.update_connected_folder_status(folder_id, sync_status='syncing')

            # Get drive service
            drive_service = self._get_drive_service(folder)

            # Streamed discovery + download: start downloading files as soon as
            # they are discovered instead of waiting for full BFS to finish.
            download_queue = collections.deque()
            download_queue_lock = threading.Lock()
            discovery_done = threading.Event()

            db.update_connected_folder_stage(folder_id, sync_stage='discovering')
            if data_room_id:
                db.update_data_room_status(data_room_id, 'parsing', progress=2)

            def _download_consumer():
                """Consume discovered files and download+enqueue them in parallel."""
                downloaded = 0
                failed = 0
                status_update_interval = 10
                last_status_count = 0

                with ThreadPoolExecutor(
                    max_workers=self.max_parallel_downloads,
                    thread_name_prefix="drive_download"
                ) as executor:
                    futures = {}

                    while True:
                        # Drain any pending files from the queue
                        batch = []
                        with download_queue_lock:
                            while download_queue:
                                batch.append(download_queue.popleft())

                        def _download_one(rec):
                            thread_drive = self._get_drive_service(folder)
                            self._download_and_enqueue_file(folder, rec, thread_drive)

                        for file_rec in batch:
                            if self._stop_event.is_set():
                                break
                            fut = executor.submit(_download_one, file_rec)
                            futures[fut] = file_rec

                        # Collect completed downloads
                        done_futs = [f for f in list(futures.keys()) if f.done()]
                        for fut in done_futs:
                            file_rec = futures.pop(fut)
                            try:
                                fut.result()
                                downloaded += 1
                            except Exception as e:
                                logger.error(f"Download failed for {file_rec['file_name']}: {e}")
                                failed += 1

                            completed = downloaded + failed
                            if completed - last_status_count >= status_update_interval:
                                db.update_connected_folder_status(folder_id, sync_status='syncing')
                                last_status_count = completed

                        # Exit when discovery is done and queue+futures are empty
                        if discovery_done.is_set():
                            with download_queue_lock:
                                queue_empty = len(download_queue) == 0
                            if queue_empty and not futures:
                                break

                        if not batch and not done_futs:
                            time.sleep(0.5)  # Avoid busy-waiting

                    # Wait for remaining futures
                    for fut in as_completed(futures):
                        file_rec = futures[fut] if fut in futures else None
                        try:
                            fut.result()
                            downloaded += 1
                        except Exception as e:
                            name = file_rec['file_name'] if file_rec else 'unknown'
                            logger.error(f"Download failed for {name}: {e}")
                            failed += 1

                logger.info(f"Download consumer done: {downloaded} enqueued, {failed} failed")

            # Start download consumer in parallel with discovery
            download_thread = threading.Thread(
                target=_download_consumer, daemon=True, name="download-consumer"
            )
            download_thread.start()

            # Run discovery, streaming files to the download queue
            self._discovery_phase(folder, drive_service, download_queue=download_queue, download_queue_lock=download_queue_lock)

            # Signal discovery is complete so download consumer can drain and exit
            discovery_done.set()
            pending_count = db.count_pending_synced_files(folder_id)
            logger.info(f"Discovery complete for {folder_name}: {pending_count} files, waiting for downloads to finish")

            db.update_connected_folder_stage(folder_id, sync_stage='processing')
            if data_room_id:
                db.update_data_room_status(data_room_id, 'parsing', progress=10)

            # Wait for all downloads to complete
            download_thread.join()

            if pending_count == 0:
                logger.info(f"No files to process in {folder_name}")
                if data_room_id:
                    db.update_data_room_status(data_room_id, 'complete', progress=100)
                db.update_connected_folder_stage(folder_id, sync_stage='complete')
            else:
                db.update_connected_folder_stage(folder_id, sync_stage='queued')
                logger.success(f"Downloads enqueued for {folder_name}, worker pool processing")

        except Exception as e:
            # Signal download consumer to stop if it's running
            try:
                discovery_done.set()
            except NameError:
                pass
            logger.error(f"Sync failed for folder {folder_name}: {e}")
            logger.error(traceback.format_exc())
            db.update_connected_folder_stage(folder_id, sync_stage='error', error_message=str(e))
            db.update_connected_folder_status(
                folder_id,
                sync_status='error',
                error_message=str(e)
            )
            if data_room_id:
                db.update_data_room_status(
                    data_room_id, 'failed',
                    error_message=f"Drive sync failed: {str(e)[:500]}"
                )
            raise
        finally:
            folder_lock.release()

    def _get_drive_service(self, folder: Dict[str, Any]) -> GoogleDriveService:
        """
        Get Google Drive service with fresh tokens.
        Uses per-user locking to prevent thundering herd of concurrent refreshes.

        Args:
            folder: Connected folder record with user tokens

        Returns:
            GoogleDriveService instance
        """
        access_token = folder['access_token']
        refresh_token = folder['refresh_token']
        expires_at = folder.get('token_expires_at')

        if self._token_needs_refresh(expires_at):
            user_id = folder['user_id']

            # Per-user lock prevents thundering herd of concurrent refreshes
            with self._locks_lock:
                if user_id not in self._token_refresh_locks:
                    self._token_refresh_locks[user_id] = threading.Lock()
                refresh_lock = self._token_refresh_locks[user_id]

            with refresh_lock:
                # Re-check after acquiring lock — another thread may have already refreshed
                if self._token_needs_refresh(folder.get('token_expires_at')):
                    new_tokens = google_oauth_service.refresh_access_token(refresh_token)
                    if new_tokens:
                        access_token = new_tokens['access_token']
                        expires_at = new_tokens.get('expires_at')
                        db.update_user_tokens(
                            user_id,
                            access_token=access_token,
                            token_expires_at=expires_at
                        )
                        # Update the shared folder dict so other threads see the fresh token
                        folder['access_token'] = access_token
                        folder['token_expires_at'] = expires_at
                    else:
                        raise Exception("Failed to refresh access token")
                else:
                    # Another thread already refreshed — use the updated values
                    access_token = folder['access_token']
                    expires_at = folder.get('token_expires_at')

        return GoogleDriveService.from_tokens(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at
        )

    def _discovery_phase(self, folder: Dict[str, Any], drive_service: GoogleDriveService,
                         download_queue=None, download_queue_lock=None):
        """
        Stage 1: Discover all folders and files without downloading.
        Uses parallel breadth-first search to scan multiple subfolders concurrently.

        Args:
            folder: Connected folder record
            drive_service: Google Drive service instance
            download_queue: Optional deque to push discovered files for concurrent download
            download_queue_lock: Lock for thread-safe access to download_queue
        """
        folder_id = folder['id']
        drive_folder_id = folder['folder_id']
        folder_name = folder['folder_name']

        logger.info(f"Discovery phase started for: {folder_name}")

        # Check if we can resume from a previous partial discovery
        existing_stage = folder.get('sync_stage')
        if existing_stage in ('discovered', 'processing', 'queued'):
            pending_count = db.count_pending_synced_files(folder_id)
            if pending_count > 0:
                logger.info(f"Resuming from previous discovery: {pending_count} pending files")
                # Load pending files into the download queue so the consumer can process them
                if download_queue is not None and download_queue_lock is not None:
                    pending_files = db.get_pending_synced_files(folder_id, limit=pending_count)
                    with download_queue_lock:
                        download_queue.extend(pending_files)
                    logger.info(f"Loaded {len(pending_files)} pending files into download queue")
                return

        if existing_stage == 'discovering':
            unscanned = db.get_unscanned_inventory_folders(folder_id)
            if unscanned:
                logger.info(f"Resuming partial discovery: {len(unscanned)} folders remaining")
                initial_items = [
                    (f['drive_folder_id'], f.get('folder_path') or '', f.get('parent_folder_id'))
                    for f in unscanned
                ]
                # Count already-discovered items
                discovered_files = db.count_pending_synced_files(folder_id)
                existing_inventory = db.get_folder_inventory(folder_id)
                discovered_folders = len(existing_inventory)
            else:
                db.delete_folder_inventory(folder_id)
                initial_items = [(drive_folder_id, '', None)]
                discovered_files = 0
                discovered_folders = 0
        else:
            db.delete_folder_inventory(folder_id)
            initial_items = [(drive_folder_id, '', None)]
            discovered_files = 0
            discovered_folders = 0

        # Thread-safe queue and counters for parallel scanning
        scan_queue = collections.deque(initial_items)
        queue_lock = threading.Lock()
        counter_lock = threading.Lock()

        max_scan_workers = settings.drive_scan_max_workers

        # Thread-local storage so each worker reuses one DriveService
        _thread_local = threading.local()

        def _get_thread_drive():
            """Get or create a per-thread DriveService instance."""
            if not hasattr(_thread_local, 'drive_service'):
                _thread_local.drive_service = self._get_drive_service(folder)
            return _thread_local.drive_service

        def _scan_one_folder(drive_id, path, parent_inv_id):
            """Scan a single folder: API call + batch DB writes."""
            nonlocal discovered_files, discovered_folders

            thread_drive = _get_thread_drive()
            current_name = path.split('/')[-1] if path else folder_name

            contents = thread_drive.list_folder_contents_only(drive_id)

            # Batch-insert the inventory record
            inv_ids = db.create_and_update_folder_inventory_batch([{
                'connected_folder_id': folder_id,
                'drive_folder_id': drive_id,
                'folder_name': current_name,
                'folder_path': path,
                'parent_folder_id': parent_inv_id,
                'file_count': contents['file_count'],
                'total_size_bytes': contents['total_size'],
            }])
            inventory_id = inv_ids[0] if inv_ids else None

            # Batch-insert discovered files
            if contents['files']:
                file_records = []
                for file in contents['files']:
                    file_path = f"{path}/{file['name']}" if path else file['name']
                    file_records.append({
                        'connected_folder_id': folder_id,
                        'drive_file_id': file['id'],
                        'file_name': file['name'],
                        'file_path': file_path,
                        'mime_type': file.get('mimeType'),
                        'file_size': file.get('size'),
                        'drive_modified_time': file.get('modifiedTime'),
                    })
                synced_ids = db.create_synced_files_batch(file_records)

                # Stream discovered files to download consumer (if available)
                if download_queue is not None and download_queue_lock is not None:
                    enriched = []
                    for sf_id, rec in zip(synced_ids, file_records):
                        enriched.append({**rec, 'id': sf_id, 'sync_status': 'pending'})
                    with download_queue_lock:
                        download_queue.extend(enriched)

            # Enqueue subfolders for scanning
            new_subfolders = []
            for subfolder in contents['subfolders']:
                sub_path = f"{path}/{subfolder['name']}" if path else subfolder['name']
                new_subfolders.append((subfolder['id'], sub_path, inventory_id))

            with queue_lock:
                scan_queue.extend(new_subfolders)

            # Log skipped files for visibility
            if contents.get('skipped_count', 0) > 0:
                skipped_names = ', '.join(f['name'] for f in contents['skipped_files'][:5])
                suffix = f" (and {contents['skipped_count'] - 5} more)" if contents['skipped_count'] > 5 else ""
                logger.warning(f"Skipped {contents['skipped_count']} unsupported files in {current_name}: {skipped_names}{suffix}")

            # Update shared counters
            with counter_lock:
                discovered_files += contents['file_count']
                discovered_folders += len(contents['subfolders'])

            return contents['file_count'], len(contents['subfolders'])

        # Parallel BFS loop
        data_room_id = folder.get('data_room_id')
        discovery_timeout = 600  # 10 minutes max for discovery phase
        discovery_start_time = time.time()
        logger.info(
            f"Discovery BFS starting: max_workers={max_scan_workers}, "
            f"timeout={discovery_timeout}s, initial_queue={len(initial_items)} items"
        )

        with ThreadPoolExecutor(max_workers=max_scan_workers, thread_name_prefix="discovery") as executor:
            futures = {}
            last_progress_update = time.time()

            while True:
                if self._stop_event.is_set():
                    break

                # Check discovery timeout
                elapsed = time.time() - discovery_start_time
                if elapsed > discovery_timeout:
                    logger.error(
                        f"Discovery timeout after {elapsed:.0f}s for {folder_name}. "
                        f"Discovered {discovered_files} files in {discovered_folders} folders so far."
                    )
                    for fut in futures:
                        fut.cancel()
                    raise TimeoutError(
                        f"Discovery timed out after {int(elapsed)}s. "
                        f"Found {discovered_files} files in {discovered_folders} folders before timeout."
                    )

                # Submit work from the queue
                with queue_lock:
                    while scan_queue and len(futures) < max_scan_workers:
                        drive_id, path, parent_inv_id = scan_queue.popleft()
                        fut = executor.submit(_scan_one_folder, drive_id, path, parent_inv_id)
                        futures[fut] = path

                if not futures:
                    # No active work and nothing in queue — we're done
                    break

                # Wait for at least one future to complete (timeout prevents deadlock)
                done_futures = set()
                try:
                    for fut in as_completed(futures, timeout=1.0):
                        done_futures.add(fut)
                        path = futures[fut]
                        try:
                            fut.result()
                        except Exception as e:
                            logger.error(f"Error scanning folder {path or 'Root'}: {e}")
                            logger.error(traceback.format_exc())
                except TimeoutError:
                    pass  # Timeout just means no more futures completed yet; loop back

                for fut in done_futures:
                    del futures[fut]

                # Throttle progress updates to at most once per second
                now = time.time()
                if now - last_progress_update >= 1.0:
                    with counter_lock:
                        cur_files = discovered_files
                        cur_folders = discovered_folders
                    db.update_connected_folder_stage(
                        folder_id,
                        sync_stage='discovering',
                        discovered_files=cur_files,
                        discovered_folders=cur_folders
                    )
                    # Update data room progress during discovery (5% -> 9%)
                    if data_room_id and cur_folders > 0:
                        discovery_progress = 5 + min(4, 4 * math.log1p(cur_folders) / math.log1p(50))
                        db.update_data_room_status(data_room_id, 'parsing', progress=discovery_progress)
                    last_progress_update = now

        # Mark discovery complete
        db.update_connected_folder_stage(
            folder_id,
            sync_stage='discovered',
            discovered_files=discovered_files,
            discovered_folders=discovered_folders,
            current_folder_path=None
        )

        # Use actual DB count for total_files instead of the in-memory counter.
        # The counter can overcount when discovery resumes and re-discovers files
        # that already exist (ON CONFLICT upsert keeps the existing row but the
        # counter still increments). Using the DB count prevents a total_files
        # mismatch that blocks the completion check in _update_folder_progress.
        actual_file_count = db.count_all_synced_files(folder_id)
        db.update_connected_folder_status(
            folder_id,
            sync_status='syncing',
            total_files=actual_file_count
        )

        logger.info(f"Discovery complete: {actual_file_count} files in DB ({discovered_files} discovered) across {discovered_folders} folders")

    def _processing_phase(self, folder: Dict[str, Any], drive_service: GoogleDriveService):
        """
        Stage 2: Download all discovered files and enqueue them for processing.

        Downloads files sequentially (to avoid SSL conflicts), then enqueues
        each as an individual PROCESS_FILE job for the worker pool to handle
        with built-in retry and crash recovery.

        Args:
            folder: Connected folder record
            drive_service: Google Drive service instance
        """
        folder_id = folder['id']
        folder_name = folder['folder_name']

        logger.info(f"Processing phase started for: {folder_name}")

        pending_files = db.get_all_pending_synced_files(folder_id)
        total_files = len(pending_files)
        downloaded_count = 0
        failed_count = 0

        if total_files == 0:
            logger.info("No files to process")
            # Mark data room as complete for empty folders
            data_room_id = folder.get('data_room_id')
            if data_room_id:
                db.update_data_room_status(data_room_id, 'complete', progress=100)
            return

        def _download_one(file_record):
            """Download a single file using a per-thread Drive service to isolate SSL sessions."""
            thread_drive_service = self._get_drive_service(folder)
            self._download_and_enqueue_file(folder, file_record, thread_drive_service)

        futures = {}
        status_update_interval = 10  # Only write status every N downloads
        last_status_count = 0

        with ThreadPoolExecutor(
            max_workers=self.max_parallel_downloads,
            thread_name_prefix="drive_download"
        ) as executor:
            for file_record in pending_files:
                if self._stop_event.is_set():
                    break
                fut = executor.submit(_download_one, file_record)
                futures[fut] = file_record

            for fut in as_completed(futures):
                file_rec = futures[fut]
                try:
                    fut.result()
                    downloaded_count += 1
                except Exception as e:
                    logger.error(f"Download failed for {file_rec['file_name']}: {e}")
                    failed_count += 1

                # Throttle status updates: write every N downloads instead of every one
                completed = downloaded_count + failed_count
                if completed - last_status_count >= status_update_interval or completed == total_files:
                    db.update_connected_folder_status(
                        folder_id,
                        sync_status='syncing'
                    )
                    last_status_count = completed

        logger.info(
            f"Download phase complete: {downloaded_count} enqueued, "
            f"{failed_count} failed out of {total_files} total"
        )

    def _download_and_enqueue_file(
        self,
        folder: Dict[str, Any],
        file_record: Dict[str, Any],
        drive_service: GoogleDriveService
    ) -> bool:
        """
        Download a file and enqueue a PROCESS_FILE job for the worker pool.

        Args:
            folder: Connected folder record
            file_record: Synced file record from database
            drive_service: Google Drive service instance

        Returns:
            True if file was enqueued successfully
        """
        from app.job_queue import job_queue, JobType

        synced_file_id = file_record['id']
        drive_file_id = file_record['drive_file_id']
        file_name = file_record['file_name']
        data_room_id = folder.get('data_room_id')

        # Skip already processed files (re-sync scenarios)
        if file_record.get('sync_status') in ('complete', 'queued'):
            return False

        db.update_synced_file_status(synced_file_id, sync_status='downloading')

        if not self._download_rate_limiter.acquire(timeout=120):
            raise RuntimeError(f"Download rate limit timeout for {file_name} after 120s")

        # Memory check — wait if below threshold, skip if critically low
        available_mb = get_available_memory_mb()
        if available_mb < 500:
            logger.warning(f"Low memory ({available_mb:.0f}MB), waiting before download of {file_name}")
            for _ in range(12):
                time.sleep(5)
                if get_available_memory_mb() > 500:
                    break
            else:
                logger.error(f"Memory still low, skipping {file_name}")
                db.update_synced_file_status(
                    synced_file_id, sync_status='failed', error_message="Insufficient memory"
                )
                raise MemoryError(f"Insufficient memory to download {file_name}")

        # Download with exponential backoff retry
        download_dir = Path(f".tmp/data_rooms/{data_room_id or folder['id']}/drive_sync")
        download_dir.mkdir(parents=True, exist_ok=True)
        local_path = download_dir / f"{drive_file_id}_{file_name}"

        RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
        max_retries = settings.drive_sync_retry_attempts
        base_delay = settings.drive_sync_retry_base_delay

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.info(f"Retry {attempt}/{max_retries-1} for {file_name}, waiting {delay}s")
                    time.sleep(delay)
                actual_path = drive_service.download_file(drive_file_id, str(local_path))
                local_path = Path(actual_path)  # may have extension appended for Google Docs
                break
            except requests_lib.HTTPError as e:
                status_code = e.response.status_code if e.response else 0
                if status_code not in RETRYABLE_STATUS_CODES:
                    db.update_synced_file_status(
                        synced_file_id, sync_status='failed',
                        error_message=f"Permanent error {status_code}: {str(e)[:500]}"
                    )
                    raise
                if attempt >= max_retries - 1:
                    db.update_synced_file_status(
                        synced_file_id, sync_status='failed',
                        error_message=f"Failed after {max_retries} retries: {str(e)[:500]}"
                    )
                    raise
            except Exception as e:
                if attempt >= max_retries - 1:
                    db.update_synced_file_status(
                        synced_file_id, sync_status='failed', error_message=str(e)[:500]
                    )
                    raise
                logger.warning(f"Download error for {file_name} (attempt {attempt+1}/{max_retries}): {e}")

        # Check if document already exists (prevent duplicates on re-sync)
        existing_doc = db.get_document_by_name(data_room_id, file_name)
        if existing_doc:
            document_id = existing_doc['id']
            logger.info(f"Document already exists: {file_name} ({document_id})")
            # Skip enqueuing if already parsed — prevents re-parsing on restart
            if existing_doc.get('parse_status') == 'parsed':
                logger.info(f"Document already parsed, skipping enqueue: {file_name}")
                db.update_synced_file_status(
                    synced_file_id, sync_status='complete', document_id=document_id
                )
                return True
        else:
            # Create document record
            file_size = file_record.get('file_size') or local_path.stat().st_size
            document_id = db.create_document(
                data_room_id=data_room_id,
                file_name=file_name,
                file_path=str(local_path),
                file_size=file_size,
                file_type=local_path.suffix.lower().replace('.', '')
            )

        # Enqueue for processing by the worker pool
        job_queue.enqueue(
            job_type=JobType.PROCESS_FILE.value,
            payload={
                'data_room_id': data_room_id,
                'file_path': str(local_path),
                'file_name': file_name,
                'document_id': document_id,
                'synced_file_id': synced_file_id,
                'connected_folder_id': folder['id'],
            },
            data_room_id=data_room_id,
            file_name=file_name,
        )

        db.update_synced_file_status(
            synced_file_id, sync_status='queued', document_id=document_id
        )
        logger.info(f"Downloaded and enqueued: {file_name}")
        return True

    def _token_needs_refresh(self, expires_at: Optional[str]) -> bool:
        """Check if token needs refresh (expired or expires within 5 minutes)."""
        if not expires_at:
            return True

        try:
            expiry = datetime.fromisoformat(expires_at.replace('Z', '+00:00'))
            # Refresh if expires within 5 minutes
            return expiry <= datetime.now(expiry.tzinfo) + timedelta(minutes=5)
        except Exception:
            return True

    def _download_and_prepare_file(
        self,
        folder: Dict[str, Any],
        file: Dict[str, Any],
        drive_service: GoogleDriveService
    ) -> Optional[Dict[str, Any]]:
        """
        Download a file and prepare it for processing.

        Args:
            folder: Connected folder record
            file: File info from Google Drive
            drive_service: Drive service instance

        Returns:
            Dict with download result or None if skipped/failed
        """
        folder_id = folder['id']
        data_room_id = folder.get('data_room_id')
        drive_file_id = file['id']
        file_name = file['name']
        modified_time = file.get('modifiedTime')

        # Check if file already synced and unchanged
        existing = db.check_file_exists_in_sync(folder_id, drive_file_id)
        is_new = existing is None

        if existing:
            if existing.get('drive_modified_time') == modified_time:
                logger.debug(f"File unchanged, skipping: {file_name}")
                return None  # Skip unchanged files
            synced_file_id = existing['id']
            logger.info(f"File updated: {file_name}")
        else:
            synced_file_id = db.create_synced_file(
                connected_folder_id=folder_id,
                drive_file_id=drive_file_id,
                file_name=file_name,
                file_path=file.get('path'),
                mime_type=file.get('mimeType'),
                file_size=file.get('size'),
                drive_modified_time=modified_time
            )
            logger.info(f"New file: {file_name}")

        # Update status to downloading
        db.update_synced_file_status(synced_file_id, sync_status='downloading')

        # Download with application-level retry (transport-level SSL retries
        # are handled by the requests.Session adapter in GoogleDriveService)
        download_dir = Path(f".tmp/data_rooms/{data_room_id or folder_id}/drive_sync")
        download_dir.mkdir(parents=True, exist_ok=True)

        local_filename = f"{drive_file_id}_{file_name}"
        local_path = download_dir / local_filename

        max_retries = settings.drive_sync_retry_attempts
        base_delay = settings.drive_sync_retry_base_delay
        last_error = None

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.info(f"Retry {attempt}/{max_retries-1} for {file_name}, waiting {delay}s")
                    time.sleep(delay)

                actual_path = drive_service.download_file(drive_file_id, str(local_path))
                local_path = Path(actual_path)  # may have extension appended for Google Docs

                return {
                    'synced_file_id': synced_file_id,
                    'data_room_id': data_room_id,
                    'file_path': str(local_path),
                    'file_name': file_name,
                    'file_size': file.get('size') or local_path.stat().st_size,
                    'is_new': is_new
                }

            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"Download error for {file_name} (attempt {attempt + 1}/{max_retries}): {e}")
                    continue
                break

        # All retries failed
        logger.error(f"Download failed for {file_name} after {max_retries} attempts: {last_error}")
        db.update_synced_file_status(
            synced_file_id,
            sync_status='failed',
            error_message=str(last_error)[:500]
        )
        return None

    def _process_downloaded_file_wrapper(self, download_result: Dict[str, Any]) -> bool:
        """
        Wrapper to process a downloaded file.

        Args:
            download_result: Result from download step

        Returns:
            True if file is new, False if updated
        """
        synced_file_id = download_result['synced_file_id']
        data_room_id = download_result['data_room_id']
        file_path = download_result['file_path']
        file_name = download_result['file_name']
        file_size = download_result['file_size']
        is_new = download_result['is_new']

        # Update status to processing
        db.update_synced_file_status(
            synced_file_id,
            sync_status='processing',
            local_file_path=file_path
        )

        try:
            if data_room_id:
                document_id = self._process_downloaded_file(
                    data_room_id=data_room_id,
                    file_path=file_path,
                    file_name=file_name,
                    file_size=file_size
                )
                db.update_synced_file_status(
                    synced_file_id,
                    sync_status='complete',
                    document_id=document_id
                )
            else:
                db.update_synced_file_status(synced_file_id, sync_status='complete')

            return is_new

        except Exception as e:
            logger.error(f"Processing failed for {file_name}: {e}")
            db.update_synced_file_status(
                synced_file_id,
                sync_status='failed',
                error_message=str(e)
            )
            raise

    def _process_file(
        self,
        folder: Dict[str, Any],
        file: Dict[str, Any],
        drive_service: GoogleDriveService
    ) -> bool:
        """
        Process a single file - check if new/updated and download if needed.

        Args:
            folder: Connected folder record
            file: File info from Google Drive
            drive_service: Drive service instance

        Returns:
            True if file is new, False if updated/existing
        """
        folder_id = folder['id']
        data_room_id = folder.get('data_room_id')
        drive_file_id = file['id']
        file_name = file['name']
        modified_time = file.get('modifiedTime')

        # Check if file already synced
        existing = db.check_file_exists_in_sync(folder_id, drive_file_id)

        if existing:
            # Check if file was modified
            if existing.get('drive_modified_time') == modified_time:
                logger.debug(f"File unchanged: {file_name}")
                return False

            # File was updated - re-download
            synced_file_id = existing['id']
            logger.info(f"File updated: {file_name}")
        else:
            # New file
            synced_file_id = db.create_synced_file(
                connected_folder_id=folder_id,
                drive_file_id=drive_file_id,
                file_name=file_name,
                file_path=file.get('path'),
                mime_type=file.get('mimeType'),
                file_size=file.get('size'),
                drive_modified_time=modified_time
            )
            logger.info(f"New file: {file_name}")

        # Update status to downloading
        db.update_synced_file_status(synced_file_id, sync_status='downloading')

        try:
            # Download file
            download_dir = Path(f".tmp/data_rooms/{data_room_id or folder_id}/drive_sync")
            download_dir.mkdir(parents=True, exist_ok=True)

            # Create unique filename to avoid collisions
            local_filename = f"{drive_file_id}_{file_name}"
            local_path = download_dir / local_filename

            actual_path = drive_service.download_file(drive_file_id, str(local_path))
            local_path = Path(actual_path)  # may have extension appended for Google Docs

            # Update status to processing
            db.update_synced_file_status(
                synced_file_id,
                sync_status='processing',
                local_file_path=str(local_path)
            )

            # Process the file (create document record and add to data room)
            if data_room_id:
                document_id = self._process_downloaded_file(
                    data_room_id=data_room_id,
                    file_path=str(local_path),
                    file_name=file_name,
                    file_size=file.get('size') or local_path.stat().st_size
                )

                db.update_synced_file_status(
                    synced_file_id,
                    sync_status='complete',
                    document_id=document_id
                )
            else:
                # No data room associated - just mark as complete
                db.update_synced_file_status(synced_file_id, sync_status='complete')

            return not existing

        except Exception as e:
            logger.error(f"Failed to process file {file_name}: {e}")
            db.update_synced_file_status(
                synced_file_id,
                sync_status='failed',
                error_message=str(e)
            )
            raise

    def _process_downloaded_file(
        self,
        data_room_id: str,
        file_path: str,
        file_name: str,
        file_size: int
    ) -> str:
        """
        Process a downloaded file - parse, chunk, embed, and index.

        Args:
            data_room_id: Data room ID
            file_path: Local file path
            file_name: Original file name
            file_size: File size in bytes

        Returns:
            Document ID
        """
        file_type = Path(file_name).suffix.lower().replace('.', '')

        # Create document record
        document_id = db.create_document(
            data_room_id=data_room_id,
            file_name=file_name,
            file_path=file_path,
            file_size=file_size,
            file_type=file_type
        )

        # Import processing tools
        from tools.ingest_data_room import parse_file_by_type
        from tools.chunk_documents import chunk_documents
        from tools.generate_embeddings import generate_embeddings
        from tools.index_to_vectordb import index_to_vectordb

        try:
            # Parse using shared dispatcher (handles all file types)
            parsed = parse_file_by_type(file_path)

            # Chunk the document
            chunks = chunk_documents(parsed, chunk_size=settings.max_chunk_size, overlap=settings.chunk_overlap)

            if not chunks:
                raise ValueError("No content extracted from document")

            # Add metadata to chunks
            for chunk in chunks:
                chunk['document_id'] = document_id
                chunk['data_room_id'] = data_room_id

            # Generate embeddings
            from app.config import settings
            logger.info(f"[{data_room_id}] Generating embeddings (batch_size={settings.batch_size}, max_concurrent={settings.embedding_max_concurrent})")
            chunks_with_embeddings = generate_embeddings(
                chunks=chunks,
                batch_size=settings.batch_size,
                max_concurrent=settings.embedding_max_concurrent
            )

            # Index to vector DB
            index_to_vectordb(data_room_id=data_room_id, chunks_with_embeddings=chunks_with_embeddings)

            # Save chunks to database
            db.create_chunks(chunks_with_embeddings)

            # Update document status
            db.update_document_status(
                document_id,
                status="parsed",
                page_count=parsed.get('page_count') or parsed.get('slide_count') or 1,
                token_count=sum(c.get('token_count', 0) for c in chunks)
            )

            logger.success(f"Processed synced file: {file_name} -> {len(chunks)} chunks")

            return document_id

        except Exception as e:
            db.update_document_status(document_id, status="failed", error_message=str(e))
            raise

    def trigger_sync(self, folder_id: str):
        """
        Trigger immediate sync for a specific folder.

        Args:
            folder_id: Connected folder ID
        """
        logger.info(f"Triggering immediate sync for folder: {folder_id}")

        folder = db.get_connected_folder(folder_id)
        if not folder:
            logger.error(f"trigger_sync: Folder not found: {folder_id}")
            return

        data_room_id = folder.get('data_room_id')

        try:
            # Get user tokens
            user = db.get_user_by_id(folder['user_id'])
            if not user:
                raise ValueError(f"User not found for folder: {folder_id}")

            folder_with_tokens = {
                **folder,
                'access_token': user['access_token'],
                'refresh_token': user['refresh_token'],
                'token_expires_at': user.get('token_expires_at')
            }

            def _safe_sync():
                try:
                    self._sync_folder(folder_with_tokens)
                except Exception as e:
                    logger.error(f"Sync thread crashed for folder {folder_id}: {e}")
                    logger.error(traceback.format_exc())
                    if data_room_id:
                        try:
                            db.update_data_room_status(
                                data_room_id, 'failed',
                                error_message=f"Sync failed: {str(e)[:500]}"
                            )
                        except Exception:
                            logger.error("Failed to update data room status after sync crash")

            # Run sync in background thread
            sync_thread = threading.Thread(
                target=_safe_sync,
                daemon=True,
                name=f"sync-{folder_id[:8]}"
            )
            sync_thread.start()
            logger.info(f"Sync thread started for folder {folder_id} (thread={sync_thread.name})")

        except Exception as e:
            logger.error(f"trigger_sync setup failed for {folder_id}: {e}")
            logger.error(traceback.format_exc())
            if data_room_id:
                try:
                    db.update_data_room_status(
                        data_room_id, 'failed',
                        error_message=f"Sync setup failed: {str(e)[:500]}"
                    )
                except Exception:
                    logger.error("Failed to update data room status after trigger_sync error")


# Global sync service instance
sync_service = SyncService()
