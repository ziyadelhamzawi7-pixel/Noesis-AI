"""
SQLite-based persistent job queue for background processing.

Provides reliable job processing with:
- Job persistence across server restarts
- Automatic retry on failure
- Priority support
- Job status tracking
"""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from contextlib import contextmanager
from loguru import logger

from app.config import settings


class JobStatus(str, Enum):
    """Job status values."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Job type identifiers."""
    PROCESS_DATA_ROOM = "process_data_room"
    PROCESS_FILE = "process_file"
    GENERATE_EMBEDDINGS = "generate_embeddings"
    SYNC_DRIVE_FOLDER = "sync_drive_folder"
    FINANCIAL_ANALYSIS = "financial_analysis"


@contextmanager
def _get_queue_connection():
    """Get database connection for job queue operations."""
    conn = sqlite3.connect(settings.database_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    try:
        yield conn
    finally:
        conn.close()


class JobQueue:
    """
    SQLite-based job queue for persistent background task processing.

    Features:
    - Jobs persist across server restarts
    - Automatic retry with configurable attempts
    - Priority-based processing
    - Job status tracking and history
    """

    def __init__(self, max_workers: int = 3):
        """
        Initialize job queue.

        Args:
            max_workers: Maximum concurrent workers (for reference)
        """
        self.max_workers = max_workers
        self._init_table()

    def _init_table(self):
        """Create job queue table if not exists."""
        with _get_queue_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS job_queue (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    priority INTEGER DEFAULT 0,
                    attempts INTEGER DEFAULT 0,
                    max_attempts INTEGER DEFAULT 3,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    worker_id TEXT,
                    data_room_id TEXT,
                    file_name TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_job_status
                ON job_queue(status, priority DESC, created_at ASC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_job_data_room
                ON job_queue(data_room_id)
            """)
            conn.commit()
            logger.debug("Job queue table initialized")

    def enqueue(
        self,
        job_type: str,
        payload: Dict[str, Any],
        priority: int = 0,
        data_room_id: Optional[str] = None,
        file_name: Optional[str] = None,
        max_attempts: int = 3
    ) -> str:
        """
        Add a job to the queue.

        Args:
            job_type: Type of job (from JobType enum)
            payload: Job data as dictionary
            priority: Higher priority jobs run first (default 0)
            data_room_id: Associated data room (for tracking)
            file_name: Associated file name (for tracking)
            max_attempts: Maximum retry attempts

        Returns:
            Job ID
        """
        job_id = f"job_{uuid.uuid4().hex[:12]}"

        with _get_queue_connection() as conn:
            conn.execute("""
                INSERT INTO job_queue
                (id, job_type, payload, priority, data_room_id, file_name, max_attempts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                job_id,
                job_type,
                json.dumps(payload),
                priority,
                data_room_id,
                file_name,
                max_attempts
            ))
            conn.commit()

        logger.info(f"Enqueued job {job_id} (type={job_type}, priority={priority})")
        return job_id

    def claim_job(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Claim the next available job for processing.

        Uses atomic UPDATE with RETURNING to prevent race conditions.

        Args:
            worker_id: Identifier for the worker claiming the job

        Returns:
            Job dict or None if no jobs available
        """
        with _get_queue_connection() as conn:
            # Find and claim the highest priority pending job
            cursor = conn.execute("""
                UPDATE job_queue
                SET status = 'running',
                    started_at = CURRENT_TIMESTAMP,
                    worker_id = ?,
                    attempts = attempts + 1
                WHERE id = (
                    SELECT id FROM job_queue
                    WHERE status = 'pending'
                    AND attempts < max_attempts
                    ORDER BY priority DESC, created_at ASC
                    LIMIT 1
                )
                RETURNING id, job_type, payload, attempts, data_room_id, file_name
            """, (worker_id,))

            row = cursor.fetchone()
            conn.commit()

            if row:
                job = {
                    'id': row['id'],
                    'job_type': row['job_type'],
                    'payload': json.loads(row['payload']),
                    'attempts': row['attempts'],
                    'data_room_id': row['data_room_id'],
                    'file_name': row['file_name']
                }
                logger.debug(f"Worker {worker_id} claimed job {job['id']}")
                return job

        return None

    def complete_job(self, job_id: str):
        """
        Mark a job as completed.

        Args:
            job_id: Job ID to complete
        """
        with _get_queue_connection() as conn:
            conn.execute("""
                UPDATE job_queue
                SET status = 'completed',
                    completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (job_id,))
            conn.commit()

        logger.info(f"Job {job_id} completed")

    def fail_job(self, job_id: str, error: str, retry: bool = True):
        """
        Mark a job as failed.

        Args:
            job_id: Job ID that failed
            error: Error message
            retry: If True and attempts remain, job returns to pending
        """
        with _get_queue_connection() as conn:
            if retry:
                # Check if retries remain
                cursor = conn.execute(
                    "SELECT attempts, max_attempts FROM job_queue WHERE id = ?",
                    (job_id,)
                )
                row = cursor.fetchone()

                if row and row['attempts'] < row['max_attempts']:
                    conn.execute("""
                        UPDATE job_queue
                        SET status = 'pending',
                            error_message = ?,
                            worker_id = NULL
                        WHERE id = ?
                    """, (error, job_id))
                    conn.commit()
                    logger.warning(f"Job {job_id} failed, will retry (attempt {row['attempts']}/{row['max_attempts']}): {error}")
                    return

            # No retries - mark as failed
            conn.execute("""
                UPDATE job_queue
                SET status = 'failed',
                    error_message = ?,
                    completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (error, job_id))
            conn.commit()

        logger.error(f"Job {job_id} failed permanently: {error}")

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled, False if job not found or not pending
        """
        with _get_queue_connection() as conn:
            cursor = conn.execute("""
                UPDATE job_queue
                SET status = 'cancelled',
                    completed_at = CURRENT_TIMESTAMP
                WHERE id = ? AND status = 'pending'
            """, (job_id,))
            conn.commit()
            return cursor.rowcount > 0

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get job details by ID.

        Args:
            job_id: Job ID

        Returns:
            Job dict or None
        """
        with _get_queue_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM job_queue WHERE id = ?",
                (job_id,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def get_jobs_by_data_room(self, data_room_id: str) -> List[Dict[str, Any]]:
        """
        Get all jobs for a data room.

        Args:
            data_room_id: Data room ID

        Returns:
            List of job dicts
        """
        with _get_queue_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM job_queue WHERE data_room_id = ? ORDER BY created_at DESC",
                (data_room_id,)
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_pending_count(self) -> int:
        """Get count of pending jobs."""
        with _get_queue_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM job_queue WHERE status = 'pending'"
            )
            return cursor.fetchone()[0]

    def get_running_count(self) -> int:
        """Get count of running jobs."""
        with _get_queue_connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM job_queue WHERE status = 'running'"
            )
            return cursor.fetchone()[0]

    def get_queue_stats(self) -> Dict[str, int]:
        """
        Get queue statistics.

        Returns:
            Dict with counts by status
        """
        with _get_queue_connection() as conn:
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count
                FROM job_queue
                GROUP BY status
            """)
            stats = {row['status']: row['count'] for row in cursor.fetchall()}

        return {
            'pending': stats.get('pending', 0),
            'running': stats.get('running', 0),
            'completed': stats.get('completed', 0),
            'failed': stats.get('failed', 0),
            'cancelled': stats.get('cancelled', 0),
            'total': sum(stats.values())
        }

    def cleanup_old_jobs(self, days: int = 7) -> int:
        """
        Remove completed/failed jobs older than specified days.

        Args:
            days: Remove jobs older than this many days

        Returns:
            Number of jobs removed
        """
        with _get_queue_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM job_queue
                WHERE status IN ('completed', 'failed', 'cancelled')
                AND completed_at < datetime('now', ?)
            """, (f'-{days} days',))
            conn.commit()
            count = cursor.rowcount

        if count > 0:
            logger.info(f"Cleaned up {count} old jobs")

        return count

    def recover_stale_jobs(self, timeout_minutes: int = 30) -> int:
        """
        Reset jobs that have been running too long (likely crashed workers).

        Args:
            timeout_minutes: Consider jobs stale after this many minutes

        Returns:
            Number of jobs recovered
        """
        with _get_queue_connection() as conn:
            cursor = conn.execute("""
                UPDATE job_queue
                SET status = 'pending',
                    worker_id = NULL,
                    error_message = 'Job recovered after timeout'
                WHERE status = 'running'
                AND started_at < datetime('now', ?)
            """, (f'-{timeout_minutes} minutes',))
            conn.commit()
            count = cursor.rowcount

        if count > 0:
            logger.warning(f"Recovered {count} stale jobs")

        return count


# Global job queue instance
job_queue = JobQueue(max_workers=settings.max_concurrent_jobs)
