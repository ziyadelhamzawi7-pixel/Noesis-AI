"""
Database utilities for VC Due Diligence system.
Provides functions for querying and updating PostgreSQL database.

Includes:
- Connection pooling via psycopg2
- Batch insert optimization
"""

import psycopg2
import psycopg2.pool
import psycopg2.extras
import json
import uuid
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager
from loguru import logger

from app.config import settings


# ============================================================================
# Connection Pool Implementation
# ============================================================================

class ConnectionPool:
    """
    Thread-safe PostgreSQL connection pool using psycopg2.

    Features:
    - Reusable connections to reduce connection overhead
    - Configurable pool size
    - Thread-safe connection management
    """

    def __init__(
        self,
        database_url: str,
        pool_size: int = 10,
        timeout: float = 30.0
    ):
        self.database_url = database_url
        self.pool_size = pool_size
        self.timeout = timeout
        self._pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self._lock = threading.Lock()
        self._initialized = False

    def _initialize_pool(self) -> None:
        """Initialize the connection pool."""
        with self._lock:
            if self._initialized:
                return
            self._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=self.pool_size,
                dsn=self.database_url
            )
            self._initialized = True
            logger.info(f"PostgreSQL connection pool initialized (max={self.pool_size})")

    @contextmanager
    def get_connection(self):
        """Get a connection from the pool."""
        if not self._initialized:
            self._initialize_pool()

        conn = self._pool.getconn()
        conn.cursor_factory = psycopg2.extras.RealDictCursor
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def close_all(self) -> None:
        """Close all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            self._initialized = False
            logger.info("Connection pool closed")

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_size": self.pool_size,
            "initialized": self._initialized
        }


# Global connection pool
_db_pool: Optional[ConnectionPool] = None


def get_db_pool() -> ConnectionPool:
    """Get or create the global connection pool."""
    global _db_pool
    if _db_pool is None:
        _db_pool = ConnectionPool(
            database_url=settings.database_url,
            pool_size=settings.db_pool_size,
            timeout=30.0
        )
    return _db_pool


@contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    Uses the connection pool for efficient connection reuse.
    """
    pool = get_db_pool()
    with pool.get_connection() as conn:
        yield conn


@contextmanager
def get_db_connection_simple():
    """
    Simple context manager for database connections (non-pooled).
    Use for one-off operations or when pool is not needed.
    """
    conn = psycopg2.connect(settings.database_url, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def dict_from_row(row) -> Optional[Dict[str, Any]]:
    """Convert database row to dictionary."""
    return dict(row) if row else None


# ============================================================================
# Database Migrations
# ============================================================================

_migrations_run = False


def run_migrations():
    """Run database migrations to add new tables/columns."""
    global _migrations_run
    if _migrations_run:
        return

    try:
        with get_db_connection_simple() as conn:
            cursor = conn.cursor()

            # Create connected_files table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS connected_files (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    drive_file_id TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_path TEXT,
                    mime_type TEXT,
                    file_size INTEGER,
                    data_room_id TEXT,
                    document_id TEXT,
                    sync_status TEXT CHECK(sync_status IN ('pending', 'downloading', 'processing', 'complete', 'failed')) DEFAULT 'pending',
                    local_file_path TEXT,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (data_room_id) REFERENCES data_rooms(id) ON DELETE SET NULL,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL,
                    UNIQUE(user_id, drive_file_id)
                )
            """)

            # Create indexes for connected_files
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_connected_files_user ON connected_files(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_connected_files_data_room ON connected_files(data_room_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_connected_files_drive_id ON connected_files(drive_file_id)")

            # Migration: Add user_id column to data_rooms for ownership
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'data_rooms'")
            dr_columns = [col['column_name'] for col in cursor.fetchall()]
            if 'user_id' not in dr_columns:
                cursor.execute("ALTER TABLE data_rooms ADD COLUMN user_id TEXT REFERENCES users(id)")
                logger.info("Added user_id column to data_rooms table")

            # Migration: Add user_id column to queries for per-user filtering
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'queries'")
            q_columns = [col['column_name'] for col in cursor.fetchall()]
            if 'user_id' not in q_columns:
                cursor.execute("ALTER TABLE queries ADD COLUMN user_id TEXT REFERENCES users(id)")
                logger.info("Added user_id column to queries table")

            # Create data_room_members table for sharing
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_room_members (
                    id TEXT PRIMARY KEY,
                    data_room_id TEXT NOT NULL,
                    user_id TEXT,
                    invited_email TEXT NOT NULL,
                    role TEXT CHECK(role IN ('owner', 'member')) NOT NULL,
                    status TEXT CHECK(status IN ('pending', 'accepted', 'revoked')) DEFAULT 'pending',
                    invited_by TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    accepted_at TIMESTAMP,
                    FOREIGN KEY (data_room_id) REFERENCES data_rooms(id) ON DELETE CASCADE,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
                    UNIQUE(data_room_id, invited_email)
                )
            """)

            # Create indexes for data_room_members
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_drm_data_room ON data_room_members(data_room_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_drm_user ON data_room_members(user_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_drm_email ON data_room_members(invited_email)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_drm_access ON data_room_members(data_room_id, status, user_id, invited_email)")

            # Create memo_chat_messages table for memo chat feature
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memo_chat_messages (
                    id TEXT PRIMARY KEY,
                    memo_id TEXT NOT NULL,
                    data_room_id TEXT NOT NULL,
                    role TEXT CHECK(role IN ('user', 'assistant')) NOT NULL,
                    content TEXT NOT NULL,
                    updated_section_key TEXT,
                    updated_section_content TEXT,
                    tokens_used INTEGER DEFAULT 0,
                    cost REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (data_room_id) REFERENCES data_rooms(id) ON DELETE CASCADE
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mcm_memo ON memo_chat_messages(memo_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mcm_data_room ON memo_chat_messages(data_room_id)")

            # Migration: Add missing memo columns for existing databases
            cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'memos'")
            memo_columns = {col['column_name'] for col in cursor.fetchall()}
            memo_additions = {
                'proposed_investment_terms': 'TEXT',
                'valuation_analysis': 'TEXT',
                'ticket_size': 'REAL',
                'post_money_valuation': 'REAL',
                'valuation_methods': 'TEXT',
            }
            for col_name, col_type in memo_additions.items():
                if col_name not in memo_columns:
                    cursor.execute(f"ALTER TABLE memos ADD COLUMN {col_name} {col_type}")
                    logger.info(f"Added {col_name} column to memos table")

            # Migration: Deduplicate documents and add UNIQUE(data_room_id, file_name)
            # Check if the unique index already exists
            cursor.execute("SELECT indexname FROM pg_indexes WHERE indexname = 'idx_documents_unique_file'")
            if not cursor.fetchone():
                # Deduplicate: keep the record with best parse_status for each (data_room_id, file_name)
                cursor.execute("""
                    SELECT data_room_id, file_name, COUNT(*) as cnt
                    FROM documents
                    GROUP BY data_room_id, file_name
                    HAVING COUNT(*) > 1
                """)
                dup_groups = cursor.fetchall()
                if dup_groups:
                    total_removed = 0
                    for dup in dup_groups:
                        dr_id = dup['data_room_id']
                        fname = dup['file_name']
                        # Keep the one with best status; among ties, keep latest
                        cursor.execute("""
                            SELECT id FROM documents
                            WHERE data_room_id = %s AND file_name = %s
                            ORDER BY
                                CASE parse_status
                                    WHEN 'parsed' THEN 1
                                    WHEN 'parsing' THEN 2
                                    WHEN 'failed' THEN 3
                                    WHEN 'pending' THEN 4
                                END,
                                uploaded_at DESC
                            LIMIT 1
                        """, (dr_id, fname))
                        keep_row = cursor.fetchone()
                        if keep_row:
                            keep_id = keep_row['id']
                            cursor.execute(
                                "DELETE FROM documents WHERE data_room_id = %s AND file_name = %s AND id != %s",
                                (dr_id, fname, keep_id)
                            )
                            total_removed += cursor.rowcount
                    logger.info(f"Deduplicated documents: removed {total_removed} duplicate records")

                    # Clean up orphaned chunks referencing deleted documents
                    cursor.execute("DELETE FROM chunks WHERE document_id NOT IN (SELECT id FROM documents)")
                    orphaned_chunks = cursor.rowcount
                    if orphaned_chunks:
                        logger.info(f"Cleaned up {orphaned_chunks} orphaned chunks from deduplication")

                # Now safe to create the unique index
                cursor.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_unique_file
                    ON documents(data_room_id, file_name)
                """)
                logger.info("Created UNIQUE index on documents(data_room_id, file_name)")

            # Migration: Un-own upload-based data rooms so they're accessible without login
            # Drive-connected data rooms (which have connected_folders entries) keep their owners
            cursor.execute("""
                UPDATE data_rooms SET user_id = NULL
                WHERE user_id IS NOT NULL
                AND id NOT IN (
                    SELECT DISTINCT data_room_id FROM connected_folders
                    WHERE data_room_id IS NOT NULL
                )
            """)
            unowned = cursor.rowcount
            if unowned > 0:
                logger.info(f"Made {unowned} upload-based data rooms accessible as legacy (cleared user_id)")

            conn.commit()
            logger.info("Database migrations completed successfully")

    except Exception as e:
        logger.error(f"Error running migrations: {e}")

    _migrations_run = True


# Run migrations when module is loaded
run_migrations()


# ============================================================================
# Data Room Operations
# ============================================================================

def create_data_room(
    company_name: str,
    analyst_name: str,
    analyst_email: Optional[str] = None,
    security_level: str = "local_only",
    total_documents: int = 0,
    data_room_id: Optional[str] = None,
    user_id: Optional[str] = None
) -> str:
    """
    Create a new data room record.

    Args:
        company_name: Name of the company
        analyst_name: Name of the analyst
        analyst_email: Email of the analyst
        security_level: Security level (local_only or cloud_enabled)
        total_documents: Total number of documents
        data_room_id: Optional custom ID (auto-generated if not provided)
        user_id: Optional owner user ID (for sharing feature)

    Returns:
        Data room ID
    """
    if not data_room_id:
        data_room_id = f"dr_{uuid.uuid4().hex[:12]}"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO data_rooms (
                id, company_name, analyst_name, analyst_email,
                security_level, total_documents, processing_status, user_id
            ) VALUES (%s, %s, %s, %s, %s, %s, 'uploading', %s)
        """, (data_room_id, company_name, analyst_name, analyst_email,
              security_level, total_documents, user_id))
        conn.commit()

    # Auto-create owner membership record when sharing is enabled
    if user_id and settings.enable_sharing and analyst_email:
        try:
            add_data_room_member(data_room_id, analyst_email, 'owner', user_id)
        except Exception as e:
            logger.warning(f"Failed to create owner membership for {data_room_id}: {e}")

    logger.info(f"Created data room: {data_room_id} for {company_name} (owner: {user_id})")
    return data_room_id


def get_data_room(data_room_id: str) -> Optional[Dict[str, Any]]:
    """
    Get data room by ID.

    Args:
        data_room_id: Data room ID

    Returns:
        Data room record or None if not found
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM data_rooms WHERE id = %s", (data_room_id,))
        row = cursor.fetchone()
        return dict_from_row(row)


def update_data_room_status(
    data_room_id: str,
    status: str,
    progress: Optional[float] = None,
    completed_at: Optional[str] = None,
    error_message: Optional[str] = None,
    total_chunks: Optional[int] = None,
    total_documents: Optional[int] = None
) -> bool:
    """
    Update data room processing status.

    Args:
        data_room_id: Data room ID
        status: New status (uploading, parsing, indexing, extracting, complete, failed)
        progress: Progress percentage (0-100)
        completed_at: Completion timestamp
        error_message: Error message if status is 'failed'
        total_chunks: Total number of indexed chunks
        total_documents: Total number of processed documents

    Returns:
        True if updated successfully
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Build dynamic query based on provided parameters
        fields = ["processing_status = %s"]
        values = [status]

        if progress is not None:
            # Enforce monotonic progress — never decrease (except reset to 0 for reprocessing)
            if progress == 0:
                fields.append("progress_percent = %s")
            else:
                fields.append("progress_percent = GREATEST(COALESCE(progress_percent, 0), %s)")
            values.append(progress)

        if completed_at is not None:
            fields.append("completed_at = %s")
            values.append(completed_at)

        if error_message is not None:
            fields.append("error_message = %s")
            values.append(error_message)

        if total_chunks is not None:
            fields.append("total_chunks = %s")
            values.append(total_chunks)

        if total_documents is not None:
            fields.append("total_documents = %s")
            values.append(total_documents)

        values.append(data_room_id)

        query = f"UPDATE data_rooms SET {', '.join(fields)} WHERE id = %s"
        cursor.execute(query, values)
        conn.commit()
        return cursor.rowcount > 0


def delete_data_room(data_room_id: str) -> bool:
    """
    Delete a data room and all associated records (for rollback on failure).

    Args:
        data_room_id: Data room ID to delete

    Returns:
        True if deleted successfully
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Delete in order of dependencies (child tables first)
        cursor.execute("DELETE FROM chunks WHERE data_room_id = %s", (data_room_id,))
        cursor.execute("DELETE FROM queries WHERE data_room_id = %s", (data_room_id,))
        cursor.execute("DELETE FROM documents WHERE data_room_id = %s", (data_room_id,))
        cursor.execute("DELETE FROM processing_logs WHERE data_room_id = %s", (data_room_id,))
        cursor.execute("DELETE FROM analysis_cache WHERE data_room_id = %s", (data_room_id,))
        cursor.execute("DELETE FROM api_usage WHERE data_room_id = %s", (data_room_id,))
        cursor.execute("DELETE FROM job_queue WHERE data_room_id = %s", (data_room_id,))
        cursor.execute("DELETE FROM data_rooms WHERE id = %s", (data_room_id,))

        conn.commit()
        logger.info(f"Deleted data room: {data_room_id}")
        return cursor.rowcount > 0


def update_data_room_cost(data_room_id: str, cost: float) -> bool:
    """
    Update actual cost for data room.

    Args:
        data_room_id: Data room ID
        cost: Cost to add to actual_cost

    Returns:
        True if updated successfully
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE data_rooms
            SET actual_cost = COALESCE(actual_cost, 0) + %s
            WHERE id = %s
        """, (cost, data_room_id))
        conn.commit()
        return cursor.rowcount > 0


def list_data_rooms(limit: int = 50) -> List[Dict[str, Any]]:
    """
    List all data rooms.

    Args:
        limit: Maximum number of records to return

    Returns:
        List of data room records
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT *
            FROM data_rooms
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        return [dict_from_row(row) for row in cursor.fetchall()]


# ============================================================================
# Data Room Sharing Operations
# ============================================================================

def add_data_room_member(
    data_room_id: str,
    invited_email: str,
    role: str,
    invited_by: Optional[str] = None
) -> str:
    """
    Add a member to a data room.

    Args:
        data_room_id: Data room ID
        invited_email: Email address of invited user
        role: Role (owner or member)
        invited_by: User ID of the person who invited

    Returns:
        Member record ID
    """
    member_id = f"drm_{uuid.uuid4().hex[:12]}"

    # Check if invited email matches an existing user
    existing_user_id = None
    accepted_at = None
    status = 'pending'

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = %s", (invited_email,))
        user_row = cursor.fetchone()
        if user_row:
            existing_user_id = user_row[0] if isinstance(user_row, tuple) else user_row['id']
            status = 'accepted'
            accepted_at = datetime.now().isoformat()

        cursor.execute("""
            INSERT INTO data_room_members (
                id, data_room_id, user_id, invited_email, role, status, invited_by, accepted_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (member_id, data_room_id, existing_user_id, invited_email, role, status, invited_by, accepted_at))
        conn.commit()

    logger.info(f"Added member {invited_email} to data room {data_room_id} as {role} (status: {status})")
    return member_id


def accept_data_room_invite(data_room_id: str, user_id: str, user_email: str) -> bool:
    """
    Accept a pending data room invite.

    Args:
        data_room_id: Data room ID
        user_id: User ID accepting the invite
        user_email: Email of the user

    Returns:
        True if invite was accepted
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE data_room_members
            SET user_id = %s, status = 'accepted', accepted_at = %s
            WHERE data_room_id = %s AND invited_email = %s AND status = 'pending'
        """, (user_id, datetime.now().isoformat(), data_room_id, user_email))
        conn.commit()
        return cursor.rowcount > 0


def revoke_data_room_member(data_room_id: str, member_id: str) -> bool:
    """
    Remove a member from a data room.

    Args:
        data_room_id: Data room ID
        member_id: Member record ID to remove

    Returns:
        True if member was removed
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM data_room_members
            WHERE id = %s AND data_room_id = %s AND role != 'owner'
        """, (member_id, data_room_id))
        conn.commit()
        return cursor.rowcount > 0


def get_data_room_members(data_room_id: str) -> List[Dict[str, Any]]:
    """
    Get all members of a data room.

    Args:
        data_room_id: Data room ID

    Returns:
        List of member records with user info joined
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT drm.*, u.name, u.picture_url
            FROM data_room_members drm
            LEFT JOIN users u ON drm.user_id = u.id
            WHERE drm.data_room_id = %s AND drm.status != 'revoked'
            ORDER BY drm.role ASC, drm.created_at ASC
        """, (data_room_id,))
        return [dict_from_row(row) for row in cursor.fetchall()]


def get_user_data_rooms(user_id: str, user_email: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get data rooms accessible to a user (owned + shared).

    Args:
        user_id: User ID
        user_email: User email
        limit: Maximum number of records

    Returns:
        List of data room records with user_role field
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT dr.*,
                CASE
                    WHEN dr.user_id = %s THEN 'owner'
                    ELSE 'member'
                END as user_role
            FROM data_rooms dr
            LEFT JOIN data_room_members drm ON dr.id = drm.data_room_id
                AND (drm.user_id = %s OR drm.invited_email = %s)
                AND drm.status = 'accepted'
            WHERE dr.user_id = %s
                OR drm.id IS NOT NULL
            GROUP BY dr.id
            ORDER BY dr.created_at DESC
            LIMIT %s
        """, (user_id, user_id, user_email, user_id, limit))
        return [dict_from_row(row) for row in cursor.fetchall()]


def check_data_room_access(
    data_room_id: str,
    user_id: Optional[str] = None,
    user_email: Optional[str] = None
) -> Optional[str]:
    """
    Check if a user has access to a data room.

    Args:
        data_room_id: Data room ID
        user_id: User ID
        user_email: User email

    Returns:
        Role string ('owner', 'member', 'legacy') or None if no access
    """
    if not settings.enable_sharing:
        return 'owner'

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Check if data room exists
        cursor.execute("SELECT user_id FROM data_rooms WHERE id = %s", (data_room_id,))
        room = cursor.fetchone()
        if not room:
            return None

        room_owner = room['user_id'] if room else None

        # Unowned rooms are not accessible (no legacy sharing)
        if room_owner is None:
            return None

        # Direct owner
        if user_id and room_owner == user_id:
            return 'owner'

        # Check membership
        if user_id or user_email:
            conditions = []
            params = [data_room_id]
            if user_id:
                conditions.append("user_id = %s")
                params.append(user_id)
            if user_email:
                conditions.append("invited_email = %s")
                params.append(user_email)

            where_clause = " OR ".join(conditions)
            cursor.execute(f"""
                SELECT role FROM data_room_members
                WHERE data_room_id = %s AND ({where_clause}) AND status = 'accepted'
                LIMIT 1
            """, params)
            member = cursor.fetchone()
            if member:
                return member['role']

        return None


def auto_accept_pending_invites(user_id: str, user_email: str) -> int:
    """
    Auto-accept all pending invites for a user on login.

    Args:
        user_id: User ID
        user_email: User email

    Returns:
        Number of invites accepted
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE data_room_members
            SET user_id = %s, status = 'accepted', accepted_at = %s
            WHERE invited_email = %s AND status = 'pending'
        """, (user_id, datetime.now().isoformat(), user_email))
        conn.commit()
        count = cursor.rowcount
        if count > 0:
            logger.info(f"Auto-accepted {count} pending invites for {user_email}")
        return count


# ============================================================================
# Document Operations
# ============================================================================

def create_document(
    data_room_id: str,
    file_name: str,
    file_path: str,
    file_size: int,
    file_type: Optional[str] = None,
    document_id: Optional[str] = None,
    relative_path: Optional[str] = None
) -> str:
    """
    Create a new document record.

    Args:
        data_room_id: Parent data room ID
        file_name: Original file name
        file_path: Path to stored file
        file_size: File size in bytes
        file_type: File type (pdf, xlsx, etc.)
        document_id: Optional custom ID
        relative_path: Optional folder-relative path for display in document tree

    Returns:
        Document ID
    """
    if not document_id:
        document_id = f"doc_{uuid.uuid4().hex[:12]}"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO documents (
                id, data_room_id, file_name, file_path,
                file_size, file_type, parse_status, relative_path
            ) VALUES (%s, %s, %s, %s, %s, %s, 'pending', %s)
            ON CONFLICT (data_room_id, file_name) DO NOTHING
        """, (document_id, data_room_id, file_name, file_path,
              file_size, file_type, relative_path))
        conn.commit()

        # If insert was ignored (duplicate), return the existing ID
        if cursor.rowcount == 0:
            cursor.execute(
                "SELECT id FROM documents WHERE data_room_id = %s AND file_name = %s",
                (data_room_id, file_name)
            )
            row = cursor.fetchone()
            if row:
                logger.info(f"Document '{file_name}' already exists in {data_room_id}, returning existing ID")
                return row['id'] if isinstance(row, dict) else row[0]

    return document_id


def get_document_by_name(data_room_id: str, file_name: str) -> Optional[Dict[str, Any]]:
    """
    Get document by data room ID and file name.

    Args:
        data_room_id: Parent data room ID
        file_name: Original file name

    Returns:
        Document record or None if not found
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM documents WHERE data_room_id = %s AND file_name = %s",
            (data_room_id, file_name)
        )
        row = cursor.fetchone()
        return dict_from_row(row)


def update_document_status(
    document_id: str,
    status: str,
    page_count: Optional[int] = None,
    token_count: Optional[int] = None,
    error_message: Optional[str] = None
) -> bool:
    """
    Update document parsing status.

    Args:
        document_id: Document ID
        status: New status (pending, parsing, parsed, failed)
        page_count: Number of pages
        token_count: Number of tokens
        error_message: Error message if failed

    Returns:
        True if updated successfully
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        fields = ["parse_status = %s", "parsed_at = CURRENT_TIMESTAMP"]
        values = [status]

        if page_count is not None:
            fields.append("page_count = %s")
            values.append(page_count)

        if token_count is not None:
            fields.append("token_count = %s")
            values.append(token_count)

        if error_message is not None:
            fields.append("error_message = %s")
            values.append(error_message)

        values.append(document_id)

        query = f"UPDATE documents SET {', '.join(fields)} WHERE id = %s"
        cursor.execute(query, values)
        conn.commit()

        return cursor.rowcount > 0


def get_documents_by_data_room(data_room_id: str) -> List[Dict[str, Any]]:
    """
    Get all documents for a data room.

    Args:
        data_room_id: Data room ID

    Returns:
        List of document records
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM documents
            WHERE data_room_id = %s
            ORDER BY uploaded_at DESC
        """, (data_room_id,))
        return [dict_from_row(row) for row in cursor.fetchall()]


def find_document_by_filename(
    data_room_id: str,
    target_filename: str,
    create_if_missing: bool = False,
    file_path: str = None,
    file_size: int = 0,
) -> Optional[str]:
    """
    Find a document ID by filename with resilient multi-tier matching.

    Uses a single DB connection and atomic INSERT OR IGNORE to prevent
    race conditions when called from parallel threads.

    Matching priority:
    1. Exact match on file_name
    2. Case-insensitive match
    3. Stem match (handles .doc vs .docx, sanitized names, etc.)
    4. If create_if_missing=True, atomic find-or-create via INSERT OR IGNORE

    Args:
        data_room_id: Parent data room ID
        target_filename: Filename to search for
        create_if_missing: If True, create a document record when no match is found
        file_path: File path (required if create_if_missing=True)
        file_size: File size in bytes (used for auto-creation)

    Returns:
        Document ID string, or None if not found and create_if_missing=False
    """
    from pathlib import Path as _Path

    target_lower = target_filename.lower()
    target_stem = _Path(target_filename).stem.lower()

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Tier 1: exact match
        cursor.execute(
            "SELECT id FROM documents WHERE data_room_id = %s AND file_name = %s",
            (data_room_id, target_filename)
        )
        row = cursor.fetchone()
        if row:
            return row['id'] if isinstance(row, dict) else row[0]

        # Tier 2: case-insensitive match
        cursor.execute(
            "SELECT id, file_name FROM documents WHERE data_room_id = %s AND LOWER(file_name) = %s",
            (data_room_id, target_lower)
        )
        row = cursor.fetchone()
        if row:
            matched_name = row['file_name'] if isinstance(row, dict) else row[1]
            matched_id = row['id'] if isinstance(row, dict) else row[0]
            logger.warning(
                f"Document matched case-insensitively: '{target_filename}' ~ '{matched_name}'"
            )
            return matched_id

        # Tier 3: stem match (fetch all for this data room since no SQL stem function)
        cursor.execute(
            "SELECT id, file_name FROM documents WHERE data_room_id = %s",
            (data_room_id,)
        )
        for doc_row in cursor.fetchall():
            doc_name = doc_row['file_name'] if isinstance(doc_row, dict) else doc_row[1]
            doc_id = doc_row['id'] if isinstance(doc_row, dict) else doc_row[0]
            if _Path(doc_name).stem.lower() == target_stem:
                logger.warning(
                    f"Document matched by stem: '{target_filename}' ~ '{doc_name}'"
                )
                return doc_id

        # Tier 4: atomic find-or-create using INSERT ON CONFLICT
        if create_if_missing and file_path:
            file_type = _Path(target_filename).suffix.lower().lstrip('.')
            new_id = f"doc_{uuid.uuid4().hex[:12]}"

            cursor.execute("""
                INSERT INTO documents (
                    id, data_room_id, file_name, file_path,
                    file_size, file_type, parse_status
                ) VALUES (%s, %s, %s, %s, %s, %s, 'pending')
                ON CONFLICT (data_room_id, file_name) DO NOTHING
            """, (new_id, data_room_id, target_filename, file_path,
                  file_size, file_type))
            conn.commit()

            # SELECT to get the actual ID (ours or the concurrent winner's)
            cursor.execute(
                "SELECT id FROM documents WHERE data_room_id = %s AND file_name = %s",
                (data_room_id, target_filename)
            )
            row = cursor.fetchone()
            if row:
                actual_id = row['id'] if isinstance(row, dict) else row[0]
                if actual_id == new_id:
                    logger.error(
                        f"No document record found for '{target_filename}' in {data_room_id}. "
                        f"Auto-created record {new_id}."
                    )
                else:
                    logger.info(
                        f"Document '{target_filename}' was concurrently created by another thread"
                    )
                return actual_id

    return None


def get_documents_with_paths(data_room_id: str) -> List[Dict[str, Any]]:
    """
    Get documents with their folder paths from synced_files.

    Joins documents with synced_files to get file_path for folder hierarchy.
    Documents without synced_file records (direct uploads) have file_path = None.
    Returns ALL documents regardless of parse_status so UI can show processing indicators.

    Args:
        data_room_id: Data room ID

    Returns:
        List of document records with file_path field
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                d.id,
                d.file_name,
                d.file_type,
                d.file_size,
                d.page_count,
                d.parse_status,
                d.uploaded_at,
                d.parsed_at,
                COALESCE(sf.file_path, d.relative_path) as file_path
            FROM documents d
            LEFT JOIN synced_files sf ON sf.document_id = d.id
            WHERE d.data_room_id = %s
            ORDER BY COALESCE(sf.file_path, d.relative_path) ASC NULLS LAST, d.file_name ASC
        """, (data_room_id,))
        return [dict_from_row(row) for row in cursor.fetchall()]


# ============================================================================
# Chunk Operations
# ============================================================================

def create_chunks(chunks: List[Dict[str, Any]]) -> int:
    """
    Create multiple chunk records in batch using executemany for performance.

    Args:
        chunks: List of chunk dictionaries with keys:
            - document_id
            - data_room_id
            - chunk_index
            - chunk_text
            - token_count (optional)
            - page_number (optional)
            - section_title (optional)
            - chunk_type (optional)
            - embedding_id (optional)

    Returns:
        Number of chunks created
    """
    if not chunks:
        return 0

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Prepare all values for batch insert (much faster than individual inserts)
        # Strip NUL bytes from text — PostgreSQL TEXT columns reject \x00
        values = [
            (
                f"chunk_{uuid.uuid4().hex[:12]}",
                chunk['document_id'],
                chunk['data_room_id'],
                chunk['chunk_index'],
                chunk['chunk_text'].replace('\x00', ''),
                chunk.get('token_count'),
                chunk.get('page_number'),
                chunk.get('section_title'),
                chunk.get('chunk_type'),
                chunk.get('embedding_id')
            )
            for chunk in chunks
        ]

        cursor.executemany("""
            INSERT INTO chunks (
                id, document_id, data_room_id, chunk_index,
                chunk_text, token_count, page_number,
                section_title, chunk_type, embedding_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, values)

        conn.commit()
        return len(chunks)


def create_chunks_batch_optimized(
    chunks: List[Dict[str, Any]],
    batch_size: int = 5000
) -> int:
    """
    Create chunks in optimized batches with transaction batching.

    This function is optimized for high-throughput bulk inserts:
    - Processes in configurable batch sizes
    - Uses executemany for 10-50x faster inserts

    Args:
        chunks: List of chunk dictionaries
        batch_size: Number of chunks per transaction batch

    Returns:
        Total number of chunks created
    """
    if not chunks:
        return 0

    total_created = 0
    pool = get_db_pool()

    with pool.get_connection() as conn:
        cursor = conn.cursor()

        # Process in batches for memory efficiency
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # Strip NUL bytes from text — PostgreSQL TEXT columns reject \x00
            values = [
                (
                    f"chunk_{uuid.uuid4().hex[:12]}",
                    chunk['document_id'],
                    chunk['data_room_id'],
                    chunk['chunk_index'],
                    chunk['chunk_text'].replace('\x00', ''),
                    chunk.get('token_count'),
                    chunk.get('page_number'),
                    chunk.get('section_title'),
                    chunk.get('chunk_type'),
                    chunk.get('embedding_id')
                )
                for chunk in batch
            ]

            try:
                cursor.executemany("""
                    INSERT INTO chunks (
                        id, document_id, data_room_id, chunk_index,
                        chunk_text, token_count, page_number,
                        section_title, chunk_type, embedding_id
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, values)

                conn.commit()
                total_created += len(batch)

                logger.debug(
                    f"Batch insert: {len(batch)} chunks "
                    f"({i + len(batch)}/{len(chunks)})"
                )

            except Exception as e:
                conn.rollback()
                logger.error(f"Batch insert failed: {e}")
                raise

    logger.info(f"Created {total_created} chunks in optimized batches")
    return total_created


def ensure_indexes() -> None:
    """
    Ensure database indexes exist for optimal query performance.

    Call this during application startup.
    """
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_chunks_data_room ON chunks(data_room_id)",
        "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)",
        "CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks(embedding_id)",
        "CREATE INDEX IF NOT EXISTS idx_documents_data_room ON documents(data_room_id)",
        "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(parse_status)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_status_priority ON job_queue(status, priority)",
        "CREATE INDEX IF NOT EXISTS idx_jobs_data_room ON job_queue(data_room_id)",
        "CREATE INDEX IF NOT EXISTS idx_queries_data_room ON queries(data_room_id)",
    ]

    with get_db_connection() as conn:
        cursor = conn.cursor()
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except Exception as e:
                logger.warning(f"Index creation skipped: {e}")

        conn.commit()
        logger.info("Database indexes verified/created")


def get_chunks_without_embeddings(data_room_id: str) -> List[Dict[str, Any]]:
    """
    Get chunks that don't have embeddings (for re-embedding).

    Args:
        data_room_id: Data room ID

    Returns:
        List of chunk records without embeddings
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM chunks
            WHERE data_room_id = %s AND embedding_id IS NULL
            ORDER BY chunk_index ASC
        """, (data_room_id,))
        return [dict_from_row(row) for row in cursor.fetchall()]


def update_chunk_embedding_id(chunk_id: str, embedding_id: str) -> bool:
    """
    Update the embedding_id for a chunk after re-embedding.

    Args:
        chunk_id: Chunk ID
        embedding_id: New embedding ID

    Returns:
        True if updated successfully
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE chunks SET embedding_id = %s WHERE id = %s",
            (embedding_id, chunk_id)
        )
        conn.commit()
        return cursor.rowcount > 0


def get_chunks_by_data_room(
    data_room_id: str,
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get chunks for a data room.

    Args:
        data_room_id: Data room ID
        limit: Maximum number of chunks to return

    Returns:
        List of chunk records
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        if limit:
            cursor.execute("""
                SELECT * FROM chunks
                WHERE data_room_id = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (data_room_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM chunks
                WHERE data_room_id = %s
                ORDER BY created_at DESC
            """, (data_room_id,))

        return [dict_from_row(row) for row in cursor.fetchall()]


def search_chunks_by_keywords(
    data_room_id: str,
    keywords: List[str],
    limit: int = 15
) -> List[Dict[str, Any]]:
    """
    Search chunks by keyword matching (ILIKE) on chunk_text.
    Joins with documents table to include file_name and document_category.

    Args:
        data_room_id: Data room ID
        keywords: List of keywords to search for (OR logic)
        limit: Maximum results to return

    Returns:
        List of chunk records with document metadata
    """
    if not keywords:
        return []

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Build OR conditions for each keyword
        conditions = []
        params: list = [data_room_id]
        for kw in keywords:
            conditions.append("c.chunk_text ILIKE %s")
            params.append(f"%{kw}%")

        where_clause = " OR ".join(conditions)
        params.append(limit)

        cursor.execute(f"""
            SELECT c.id, c.document_id, c.data_room_id, c.chunk_index,
                   c.chunk_text, c.token_count, c.page_number, c.section_title,
                   c.chunk_type, c.embedding_id, c.metadata,
                   d.file_name, d.document_category
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.data_room_id = %s AND ({where_clause})
            ORDER BY c.chunk_index ASC
            LIMIT %s
        """, params)

        return [dict_from_row(row) for row in cursor.fetchall()]


# ============================================================================
# Query Operations
# ============================================================================

def save_query(
    data_room_id: str,
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
    confidence_score: Optional[float] = None,
    analyst_email: Optional[str] = None,
    conversation_id: Optional[str] = None,
    response_time_ms: Optional[int] = None,
    tokens_used: Optional[int] = None,
    cost: Optional[float] = None,
    user_id: Optional[str] = None
) -> str:
    """
    Save a query and answer to the database.

    Args:
        data_room_id: Data room ID
        question: Question text
        answer: Answer text
        sources: List of source citations
        confidence_score: Confidence score (0-1)
        analyst_email: Email of analyst who asked
        conversation_id: Conversation ID for threading
        response_time_ms: Response time in milliseconds
        tokens_used: Number of tokens used
        cost: API cost
        user_id: User ID who asked the question

    Returns:
        Query ID
    """
    query_id = f"query_{uuid.uuid4().hex[:12]}"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO queries (
                id, data_room_id, question, answer, sources,
                confidence_score, analyst_email, conversation_id,
                response_time_ms, tokens_used, cost, user_id
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            query_id, data_room_id, question, answer,
            json.dumps(sources), confidence_score, analyst_email,
            conversation_id, response_time_ms, tokens_used, cost, user_id
        ))
        conn.commit()

    # Update data room cost
    if cost:
        update_data_room_cost(data_room_id, cost)

    return query_id


def get_query_history(
    data_room_id: str,
    limit: int = 50,
    conversation_id: Optional[str] = None,
    filter_user_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get query history for a data room.

    Args:
        data_room_id: Data room ID
        limit: Maximum number of queries to return
        conversation_id: Filter by conversation ID
        filter_user_id: Filter by user ID (for "my questions" view)

    Returns:
        List of query records
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        conditions = ["data_room_id = %s"]
        params: list = [data_room_id]

        if conversation_id:
            conditions.append("conversation_id = %s")
            params.append(conversation_id)

        if filter_user_id:
            conditions.append("user_id = %s")
            params.append(filter_user_id)

        where_clause = " AND ".join(conditions)
        params.append(limit)

        # Join with users table to get name/picture for team view
        cursor.execute(f"""
            SELECT q.*, u.name as user_name, u.picture_url as user_picture_url
            FROM queries q
            LEFT JOIN users u ON q.user_id = u.id
            WHERE {where_clause}
            ORDER BY q.created_at DESC
            LIMIT %s
        """, params)

        queries = []
        for row in cursor.fetchall():
            query = dict_from_row(row)
            # Parse JSON sources back to list
            if query and query.get('sources'):
                query['sources'] = json.loads(query['sources'])
            queries.append(query)

        return queries


def delete_question(question_id: str, user_id: str) -> bool:
    """Delete a question. Only the owner can delete their own question."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM queries WHERE id = %s AND user_id = %s",
            (question_id, user_id)
        )
        conn.commit()
        return cursor.rowcount > 0


# ============================================================================
# Processing Log Operations
# ============================================================================

def log_processing_stage(
    stage: str,
    status: str,
    data_room_id: Optional[str] = None,
    document_id: Optional[str] = None,
    message: Optional[str] = None,
    error_details: Optional[str] = None,
    duration_ms: Optional[int] = None
) -> str:
    """
    Log a processing stage.

    Args:
        stage: Processing stage name
        status: Status (started, completed, failed)
        data_room_id: Data room ID
        document_id: Document ID
        message: Log message
        error_details: Error details if failed
        duration_ms: Duration in milliseconds

    Returns:
        Log ID
    """
    log_id = f"log_{uuid.uuid4().hex[:12]}"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO processing_logs (
                id, data_room_id, document_id, stage, status,
                message, error_details, duration_ms
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            log_id, data_room_id, document_id, stage, status,
            message, error_details, duration_ms
        ))
        conn.commit()

    return log_id


def get_processing_logs(
    data_room_id: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get processing logs.

    Args:
        data_room_id: Filter by data room ID
        limit: Maximum number of logs to return

    Returns:
        List of log records
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        if data_room_id:
            cursor.execute("""
                SELECT * FROM processing_logs
                WHERE data_room_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (data_room_id, limit))
        else:
            cursor.execute("""
                SELECT * FROM processing_logs
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))

        return [dict_from_row(row) for row in cursor.fetchall()]


# ============================================================================
# API Usage Tracking
# ============================================================================

def track_api_usage(
    provider: str,
    model: str,
    operation: str,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    data_room_id: Optional[str] = None
) -> str:
    """
    Track API usage for cost monitoring.

    Args:
        provider: API provider (openai, anthropic)
        model: Model name
        operation: Operation type (embedding, completion, etc.)
        input_tokens: Input tokens used
        output_tokens: Output tokens used
        cost: Cost in USD
        data_room_id: Associated data room ID

    Returns:
        Usage record ID
    """
    usage_id = f"usage_{uuid.uuid4().hex[:12]}"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO api_usage (
                id, data_room_id, provider, model, operation,
                input_tokens, output_tokens, cost
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            usage_id, data_room_id, provider, model, operation,
            input_tokens, output_tokens, cost
        ))
        conn.commit()

    # Update data room cost if applicable
    if data_room_id and cost:
        update_data_room_cost(data_room_id, cost)

    return usage_id


def get_api_costs(
    days: int = 30,
    data_room_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get API cost summary.

    Args:
        days: Number of days to include
        data_room_id: Filter by data room ID

    Returns:
        Cost summary with breakdowns
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Total cost
        if data_room_id:
            cursor.execute("""
                SELECT
                    SUM(cost) as total_cost,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    COUNT(*) as total_calls
                FROM api_usage
                WHERE data_room_id = %s
                AND timestamp >= CURRENT_TIMESTAMP - (%s * interval '1 day')
            """, (data_room_id, days))
        else:
            cursor.execute("""
                SELECT
                    SUM(cost) as total_cost,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens,
                    COUNT(*) as total_calls
                FROM api_usage
                WHERE timestamp >= CURRENT_TIMESTAMP - (%s * interval '1 day')
            """, (days,))

        summary = dict_from_row(cursor.fetchone())

        # Breakdown by provider
        if data_room_id:
            cursor.execute("""
                SELECT provider, SUM(cost) as cost, COUNT(*) as calls
                FROM api_usage
                WHERE data_room_id = %s
                AND timestamp >= CURRENT_TIMESTAMP - (%s * interval '1 day')
                GROUP BY provider
            """, (data_room_id, days))
        else:
            cursor.execute("""
                SELECT provider, SUM(cost) as cost, COUNT(*) as calls
                FROM api_usage
                WHERE timestamp >= CURRENT_TIMESTAMP - (%s * interval '1 day')
                GROUP BY provider
            """, (days,))

        summary['by_provider'] = [dict_from_row(row) for row in cursor.fetchall()]

        return summary


# ============================================================================
# Analysis Cache Operations
# ============================================================================

def save_analysis_cache(
    data_room_id: str,
    analysis_type: str,
    extracted_data: Dict[str, Any],
    version: int = 1
) -> str:
    """
    Save extracted/analyzed data to cache.

    Args:
        data_room_id: Data room ID
        analysis_type: Type of analysis (financials, team, market, etc.)
        extracted_data: Extracted data as dictionary
        version: Version number

    Returns:
        Cache ID
    """
    cache_id = f"cache_{uuid.uuid4().hex[:12]}"

    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Delete old version if exists
        cursor.execute("""
            DELETE FROM analysis_cache
            WHERE data_room_id = %s AND analysis_type = %s AND version = %s
        """, (data_room_id, analysis_type, version))

        # Insert new cache
        cursor.execute("""
            INSERT INTO analysis_cache (
                id, data_room_id, analysis_type, extracted_data, version
            ) VALUES (%s, %s, %s, %s, %s)
        """, (
            cache_id, data_room_id, analysis_type,
            json.dumps(extracted_data), version
        ))
        conn.commit()

    return cache_id


def get_analysis_cache(
    data_room_id: str,
    analysis_type: str,
    version: int = 1
) -> Optional[Dict[str, Any]]:
    """
    Get cached analysis data.

    Args:
        data_room_id: Data room ID
        analysis_type: Type of analysis
        version: Version number

    Returns:
        Extracted data or None if not cached
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT extracted_data FROM analysis_cache
            WHERE data_room_id = %s AND analysis_type = %s AND version = %s
        """, (data_room_id, analysis_type, version))

        row = cursor.fetchone()
        if row:
            return json.loads(row['extracted_data'])
        return None


# ============================================================================
# User Operations (Google OAuth)
# ============================================================================

def create_or_update_user(
    email: str,
    name: Optional[str] = None,
    picture_url: Optional[str] = None,
    google_id: Optional[str] = None,
    access_token: Optional[str] = None,
    refresh_token: Optional[str] = None,
    token_expires_at: Optional[str] = None
) -> str:
    """
    Create a new user or update existing user by email.

    Args:
        email: User email (unique identifier)
        name: User display name
        picture_url: Profile picture URL
        google_id: Google user ID
        access_token: OAuth access token
        refresh_token: OAuth refresh token
        token_expires_at: Token expiry timestamp

    Returns:
        User ID
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        existing = cursor.fetchone()

        if existing:
            # Update existing user
            user_id = existing['id']
            fields = ["last_login_at = CURRENT_TIMESTAMP"]
            values = []

            if name:
                fields.append("name = %s")
                values.append(name)
            if picture_url:
                fields.append("picture_url = %s")
                values.append(picture_url)
            if google_id:
                fields.append("google_id = %s")
                values.append(google_id)
            if access_token:
                fields.append("access_token = %s")
                values.append(access_token)
            if refresh_token:
                fields.append("refresh_token = %s")
                values.append(refresh_token)
            if token_expires_at:
                fields.append("token_expires_at = %s")
                values.append(token_expires_at)

            values.append(user_id)
            query = f"UPDATE users SET {', '.join(fields)} WHERE id = %s"
            cursor.execute(query, values)
            logger.info(f"Updated user: {email}")
        else:
            # Create new user
            user_id = f"user_{uuid.uuid4().hex[:12]}"
            cursor.execute("""
                INSERT INTO users (
                    id, email, name, picture_url, google_id,
                    access_token, refresh_token, token_expires_at, last_login_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
            """, (user_id, email, name, picture_url, google_id,
                  access_token, refresh_token, token_expires_at))
            logger.info(f"Created new user: {email}")

        conn.commit()
        return user_id


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        return dict_from_row(cursor.fetchone())


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        return dict_from_row(cursor.fetchone())


def update_user_tokens(
    user_id: str,
    access_token: str,
    refresh_token: Optional[str] = None,
    token_expires_at: Optional[str] = None
) -> bool:
    """Update user OAuth tokens."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        fields = ["access_token = %s"]
        values = [access_token]

        if refresh_token:
            fields.append("refresh_token = %s")
            values.append(refresh_token)
        if token_expires_at:
            fields.append("token_expires_at = %s")
            values.append(token_expires_at)

        values.append(user_id)
        query = f"UPDATE users SET {', '.join(fields)} WHERE id = %s"
        cursor.execute(query, values)
        conn.commit()

        return cursor.rowcount > 0


def delete_user(user_id: str) -> bool:
    """Delete a user and revoke access."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        return cursor.rowcount > 0


# ============================================================================
# Connected Folders Operations (Google Drive Sync)
# ============================================================================

def create_connected_folder(
    user_id: str,
    folder_id: str,
    folder_name: str,
    folder_path: Optional[str] = None,
    data_room_id: Optional[str] = None
) -> str:
    """
    Create a new connected folder for syncing.

    Args:
        user_id: User ID
        folder_id: Google Drive folder ID
        folder_name: Folder display name
        folder_path: Full folder path
        data_room_id: Associated data room ID

    Returns:
        Connected folder ID
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Check if folder is already connected for this user
        cursor.execute(
            "SELECT id FROM connected_folders WHERE user_id = %s AND folder_id = %s",
            (user_id, folder_id)
        )
        existing = cursor.fetchone()
        if existing:
            # Update data_room_id if a new one is provided (reconnection scenario)
            if data_room_id:
                cursor.execute(
                    "UPDATE connected_folders SET data_room_id = %s, sync_status = 'syncing' WHERE id = %s",
                    (data_room_id, existing['id'])
                )
                conn.commit()
                logger.info(f"Updated existing folder connection with new data room: {existing['id']} -> {data_room_id}")
            else:
                logger.info(f"Folder already connected: {folder_name} ({folder_id}) -> {existing['id']}")
            return existing['id']

        # Create new connection
        connection_id = f"cf_{uuid.uuid4().hex[:12]}"
        cursor.execute("""
            INSERT INTO connected_folders (
                id, user_id, folder_id, folder_name, folder_path, data_room_id, sync_status, sync_stage
            ) VALUES (%s, %s, %s, %s, %s, %s, 'syncing', 'idle')
        """, (connection_id, user_id, folder_id, folder_name, folder_path, data_room_id))
        conn.commit()

    logger.info(f"Created connected folder: {folder_name} ({folder_id})")
    return connection_id


def get_connected_folder(connection_id: str) -> Optional[Dict[str, Any]]:
    """Get connected folder by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM connected_folders WHERE id = %s", (connection_id,))
        return dict_from_row(cursor.fetchone())


def get_connected_folders_by_user(user_id: str) -> List[Dict[str, Any]]:
    """Get all connected folders for a user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM connected_folders
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id,))
        return [dict_from_row(row) for row in cursor.fetchall()]


def get_active_connected_folders() -> List[Dict[str, Any]]:
    """Get all active connected folders for sync processing."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT cf.*, u.access_token, u.refresh_token, u.token_expires_at
            FROM connected_folders cf
            JOIN users u ON cf.user_id = u.id
            WHERE cf.sync_status IN ('active', 'syncing')
            AND u.is_active = TRUE
            ORDER BY cf.last_sync_at ASC NULLS FIRST
        """)
        return [dict_from_row(row) for row in cursor.fetchall()]


def update_connected_folder_status(
    connection_id: str,
    sync_status: str,
    sync_page_token: Optional[str] = None,
    total_files: Optional[int] = None,
    processed_files: Optional[int] = None,
    error_message: Optional[str] = None
) -> bool:
    """Update connected folder sync status."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        fields = ["sync_status = %s"]
        values = [sync_status]

        if sync_status == 'syncing' or sync_status == 'active':
            fields.append("last_sync_at = CURRENT_TIMESTAMP")

        if sync_page_token is not None:
            fields.append("sync_page_token = %s")
            values.append(sync_page_token)
        if total_files is not None:
            fields.append("total_files = %s")
            values.append(total_files)
        if processed_files is not None:
            fields.append("processed_files = %s")
            values.append(processed_files)
        if error_message is not None:
            fields.append("error_message = %s")
            values.append(error_message)

        values.append(connection_id)
        query = f"UPDATE connected_folders SET {', '.join(fields)} WHERE id = %s"
        cursor.execute(query, values)
        conn.commit()

        return cursor.rowcount > 0


def delete_connected_folder(connection_id: str) -> bool:
    """Delete a connected folder."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM connected_folders WHERE id = %s", (connection_id,))
        conn.commit()
        logger.info(f"Deleted connected folder: {connection_id}")
        return cursor.rowcount > 0


def update_connected_folder_stage(
    connection_id: str,
    sync_stage: str,
    discovered_files: Optional[int] = None,
    discovered_folders: Optional[int] = None,
    current_folder_path: Optional[str] = None,
    error_message: Optional[str] = None
) -> bool:
    """
    Update connected folder sync stage and discovery progress.

    Args:
        connection_id: Connected folder ID
        sync_stage: New stage (idle, discovering, discovered, processing, complete, error)
        discovered_files: Number of files discovered
        discovered_folders: Number of folders discovered
        current_folder_path: Current folder being scanned
        error_message: Error message if stage is 'error'

    Returns:
        True if updated successfully
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        fields = ["sync_stage = %s"]
        values = [sync_stage]

        if discovered_files is not None:
            fields.append("discovered_files = %s")
            values.append(discovered_files)
        if discovered_folders is not None:
            fields.append("discovered_folders = %s")
            values.append(discovered_folders)
        if current_folder_path is not None:
            fields.append("current_folder_path = %s")
            values.append(current_folder_path)
        if error_message is not None:
            fields.append("error_message = %s")
            values.append(error_message)

        values.append(connection_id)
        query = f"UPDATE connected_folders SET {', '.join(fields)} WHERE id = %s"
        cursor.execute(query, values)
        conn.commit()

        return cursor.rowcount > 0


# ============================================================================
# Connected Files Operations (Individual File Connections)
# ============================================================================

def create_connected_file(
    user_id: str,
    drive_file_id: str,
    file_name: str,
    file_path: Optional[str] = None,
    mime_type: Optional[str] = None,
    file_size: Optional[int] = None,
    data_room_id: Optional[str] = None
) -> str:
    """
    Create a connected file record for an individual Google Drive file.

    Args:
        user_id: User ID
        drive_file_id: Google Drive file ID
        file_name: File display name
        file_path: Full file path in Drive
        mime_type: MIME type of the file
        file_size: File size in bytes
        data_room_id: Associated data room ID

    Returns:
        Connected file ID
    """
    file_id = f"cf_{uuid.uuid4().hex[:12]}"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO connected_files (
                id, user_id, drive_file_id, file_name, file_path,
                mime_type, file_size, data_room_id, sync_status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'pending')
        """, (file_id, user_id, drive_file_id, file_name, file_path,
              mime_type, file_size, data_room_id))
        conn.commit()

    logger.info(f"Created connected file: {file_name} (ID: {file_id})")
    return file_id


def get_connected_file(file_id: str) -> Optional[Dict[str, Any]]:
    """Get a connected file by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM connected_files WHERE id = %s", (file_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
    return None


def get_connected_files_by_user(user_id: str) -> List[Dict[str, Any]]:
    """Get all connected files for a user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM connected_files
            WHERE user_id = %s
            ORDER BY created_at DESC
        """, (user_id,))
        return [dict(row) for row in cursor.fetchall()]


def get_connected_files_by_data_room(data_room_id: str) -> List[Dict[str, Any]]:
    """Get all connected files for a data room."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM connected_files
            WHERE data_room_id = %s
            ORDER BY created_at DESC
        """, (data_room_id,))
        return [dict(row) for row in cursor.fetchall()]


def update_connected_file_status(
    file_id: str,
    sync_status: str,
    local_file_path: Optional[str] = None,
    document_id: Optional[str] = None,
    error_message: Optional[str] = None
) -> bool:
    """
    Update connected file status.

    Args:
        file_id: Connected file ID
        sync_status: New status (pending, downloading, processing, complete, failed)
        local_file_path: Local file path after download
        document_id: Associated document ID after processing
        error_message: Error message if failed

    Returns:
        True if updated successfully
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        fields = ["sync_status = %s"]
        values = [sync_status]

        if local_file_path is not None:
            fields.append("local_file_path = %s")
            values.append(local_file_path)
        if document_id is not None:
            fields.append("document_id = %s")
            values.append(document_id)
        if error_message is not None:
            fields.append("error_message = %s")
            values.append(error_message)

        values.append(file_id)
        query = f"UPDATE connected_files SET {', '.join(fields)} WHERE id = %s"
        cursor.execute(query, values)
        conn.commit()

        return cursor.rowcount > 0


def delete_connected_file(file_id: str) -> bool:
    """Delete a connected file."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM connected_files WHERE id = %s", (file_id,))
        conn.commit()
        logger.info(f"Deleted connected file: {file_id}")
        return cursor.rowcount > 0


def check_connected_file_exists(user_id: str, drive_file_id: str) -> Optional[str]:
    """
    Check if a connected file already exists for this user and drive file.

    Returns:
        Connected file ID if exists, None otherwise
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM connected_files
            WHERE user_id = %s AND drive_file_id = %s
        """, (user_id, drive_file_id))
        row = cursor.fetchone()
        if row:
            return row['id']
    return None


def get_connected_file_with_data_room(user_id: str, drive_file_id: str) -> Optional[Dict[str, Any]]:
    """
    Get connected file along with its associated data room info for deduplication.

    Returns:
        Dict with connected file info and data room status, or None if not found
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT cf.*, dr.processing_status as dr_status, dr.progress_percent as dr_progress
            FROM connected_files cf
            LEFT JOIN data_rooms dr ON cf.data_room_id = dr.id
            WHERE cf.user_id = %s AND cf.drive_file_id = %s
        """, (user_id, drive_file_id))
        row = cursor.fetchone()
        return dict(row) if row else None


def cleanup_orphaned_data_rooms(older_than_minutes: int = 60) -> int:
    """
    Delete data rooms that have no associated documents or connected files
    and are stuck at uploading 0%.

    Args:
        older_than_minutes: Only delete data rooms older than this many minutes

    Returns:
        Number of deleted data rooms
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM data_rooms
            WHERE id NOT IN (SELECT DISTINCT data_room_id FROM documents WHERE data_room_id IS NOT NULL)
            AND id NOT IN (SELECT DISTINCT data_room_id FROM connected_files WHERE data_room_id IS NOT NULL)
            AND id NOT IN (SELECT DISTINCT data_room_id FROM connected_folders WHERE data_room_id IS NOT NULL)
            AND processing_status = 'uploading'
            AND progress_percent = 0
            AND created_at < CURRENT_TIMESTAMP - (%s * interval '1 minute')
        """, (older_than_minutes,))
        conn.commit()
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} orphaned data rooms")
        return deleted


# ============================================================================
# Folder Inventory Operations (Discovery Stage)
# ============================================================================

def create_folder_inventory(
    connected_folder_id: str,
    drive_folder_id: str,
    folder_name: str,
    folder_path: Optional[str] = None,
    parent_folder_id: Optional[str] = None
) -> str:
    """
    Create a folder inventory record during discovery.

    Args:
        connected_folder_id: Parent connected folder ID
        drive_folder_id: Google Drive folder ID
        folder_name: Folder display name
        folder_path: Full folder path from root
        parent_folder_id: Parent folder inventory ID

    Returns:
        Folder inventory ID
    """
    inventory_id = f"fi_{uuid.uuid4().hex[:12]}"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO folder_inventory (
                id, connected_folder_id, drive_folder_id, folder_name,
                folder_path, parent_folder_id, scan_status
            ) VALUES (%s, %s, %s, %s, %s, %s, 'scanning')
            ON CONFLICT (connected_folder_id, drive_folder_id) DO UPDATE SET
                folder_name = EXCLUDED.folder_name,
                folder_path = EXCLUDED.folder_path,
                parent_folder_id = EXCLUDED.parent_folder_id,
                scan_status = EXCLUDED.scan_status
        """, (inventory_id, connected_folder_id, drive_folder_id, folder_name,
              folder_path, parent_folder_id))
        conn.commit()

    return inventory_id


def update_folder_inventory_counts(
    inventory_id: str,
    file_count: int,
    total_size_bytes: int = 0,
    scan_status: str = 'scanned',
    error_message: Optional[str] = None
) -> bool:
    """
    Update folder inventory after scanning.

    Args:
        inventory_id: Folder inventory ID
        file_count: Number of supported files in folder
        total_size_bytes: Total size of files
        scan_status: New status (scanned, error)
        error_message: Error message if status is 'error'

    Returns:
        True if updated successfully
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        fields = ["file_count = %s", "total_size_bytes = %s", "scan_status = %s"]
        values = [file_count, total_size_bytes, scan_status]

        if error_message is not None:
            fields.append("error_message = %s")
            values.append(error_message)

        values.append(inventory_id)
        query = f"UPDATE folder_inventory SET {', '.join(fields)} WHERE id = %s"
        cursor.execute(query, values)
        conn.commit()

        return cursor.rowcount > 0


def create_and_update_folder_inventory_batch(
    records: List[Dict[str, Any]]
) -> List[str]:
    """
    Create folder inventory records and set their counts in a single transaction.

    Args:
        records: List of dicts with keys: connected_folder_id, drive_folder_id,
                 folder_name, folder_path, parent_folder_id, file_count, total_size_bytes

    Returns:
        List of generated inventory IDs
    """
    if not records:
        return []

    ids = [f"fi_{uuid.uuid4().hex[:12]}" for _ in records]

    insert_values = [
        (
            inv_id,
            r['connected_folder_id'],
            r['drive_folder_id'],
            r['folder_name'],
            r.get('folder_path'),
            r.get('parent_folder_id'),
            'scanned',
            r.get('file_count', 0),
            r.get('total_size_bytes', 0),
        )
        for inv_id, r in zip(ids, records)
    ]

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany("""
            INSERT INTO folder_inventory (
                id, connected_folder_id, drive_folder_id, folder_name,
                folder_path, parent_folder_id, scan_status,
                file_count, total_size_bytes
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (connected_folder_id, drive_folder_id) DO UPDATE SET
                folder_name = EXCLUDED.folder_name,
                folder_path = EXCLUDED.folder_path,
                parent_folder_id = EXCLUDED.parent_folder_id,
                scan_status = EXCLUDED.scan_status,
                file_count = EXCLUDED.file_count,
                total_size_bytes = EXCLUDED.total_size_bytes
        """, insert_values)
        conn.commit()

    return ids


def reset_failed_synced_files(connection_id: Optional[str] = None) -> int:
    """
    Reset failed and stuck synced files to pending status for retry.

    Resets files with status: 'failed', 'downloading', 'processing'
    (handles files stuck from interrupted syncs)

    Args:
        connection_id: Optional - only reset files for this folder

    Returns:
        Number of files reset
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if connection_id:
            cursor.execute("""
                UPDATE synced_files
                SET sync_status = 'pending', error_message = NULL
                WHERE sync_status IN ('failed', 'downloading', 'processing')
                AND connected_folder_id = %s
            """, (connection_id,))
        else:
            cursor.execute("""
                UPDATE synced_files
                SET sync_status = 'pending', error_message = NULL
                WHERE sync_status IN ('failed', 'downloading', 'processing')
            """)
        conn.commit()
        return cursor.rowcount


def get_folder_inventory(connected_folder_id: str) -> List[Dict[str, Any]]:
    """
    Get all folder inventory for a connected folder.

    Args:
        connected_folder_id: Connected folder ID

    Returns:
        List of folder inventory records
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM folder_inventory
            WHERE connected_folder_id = %s
            ORDER BY folder_path ASC
        """, (connected_folder_id,))
        return [dict_from_row(row) for row in cursor.fetchall()]


def count_pending_synced_files(connected_folder_id: str) -> int:
    """Count pending synced files for a connected folder (for sync recovery)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM synced_files WHERE connected_folder_id = %s AND sync_status = 'pending'",
            (connected_folder_id,)
        )
        return cursor.fetchone()['cnt']


def get_unscanned_inventory_folders(connected_folder_id: str) -> List[Dict[str, Any]]:
    """Get folder inventory entries that haven't been scanned yet (for crash recovery)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM folder_inventory
            WHERE connected_folder_id = %s AND scan_status IN ('pending', 'scanning')
            ORDER BY folder_path ASC
        """, (connected_folder_id,))
        return [dict_from_row(row) for row in cursor.fetchall()]


def update_folder_inventory_scan_status(inventory_id: str, scan_status: str) -> bool:
    """Update the scan status of a folder inventory entry."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE folder_inventory SET scan_status = %s WHERE id = %s",
            (scan_status, inventory_id)
        )
        conn.commit()
        return cursor.rowcount > 0


def delete_folder_inventory(connected_folder_id: str) -> bool:
    """
    Delete all folder inventory for a connected folder (for re-discovery).

    Args:
        connected_folder_id: Connected folder ID

    Returns:
        True if deleted successfully
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM folder_inventory WHERE connected_folder_id = %s",
            (connected_folder_id,)
        )
        conn.commit()
        return cursor.rowcount > 0


def get_all_pending_synced_files(connected_folder_id: str) -> List[Dict[str, Any]]:
    """
    Get all pending files that need to be processed (no limit).

    Args:
        connected_folder_id: Connected folder ID

    Returns:
        List of pending synced file records
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM synced_files
            WHERE connected_folder_id = %s
            AND sync_status = 'pending'
            ORDER BY created_at ASC
        """, (connected_folder_id,))
        return [dict_from_row(row) for row in cursor.fetchall()]


# ============================================================================
# Synced Files Operations
# ============================================================================

def create_synced_file(
    connected_folder_id: str,
    drive_file_id: str,
    file_name: str,
    file_path: Optional[str] = None,
    mime_type: Optional[str] = None,
    file_size: Optional[int] = None,
    drive_modified_time: Optional[str] = None
) -> str:
    """Create a record for a synced file."""
    synced_file_id = f"sf_{uuid.uuid4().hex[:12]}"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO synced_files (
                id, connected_folder_id, drive_file_id, file_name,
                file_path, mime_type, file_size, drive_modified_time
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (connected_folder_id, drive_file_id) DO UPDATE SET
                file_name = EXCLUDED.file_name,
                file_path = EXCLUDED.file_path,
                mime_type = EXCLUDED.mime_type,
                file_size = EXCLUDED.file_size,
                drive_modified_time = EXCLUDED.drive_modified_time
        """, (synced_file_id, connected_folder_id, drive_file_id, file_name,
              file_path, mime_type, file_size, drive_modified_time))
        conn.commit()

    return synced_file_id


def create_synced_files_batch(files: List[Dict[str, Any]]) -> List[str]:
    """
    Create multiple synced file records in a single transaction.

    Uses execute_values with RETURNING to get the actual IDs from the database,
    not pre-generated ones. This is critical for ON CONFLICT (upsert) scenarios:
    when a row already exists, the DB keeps its original ID, so we must read it
    back instead of assuming our generated ID was inserted.

    Args:
        files: List of dicts with keys: connected_folder_id, drive_file_id,
               file_name, file_path, mime_type, file_size, drive_modified_time

    Returns:
        List of actual synced file IDs from the database
    """
    if not files:
        return []

    ids = [f"sf_{uuid.uuid4().hex[:12]}" for _ in files]

    values = [
        (
            sf_id,
            f['connected_folder_id'],
            f['drive_file_id'],
            f['file_name'],
            f.get('file_path'),
            f.get('mime_type'),
            f.get('file_size'),
            f.get('drive_modified_time'),
        )
        for sf_id, f in zip(ids, files)
    ]

    with get_db_connection() as conn:
        cursor = conn.cursor()
        result = psycopg2.extras.execute_values(
            cursor,
            """
            INSERT INTO synced_files (
                id, connected_folder_id, drive_file_id, file_name,
                file_path, mime_type, file_size, drive_modified_time
            ) VALUES %s
            ON CONFLICT (connected_folder_id, drive_file_id) DO UPDATE SET
                file_name = EXCLUDED.file_name,
                file_path = EXCLUDED.file_path,
                mime_type = EXCLUDED.mime_type,
                file_size = EXCLUDED.file_size,
                drive_modified_time = EXCLUDED.drive_modified_time
            RETURNING id
            """,
            values,
            fetch=True
        )
        conn.commit()

    return [row[0] for row in result]


def get_synced_files_by_folder(connected_folder_id: str) -> List[Dict[str, Any]]:
    """Get all synced files for a connected folder."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM synced_files
            WHERE connected_folder_id = %s
            ORDER BY file_name ASC
        """, (connected_folder_id,))
        return [dict_from_row(row) for row in cursor.fetchall()]


def get_pending_synced_files(connected_folder_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get pending files that need to be processed."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM synced_files
            WHERE connected_folder_id = %s
            AND sync_status IN ('pending', 'failed')
            ORDER BY created_at ASC
            LIMIT %s
        """, (connected_folder_id, limit))
        return [dict_from_row(row) for row in cursor.fetchall()]


def update_synced_file_status(
    synced_file_id: str,
    sync_status: str,
    local_file_path: Optional[str] = None,
    document_id: Optional[str] = None,
    error_message: Optional[str] = None
) -> bool:
    """Update synced file status."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        fields = ["sync_status = %s", "last_synced_at = CURRENT_TIMESTAMP"]
        values = [sync_status]

        if local_file_path:
            fields.append("local_file_path = %s")
            values.append(local_file_path)
        if document_id:
            fields.append("document_id = %s")
            values.append(document_id)
        if error_message is not None:
            fields.append("error_message = %s")
            values.append(error_message)

        values.append(synced_file_id)
        query = f"UPDATE synced_files SET {', '.join(fields)} WHERE id = %s"
        cursor.execute(query, values)
        conn.commit()

        return cursor.rowcount > 0


def count_synced_files_by_status(connected_folder_id: str, sync_status: str = None) -> int:
    """Count synced files for a folder, optionally filtered by status."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if sync_status:
            cursor.execute(
                "SELECT COUNT(*) as cnt FROM synced_files WHERE connected_folder_id = %s AND sync_status = %s",
                (connected_folder_id, sync_status)
            )
        else:
            cursor.execute(
                "SELECT COUNT(*) as cnt FROM synced_files WHERE connected_folder_id = %s",
                (connected_folder_id,)
            )
        return cursor.fetchone()['cnt']


def count_processed_synced_files(connected_folder_id: str) -> int:
    """Count processed synced files (complete + failed) for a folder."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM synced_files WHERE connected_folder_id = %s AND sync_status IN ('complete', 'failed')",
            (connected_folder_id,)
        )
        return cursor.fetchone()['cnt']


def count_all_synced_files(connected_folder_id: str) -> int:
    """Count ALL synced files for a folder (regardless of status).
    Used as the authoritative total for completion checks instead of the
    in-memory discovery counter which can drift on conflict/resume."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM synced_files WHERE connected_folder_id = %s",
            (connected_folder_id,)
        )
        return cursor.fetchone()['cnt']


def check_file_exists_in_sync(connected_folder_id: str, drive_file_id: str) -> Optional[Dict[str, Any]]:
    """Check if a file is already synced."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM synced_files
            WHERE connected_folder_id = %s AND drive_file_id = %s
        """, (connected_folder_id, drive_file_id))
        return dict_from_row(cursor.fetchone())


def get_recently_completed_synced_files(connected_folder_id: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Get the most recently completed synced files for progress UI."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT file_name, mime_type
            FROM synced_files
            WHERE connected_folder_id = %s AND sync_status = 'complete'
            ORDER BY last_synced_at DESC
            LIMIT %s
        """, (connected_folder_id, limit))
        return [dict_from_row(row) for row in cursor.fetchall()]


def get_synced_file_type_counts(connected_folder_id: str) -> Dict[str, int]:
    """Get file type counts grouped by category for progress UI."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                CASE
                    WHEN mime_type LIKE '%%pdf%%' THEN 'pdf'
                    WHEN mime_type LIKE '%%spreadsheet%%' OR mime_type LIKE '%%excel%%'
                         OR mime_type LIKE '%%csv%%' THEN 'spreadsheet'
                    WHEN mime_type LIKE '%%document%%' OR mime_type LIKE '%%msword%%'
                         OR mime_type LIKE '%%wordprocessing%%' THEN 'document'
                    WHEN mime_type LIKE '%%presentation%%' OR mime_type LIKE '%%powerpoint%%' THEN 'presentation'
                    ELSE 'other'
                END AS file_category,
                COUNT(*) AS count
            FROM synced_files
            WHERE connected_folder_id = %s
            GROUP BY file_category
        """, (connected_folder_id,))
        return {row['file_category']: row['count'] for row in cursor.fetchall()}


def get_current_processing_file(connected_folder_id: str) -> Optional[Dict[str, Any]]:
    """Get the file currently being processed for progress UI."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT file_name, sync_status
            FROM synced_files
            WHERE connected_folder_id = %s AND sync_status IN ('downloading', 'processing')
            ORDER BY last_synced_at DESC
            LIMIT 1
        """, (connected_folder_id,))
        row = cursor.fetchone()
        return dict_from_row(row) if row else None


# ============================================================================
# Financial Analysis Operations
# ============================================================================

def save_financial_analysis(
    analysis_id: str,
    data_room_id: str,
    document_id: str,
    file_name: str,
    status: str,
    model_structure: Optional[Dict[str, Any]] = None,
    extracted_metrics: Optional[List[Dict[str, Any]]] = None,
    time_series: Optional[List[Dict[str, Any]]] = None,
    missing_metrics: Optional[List[Dict[str, Any]]] = None,
    validation_results: Optional[Dict[str, Any]] = None,
    insights: Optional[List[Dict[str, Any]]] = None,
    follow_up_questions: Optional[List[Dict[str, Any]]] = None,
    key_metrics_summary: Optional[Dict[str, Any]] = None,
    risk_assessment: Optional[Dict[str, Any]] = None,
    investment_thesis_notes: Optional[Dict[str, Any]] = None,
    executive_summary: Optional[str] = None,
    analysis_cost: float = 0.0,
    tokens_used: int = 0,
    processing_time_ms: int = 0,
    error_message: Optional[str] = None
) -> str:
    """
    Save financial analysis results.

    Args:
        analysis_id: Unique analysis ID
        data_room_id: Data room ID
        document_id: Document ID
        file_name: Name of the analyzed file
        status: Analysis status (in_progress, complete, failed)
        model_structure: Model structure analysis
        extracted_metrics: List of extracted metrics
        time_series: Time series data
        missing_metrics: Metrics that were expected but not found
        validation_results: Validation check results
        insights: AI-generated insights
        follow_up_questions: Questions for founders
        key_metrics_summary: Summary of key metrics
        risk_assessment: Risk assessment
        investment_thesis_notes: Notes for investment thesis
        executive_summary: Executive summary text
        analysis_cost: Cost of analysis
        tokens_used: Total tokens used
        processing_time_ms: Processing time
        error_message: Error message if failed

    Returns:
        Analysis ID
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Check if analysis already exists
        cursor.execute("SELECT id FROM financial_analyses WHERE id = %s", (analysis_id,))
        existing = cursor.fetchone()

        if existing:
            # Update existing analysis
            cursor.execute("""
                UPDATE financial_analyses SET
                    status = %s,
                    model_structure = %s,
                    extracted_metrics = %s,
                    time_series = %s,
                    missing_metrics = %s,
                    validation_results = %s,
                    insights = %s,
                    follow_up_questions = %s,
                    key_metrics_summary = %s,
                    risk_assessment = %s,
                    investment_thesis_notes = %s,
                    executive_summary = %s,
                    analysis_cost = %s,
                    tokens_used = %s,
                    processing_time_ms = %s,
                    error_message = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (
                status,
                json.dumps(model_structure) if model_structure else None,
                json.dumps(extracted_metrics) if extracted_metrics else None,
                json.dumps(time_series) if time_series else None,
                json.dumps(missing_metrics) if missing_metrics else None,
                json.dumps(validation_results) if validation_results else None,
                json.dumps(insights) if insights else None,
                json.dumps(follow_up_questions) if follow_up_questions else None,
                json.dumps(key_metrics_summary) if key_metrics_summary else None,
                json.dumps(risk_assessment) if risk_assessment else None,
                json.dumps(investment_thesis_notes) if investment_thesis_notes else None,
                executive_summary,
                analysis_cost,
                tokens_used,
                processing_time_ms,
                error_message,
                analysis_id
            ))
        else:
            # Insert new analysis
            cursor.execute("""
                INSERT INTO financial_analyses (
                    id, data_room_id, document_id, file_name, status,
                    model_structure, extracted_metrics, time_series, missing_metrics,
                    validation_results, insights, follow_up_questions,
                    key_metrics_summary, risk_assessment, investment_thesis_notes,
                    executive_summary, analysis_cost, tokens_used, processing_time_ms,
                    error_message
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                analysis_id, data_room_id, document_id, file_name, status,
                json.dumps(model_structure) if model_structure else None,
                json.dumps(extracted_metrics) if extracted_metrics else None,
                json.dumps(time_series) if time_series else None,
                json.dumps(missing_metrics) if missing_metrics else None,
                json.dumps(validation_results) if validation_results else None,
                json.dumps(insights) if insights else None,
                json.dumps(follow_up_questions) if follow_up_questions else None,
                json.dumps(key_metrics_summary) if key_metrics_summary else None,
                json.dumps(risk_assessment) if risk_assessment else None,
                json.dumps(investment_thesis_notes) if investment_thesis_notes else None,
                executive_summary,
                analysis_cost, tokens_used, processing_time_ms, error_message
            ))

        conn.commit()

        # Save individual metrics for quick querying
        if extracted_metrics and status == 'complete':
            save_financial_metrics(analysis_id, data_room_id, extracted_metrics)

    # Update data room cost
    if analysis_cost > 0:
        update_data_room_cost(data_room_id, analysis_cost)

    logger.info(f"Saved financial analysis: {analysis_id} ({status})")
    return analysis_id


def save_financial_metrics(
    financial_analysis_id: str,
    data_room_id: str,
    metrics: List[Dict[str, Any]]
) -> int:
    """
    Save individual financial metrics for quick querying.

    Args:
        financial_analysis_id: Parent analysis ID
        data_room_id: Data room ID
        metrics: List of metric dictionaries

    Returns:
        Number of metrics saved
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Delete existing metrics for this analysis
        cursor.execute(
            "DELETE FROM financial_metrics WHERE financial_analysis_id = %s",
            (financial_analysis_id,)
        )

        # Insert new metrics
        for metric in metrics:
            metric_id = f"fm_{uuid.uuid4().hex[:12]}"
            cursor.execute("""
                INSERT INTO financial_metrics (
                    id, financial_analysis_id, data_room_id, metric_name,
                    category, metric_value, metric_unit, period,
                    cell_reference, confidence, source_sheet, notes
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                metric_id,
                financial_analysis_id,
                data_room_id,
                metric.get('name'),
                metric.get('category'),
                metric.get('value'),
                metric.get('unit'),
                metric.get('period'),
                metric.get('cell_reference'),
                metric.get('confidence', 'medium'),
                metric.get('source_sheet'),
                metric.get('notes')
            ))

        conn.commit()
        return len(metrics)


def get_financial_analysis(
    document_id: str,
    data_room_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Get financial analysis for a document.

    Args:
        document_id: Document ID
        data_room_id: Optional data room ID for validation

    Returns:
        Financial analysis record with parsed JSON fields
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        if data_room_id:
            cursor.execute("""
                SELECT * FROM financial_analyses
                WHERE document_id = %s AND data_room_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (document_id, data_room_id))
        else:
            cursor.execute("""
                SELECT * FROM financial_analyses
                WHERE document_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """, (document_id,))

        row = cursor.fetchone()
        if not row:
            return None

        result = dict_from_row(row)

        # Parse JSON fields
        json_fields = [
            'model_structure', 'extracted_metrics', 'time_series',
            'missing_metrics', 'validation_results', 'insights',
            'follow_up_questions', 'key_metrics_summary',
            'risk_assessment', 'investment_thesis_notes'
        ]

        for field in json_fields:
            if result.get(field):
                try:
                    result[field] = json.loads(result[field])
                except json.JSONDecodeError:
                    result[field] = None

        return result


def get_financial_analyses_by_data_room(data_room_id: str) -> List[Dict[str, Any]]:
    """
    Get all financial analyses for a data room.

    Args:
        data_room_id: Data room ID

    Returns:
        List of financial analysis records
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM financial_analyses
            WHERE data_room_id = %s
            ORDER BY created_at DESC
        """, (data_room_id,))

        results = []
        for row in cursor.fetchall():
            result = dict_from_row(row)

            # Parse JSON fields
            json_fields = [
                'model_structure', 'extracted_metrics', 'time_series',
                'missing_metrics', 'validation_results', 'insights',
                'follow_up_questions', 'key_metrics_summary',
                'risk_assessment', 'investment_thesis_notes'
            ]

            for field in json_fields:
                if result.get(field):
                    try:
                        result[field] = json.loads(result[field])
                    except json.JSONDecodeError:
                        result[field] = None

            results.append(result)

        return results


def get_financial_metrics_by_data_room(
    data_room_id: str,
    category: Optional[str] = None,
    metric_name: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get financial metrics for a data room.

    Args:
        data_room_id: Data room ID
        category: Filter by category (revenue, profitability, cash, saas, etc.)
        metric_name: Filter by metric name

    Returns:
        List of financial metric records
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()

        query = "SELECT * FROM financial_metrics WHERE data_room_id = %s"
        params = [data_room_id]

        if category:
            query += " AND category = %s"
            params.append(category)

        if metric_name:
            query += " AND metric_name LIKE %s"
            params.append(f"%{metric_name}%")

        query += " ORDER BY period DESC, metric_name ASC"

        cursor.execute(query, params)
        return [dict_from_row(row) for row in cursor.fetchall()]


def get_document_by_id(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Get document by ID.

    Args:
        document_id: Document ID

    Returns:
        Document record or None if not found
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = %s", (document_id,))
        return dict_from_row(cursor.fetchone())


def check_financial_analysis_exists(document_id: str) -> bool:
    """
    Check if a financial analysis already exists for a document.

    Args:
        document_id: Document ID

    Returns:
        True if analysis exists
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id FROM financial_analyses
            WHERE document_id = %s AND status = 'complete'
        """, (document_id,))
        return cursor.fetchone() is not None


# ============================================================================
# Memo Functions
# ============================================================================

def save_memo(
    memo_id: str,
    data_room_id: str,
    version: int = 1,
    ticket_size: Optional[float] = None,
    post_money_valuation: Optional[float] = None,
    valuation_methods: Optional[List[str]] = None
) -> str:
    """Create a new memo record with status 'generating'."""
    import json
    with get_db_connection() as conn:
        cursor = conn.cursor()
        valuation_methods_json = json.dumps(valuation_methods) if valuation_methods else None
        cursor.execute("""
            INSERT INTO memos (id, data_room_id, version, status, created_at, ticket_size, post_money_valuation, valuation_methods)
            VALUES (%s, %s, %s, 'generating', %s, %s, %s, %s)
        """, (memo_id, data_room_id, version, datetime.now().isoformat(), ticket_size, post_money_valuation, valuation_methods_json))
        conn.commit()
    return memo_id


def update_memo_section(
    memo_id: str,
    section_key: str,
    content: str,
    tokens_used: int = 0,
    cost: float = 0.0
):
    """Update a single section of a memo and accumulate tokens/cost."""
    valid_sections = [
        'proposed_investment_terms', 'executive_summary', 'market_analysis',
        'team_assessment', 'product_technology', 'financial_analysis',
        'valuation_analysis', 'risks_concerns', 'outcome_scenario_analysis',
        'investment_recommendation'
    ]
    if section_key not in valid_sections:
        raise ValueError(f"Invalid section key: {section_key}")

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"""
            UPDATE memos
            SET {section_key} = %s,
                tokens_used = COALESCE(tokens_used, 0) + %s,
                cost = COALESCE(cost, 0) + %s
            WHERE id = %s
        """, (content, tokens_used, cost, memo_id))
        conn.commit()


def update_memo_status(
    memo_id: str,
    status: str,
    full_memo: Optional[str] = None
):
    """Update memo status and optionally set full_memo."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if full_memo:
            cursor.execute("""
                UPDATE memos
                SET status = %s, full_memo = %s, completed_at = %s
                WHERE id = %s
            """, (status, full_memo, datetime.now().isoformat(), memo_id))
        else:
            cursor.execute("""
                UPDATE memos SET status = %s WHERE id = %s
            """, (status, memo_id))
        conn.commit()


def update_memo_deal_terms(
    memo_id: str,
    ticket_size: Optional[float] = None,
    post_money_valuation: Optional[float] = None
) -> None:
    """Update deal terms on an existing memo."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE memos
            SET ticket_size = %s, post_money_valuation = %s
            WHERE id = %s
        """, (ticket_size, post_money_valuation, memo_id))
        conn.commit()


def update_memo_metadata(memo_id: str, metadata: dict) -> None:
    """Save chart data / metadata JSON to memo record."""
    import json
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE memos SET metadata = %s WHERE id = %s",
            (json.dumps(metadata), memo_id)
        )
        conn.commit()


def get_latest_memo(data_room_id: str) -> Optional[Dict[str, Any]]:
    """Get the most recent memo for a data room."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM memos
            WHERE data_room_id = %s
            ORDER BY created_at DESC
            LIMIT 1
        """, (data_room_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_memo_by_id(memo_id: str) -> Optional[Dict[str, Any]]:
    """Get a memo by its ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM memos WHERE id = %s", (memo_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def check_memo_cancelled(memo_id: str) -> bool:
    """Check if a memo has been cancelled. Used by the generator loop."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM memos WHERE id = %s", (memo_id,))
        row = cursor.fetchone()
        return row is not None and row['status'] == "cancelled"


# ============================================================================
# Memo Chat Message Functions
# ============================================================================

def save_memo_chat_message(
    memo_id: str,
    data_room_id: str,
    role: str,
    content: str,
    updated_section_key: Optional[str] = None,
    updated_section_content: Optional[str] = None,
    tokens_used: int = 0,
    cost: float = 0.0
) -> str:
    """
    Save a chat message for a memo.

    Args:
        memo_id: Memo ID
        data_room_id: Data room ID
        role: 'user' or 'assistant'
        content: Message content
        updated_section_key: If assistant updated a section, the key
        updated_section_content: If assistant updated a section, the content
        tokens_used: Tokens used for assistant response
        cost: API cost for assistant response

    Returns:
        Message ID
    """
    message_id = f"mcm_{uuid.uuid4().hex[:12]}"

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO memo_chat_messages (
                id, memo_id, data_room_id, role, content,
                updated_section_key, updated_section_content,
                tokens_used, cost
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            message_id, memo_id, data_room_id, role, content,
            updated_section_key, updated_section_content,
            tokens_used, cost
        ))
        conn.commit()

    return message_id


def get_memo_chat_history(
    memo_id: str,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """
    Get chat history for a memo.

    Args:
        memo_id: Memo ID
        limit: Maximum number of messages to return

    Returns:
        List of chat message records ordered by created_at ASC (oldest first)
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, memo_id, data_room_id, role, content,
                   updated_section_key, updated_section_content,
                   tokens_used, cost, created_at
            FROM memo_chat_messages
            WHERE memo_id = %s
            ORDER BY created_at ASC
            LIMIT %s
        """, (memo_id, limit))

        messages = []
        for row in cursor.fetchall():
            messages.append(dict(row))

        return messages
