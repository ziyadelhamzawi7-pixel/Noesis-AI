"""
FastAPI application for VC Due Diligence tool.
Provides API endpoints for data room processing, Q&A, and memo generation.
"""

import sys
import os
import time
import json
import asyncio
import hashlib
import secrets
from pathlib import Path
from typing import List, Optional, Dict
import uuid
from datetime import datetime
from threading import Semaphore, Lock
import psutil

# Add parent directory to path to import tools
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Query, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, FileResponse, StreamingResponse, Response
from loguru import logger

from app.config import settings, validate_settings
from app.job_queue import job_queue, JobStatus
from app.models import (
    DataRoomCreate,
    DataRoomStatus,
    QuestionRequest,
    QuestionAnswer,
    HealthCheck,
    ErrorResponse,
    MemoGenerateRequest,
    GoogleAuthURL,
    GoogleAuthCallback,
    UserInfo,
    DriveFile,
    DriveFileList,
    ConnectFolderRequest,
    ConnectedFolder,
    ConnectFilesRequest,
    ConnectedFile,
    SyncedFile,
    DocumentPreview,
    FinancialAnalysisResult,
    FinancialAnalysisTriggerRequest,
    FinancialAnalysisTriggerResponse,
    FinancialSummary,
    MemoChatRequest,
    FolderNode,
    DocumentWithPath,
    DocumentTreeResponse,
    InviteMemberRequest,
    DataRoomMember,
)
from app import database as db
from app.google_oauth import google_oauth_service
from app.google_drive import GoogleDriveService
from app.sync_service import sync_service
from app.email_service import send_invite_email
from app.worker import worker_pool

# Initialize FastAPI app
app = FastAPI(
    title="VC Due Diligence Assistant",
    description="AI-powered tool for analyzing startup data rooms and generating investment memos",
    version="0.1.0"
)

# Add CORS middleware for frontend
_cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Access Password Protection ---
# When ACCESS_PASSWORD is set, all routes are gated behind a login page.
# A signed session token is stored in a cookie to maintain the session.
_access_password = settings.access_password
_auth_secret = os.getenv("AUTH_SECRET", secrets.token_hex(32))

def _make_session_token() -> str:
    """Create a signed session token for the access password."""
    return hashlib.sha256(f"{_auth_secret}:authenticated".encode()).hexdigest()

if _access_password:
    from starlette.middleware.base import BaseHTTPMiddleware

    _LOGIN_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Noesis AI — Login</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0a0f; color: #e0e0e0; display: flex; align-items: center; justify-content: center; min-height: 100vh; }
  .card { background: #14141f; border: 1px solid #2a2a3a; border-radius: 12px; padding: 40px; width: 100%; max-width: 400px; }
  h1 { font-size: 24px; margin-bottom: 8px; color: #fff; }
  p.sub { font-size: 14px; color: #888; margin-bottom: 24px; }
  label { display: block; font-size: 13px; color: #aaa; margin-bottom: 6px; }
  input { width: 100%; padding: 10px 14px; border-radius: 8px; border: 1px solid #2a2a3a; background: #0a0a0f; color: #fff; font-size: 15px; outline: none; }
  input:focus { border-color: #6c63ff; }
  button { width: 100%; padding: 11px; margin-top: 20px; border: none; border-radius: 8px; background: #6c63ff; color: #fff; font-size: 15px; font-weight: 600; cursor: pointer; }
  button:hover { background: #5a52d5; }
  .error { color: #ff6b6b; font-size: 13px; margin-top: 12px; display: none; }
</style>
</head>
<body>
<div class="card">
  <h1>Noesis AI</h1>
  <p class="sub">Enter the access password to continue.</p>
  <form method="POST" action="/login">
    <label for="password">Password</label>
    <input type="password" id="password" name="password" placeholder="Access password" autofocus required>
    <button type="submit">Sign in</button>
    <p class="error" id="err">Incorrect password. Please try again.</p>
  </form>
  <script>
    if (new URLSearchParams(window.location.search).get('error')) {
      document.getElementById('err').style.display = 'block';
    }
  </script>
</div>
</body>
</html>"""

    class AccessPasswordMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            path = request.url.path
            # Allow login page, health check, and static login assets through
            if path in ("/login", "/api/health"):
                return await call_next(request)
            # Check session cookie
            token = request.cookies.get("noesis_session")
            if token == _make_session_token():
                return await call_next(request)
            # Not authenticated — redirect browsers, reject API calls
            if path.startswith("/api/"):
                return JSONResponse(status_code=401, content={"detail": "Authentication required"})
            return RedirectResponse(url="/login", status_code=302)

    app.add_middleware(AccessPasswordMiddleware)

    @app.get("/login")
    async def login_page():
        """Serve the login page."""
        from fastapi.responses import HTMLResponse
        return HTMLResponse(content=_LOGIN_PAGE_HTML)

    @app.post("/login")
    async def login_submit(request: Request):
        """Validate the access password and set a session cookie."""
        form = await request.form()
        password = form.get("password", "")
        if password == _access_password:
            response = RedirectResponse(url="/", status_code=302)
            response.set_cookie(
                key="noesis_session",
                value=_make_session_token(),
                httponly=True,
                samesite="lax",
                max_age=60 * 60 * 24 * 7,  # 7 days
            )
            return response
        return RedirectResponse(url="/login?error=1", status_code=302)

    logger.info("Access password protection is ENABLED")

# Processing semaphore to limit concurrent background jobs
# Prevents memory exhaustion from too many simultaneous file processing tasks
_processing_semaphore = Semaphore(settings.max_concurrent_jobs)
_active_processing_count = 0  # Track for health endpoint

# Track data rooms actively processed by BackgroundTasks (not via job_queue).
# Used by stall guard to avoid killing legitimate in-progress processing.
_active_background_tasks: set = set()
_active_background_tasks_lock = Lock()

# Validate configuration on startup
@app.on_event("startup")
async def startup_event():
    """Validate configuration and initialize services."""
    logger.info("Starting VC Due Diligence API server...")

    if not validate_settings():
        logger.error("Configuration validation failed!")
        logger.error("Please ensure ANTHROPIC_API_KEY and OPENAI_API_KEY are set in .env file")
    else:
        logger.success("Configuration validated successfully")

    # Cancel stale running jobs from previous run (they caused OOM crashes)
    try:
        import sqlite3 as _sqlite3
        _conn = _sqlite3.connect(settings.database_path, timeout=30.0)
        stale_count = _conn.execute(
            "UPDATE job_queue SET status = 'cancelled' WHERE status = 'running'"
        ).rowcount
        _conn.commit()
        _conn.close()
        if stale_count > 0:
            logger.info(f"Cancelled {stale_count} stale running jobs from previous run")

        # Clean up orphaned connected_folders and jobs (referencing deleted data rooms)
        import sqlite3
        conn = sqlite3.connect(settings.database_path, timeout=30.0)
        try:
            cursor = conn.cursor()
            # Delete orphaned connected_folders first (this is the source of orphaned jobs)
            cursor.execute("""
                DELETE FROM connected_folders
                WHERE data_room_id IS NOT NULL
                AND data_room_id NOT IN (SELECT id FROM data_rooms WHERE id IS NOT NULL)
            """)
            orphaned_folders = cursor.rowcount
            if orphaned_folders > 0:
                logger.info(f"Cleaned up {orphaned_folders} orphaned connected_folders")

            # Delete orphaned jobs
            cursor.execute("""
                DELETE FROM job_queue
                WHERE data_room_id IS NOT NULL
                AND data_room_id NOT IN (SELECT id FROM data_rooms WHERE id IS NOT NULL)
            """)
            orphaned_jobs = cursor.rowcount
            if orphaned_jobs > 0:
                logger.info(f"Cleaned up {orphaned_jobs} orphaned jobs")

            # Reset documents stuck at 'parsing' with no active job
            cursor.execute("""
                UPDATE documents
                SET parse_status = 'pending'
                WHERE parse_status = 'parsing'
            """)
            orphaned_docs = cursor.rowcount
            if orphaned_docs > 0:
                logger.info(f"Reset {orphaned_docs} orphaned documents to pending")

            conn.commit()
        finally:
            conn.close()

        # Clean up old completed jobs
        cleaned = job_queue.cleanup_old_jobs(days=7)
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} old jobs")
    except Exception as e:
        logger.warning(f"Job queue maintenance failed: {e}")

    # Clean up orphaned data rooms (stuck at uploading 0% with no files)
    try:
        orphaned = db.cleanup_orphaned_data_rooms(older_than_minutes=60)
        if orphaned > 0:
            logger.info(f"Cleaned up {orphaned} orphaned data rooms")
    except Exception as e:
        logger.warning(f"Failed to cleanup orphaned data rooms: {e}")

    # Reset ALL orphaned 'running' jobs back to 'pending'.
    # No workers survived the server restart, so every 'running' job is orphaned.
    # This must happen BEFORE the worker pool starts so the jobs can be picked up.
    try:
        recovered = job_queue.recover_stale_jobs(timeout_minutes=0)
        if recovered > 0:
            logger.info(f"[startup] Reset {recovered} orphaned running jobs back to pending")
    except Exception as e:
        logger.warning(f"[startup] Failed to recover orphaned jobs: {e}")

    # Recover stalled data rooms — re-trigger processing instead of killing them
    try:
        import sqlite3 as _sqlite3_recovery
        import threading as _threading_recovery
        _rconn = _sqlite3_recovery.connect(settings.database_path, timeout=30.0)
        _rconn.row_factory = _sqlite3_recovery.Row
        try:
            _rcur = _rconn.cursor()
            # Find data rooms with pending documents but no active jobs.
            # (all running jobs were already reset to pending/failed above).
            # Exclude pending jobs that have exhausted their attempts — they
            # can never be claimed by claim_job and shouldn't block recovery.
            _rcur.execute("""
                SELECT dr.id
                FROM data_rooms dr
                WHERE dr.processing_status IN ('parsing', 'indexing', 'extracting')
                AND EXISTS (
                    SELECT 1 FROM documents d
                    WHERE d.data_room_id = dr.id
                    AND d.parse_status = 'pending'
                )
                AND NOT EXISTS (
                    SELECT 1 FROM job_queue jq
                    WHERE jq.data_room_id = dr.id
                    AND jq.status IN ('pending', 'running')
                    AND jq.attempts < jq.max_attempts
                )
            """)
            rooms_to_reprocess = [row['id'] for row in _rcur.fetchall()]
        finally:
            _rconn.close()

        if rooms_to_reprocess:
            logger.info(f"Found {len(rooms_to_reprocess)} data rooms needing re-processing after restart")

            async def _delayed_reprocess():
                await asyncio.sleep(60)  # Wait for server to fully stabilize
                for room_id in rooms_to_reprocess:
                    try:
                        documents = db.get_documents_by_data_room(room_id)
                        file_paths = []
                        for doc in documents:
                            if doc['parse_status'] == 'pending':
                                fp = os.path.join(settings.data_rooms_path, room_id, "raw", doc['file_name'])
                                if Path(fp).exists():
                                    file_paths.append(fp)

                        if file_paths:
                            logger.info(
                                f"[startup-recovery] Re-triggering processing for {room_id}: "
                                f"{len(file_paths)} pending files"
                            )
                            t = _threading_recovery.Thread(
                                target=process_data_room_background,
                                args=(room_id, file_paths),
                                name=f"recovery-{room_id[:8]}",
                                daemon=True
                            )
                            t.start()
                        else:
                            # No files on disk — mark as complete with error
                            logger.warning(
                                f"[startup-recovery] No files found on disk for {room_id}, marking complete"
                            )
                            db.update_data_room_status(
                                room_id, "complete", progress=100,
                                completed_at=datetime.now().isoformat(),
                                error_message="Some files were lost during restart; re-upload to retry"
                            )
                    except Exception as e:
                        logger.error(f"[startup-recovery] Failed to re-process {room_id}: {e}")

            asyncio.create_task(_delayed_reprocess())

    except Exception as e:
        logger.warning(f"Startup recovery failed: {e}")

    # Start worker pool first (before sync, so sync jobs have workers)
    try:
        worker_pool.start()
        logger.info("Worker pool started for file processing")
    except Exception as e:
        logger.warning(f"Failed to start worker pool: {e}")

    # Periodic stall detection — recover data rooms stuck in processing with no active jobs
    async def _stall_detection_loop():
        import asyncio
        import sqlite3 as _sqlite3_stall
        await asyncio.sleep(60)  # Wait 1 minute before first check (after startup recovery settles)
        while True:
            try:
                # Recover jobs stuck in 'running' for >5 minutes (crashed/hung workers)
                recovered = job_queue.recover_stale_jobs(timeout_minutes=5)
                if recovered > 0:
                    logger.info(f"[stall-guard] Recovered {recovered} stale running jobs")

                _sconn = _sqlite3_stall.connect(settings.database_path, timeout=30.0)
                _sconn.row_factory = _sqlite3_stall.Row
                try:
                    _scur = _sconn.cursor()
                    # Find data rooms stuck in processing for >5 minutes with no active jobs.
                    # Treat jobs running for >5 minutes as effectively dead (crashed workers).
                    # Exclude pending jobs that have exhausted their attempts — they can
                    # never be claimed (claim_job requires attempts < max_attempts) and
                    # would otherwise permanently block stall detection.
                    _scur.execute("""
                        SELECT dr.id
                        FROM data_rooms dr
                        WHERE dr.processing_status IN ('parsing', 'indexing', 'extracting')
                        AND dr.created_at < datetime('now', '-5 minutes')
                        AND EXISTS (SELECT 1 FROM documents d WHERE d.data_room_id = dr.id)
                        AND NOT EXISTS (
                            SELECT 1 FROM job_queue jq
                            WHERE jq.data_room_id = dr.id
                            AND (
                                (jq.status = 'pending' AND jq.attempts < jq.max_attempts)
                                 OR (jq.status = 'running' AND jq.started_at > datetime('now', '-5 minutes'))
                            )
                        )
                    """)
                    stalled_ids = [row['id'] for row in _scur.fetchall()]

                    # Exclude data rooms actively processed by BackgroundTasks
                    with _active_background_tasks_lock:
                        active_bg = set(_active_background_tasks)
                    skipped = [rid for rid in stalled_ids if rid in active_bg]
                    if skipped:
                        logger.debug(f"[stall-guard] Skipping {len(skipped)} data rooms with active BackgroundTasks")
                    stalled_ids = [rid for rid in stalled_ids if rid not in active_bg]

                    for room_id in stalled_ids:
                        _scur.execute("""
                            UPDATE documents
                            SET parse_status = 'failed',
                                error_message = 'Processing interrupted - use Reprocess to retry'
                            WHERE data_room_id = ? AND parse_status IN ('pending', 'parsing')
                        """, (room_id,))

                        _scur.execute(
                            "SELECT COUNT(*) as cnt FROM documents WHERE data_room_id = ?",
                            (room_id,)
                        )
                        doc_count = _scur.fetchone()['cnt']

                        _scur.execute("""
                            UPDATE data_rooms
                            SET processing_status = 'complete',
                                progress_percent = 100,
                                total_documents = ?,
                                completed_at = datetime('now')
                            WHERE id = ?
                        """, (doc_count, room_id))

                        logger.info(f"[stall-guard] Recovered stalled data room {room_id} ({doc_count} docs)")

                    if stalled_ids:
                        _sconn.commit()

                    # Also check for complete data rooms with chunks but 0 embeddings
                    _scur.execute("""
                        SELECT dr.id
                        FROM data_rooms dr
                        WHERE dr.processing_status = 'complete'
                        AND EXISTS (
                            SELECT 1 FROM chunks c
                            WHERE c.data_room_id = dr.id AND c.embedding_id IS NULL
                        )
                        AND NOT EXISTS (
                            SELECT 1 FROM chunks c2
                            WHERE c2.data_room_id = dr.id AND c2.embedding_id IS NOT NULL
                        )
                    """)
                    needs_reembed = [row['id'] for row in _scur.fetchall()]
                finally:
                    _sconn.close()

                # Trigger reembed in a background thread so the event loop stays free
                for room_id in needs_reembed:
                    try:
                        logger.info(f"[stall-guard] Data room {room_id} has chunks but 0 embeddings — auto-triggering reembed")
                        from tools.ingest_data_room import reembed_data_room
                        reembed_result = await asyncio.to_thread(reembed_data_room, room_id)
                        logger.info(f"[stall-guard] Reembed result for {room_id}: {reembed_result}")
                    except Exception as e:
                        logger.error(f"[stall-guard] Reembed failed for {room_id}: {e}")

            except Exception as e:
                logger.warning(f"[stall-guard] Check failed: {e}")

            await asyncio.sleep(60)

    import asyncio
    asyncio.create_task(_stall_detection_loop())

    # Start sync service with a delay so the server can respond to requests first
    async def _delayed_sync_start():
        await asyncio.sleep(30)  # 30s delay before first sync
        try:
            sync_service.start()
            logger.info("Google Drive sync service started (delayed)")
        except Exception as e:
            logger.warning(f"Failed to start sync service: {e}")

    asyncio.create_task(_delayed_sync_start())

    # Log server info
    logger.info(f"API server: http://{settings.host}:{settings.port}")
    logger.info(f"API docs: http://{settings.host}:{settings.port}/docs")
    logger.info(f"Database: {settings.database_path}")
    logger.info(f"Vector DB: {settings.chroma_db_path}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down VC Due Diligence API server...")
    sync_service.stop()
    worker_pool.stop()


# ============================================================================
# Access Control Helpers
# ============================================================================

def get_user_identity(
    x_user_id: Optional[str] = Header(None),
    x_user_email: Optional[str] = Header(None)
) -> Dict[str, Optional[str]]:
    """Extract user identity from request headers."""
    return {"user_id": x_user_id, "user_email": x_user_email}


def require_data_room_access(
    data_room_id: str,
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
    require_owner: bool = False
) -> str:
    """
    Check if user has access to a data room. Returns role.
    Raises HTTPException(403) if no access.
    When sharing is disabled (feature flag), always allows access.
    """
    if not settings.enable_sharing:
        return 'owner'

    role = db.check_data_room_access(data_room_id, user_id, user_email)
    if role is None:
        raise HTTPException(status_code=403, detail="You do not have access to this data room")
    if require_owner and role not in ('owner', 'legacy'):
        raise HTTPException(status_code=403, detail="Only the data room owner can perform this action")
    return role


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """
    Health check endpoint to verify system status.
    Returns comprehensive system metrics including Tier 3 components.
    """
    # Check database exists
    db_exists = Path(settings.database_path).exists()

    # Check vector DB exists
    vector_db_exists = Path(settings.chroma_db_path).exists()

    # Check API keys configured
    api_keys_ok = bool(settings.anthropic_api_key and settings.openai_api_key)

    # Get system metrics
    process = psutil.Process()
    memory_info = process.memory_info()
    system_memory = psutil.virtual_memory()

    # Build response
    response = {
        "status": "healthy",
        "version": "0.1.0",
        "tier": "3",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {
            "database": "ok" if db_exists else "missing",
            "vector_db": "ok" if vector_db_exists else "missing",
            "api_keys": "configured" if api_keys_ok else "missing"
        },
        "system": {
            "memory_used_mb": round(memory_info.rss / (1024 * 1024), 1),
            "memory_available_mb": round(system_memory.available / (1024 * 1024), 1),
            "memory_percent": system_memory.percent,
            "cpu_percent": process.cpu_percent(),
            "active_processing_jobs": _active_processing_count,
            "max_concurrent_jobs": settings.max_concurrent_jobs,
            "sync_service_running": sync_service.is_running if hasattr(sync_service, 'is_running') else False
        }
    }

    # Add Tier 3 component status
    try:
        from app.circuit_breaker import openai_circuit, chromadb_circuit
        response["components"]["openai_api"] = openai_circuit.state.value
        response["components"]["chromadb"] = chromadb_circuit.state.value

        # Mark as degraded if circuit breakers are open
        if openai_circuit.state.value == "open":
            response["status"] = "degraded"
    except ImportError:
        pass

    try:
        from app.rate_limiter import embedding_rate_limiter
        rate_metrics = embedding_rate_limiter.get_metrics()
        response["rate_limiter"] = {
            "rpm_utilization": round(rate_metrics["rpm_utilization"] * 100, 1),
            "tpm_utilization": round(rate_metrics["tpm_utilization"] * 100, 1),
            "total_requests": rate_metrics["total_requests"],
            "rate_limit_hits": rate_metrics["rate_limit_hits"],
            "in_backoff": rate_metrics["in_backoff"]
        }
    except ImportError:
        pass

    try:
        from app.memory_manager import memory_manager
        mem_status = memory_manager.get_status()
        response["memory_manager"] = {
            "tracked_allocated_mb": round(mem_status["tracked_allocated_mb"], 1),
            "active_allocations": mem_status["active_allocations"],
            "can_allocate_500mb": mem_status["can_allocate_500mb"]
        }
    except ImportError:
        pass

    try:
        from app.redis_cache import redis_cache
        cache_stats = redis_cache.get_stats()
        response["cache"] = {
            "connected": cache_stats["connected"],
            "hit_rate": round(cache_stats["hit_rate"] * 100, 1),
            "hits": cache_stats["hits"],
            "misses": cache_stats["misses"]
        }
    except ImportError:
        pass

    # Final status check
    if not (db_exists and vector_db_exists and api_keys_ok):
        response["status"] = "unhealthy"

    return response


@app.get("/api/metrics")
async def get_metrics():
    """
    Prometheus-compatible metrics endpoint for monitoring.
    Returns metrics in text format for scraping.
    """
    try:
        from app.metrics import metrics
        return Response(
            content=metrics.get_prometheus_format(),
            media_type="text/plain"
        )
    except ImportError:
        return Response(
            content="# Metrics module not available\n",
            media_type="text/plain"
        )


@app.get("/api/tier3/status")
async def get_tier3_status():
    """
    Detailed Tier 3 component status for debugging and monitoring.
    """
    status = {
        "tier": 3,
        "timestamp": datetime.utcnow().isoformat(),
        "components": {}
    }

    # Rate limiter
    try:
        from app.rate_limiter import embedding_rate_limiter
        status["components"]["rate_limiter"] = embedding_rate_limiter.get_metrics()
    except ImportError:
        status["components"]["rate_limiter"] = {"available": False}

    # Circuit breakers
    try:
        from app.circuit_breaker import circuit_registry
        status["components"]["circuit_breakers"] = circuit_registry.get_all_metrics()
    except ImportError:
        status["components"]["circuit_breakers"] = {"available": False}

    # Memory manager
    try:
        from app.memory_manager import memory_manager
        status["components"]["memory"] = memory_manager.get_metrics()
    except ImportError:
        status["components"]["memory"] = {"available": False}

    # Cache
    try:
        from app.redis_cache import redis_cache, embedding_cache, query_cache
        status["components"]["cache"] = {
            "redis": redis_cache.get_stats(),
            "embedding_cache_hits": embedding_cache.cache.stats.hits,
            "query_cache_hits": query_cache.cache.stats.hits
        }
    except ImportError:
        status["components"]["cache"] = {"available": False}

    # Database pool
    try:
        from app.database import get_db_pool
        pool = get_db_pool()
        status["components"]["db_pool"] = pool.get_stats()
    except Exception:
        status["components"]["db_pool"] = {"available": False}

    return status


# Job queue status endpoint
@app.get("/api/jobs/status")
async def get_job_queue_status():
    """
    Get job queue statistics.
    Returns counts of pending, running, completed, and failed jobs.
    """
    stats = job_queue.get_queue_stats()
    return {
        "queue_stats": stats,
        "processing": {
            "active_jobs": _active_processing_count,
            "max_concurrent": settings.max_concurrent_jobs,
            "queue_depth": stats['pending']
        }
    }


@app.get("/api/jobs/{job_id}")
async def get_job_details(job_id: str):
    """
    Get details of a specific job.
    """
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a pending job.
    """
    if job_queue.cancel_job(job_id):
        return {"message": f"Job {job_id} cancelled"}
    raise HTTPException(
        status_code=400,
        detail=f"Cannot cancel job {job_id}. Job may not exist or is not pending."
    )


# Root endpoint
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
_serve_frontend = os.getenv("SERVE_FRONTEND", "false").lower() == "true" and _frontend_dist.exists()

@app.get("/")
async def root():
    """Root endpoint — serves frontend in production, API info in development."""
    if _serve_frontend:
        return FileResponse(str(_frontend_dist / "index.html"))
    return {
        "name": "VC Due Diligence Assistant API",
        "version": "0.1.0",
        "status": "running",
        "documentation": "/docs",
        "health_check": "/api/health"
    }


# Background Processing Functions

def process_data_room_background(data_room_id: str, file_paths: List[str]):
    """
    Background task to process data room files.

    Pipeline:
    1. Parse all documents
    2. Chunk text
    3. Generate embeddings
    4. Index to vector DB

    Uses a semaphore to limit concurrent processing and prevent memory exhaustion.
    """
    import time
    import traceback
    global _active_processing_count

    # Try to acquire semaphore (non-blocking)
    if not _processing_semaphore.acquire(blocking=False):
        logger.warning(f"[{data_room_id}] Processing queue full ({settings.max_concurrent_jobs} jobs running). Queuing...")
        db.update_data_room_status(data_room_id, "queued", progress=0)
        # Wait for semaphore (blocking)
        _processing_semaphore.acquire(blocking=True)
        logger.info(f"[{data_room_id}] Acquired processing slot, starting...")

    _active_processing_count += 1
    with _active_background_tasks_lock:
        _active_background_tasks.add(data_room_id)
    start_time = time.time()

    try:
        logger.info(f"[{data_room_id}] Starting background processing for {len(file_paths)} files (active jobs: {_active_processing_count})")

        # Update status to parsing
        db.update_data_room_status(data_room_id, "parsing", progress=10)
        db.log_processing_stage("processing", "started", data_room_id=data_room_id,
                                message=f"Processing {len(file_paths)} files")

        # Import processing tools with detailed error logging
        try:
            logger.info(f"[{data_room_id}] Importing processing tools...")
            from tools.ingest_data_room import parse_file_by_type
            from tools.chunk_documents import chunk_documents
            from tools.generate_embeddings import generate_embeddings
            from tools.index_to_vectordb import index_to_vectordb, VectorDBIndexer
            logger.info(f"[{data_room_id}] All tools imported successfully")
        except ImportError as e:
            error_msg = f"Failed to import processing tools: {e}"
            logger.error(f"[{data_room_id}] {error_msg}", exc_info=True)
            db.update_data_room_status(data_room_id, "failed", error_message=error_msg)
            db.log_processing_stage("processing", "failed", data_room_id=data_room_id,
                                    error_details=f"{error_msg}\n{traceback.format_exc()}")
            return

        all_chunks = []
        parsed_docs = 0
        failed_files = []

        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
        from tools.generate_embeddings import generate_embeddings_streaming
        import threading
        import queue

        # --- Helper: parse + chunk a single file (runs in worker thread) ---
        def _parse_and_chunk_one(file_path):
            """Parse and chunk a single file. Returns (chunks, doc_id) or raises."""
            file_path_obj = Path(file_path)
            file_type = file_path_obj.suffix.lower()

            # Find document record with resilient matching and auto-create fallback
            document_id = db.find_document_by_filename(
                data_room_id=data_room_id,
                target_filename=file_path_obj.name,
                create_if_missing=True,
                file_path=str(file_path_obj),
                file_size=file_path_obj.stat().st_size,
            )

            if not document_id:
                raise FileNotFoundError(
                    f"Document record not found and auto-creation failed for: {file_path_obj.name}"
                )

            db.update_document_status(document_id, "parsing")

            # Parse using shared dispatcher (handles all file types including .doc/.ppt/.txt)
            parsed = parse_file_by_type(
                file_path,
                use_ocr=True,
                max_ocr_pages=settings.max_ocr_pages,
                use_financial_excel=True,
            )

            # Check parse errors
            if parsed.get('error'):
                error_msg = f"Parsing returned error: {parsed.get('error')}"
                db.update_document_status(document_id, "failed", error_message=error_msg)
                raise RuntimeError(error_msg)

            # Chunk
            chunks = chunk_documents(parsed, chunk_size=settings.max_chunk_size, overlap=settings.chunk_overlap)
            if not chunks:
                if parsed.get('needs_ocr'):
                    error_msg = "Document appears to be image-based with no extractable text. OCR failed or Tesseract not installed (brew install tesseract)"
                elif file_type in ['.xlsx', '.xls', '.csv']:
                    sheet_count = len(parsed.get('sheets', []))
                    total_rows = parsed.get('total_rows', 0)
                    error_msg = (
                        f"No searchable chunks generated from Excel file "
                        f"({sheet_count} sheets, {total_rows} total rows). "
                        f"Data rows may have been filtered out during text conversion."
                    )
                else:
                    error_msg = "No chunks generated from document (possibly empty or unreadable)"
                db.update_document_status(document_id, "failed", error_message=error_msg)
                raise RuntimeError(error_msg)

            # Tag chunks
            for chunk in chunks:
                chunk['document_id'] = document_id
                chunk['data_room_id'] = data_room_id
                if 'metadata' not in chunk:
                    chunk['metadata'] = {}
                chunk['metadata']['document_id'] = document_id
                chunk['metadata']['data_room_id'] = data_room_id

            # Mark document parsed
            db.update_document_status(
                document_id, "parsed",
                page_count=parsed.get('page_count'),
                token_count=sum(c.get('token_count', 0) for c in chunks)
            )

            logger.info(f"[{data_room_id}] Parsed {file_path_obj.name}: {parsed.get('page_count', '?')} pages, {len(chunks)} chunks")
            return chunks, document_id

        # --- Pre-flight: verify OpenAI API key works before processing ---
        try:
            from openai import OpenAI as _OpenAI
            _test_client = _OpenAI(api_key=settings.openai_api_key)
            _test_client.embeddings.create(model=settings.openai_embedding_model, input=["test"])
            logger.info(f"[{data_room_id}] OpenAI embedding API pre-flight check passed")
        except Exception as e:
            error_msg = f"OpenAI embedding API is not working: {e}"
            logger.error(f"[{data_room_id}] {error_msg}")
            db.update_data_room_status(data_room_id, "failed", error_message=error_msg)
            db.log_processing_stage("processing", "failed", data_room_id=data_room_id,
                                    error_details=error_msg)
            return

        # --- Pipelined parsing + embedding (overlap I/O with API calls) ---
        chunk_queue = queue.Queue()  # Parsed chunks flow here for embedding
        _SENTINEL = None  # Signals parsing is done
        total_chunks_queued = 0
        total_embedded = 0
        embed_error = [None]  # Mutable container for cross-thread error
        shared_indexer = VectorDBIndexer()  # Shared across embedding functions

        def _embedding_consumer():
            """Consume chunks from the queue and embed+index them as they arrive."""
            nonlocal total_embedded
            try:
                pending_chunks = []
                batch_size = settings.streaming_batch_size or 256
                consecutive_timeouts = 0
                max_consecutive_timeouts = 60  # Exit after 60 seconds of no data

                while True:
                    try:
                        # Increased timeout to 30 seconds to handle slow API calls
                        item = chunk_queue.get(timeout=30)
                        consecutive_timeouts = 0  # Reset on successful get
                    except queue.Empty:
                        consecutive_timeouts += 1
                        # If we've been waiting too long with no data, check if we should exit
                        if consecutive_timeouts >= max_consecutive_timeouts // 30:
                            logger.warning(f"[{data_room_id}] Embedding consumer: no data for {consecutive_timeouts * 30}s, flushing and exiting")
                            if pending_chunks:
                                _embed_and_index_batch(pending_chunks)
                            break
                        continue

                    if item is _SENTINEL:
                        # Flush remaining
                        if pending_chunks:
                            _embed_and_index_batch(pending_chunks)
                        break

                    pending_chunks.extend(item)

                    # Embed when we have enough for a batch
                    while len(pending_chunks) >= batch_size:
                        batch = pending_chunks[:batch_size]
                        pending_chunks = pending_chunks[batch_size:]
                        _embed_and_index_batch(batch)

            except TimeoutError as e:
                # Handle rate limiter timeout gracefully - flush what we have
                logger.warning(f"[{data_room_id}] Rate limiter timeout in embedding consumer: {e}")
                embed_error[0] = e
            except Exception as e:
                embed_error[0] = e
                logger.error(f"[{data_room_id}] Embedding consumer error: {e}", exc_info=True)

        embedding_errors = []  # Collect actual API error messages

        def _embed_and_index_batch(chunks_batch, retry_count=0):
            """Embed and index a batch of chunks with retry support."""
            nonlocal total_embedded
            max_retries = 3

            try:
                for batch in generate_embeddings_streaming(
                    chunks_batch,
                    model=settings.openai_embedding_model,
                    batch_size=settings.streaming_batch_size,
                    max_concurrent=settings.embedding_max_concurrent
                ):
                    index_to_vectordb(data_room_id=data_room_id, chunks_with_embeddings=batch, indexer=shared_indexer)
                    db.create_chunks_batch_optimized(batch)
                    successfully_embedded = sum(1 for c in batch if c.get('embedding') is not None and not c.get('embedding_failed'))
                    total_embedded += successfully_embedded
                    # Collect error messages from failed chunks
                    for c in batch:
                        if c.get('embedding_error') and c['embedding_error'] not in embedding_errors:
                            embedding_errors.append(c['embedding_error'])
                    if total_chunks_queued > 0:
                        progress = 10 + (total_embedded / max(total_chunks_queued, 1)) * 85
                        db.update_data_room_status(data_room_id, "indexing", progress=min(progress, 95))
                    logger.info(f"[{data_room_id}] Indexed batch: {total_embedded} chunks embedded so far")
            except TimeoutError as e:
                if retry_count < max_retries:
                    logger.warning(f"[{data_room_id}] Rate limiter timeout, retrying batch ({retry_count + 1}/{max_retries})...")
                    time.sleep(5 * (retry_count + 1))  # Exponential backoff
                    _embed_and_index_batch(chunks_batch, retry_count + 1)
                else:
                    logger.error(f"[{data_room_id}] Rate limiter timeout after {max_retries} retries: {e}")
                    raise

        # Start embedding consumer thread
        embed_thread = threading.Thread(target=_embedding_consumer, name="embed-consumer", daemon=True)
        embed_thread.start()

        # --- Parallel file parsing (producer) ---
        parse_workers = min(settings.max_parse_workers, len(file_paths))
        logger.info(f"[{data_room_id}] Parsing {len(file_paths)} files with {parse_workers} workers (timeout: {settings.parse_timeout_seconds}s) [pipelined]")

        with ThreadPoolExecutor(max_workers=parse_workers, thread_name_prefix="parse") as executor:
            future_to_path = {
                executor.submit(_parse_and_chunk_one, fp): fp
                for fp in file_paths
            }

            for future in as_completed(future_to_path):
                fp = future_to_path[future]
                fname = Path(fp).name
                try:
                    chunks, doc_id = future.result(timeout=settings.parse_timeout_seconds)
                    all_chunks.extend(chunks)
                    total_chunks_queued += len(chunks)
                    chunk_queue.put(chunks)  # Feed to embedding consumer immediately
                    parsed_docs += 1
                    progress = 10 + (parsed_docs / len(file_paths)) * 40
                    db.update_data_room_status(data_room_id, "parsing", progress=progress)
                    logger.info(f"[{data_room_id}] Progress: {progress:.1f}% ({parsed_docs}/{len(file_paths)} files)")
                except FuturesTimeoutError:
                    error_msg = f"Parsing timed out after {settings.parse_timeout_seconds}s"
                    logger.error(f"[{data_room_id}] {error_msg}: {fname}")
                    failed_files.append({"file": fname, "error": error_msg})
                    # Mark the document as failed in the database
                    try:
                        doc_id = db.find_document_by_filename(data_room_id, fname)
                        if doc_id:
                            db.update_document_status(doc_id, "failed", error_message=error_msg)
                    except Exception:
                        pass
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"[{data_room_id}] Failed to parse {fname}: {error_msg}", exc_info=True)
                    failed_files.append({"file": fname, "error": error_msg})
                    # Mark the document as failed if we can find it
                    try:
                        doc_id = db.find_document_by_filename(data_room_id, fname)
                        if doc_id:
                            db.update_document_status(doc_id, "failed", error_message=error_msg[:500])
                    except Exception:
                        pass

        # Signal embedding consumer that parsing is done, then wait for it with timeout
        chunk_queue.put(_SENTINEL)
        embed_thread_timeout = 300  # 5 minutes max to wait for embedding to finish
        embed_thread.join(timeout=embed_thread_timeout)

        if embed_thread.is_alive():
            logger.warning(f"[{data_room_id}] Embedding consumer thread timed out after {embed_thread_timeout}s, continuing anyway")
            # Thread is still running but we won't wait forever

        # Log summary of parsing phase
        logger.info(f"[{data_room_id}] Parsing complete: {parsed_docs} succeeded, {len(failed_files)} failed")
        if failed_files:
            for f in failed_files:
                logger.error(f"[{data_room_id}] Failed file: {f['file']} - {f['error']}")

        if not all_chunks:
            error_msg = f"No chunks extracted from any document. Failed files: {[f['file'] for f in failed_files]}"
            logger.error(f"[{data_room_id}] {error_msg}")
            db.update_data_room_status(data_room_id, "failed", error_message=error_msg)
            db.log_processing_stage("processing", "failed", data_room_id=data_room_id,
                                    error_details=error_msg)
            return

        # Check if embedding consumer had an error
        if embed_error[0]:
            error_msg = f"Error during embedding/indexing: {str(embed_error[0])}"
            logger.warning(f"[{data_room_id}] {error_msg}")
            # Don't fail if we have some successful embeddings - continue to completion
            if total_embedded == 0:
                db.update_data_room_status(data_room_id, "failed", error_message=error_msg)
                db.log_processing_stage("processing", "failed", data_room_id=data_room_id,
                                        error_details=error_msg)
                return

        logger.info(f"[{data_room_id}] Total chunks embedded: {total_embedded} / {total_chunks_queued}")

        # Check if any embeddings actually succeeded
        if total_embedded == 0 and total_chunks_queued > 0:
            error_detail = f" Error: {'; '.join(embedding_errors[:3])}" if embedding_errors else ""
            error_msg = (
                f"All {total_chunks_queued} chunks failed embedding generation.{error_detail} "
                f"No documents were indexed for search. Try re-processing the data room."
            )
            logger.error(f"[{data_room_id}] {error_msg}")
            db.update_data_room_status(data_room_id, "failed", error_message=error_msg)
            db.log_processing_stage("processing", "failed", data_room_id=data_room_id,
                                    error_details=error_msg)
            return

        # Mark as complete (with error summary if some files failed)
        duration_ms = int((time.time() - start_time) * 1000)
        if failed_files:
            error_summary = f"{len(failed_files)}/{len(file_paths)} files failed: " + \
                ", ".join(f['file'] for f in failed_files[:5])
            if len(failed_files) > 5:
                error_summary += f" (and {len(failed_files) - 5} more)"
            db.update_data_room_status(
                data_room_id,
                "complete",
                progress=100,
                completed_at=datetime.now().isoformat(),
                error_message=error_summary,
                total_chunks=total_embedded,
                total_documents=parsed_docs,
            )
            db.log_processing_stage("processing", "completed", data_room_id=data_room_id,
                                    message=f"Processed {parsed_docs} files, {total_embedded} chunks. {error_summary}",
                                    duration_ms=duration_ms)
            logger.warning(f"[{data_room_id}] Processing complete with errors in {duration_ms}ms: {parsed_docs} files, {total_embedded} chunks. {error_summary}")
        else:
            db.update_data_room_status(
                data_room_id,
                "complete",
                progress=100,
                completed_at=datetime.now().isoformat(),
                total_chunks=total_embedded,
                total_documents=parsed_docs,
            )
            db.log_processing_stage("processing", "completed", data_room_id=data_room_id,
                                    message=f"Processed {parsed_docs} files, {total_embedded} chunks",
                                    duration_ms=duration_ms)
            logger.success(f"[{data_room_id}] Processing complete in {duration_ms}ms: {parsed_docs} files, {total_embedded} chunks")

        # Safety net: verify embeddings exist in database (don't trust in-memory counter)
        # This catches race conditions where the embedding thread timed out or failed silently
        try:
            missing = db.get_chunks_without_embeddings(data_room_id)
            if missing and len(missing) > 0:
                total_db_chunks = len(missing)
                logger.warning(
                    f"[{data_room_id}] Post-completion check: {total_db_chunks} chunks "
                    f"missing embeddings — auto-triggering reembed"
                )
                from tools.ingest_data_room import reembed_data_room
                reembed_result = reembed_data_room(data_room_id)
                logger.info(f"[{data_room_id}] Auto-reembed result: {reembed_result}")
        except Exception as e:
            logger.error(f"[{data_room_id}] Post-completion embedding check failed: {e}")

        # Auto-trigger financial analysis for Excel files if enabled
        if settings.enable_auto_financial_analysis:
            try:
                from tools.financial_analysis_agent import analyze_financial_model

                documents = db.get_documents_by_data_room(data_room_id)
                for doc in documents:
                    if doc['file_type'] in ['xlsx', 'xls'] and doc['parse_status'] == 'parsed':
                        logger.info(f"[{data_room_id}] Auto-triggering financial analysis for {doc['file_name']}")

                        file_path = f"{settings.data_rooms_path}/{data_room_id}/raw/{doc['file_name']}"

                        if Path(file_path).exists():
                            try:
                                result = analyze_financial_model(
                                    file_path=file_path,
                                    data_room_id=data_room_id,
                                    document_id=doc['id'],
                                    analysis_model=settings.financial_analysis_model,
                                    extraction_model=settings.financial_extraction_model,
                                    max_cost=settings.financial_analysis_max_cost
                                )

                                db.save_financial_analysis(
                                    analysis_id=result['analysis_id'],
                                    data_room_id=data_room_id,
                                    document_id=doc['id'],
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

                                logger.success(f"[{data_room_id}] Financial analysis complete for {doc['file_name']}")

                            except Exception as e:
                                logger.error(f"[{data_room_id}] Financial analysis failed for {doc['file_name']}: {e}")

            except ImportError as e:
                logger.warning(f"[{data_room_id}] Financial analysis module not available: {e}")

    except Exception as e:
        error_msg = f"Unexpected error processing data room: {str(e)}"
        logger.error(f"[{data_room_id}] {error_msg}", exc_info=True)
        db.update_data_room_status(data_room_id, "failed", error_message=error_msg)
        db.log_processing_stage("processing", "failed", data_room_id=data_room_id,
                                error_details=f"{error_msg}\n{traceback.format_exc()}")

    finally:
        # Always release the semaphore and decrement counter
        _active_processing_count -= 1
        _processing_semaphore.release()
        with _active_background_tasks_lock:
            _active_background_tasks.discard(data_room_id)
        logger.debug(f"[{data_room_id}] Released processing slot (active jobs: {_active_processing_count})")


# Data Room Endpoints

# File validation helpers
import re
import shutil
import aiofiles

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and invalid characters.
    """
    # Get just the filename (remove any path components)
    filename = Path(filename).name

    # Remove or replace dangerous characters
    # Keep alphanumeric, dots, hyphens, underscores, and spaces
    filename = re.sub(r'[^\w\s\-\.]', '_', filename)

    # Prevent leading dots (hidden files) or empty names
    filename = filename.lstrip('.')
    if not filename:
        filename = "unnamed_file"

    # Limit length (preserve extension)
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255 - len(ext)] + ext

    return filename


def validate_file_content(content: bytes, filename: str) -> tuple:
    """
    Validate that file content matches expected type using magic bytes.

    Returns:
        (is_valid: bool, error_message: str or None)
    """
    extension = Path(filename).suffix.lower()

    SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.csv', '.txt'}
    if extension not in SUPPORTED_EXTENSIONS:
        return False, f"Unsupported file type: '{extension}'. Supported: PDF, Word, Excel, PowerPoint, CSV, TXT"

    if extension == '.pdf':
        # PDF files start with %PDF-
        if not content.startswith(b'%PDF-'):
            return False, f"File '{filename}' does not appear to be a valid PDF (invalid header)"
        return True, None

    elif extension in ['.xlsx', '.pptx', '.docx']:
        # Office Open XML files are ZIP archives starting with PK
        if not content.startswith(b'PK'):
            return False, f"File '{filename}' does not appear to be a valid Office file (invalid header)"
        return True, None

    elif extension in ['.xls', '.ppt', '.doc']:
        # Old Office formats start with D0 CF 11 E0 (OLE compound document)
        if not content.startswith(b'\xd0\xcf\x11\xe0'):
            return False, f"File '{filename}' does not appear to be a valid Office file (invalid header)"
        return True, None

    elif extension == '.csv':
        # CSV is plain text - just check it's decodable
        try:
            content.decode('utf-8')
            return True, None
        except UnicodeDecodeError:
            return False, f"File '{filename}' does not appear to be a valid CSV file (encoding error)"

    else:
        # Unknown extension - allow but warn
        logger.warning(f"Unknown file type: {extension} for {filename}")
        return True, None


async def stream_upload_file(
    file: UploadFile,
    destination: Path,
    max_size_bytes: int,
    chunk_size: int = 1024 * 1024  # 1MB chunks
) -> dict:
    """
    Stream file upload to disk without loading entire file into memory.

    Args:
        file: The uploaded file
        destination: Path to save the file
        max_size_bytes: Maximum allowed file size
        chunk_size: Size of chunks to read (default 1MB)

    Returns:
        Dict with filename, path, size, and header_bytes for validation

    Raises:
        HTTPException: If file exceeds max size or other errors
    """
    total_size = 0
    header_bytes = b''

    try:
        async with aiofiles.open(destination, 'wb') as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break

                # Capture first bytes for magic byte validation
                if total_size == 0:
                    header_bytes = chunk[:20]

                total_size += len(chunk)

                # Check size limit
                if total_size > max_size_bytes:
                    # Clean up partial file
                    await f.close()
                    if destination.exists():
                        destination.unlink()
                    raise HTTPException(
                        status_code=413,
                        detail=f"File '{file.filename}' ({total_size / (1024*1024):.1f}MB) exceeds maximum size of {max_size_bytes / (1024*1024):.0f}MB"
                    )

                await f.write(chunk)

        return {
            'filename': destination.name,
            'path': str(destination),
            'size': total_size,
            'header_bytes': header_bytes
        }

    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        if destination.exists():
            destination.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


@app.post("/api/data-room/create")
async def create_data_room(
    company_name: str = Form(...),
    analyst_name: str = Form(...),
    analyst_email: Optional[str] = Form(None),
    security_level: str = Form("local_only"),
    total_documents: int = Form(0),
    files: List[UploadFile] = File(default=[]),
    identity: Dict = Depends(get_user_identity)
):
    """
    Create a new data room, optionally with files.

    When called without files (total_documents > 0), creates the record immediately
    so the frontend can navigate to the data room. Files are then uploaded separately
    via POST /api/data-room/{id}/upload-files.

    When called with files, behaves as before (upload + enqueue processing).

    Returns:
        Data room ID and initial status
    """
    data_room_id = None
    data_room_path = None

    try:
        # Generate data room ID
        data_room_id = f"dr_{uuid.uuid4().hex[:12]}"

        logger.info(f"Creating data room {data_room_id} for {company_name}")

        # ============================================================
        # Phase 1: Create directory and data room record up front
        # ============================================================
        data_room_path = Path(f"{settings.data_rooms_path}/{data_room_id}/raw")
        data_room_path.mkdir(parents=True, exist_ok=True)

        # Use file count if files provided, otherwise use the declared total_documents
        total_files = len(files) if files else total_documents

        db.create_data_room(
            data_room_id=data_room_id,
            company_name=company_name,
            analyst_name=analyst_name,
            analyst_email=analyst_email,
            security_level=security_level,
            total_documents=total_files,
            user_id=None  # Upload-page data rooms are unowned ("legacy") — accessible without login
        )
        db.update_data_room_status(data_room_id, "uploading", progress=0)

        # If no files provided, return immediately — files will be uploaded separately
        if not files:
            logger.info(f"Data room {data_room_id} created (metadata only, expecting {total_files} files)")
            return {
                "data_room_id": data_room_id,
                "company_name": company_name,
                "total_documents": total_files,
                "status": "uploading",
                "message": "Data room created. Upload files to begin processing."
            }

        # ============================================================
        # Phase 2: Stream each file to disk, create its document
        # record, and enqueue it for processing immediately.
        # Files start parsing while subsequent files are still uploading.
        # ============================================================
        max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        uploaded_count = 0
        from app.job_queue import JobType

        for file in files:
            original_filename = file.filename
            safe_filename = sanitize_filename(original_filename)

            # Log if filename was sanitized
            if safe_filename != original_filename:
                logger.warning(f"Sanitized filename: '{original_filename}' -> '{safe_filename}'")

            file_path = data_room_path / safe_filename

            # Stream file to disk (uses ~1MB memory instead of full file size)
            file_info = await stream_upload_file(
                file=file,
                destination=file_path,
                max_size_bytes=max_size_bytes
            )

            # Check for empty file
            if file_info['size'] == 0:
                file_path.unlink()
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{original_filename}' is empty"
                )

            # Validate file content using header bytes (magic bytes)
            is_valid, error_msg = validate_file_content(file_info['header_bytes'], safe_filename)
            if not is_valid:
                file_path.unlink()
                raise HTTPException(status_code=400, detail=error_msg)

            # Create document record immediately
            file_type = Path(safe_filename).suffix.lower().replace('.', '')
            document_id = db.create_document(
                data_room_id=data_room_id,
                file_name=safe_filename,
                file_path=str(file_path),
                file_size=file_info['size'],
                file_type=file_type
            )

            # Enqueue for processing right away — workers start parsing
            # while remaining files are still uploading
            job_queue.enqueue(
                job_type=JobType.PROCESS_FILE.value,
                payload={
                    'data_room_id': data_room_id,
                    'document_id': document_id,
                    'file_path': str(file_path),
                    'file_name': safe_filename,
                    'upload_total_files': total_files,
                },
                data_room_id=data_room_id,
                file_name=safe_filename
            )

            uploaded_count += 1
            logger.info(f"Streamed & enqueued file {uploaded_count}/{total_files}: {safe_filename} ({file_info['size']} bytes)")

        logger.success(f"Data room {data_room_id} created with {uploaded_count} files — processing started")

        return {
            "data_room_id": data_room_id,
            "company_name": company_name,
            "total_documents": uploaded_count,
            "status": "uploading",
            "message": "Data room created. Processing will begin shortly."
        }

    except HTTPException:
        # Re-raise HTTP exceptions (validation errors) as-is
        # Cleanup any partially created resources
        try:
            if data_room_path and data_room_path.exists():
                shutil.rmtree(data_room_path.parent)  # Remove the entire data room folder
                logger.info(f"Cleaned up data room directory after validation error")
            # Also clean up database records if they were created
            if data_room_id:
                db.delete_data_room(data_room_id)
                logger.info(f"Cleaned up data room database records after validation error")
        except Exception as cleanup_error:
            logger.error(f"Error during HTTPException cleanup: {cleanup_error}")
        raise

    except Exception as e:
        logger.error(f"Failed to create data room {data_room_id}: {e}", exc_info=True)

        # Rollback: Clean up any created resources
        try:
            # Remove uploaded files
            if data_room_path and data_room_path.exists():
                shutil.rmtree(data_room_path.parent)
                logger.info(f"Rolled back: Removed data room directory {data_room_path.parent}")

            # Remove database records
            if data_room_id:
                db.delete_data_room(data_room_id)
                logger.info(f"Rolled back: Removed data room database records for {data_room_id}")
        except Exception as cleanup_error:
            logger.error(f"Error during rollback cleanup: {cleanup_error}")

        raise HTTPException(status_code=500, detail=f"Failed to create data room: {str(e)}")


@app.post("/api/data-room/{data_room_id}/upload-files")
async def upload_files_to_data_room(
    data_room_id: str,
    files: List[UploadFile] = File(...),
    identity: Dict = Depends(get_user_identity)
):
    """
    Upload files to an existing data room and enqueue them for processing.

    Used after creating a data room via POST /api/data-room/create (without files)
    so the frontend can navigate to the data room immediately while files upload
    in the background.
    """
    try:
        data_room = db.get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(status_code=404, detail="Data room not found")

        data_room_path = Path(f"{settings.data_rooms_path}/{data_room_id}/raw")
        data_room_path.mkdir(parents=True, exist_ok=True)

        max_size_bytes = settings.max_file_size_mb * 1024 * 1024
        batch_size = len(files)
        uploaded_count = 0
        from app.job_queue import JobType

        # Use the data room's total_documents (set during creation) as the real total,
        # since files may arrive in multiple batches from the frontend.
        real_total = data_room.get('total_documents') or batch_size

        async def _save_file(file: UploadFile):
            """Stream a single file to disk and return its info."""
            original_filename = file.filename
            safe_filename = sanitize_filename(original_filename)
            if safe_filename != original_filename:
                logger.warning(f"Sanitized filename: '{original_filename}' -> '{safe_filename}'")
            dest = data_room_path / safe_filename
            file_info = await stream_upload_file(file=file, destination=dest, max_size_bytes=max_size_bytes)
            return safe_filename, dest, file_info

        # Save all files in this batch concurrently
        saved_files = await asyncio.gather(*[_save_file(f) for f in files])

        for safe_filename, file_path, file_info in saved_files:
            if file_info['size'] == 0:
                file_path.unlink()
                logger.warning(f"Skipping empty file: {safe_filename}")
                continue

            is_valid, error_msg = validate_file_content(file_info['header_bytes'], safe_filename)
            if not is_valid:
                file_path.unlink()
                logger.warning(f"Skipping invalid file {safe_filename}: {error_msg}")
                continue

            file_type = Path(safe_filename).suffix.lower().replace('.', '')
            document_id = db.create_document(
                data_room_id=data_room_id,
                file_name=safe_filename,
                file_path=str(file_path),
                file_size=file_info['size'],
                file_type=file_type
            )

            job_queue.enqueue(
                job_type=JobType.PROCESS_FILE.value,
                payload={
                    'data_room_id': data_room_id,
                    'document_id': document_id,
                    'file_path': str(file_path),
                    'file_name': safe_filename,
                    'upload_total_files': real_total,
                },
                data_room_id=data_room_id,
                file_name=safe_filename
            )

            uploaded_count += 1
            logger.info(f"[upload-files] Streamed & enqueued file {uploaded_count}/{batch_size}: {safe_filename} ({file_info['size']} bytes)")

        logger.success(f"[upload-files] {uploaded_count} files uploaded to data room {data_room_id}")

        return {
            "data_room_id": data_room_id,
            "uploaded_count": uploaded_count,
            "total_files": batch_size,
            "message": f"{uploaded_count} files uploaded. Processing started."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload files to data room {data_room_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {str(e)}")


@app.get("/api/data-room/{data_room_id}/status")
def get_data_room_status(
    data_room_id: str,
    identity: Dict = Depends(get_user_identity)
):
    """
    Get processing status for a data room.

    Note: This is a sync def (not async) so FastAPI runs it in a threadpool,
    preventing synchronous DB calls from blocking the event loop.

    Returns:
        Current status, progress, metadata, and error details if failed
    """
    try:
        # Retry logic to handle race condition on newly created data rooms
        # SQLite WAL mode + connection pool may delay visibility of new writes
        data_room = None
        for attempt in range(3):
            data_room = db.get_data_room(data_room_id)
            if data_room:
                break
            if attempt < 2:
                time.sleep(0.1)

        if not data_room:
            raise HTTPException(
                status_code=404,
                detail=f"Data room not found: {data_room_id}"
            )

        if settings.enable_sharing:
            require_data_room_access(
                data_room_id, identity["user_id"], identity["user_email"]
            )

        # Get document count and stats
        documents = db.get_documents_by_data_room(data_room_id)
        parsed_count = sum(1 for doc in documents if doc['parse_status'] == 'parsed')
        failed_count = sum(1 for doc in documents if doc['parse_status'] == 'failed')

        # Get failed documents with their error messages
        failed_documents_details = [
            {
                "file_name": doc['file_name'],
                "error_message": doc.get('error_message', 'Unknown error')
            }
            for doc in documents if doc['parse_status'] == 'failed'
        ]

        # Get recent processing logs for debugging
        processing_logs = db.get_processing_logs(data_room_id=data_room_id, limit=5)
        recent_logs = [
            {
                "stage": log['stage'],
                "status": log['status'],
                "message": log.get('message'),
                "error_details": log.get('error_details'),
                "timestamp": log.get('timestamp')
            }
            for log in processing_logs
        ]

        response = {
            "id": data_room['id'],
            "company_name": data_room['company_name'],
            "analyst_name": data_room['analyst_name'],
            "processing_status": data_room['processing_status'],
            "progress_percent": data_room['progress_percent'],
            "total_documents": data_room['total_documents'],
            "parsed_documents": parsed_count,
            "failed_documents_count": failed_count,
            "created_at": data_room['created_at'],
            "completed_at": data_room.get('completed_at'),
            "actual_cost": data_room.get('actual_cost', 0.0)
        }

        # Include error details if processing failed
        if data_room['processing_status'] == 'failed':
            response['error_message'] = data_room.get('error_message', 'Unknown error - check logs')
            response['failed_documents'] = failed_documents_details
            response['recent_logs'] = recent_logs

        # Always include failed documents if any
        elif failed_count > 0:
            response['failed_documents'] = failed_documents_details

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data-room/{data_room_id}/question")
async def ask_question(
    data_room_id: str,
    request: QuestionRequest,
    identity: Dict = Depends(get_user_identity)
):
    """
    Ask a question about the data room.

    Uses RAG (Retrieval Augmented Generation) to:
    1. Search for relevant chunks
    2. Generate answer with Claude
    3. Return answer with source citations

    Returns:
        Answer with sources and confidence score
    """
    try:
        import time
        start_time = time.time()

        # Verify data room exists
        data_room = db.get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(
                status_code=404,
                detail=f"Data room not found: {data_room_id}"
            )

        if settings.enable_sharing:
            require_data_room_access(
                data_room_id, identity["user_id"], identity["user_email"]
            )

        logger.info(f"Question for {data_room_id}: {request.question}")

        # Import answer_question tool
        from tools.answer_question import answer_question

        # Run blocking Q&A in a thread so we don't block the async event loop.
        # Without this, the synchronous OpenAI + Claude API calls freeze the
        # entire server, making it unresponsive to all other requests.
        result = await asyncio.to_thread(
            answer_question,
            question=request.question,
            data_room_id=data_room_id,
            filters=request.filters,
            model=settings.claude_model,
        )

        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)

        # Save to queries table (non-blocking: don't lose the answer if DB write fails)
        try:
            db.save_query(
                data_room_id=data_room_id,
                question=request.question,
                answer=result['answer'],
                sources=result.get('sources', []),
                confidence_score=result.get('confidence_score'),
                response_time_ms=response_time_ms,
                tokens_used=result.get('tokens_used'),
                cost=result.get('cost'),
                user_id=identity["user_id"]
            )
        except Exception as e:
            logger.error(f"Failed to save query to database (answer still returned): {e}")

        # Normalize confidence key for frontend compatibility
        if 'confidence_score' in result and 'confidence' not in result:
            result['confidence'] = result['confidence_score']

        return result

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Data room not found: {data_room_id}"
        )
    except Exception as e:
        logger.error(f"Failed to answer question: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/data-room/{data_room_id}/questions/{question_id}")
def delete_question(
    data_room_id: str,
    question_id: str,
    identity: Dict = Depends(get_user_identity)
):
    """Delete a question. Only the owner can delete their own question."""
    if not identity.get("user_id"):
        raise HTTPException(status_code=401, detail="User identity required")
    deleted = db.delete_question(question_id, identity["user_id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Question not found or not owned by you")
    return {"status": "deleted"}


@app.post("/api/data-room/{data_room_id}/reembed")
async def reembed_data_room_endpoint(data_room_id: str, background_tasks: BackgroundTasks):
    """
    Re-generate embeddings for chunks that failed embedding generation.
    Runs in the background and indexes results to ChromaDB.
    """
    try:
        # Verify data room exists
        data_room = db.get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(
                status_code=404,
                detail=f"Data room not found: {data_room_id}"
            )

        from tools.ingest_data_room import reembed_data_room
        background_tasks.add_task(reembed_data_room, data_room_id)

        return {
            "status": "reembedding_started",
            "data_room_id": data_room_id,
            "message": "Re-embedding process started in background"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start re-embedding: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data-room/{data_room_id}/reprocess")
async def reprocess_data_room_endpoint(data_room_id: str, background_tasks: BackgroundTasks):
    """
    Re-run the full processing pipeline (parse → chunk → embed → index) for an existing data room.
    Clears existing chunks and re-processes all uploaded files with the current code.
    """
    try:
        # Verify data room exists
        data_room = db.get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(status_code=404, detail=f"Data room not found: {data_room_id}")

        # Get uploaded files
        documents = db.get_documents_by_data_room(data_room_id)
        if not documents:
            raise HTTPException(status_code=400, detail="No documents found in this data room")

        # Build file paths from existing documents
        file_paths = []
        for doc in documents:
            file_path = f"{settings.data_rooms_path}/{data_room_id}/raw/{doc['file_name']}"
            if Path(file_path).exists():
                file_paths.append(file_path)
            else:
                logger.warning(f"[{data_room_id}] File not found for reprocessing: {doc['file_name']}")

        if not file_paths:
            raise HTTPException(status_code=400, detail="No uploaded files found on disk for reprocessing")

        # Clear existing chunks from SQLite
        with db.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE data_room_id = ?", (data_room_id,))
            deleted_count = cursor.rowcount
            conn.commit()
            logger.info(f"[{data_room_id}] Cleared {deleted_count} existing chunks from DB")

        # Clear existing ChromaDB collection
        try:
            from tools.index_to_vectordb import VectorDBIndexer
            indexer = VectorDBIndexer()
            indexer.delete_collection(data_room_id)
            logger.info(f"[{data_room_id}] Cleared ChromaDB collection")
        except Exception as e:
            logger.warning(f"[{data_room_id}] Failed to clear ChromaDB collection (may not exist): {e}")

        # Reset document statuses
        for doc in documents:
            db.update_document_status(doc['id'], "pending")

        # Reset data room status
        db.update_data_room_status(data_room_id, "parsing", progress=0)

        # Trigger background reprocessing
        background_tasks.add_task(process_data_room_background, data_room_id, file_paths)

        return {
            "status": "reprocessing_started",
            "data_room_id": data_room_id,
            "files_to_process": len(file_paths),
            "message": f"Reprocessing {len(file_paths)} files in background"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start reprocessing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-rooms")
def list_data_rooms(
    limit: int = 50,
    identity: Dict = Depends(get_user_identity)
):
    """
    List data rooms accessible to the current user.

    Args:
        limit: Maximum number of data rooms to return

    Returns:
        List of data rooms with metadata
    """
    try:
        if settings.enable_sharing and identity["user_id"] and identity["user_email"]:
            data_rooms = db.get_user_data_rooms(
                user_id=identity["user_id"],
                user_email=identity["user_email"],
                limit=limit
            )
        else:
            data_rooms = db.list_data_rooms(limit=limit)

        return {
            "data_rooms": data_rooms,
            "total": len(data_rooms)
        }

    except Exception as e:
        logger.error(f"Failed to list data rooms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Data Room Sharing Endpoints
# ============================================================================

@app.get("/api/data-room/{data_room_id}/members")
def get_data_room_members(
    data_room_id: str,
    identity: Dict = Depends(get_user_identity)
):
    """List all members of a data room."""
    try:
        data_room = db.get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(status_code=404, detail="Data room not found")

        if settings.enable_sharing:
            require_data_room_access(
                data_room_id, identity["user_id"], identity["user_email"]
            )

        members = db.get_data_room_members(data_room_id)
        return {"members": members, "total": len(members)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get data room members: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data-room/{data_room_id}/invite")
async def invite_data_room_member(
    data_room_id: str,
    request: InviteMemberRequest,
    background_tasks: BackgroundTasks,
    identity: Dict = Depends(get_user_identity)
):
    """Invite a member to a data room by email. Owner only."""
    try:
        data_room = db.get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(status_code=404, detail="Data room not found")

        if settings.enable_sharing:
            require_data_room_access(
                data_room_id, identity["user_id"], identity["user_email"],
                require_owner=True
            )

        # Check if already a member
        existing_members = db.get_data_room_members(data_room_id)
        for member in existing_members:
            if member and member.get("invited_email") == request.email:
                raise HTTPException(
                    status_code=409,
                    detail=f"{request.email} is already a member of this data room"
                )

        member_id = db.add_data_room_member(
            data_room_id=data_room_id,
            invited_email=request.email,
            role='member',
            invited_by=identity["user_id"]
        )

        # Fetch the created member record
        members = db.get_data_room_members(data_room_id)
        new_member = next((m for m in members if m and m.get("id") == member_id), None)

        # Send invite email in background
        inviter = db.get_user_by_id(identity["user_id"]) if identity["user_id"] else None
        inviter_name = (inviter.get("name") if inviter else None) or identity.get("user_email") or "A team member"
        background_tasks.add_task(
            send_invite_email,
            to_email=request.email,
            inviter_name=inviter_name,
            company_name=data_room.get("company_name", "a data room"),
            data_room_id=data_room_id,
        )

        return {
            "member": new_member,
            "message": f"Invited {request.email} to data room"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to invite member: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/data-room/{data_room_id}/members/{member_id}")
async def remove_data_room_member(
    data_room_id: str,
    member_id: str,
    identity: Dict = Depends(get_user_identity)
):
    """Remove a member from a data room. Owner only."""
    try:
        data_room = db.get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(status_code=404, detail="Data room not found")

        if settings.enable_sharing:
            require_data_room_access(
                data_room_id, identity["user_id"], identity["user_email"],
                require_owner=True
            )

        removed = db.revoke_data_room_member(data_room_id, member_id)
        if not removed:
            raise HTTPException(
                status_code=404,
                detail="Member not found or cannot remove owner"
            )

        return {"message": "Member removed from data room"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to remove member: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data-room/{data_room_id}/accept-invite")
async def accept_data_room_invite(
    data_room_id: str,
    identity: Dict = Depends(get_user_identity)
):
    """Accept a pending invite to a data room."""
    try:
        if not identity["user_id"] or not identity["user_email"]:
            raise HTTPException(status_code=401, detail="Authentication required")

        accepted = db.accept_data_room_invite(
            data_room_id, identity["user_id"], identity["user_email"]
        )
        if not accepted:
            raise HTTPException(
                status_code=404,
                detail="No pending invite found for this data room"
            )

        return {"message": "Invite accepted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to accept invite: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/data-room/{data_room_id}")
def delete_data_room_endpoint(
    data_room_id: str,
    identity: Dict = Depends(get_user_identity)
):
    """
    Delete a data room and all associated data.

    Args:
        data_room_id: Data room ID to delete

    Returns:
        Success message
    """
    try:
        # Verify data room exists
        data_room = db.get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(status_code=404, detail="Data room not found")

        # Only owner can delete
        if settings.enable_sharing:
            require_data_room_access(
                data_room_id, identity["user_id"], identity["user_email"],
                require_owner=True
            )

        # Delete ChromaDB collection
        try:
            from tools.index_to_vectordb import get_chroma_client
            client = get_chroma_client()
            collection_name = f"data_room_{data_room_id}"
            client.delete_collection(collection_name)
            logger.info(f"Deleted ChromaDB collection: {collection_name}")
        except Exception as e:
            logger.warning(f"Failed to delete ChromaDB collection: {e}")

        # Delete file system directory
        import shutil
        data_room_path = Path(f"{settings.data_rooms_path}/{data_room_id}")
        if data_room_path.exists():
            shutil.rmtree(data_room_path)
            logger.info(f"Deleted data room directory: {data_room_path}")

        # Delete database records (cascades to documents, chunks, queries, etc.)
        db.delete_data_room(data_room_id)
        logger.info(f"Deleted data room from database: {data_room_id}")

        return {"message": "Data room deleted successfully", "id": data_room_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete data room {data_room_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-room/{data_room_id}/documents")
def get_documents(data_room_id: str):
    """
    Get all documents for a data room.

    Args:
        data_room_id: Data room ID

    Returns:
        List of documents with metadata
    """
    try:
        # Verify data room exists
        data_room = db.get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(
                status_code=404,
                detail=f"Data room not found: {data_room_id}"
            )

        documents = db.get_documents_by_data_room(data_room_id)

        return {
            "data_room_id": data_room_id,
            "documents": documents,
            "total": len(documents)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def build_document_tree(documents: list, target_path: Optional[str] = None) -> dict:
    """
    Build a tree structure from documents with file paths.

    Args:
        documents: List of documents with file_path field
        target_path: If provided, only return children of this path

    Returns:
        Dict with folders, documents at current level, and uploads
    """
    folders = {}
    folder_documents = []
    uploads = []

    for doc in documents:
        file_path = doc.get('file_path')

        if not file_path:
            # Direct upload - no folder path
            uploads.append(doc)
            continue

        # Parse the file path to get folder and filename
        path_parts = file_path.rsplit('/', 1)
        if len(path_parts) == 1:
            # File at root of connected folder
            folder_path = ""
            file_name = path_parts[0]
        else:
            folder_path = path_parts[0]
            file_name = path_parts[1]

        # Check if this document belongs to target path level
        if target_path:
            if folder_path and not folder_path.startswith(target_path):
                continue
            if folder_path == "":
                continue  # Root files don't belong under a target path
            relative = folder_path[len(target_path):].lstrip('/') if folder_path.startswith(target_path) else folder_path
        else:
            relative = folder_path

        if not relative:
            # Document is directly in target folder (or root if no target)
            folder_documents.append(doc)
        else:
            # Document is in a subfolder
            immediate_folder = relative.split('/')[0]
            full_path = f"{target_path}/{immediate_folder}" if target_path else immediate_folder

            if full_path not in folders:
                folders[full_path] = {
                    "name": immediate_folder,
                    "path": full_path,
                    "child_count": 0,
                    "has_subfolders": False
                }

            folders[full_path]["child_count"] += 1

            # Check if there are deeper subfolders
            remaining = relative[len(immediate_folder):].lstrip('/')
            if remaining and '/' in remaining:
                folders[full_path]["has_subfolders"] = True
            elif remaining:
                folders[full_path]["has_subfolders"] = True

    return {
        "folders": sorted(folders.values(), key=lambda f: f["name"].lower()),
        "documents": folder_documents,
        "uploads": uploads if not target_path else []
    }


@app.get("/api/data-room/{data_room_id}/document-tree", response_model=DocumentTreeResponse)
def get_document_tree(
    data_room_id: str,
    folder_path: Optional[str] = None
):
    """
    Get documents organized in a folder hierarchy.

    Args:
        data_room_id: Data room ID
        folder_path: Optional path to get children of a specific folder (for lazy loading)

    Returns:
        Folder tree structure with documents at the requested level
    """
    try:
        # Verify data room exists
        data_room = db.get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(
                status_code=404,
                detail=f"Data room not found: {data_room_id}"
            )

        # Get documents with their folder paths
        documents = db.get_documents_with_paths(data_room_id)

        # Build tree structure
        tree = build_document_tree(documents, folder_path)

        return DocumentTreeResponse(
            data_room_id=data_room_id,
            current_path=folder_path,
            folders=[FolderNode(**f) for f in tree["folders"]],
            documents=[DocumentWithPath(**d) for d in tree["documents"]],
            uploads=[DocumentWithPath(**d) for d in tree["uploads"]],
            total_documents=len(documents)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document tree: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-room/{data_room_id}/document/{document_id}/file")
async def get_document_file(
    data_room_id: str,
    document_id: str,
    download: bool = False
):
    """
    Stream a document file for viewing or download.

    Args:
        data_room_id: Data room ID
        document_id: Document ID
        download: If True, sets Content-Disposition to attachment for download

    Returns:
        FileResponse with the document file
    """
    try:
        # Verify data room exists
        data_room = db.get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(
                status_code=404,
                detail=f"Data room not found: {data_room_id}"
            )

        # Get document and verify it belongs to this data room
        documents = db.get_documents_by_data_room(data_room_id)
        document = None
        for doc in documents:
            if doc['id'] == document_id:
                document = doc
                break

        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )

        # Build file path — use stored path for Drive-synced files, fall back to raw/ for uploads
        stored_path = document.get('file_path')
        if stored_path:
            file_path = Path(stored_path)
        else:
            file_path = Path(f"{settings.data_rooms_path}/{data_room_id}/raw/{document['file_name']}")

        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found on disk: {document['file_name']}"
            )

        # Determine MIME type
        mime_types = {
            '.pdf': 'application/pdf',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.csv': 'text/csv',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.txt': 'text/plain',
        }

        suffix = file_path.suffix.lower()
        media_type = mime_types.get(suffix, 'application/octet-stream')

        # Set content disposition
        disposition = 'attachment' if download else 'inline'

        return FileResponse(
            path=str(file_path),
            media_type=media_type,
            filename=document['file_name'],
            headers={
                'Content-Disposition': f'{disposition}; filename="{document["file_name"]}"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-room/{data_room_id}/document/{document_id}/preview", response_model=DocumentPreview)
def get_document_preview(
    data_room_id: str,
    document_id: str,
    sheet: Optional[str] = None,
    max_rows: int = 100
):
    """
    Get preview data for spreadsheet documents (Excel, CSV).

    Args:
        data_room_id: Data room ID
        document_id: Document ID
        sheet: Sheet name for Excel files (optional, defaults to first sheet)
        max_rows: Maximum rows to return (default 100)

    Returns:
        DocumentPreview with headers, rows, and sheet information
    """
    try:
        import pandas as pd

        # Verify data room exists
        data_room = db.get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(
                status_code=404,
                detail=f"Data room not found: {data_room_id}"
            )

        # Get document and verify it belongs to this data room
        documents = db.get_documents_by_data_room(data_room_id)
        document = None
        for doc in documents:
            if doc['id'] == document_id:
                document = doc
                break

        if not document:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )

        # Build file path — use stored path for Drive-synced files, fall back to raw/ for uploads
        stored_path = document.get('file_path')
        if stored_path:
            file_path = Path(stored_path)
        else:
            file_path = Path(f"{settings.data_rooms_path}/{data_room_id}/raw/{document['file_name']}")

        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found on disk: {document['file_name']}"
            )

        suffix = file_path.suffix.lower()

        # Handle different file types
        if suffix in ['.xlsx', '.xls']:
            # Excel file - get sheet names and data
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names

            # Use specified sheet or default to first
            current_sheet = sheet if sheet and sheet in sheet_names else sheet_names[0]

            # Read the sheet
            df = pd.read_excel(file_path, sheet_name=current_sheet)

        elif suffix == '.csv':
            # CSV file - no sheets
            df = pd.read_csv(file_path)
            sheet_names = []
            current_sheet = None

        else:
            # Unsupported file type for preview
            return DocumentPreview(
                file_name=document['file_name'],
                file_type=suffix.lstrip('.'),
                error=f"Preview not supported for {suffix} files. Use the file endpoint to download."
            )

        # Get total rows before limiting
        total_rows = len(df)

        # Limit rows
        df_limited = df.head(max_rows)

        # Convert to list format, handling NaN values
        df_limited = df_limited.fillna('')
        headers = [str(col) for col in df_limited.columns.tolist()]
        rows = df_limited.values.tolist()

        # Convert any remaining non-serializable types to strings
        rows = [[str(cell) if not isinstance(cell, (str, int, float, bool)) else cell for cell in row] for row in rows]

        return DocumentPreview(
            file_name=document['file_name'],
            file_type=suffix.lstrip('.'),
            sheets=sheet_names if suffix in ['.xlsx', '.xls'] else [],
            current_sheet=current_sheet,
            headers=headers,
            rows=rows,
            total_rows=total_rows,
            preview_rows=len(rows),
            has_more=total_rows > max_rows
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-room/{data_room_id}/questions")
def get_question_history(
    data_room_id: str,
    limit: int = 50,
    filter: Optional[str] = Query(None, description="Filter: 'mine' for user's questions, 'team' or omit for all"),
    identity: Dict = Depends(get_user_identity)
):
    """
    Get Q&A history for a data room.

    Args:
        data_room_id: Data room ID
        limit: Maximum number of questions to return
        filter: 'mine' to show only user's questions, 'team' or omit for all

    Returns:
        List of questions and answers
    """
    try:
        # Verify data room exists
        data_room = db.get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(
                status_code=404,
                detail=f"Data room not found: {data_room_id}"
            )

        if settings.enable_sharing:
            require_data_room_access(
                data_room_id, identity["user_id"], identity["user_email"]
            )

        # Apply user filter
        filter_user_id = None
        if filter == 'mine' and identity["user_id"]:
            filter_user_id = identity["user_id"]

        questions = db.get_query_history(
            data_room_id, limit=limit, filter_user_id=filter_user_id
        )

        return {
            "data_room_id": data_room_id,
            "questions": questions,
            "total": len(questions)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get questions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Memo Endpoints

def _run_memo_generation(memo_id: str, data_room_id: str, deal_params: Optional[dict] = None):
    """Background task to generate an investment memo."""
    try:
        from tools.memo_generator import MemoGenerator
        from app.database import update_memo_section, update_memo_status, check_memo_cancelled, update_memo_metadata

        generator = MemoGenerator(data_room_id)
        generator.generate_memo(
            memo_id=memo_id,
            save_section_fn=update_memo_section,
            update_status_fn=update_memo_status,
            deal_params=deal_params,
            check_cancelled_fn=check_memo_cancelled,
            save_metadata_fn=update_memo_metadata,
        )
    except Exception as e:
        logger.error(f"Memo generation failed for {memo_id}: {e}")
        try:
            from app.database import update_memo_status
            update_memo_status(memo_id, "failed")
        except Exception:
            pass


@app.post("/api/data-room/{data_room_id}/memo/generate")
async def generate_memo(
    data_room_id: str,
    background_tasks: BackgroundTasks,
    request: Optional[MemoGenerateRequest] = None,
    identity: Dict = Depends(get_user_identity)
):
    """Generate investment memo for data room."""
    try:
        from app.database import get_data_room, save_memo

        data_room = get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(status_code=404, detail="Data room not found")

        if settings.enable_sharing:
            require_data_room_access(
                data_room_id, identity["user_id"], identity["user_email"]
            )

        if data_room.get("processing_status") != "complete":
            raise HTTPException(status_code=400, detail="Data room processing not complete")

        # Extract deal parameters from request
        deal_params = None
        ticket_size = None
        post_money_valuation = None
        valuation_methods = None
        if request:
            ticket_size = request.ticket_size
            post_money_valuation = request.post_money_valuation
            valuation_methods = request.valuation_methods
            deal_params = {
                "ticket_size": ticket_size,
                "post_money_valuation": post_money_valuation,
                "valuation_methods": valuation_methods,
            }
            # Remove None values
            deal_params = {k: v for k, v in deal_params.items() if v is not None}
            if not deal_params:
                deal_params = None

        memo_id = f"memo_{uuid.uuid4().hex[:12]}"
        save_memo(memo_id, data_room_id, ticket_size=ticket_size, post_money_valuation=post_money_valuation, valuation_methods=valuation_methods)

        logger.info(f"Generating memo {memo_id} for data room {data_room_id}, deal_params={deal_params}")
        background_tasks.add_task(_run_memo_generation, memo_id, data_room_id, deal_params)

        return {
            "memo_id": memo_id,
            "data_room_id": data_room_id,
            "status": "generating",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate memo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _serialize_memo(memo: dict) -> dict:
    """Parse JSON text fields in a memo dict so the response is properly typed."""
    if not memo:
        return memo
    for field in ("metadata", "valuation_methods"):
        raw = memo.get(field)
        if raw and isinstance(raw, str):
            try:
                memo[field] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass
    return memo


@app.get("/api/data-room/{data_room_id}/memo")
def get_memo(data_room_id: str):
    """Get the latest memo for a data room."""
    try:
        from app.database import get_latest_memo

        memo = get_latest_memo(data_room_id)
        return {"data_room_id": data_room_id, "memo": _serialize_memo(memo) if memo else None}

    except Exception as e:
        logger.error(f"Failed to get memo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-room/{data_room_id}/memo/{memo_id}/status")
def get_memo_status(data_room_id: str, memo_id: str):
    """Get memo generation status by ID."""
    try:
        from app.database import get_memo_by_id

        memo = get_memo_by_id(memo_id)
        if not memo:
            raise HTTPException(status_code=404, detail="Memo not found")
        return _serialize_memo(memo)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memo status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data-room/{data_room_id}/memo/{memo_id}/cancel")
async def cancel_memo_generation(data_room_id: str, memo_id: str):
    """Cancel an in-progress memo generation."""
    try:
        from app.database import get_memo_by_id, update_memo_status

        memo = get_memo_by_id(memo_id)
        if not memo:
            raise HTTPException(status_code=404, detail="Memo not found")
        if memo.get("data_room_id") != data_room_id:
            raise HTTPException(status_code=404, detail="Memo not found for this data room")
        if memo.get("status") != "generating":
            raise HTTPException(status_code=400, detail="Memo is not currently generating")

        update_memo_status(memo_id, "cancelled")
        logger.info(f"Memo {memo_id} marked as cancelled")

        return {"memo_id": memo_id, "status": "cancelled"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel memo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _needs_document_context(message: str) -> bool:
    """Check if the message needs data room document context.

    Broadly inclusive — almost all analytical questions benefit from source docs.
    Only skip for very short, purely cosmetic requests (e.g., 'change color').
    """
    message_lower = message.lower()

    # Skip document search for purely cosmetic / chart-styling requests
    cosmetic_only = [
        'change color', 'change the color', 'make it green', 'make it blue',
        'use green', 'use blue', 'change font', 'change style',
    ]
    if any(phrase in message_lower for phrase in cosmetic_only):
        return False

    # For everything else, default to including document context
    return True


@app.post("/api/data-room/{data_room_id}/memo/{memo_id}/chat")
async def memo_chat(data_room_id: str, memo_id: str, request: MemoChatRequest):
    """Chat with the investment memo — ask questions or request edits to sections."""
    import re

    try:
        from app.database import get_memo_by_id, update_memo_section, save_memo_chat_message, update_memo_metadata
        from anthropic import Anthropic
        import json as _json

        memo = get_memo_by_id(memo_id)
        if not memo:
            raise HTTPException(status_code=404, detail="Memo not found")
        if memo.get("data_room_id") != data_room_id:
            raise HTTPException(status_code=404, detail="Memo not found for this data room")

        # Save user message to chat history
        save_memo_chat_message(
            memo_id=memo_id,
            data_room_id=data_room_id,
            role="user",
            content=request.message
        )

        # Build memo context from all sections
        section_keys = [
            ("proposed_investment_terms", "Proposed Investment Terms"),
            ("executive_summary", "Executive Summary"),
            ("market_analysis", "Market Analysis"),
            ("team_assessment", "Team Assessment"),
            ("product_technology", "Product & Technology"),
            ("financial_analysis", "Financial Analysis"),
            ("valuation_analysis", "Valuation Analysis"),
            ("risks_concerns", "Risks & Concerns"),
            ("outcome_scenario_analysis", "Outcome Scenario Analysis"),
            ("investment_recommendation", "Investment Recommendation"),
        ]
        memo_context_parts = []
        for key, label in section_keys:
            content = memo.get(key)
            if content:
                memo_context_parts.append(f"## {label}\n\n{content}")
        memo_context = "\n\n---\n\n".join(memo_context_parts)

        # Build chart context from metadata
        chart_context = ""
        current_charts = None
        metadata_raw = memo.get("metadata")
        if metadata_raw:
            try:
                metadata = _json.loads(metadata_raw) if isinstance(metadata_raw, str) else metadata_raw
                current_charts = metadata.get("chart_data", {})
                if current_charts and current_charts.get("charts"):
                    chart_context = (
                        "\n\n---\n\nCurrent Charts (JSON spec):\n\n```json\n"
                        + _json.dumps(current_charts, indent=2)
                        + "\n```"
                    )
            except (ValueError, TypeError, AttributeError):
                pass

        # Semantic search for document context when needed
        document_context = ""
        if _needs_document_context(request.message):
            try:
                from tools.semantic_search import semantic_search
                search_results = semantic_search(
                    query=request.message,
                    data_room_id=data_room_id,
                    top_k=10
                )
                if search_results:
                    doc_parts = []
                    for i, chunk in enumerate(search_results, 1):
                        source = chunk.get('source', {})
                        fn = source.get('file_name', 'Unknown')
                        page = source.get('page_number')
                        loc = f", p.{page}" if page else ""
                        doc_parts.append(f"[Document {i}: {fn}{loc}]\n{chunk['chunk_text']}")
                    document_context = "\n\n---\n\nRelevant Data Room Documents:\n\n" + "\n\n---\n\n".join(doc_parts)
            except Exception as e:
                logger.warning(f"Failed to retrieve document context: {e}")

        has_docs = bool(document_context)
        has_charts = bool(chart_context)
        system_prompt = (
            "You are an AI analyst assistant for a VC firm. You have full access to the investment memo "
            "and the data room documents for the deal under review. "
            "Help the analyst with whatever they need: answer questions, run analyses, create tables, "
            "build charts, compute metrics, compare scenarios, or edit the memo.\n\n"
            "Be conversational and proactive — if the analyst asks a question, give a thorough, "
            "data-driven answer with specific numbers from the memo and source documents. "
            "Use markdown formatting (tables, bold, bullet points) to make your responses clear.\n\n"

            "## Editing Memo Sections\n"
            "If the user requests ANY change, addition, or update to the memo, you MUST immediately "
            "produce the updated section content — do NOT just describe what you would do or say you will add it. "
            "Actually write the content and output it in the required tag.\n\n"
            "Steps:\n"
            "1. Briefly explain what you changed (1-2 sentences)\n"
            "2. Output the COMPLETE updated section in this EXACT tag format:\n\n"
            "<updated_section key=\"section_key\">\n"
            "[Full updated section content - the ENTIRE section, not just changes]\n"
            "</updated_section>\n\n"
            "Valid section keys: proposed_investment_terms, executive_summary, market_analysis, team_assessment, "
            "product_technology, financial_analysis, valuation_analysis, risks_concerns, outcome_scenario_analysis, "
            "investment_recommendation.\n"
            "CRITICAL: The <updated_section> tag is REQUIRED for any modification — without it, the memo won't update. "
            "Never just say you will update — always include the tag with the full content.\n\n"

            "## Charts & Visualizations\n"
            + (
                "The memo currently has charts (specs shown below). "
                if has_charts else
                "The memo does not currently have any charts. "
            )
            + "If the user asks to create a chart, add a visualization, modify an existing chart, "
            "change colors, switch chart types, add/remove data, or anything chart-related, you MUST:\n"
            "1. Briefly explain what you're creating or changing\n"
            "2. Output the COMPLETE charts JSON in this tag:\n\n"
            "<updated_charts>\n"
            "{\"charts\": [... full charts array ...]}\n"
            "</updated_charts>\n\n"
            "Chart spec fields:\n"
            "- id: unique string (e.g. \"revenue\", \"users\", \"margins\")\n"
            "- title: chart title\n"
            "- type: \"bar\" | \"horizontal_bar\" | \"line\"\n"
            "- x_key, y_key: data keys for axes\n"
            "- x_label, y_label: axis labels (empty string if not needed)\n"
            "- y_format: \"currency\" | \"number\" | \"percent\"\n"
            "- color_key: data key for coloring (empty string if not needed)\n"
            "- colors: object mapping color_key values to hex, OR single hex string\n"
            "- data: array of data point objects with keys matching x_key, y_key, and color_key\n"
            "Use indigo palette: #6366f1 primary, #a5b4fc light, #4f46e5 dark, #818cf8 medium.\n"
            "All numeric values must be raw numbers, not formatted strings.\n"
            + (
                "When modifying charts, include ALL existing charts in the array (not just the changed one), "
                "otherwise unchanged charts will be lost.\n"
                "If the user asks to convert a chart into a table, REMOVE that chart from the charts array "
                "in <updated_charts> and add the table as markdown in the appropriate section via <updated_section>.\n\n"
                if has_charts else "\n"
            )
            + "## REMINDER\n"
            "NEVER just describe or announce what you would change. ALWAYS output the actual <updated_section> "
            "and/or <updated_charts> tags with full content. If you don't include the tags, nothing will update.\n"
        )

        user_prompt = f"Current Investment Memo:\n\n{memo_context}{chart_context}{document_context}\n\n---\n\nUser: {request.message}"

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        # Run blocking API call in a thread to avoid freezing the event loop
        response = await asyncio.to_thread(
            client.messages.create,
            model=settings.claude_model,
            max_tokens=16000,
            temperature=0.3,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        answer_text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens

        # Calculate cost
        pricing = {"input": 3.0, "output": 15.0}
        cost = (input_tokens * pricing["input"] / 1_000_000) + (
            output_tokens * pricing["output"] / 1_000_000
        )

        # Parse for updated section
        updated_section = None
        tag_match = re.search(
            r'<updated_section\s+key="(\w+)">\s*(.*?)\s*</updated_section>',
            answer_text,
            re.DOTALL,
        )
        if tag_match:
            section_key = tag_match.group(1)
            section_content = tag_match.group(2).strip()
            valid_keys = [k for k, _ in section_keys]
            if section_key in valid_keys:
                update_memo_section(memo_id, section_key, section_content, total_tokens, cost)
                updated_section = {"key": section_key, "content": section_content}
                # Strip the tag from the displayed answer
                answer_text = answer_text[:tag_match.start()].strip()

        # Parse for updated charts
        updated_charts = None
        charts_match = re.search(
            r'<updated_charts>\s*(.*?)\s*</updated_charts>',
            answer_text,
            re.DOTALL,
        )
        if charts_match:
            try:
                charts_json = _json.loads(charts_match.group(1).strip())
                if isinstance(charts_json, dict) and "charts" in charts_json:
                    from tools.memo_generator import _validate_and_fix_chart_specs
                    charts_json = _validate_and_fix_chart_specs(charts_json)
                    if charts_json:
                        update_memo_metadata(memo_id, {"chart_data": charts_json})
                        updated_charts = charts_json
                    # Strip the tag from the displayed answer
                    answer_text = answer_text[:charts_match.start()].strip()
                    if not answer_text and tag_match:
                        # If there was also a section update, answer is already set
                        pass
            except (_json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse updated charts JSON: {e}")

        # Save assistant message to chat history
        save_memo_chat_message(
            memo_id=memo_id,
            data_room_id=data_room_id,
            role="assistant",
            content=answer_text,
            updated_section_key=updated_section["key"] if updated_section else None,
            updated_section_content=updated_section["content"] if updated_section else None,
            tokens_used=total_tokens,
            cost=cost
        )

        result = {
            "answer": answer_text,
            "tokens_used": total_tokens,
            "cost": round(cost, 6),
        }
        if updated_section:
            result["updated_section"] = updated_section
        if updated_charts:
            result["updated_charts"] = updated_charts

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Memo chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-room/{data_room_id}/memo/{memo_id}/chat-history")
def get_memo_chat_history(data_room_id: str, memo_id: str, limit: int = 100):
    """Get chat history for a memo."""
    try:
        from app.database import get_memo_by_id, get_memo_chat_history

        memo = get_memo_by_id(memo_id)
        if not memo:
            raise HTTPException(status_code=404, detail="Memo not found")
        if memo.get("data_room_id") != data_room_id:
            raise HTTPException(status_code=404, detail="Memo not found for this data room")

        messages = get_memo_chat_history(memo_id, limit=limit)

        return {
            "memo_id": memo_id,
            "messages": messages,
            "total": len(messages)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get memo chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _rewrite_section_with_deal_terms(
    existing_content: str,
    ticket_size: float | None,
    post_money_valuation: float | None,
) -> str | None:
    """
    Rewrite the proposed_investment_terms section with updated deal term numbers.
    Uses Claude for a fast rewrite. Returns updated markdown or None on failure.
    """
    from anthropic import Anthropic

    deal_parts = []
    if ticket_size:
        deal_parts.append(f"- Investment Amount (Ticket Size): ${ticket_size:,.0f}")
    if post_money_valuation:
        deal_parts.append(f"- Post-Money Valuation: ${post_money_valuation:,.0f}")
    if ticket_size and post_money_valuation:
        ownership_pct = (ticket_size / post_money_valuation) * 100
        deal_parts.append(f"- Implied Ownership: {ownership_pct:.2f}%")

    if not deal_parts:
        return None

    deal_params_text = "\n".join(deal_parts)

    prompt = (
        'You are updating the "Proposed Investment Terms" section of a VC investment memo.\n\n'
        f"The analyst has changed the deal parameters to:\n{deal_params_text}\n\n"
        f"Here is the current section content:\n\n{existing_content}\n\n"
        "Rewrite this section with the updated deal parameters. Rules:\n"
        "1. Keep the EXACT same structure, formatting, and markdown table layout\n"
        "2. Update ALL references to investment amount, ticket size, post-money valuation, and ownership percentage to match the new values\n"
        "3. Do NOT add new rows, remove existing rows, or change the analysis/commentary\n"
        "4. Do NOT add any preamble or explanation - output ONLY the updated section content\n"
        "5. If the section contains a markdown table, preserve the table format exactly, only changing the numeric values"
    )

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = await asyncio.to_thread(
        client.messages.create,
        model=settings.claude_model,
        max_tokens=2048,
        temperature=0.1,
        messages=[{"role": "user", "content": prompt}],
    )

    if response.content and response.content[0].text.strip():
        return response.content[0].text.strip()

    return None


@app.put("/api/data-room/{data_room_id}/memo/{memo_id}/deal-terms")
async def update_deal_terms(data_room_id: str, memo_id: str, request: dict):
    """Update deal terms on an existing memo and regenerate the proposed terms section."""
    try:
        from app.database import get_memo_by_id, update_memo_deal_terms, update_memo_section

        memo = get_memo_by_id(memo_id)
        if not memo:
            raise HTTPException(status_code=404, detail="Memo not found")
        if memo.get("data_room_id") != data_room_id:
            raise HTTPException(status_code=404, detail="Memo not found for this data room")

        ticket_size = request.get("ticket_size")
        post_money_valuation = request.get("post_money_valuation")

        update_memo_deal_terms(memo_id, ticket_size, post_money_valuation)

        # Regenerate proposed_investment_terms section with new deal values
        existing_section = memo.get("proposed_investment_terms")
        if existing_section and existing_section.strip():
            try:
                new_content = await _rewrite_section_with_deal_terms(
                    existing_content=existing_section,
                    ticket_size=ticket_size,
                    post_money_valuation=post_money_valuation,
                )
                if new_content:
                    update_memo_section(memo_id, "proposed_investment_terms", new_content, 0, 0.0)
            except Exception as e:
                logger.warning(f"Failed to regenerate proposed_investment_terms: {e}")

        # Return updated memo
        updated_memo = get_memo_by_id(memo_id)
        return updated_memo

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update deal terms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-room/{data_room_id}/memo/{memo_id}/export")
async def export_memo_docx(data_room_id: str, memo_id: str):
    """Export memo as a downloadable DOCX file."""
    from fastapi.responses import StreamingResponse
    from io import BytesIO

    try:
        from app.database import get_memo_by_id, get_data_room
        from tools.memo_exporter import generate_memo_docx

        # Get memo
        memo = get_memo_by_id(memo_id)
        if not memo:
            raise HTTPException(status_code=404, detail="Memo not found")
        if memo.get("data_room_id") != data_room_id:
            raise HTTPException(status_code=404, detail="Memo not found for this data room")

        if memo.get("status") != "complete":
            raise HTTPException(status_code=400, detail="Memo is not complete yet")

        # Get data room for company name
        data_room = get_data_room(data_room_id)
        company_name = data_room.get("company_name", "Company") if data_room else "Company"

        # Generate DOCX
        docx_buffer = generate_memo_docx(memo, company_name)

        # Create filename
        safe_company_name = "".join(c for c in company_name if c.isalnum() or c in (' ', '-', '_')).strip()
        filename = f"{safe_company_name}_Investment_Memo.docx"

        return StreamingResponse(
            docx_buffer,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export memo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Utility Endpoints

@app.get("/api/costs")
def get_costs(days: int = 30, data_room_id: Optional[str] = None):
    """
    Get API cost report.

    Args:
        days: Number of days to include in report
        data_room_id: Optional filter by data room

    Returns:
        Cost breakdown by provider, model, and data room
    """
    try:
        cost_summary = db.get_api_costs(days=days, data_room_id=data_room_id)

        return {
            "period_days": days,
            "total_cost": cost_summary.get('total_cost', 0.0) or 0.0,
            "total_input_tokens": cost_summary.get('total_input_tokens', 0) or 0,
            "total_output_tokens": cost_summary.get('total_output_tokens', 0) or 0,
            "total_calls": cost_summary.get('total_calls', 0) or 0,
            "by_provider": cost_summary.get('by_provider', [])
        }

    except Exception as e:
        logger.error(f"Failed to get costs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/collections")
async def list_collections():
    """
    List all data room collections in vector DB.

    Returns:
        List of collection names and stats
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        collections = client.list_collections()

        return {
            "collections": [
                {
                    "name": c.name,
                    "count": c.count(),
                    "metadata": c.metadata
                }
                for c in collections
            ],
            "total": len(collections)
        }

    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Google OAuth & Drive Endpoints
# ============================================================================

# TTL-based state storage for OAuth (auto-expires entries to prevent memory leaks)
class TTLDict:
    """Dict with automatic TTL-based expiration."""
    def __init__(self, ttl_seconds: int = 600):
        self._store = {}  # key -> (value, timestamp)
        self.ttl = ttl_seconds

    def __setitem__(self, key, value):
        self._store[key] = (value, time.time())

    def __contains__(self, key):
        self._cleanup()
        return key in self._store

    def pop(self, key, default=None):
        self._cleanup()
        if key in self._store:
            val, _ = self._store.pop(key)
            return val
        return default

    def _cleanup(self):
        now = time.time()
        expired = [k for k, (_, ts) in self._store.items() if now - ts > self.ttl]
        for k in expired:
            del self._store[k]

_oauth_states = TTLDict(ttl_seconds=settings.oauth_state_ttl_seconds)


# Per-user rate limiter for Google Drive API endpoints
class EndpointRateLimiter:
    """Per-user rate limiter for API endpoints."""
    def __init__(self, max_per_minute: int = 30):
        self.max_per_minute = max_per_minute
        self._buckets = {}  # user_id -> list of timestamps

    def check(self, user_id: str):
        now = time.time()
        bucket = self._buckets.get(user_id, [])
        self._buckets[user_id] = [t for t in bucket if now - t < 60]
        if len(self._buckets[user_id]) >= self.max_per_minute:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
        self._buckets[user_id].append(now)

_drive_rate_limiter = EndpointRateLimiter(max_per_minute=settings.drive_api_rate_limit_per_minute)


@app.get("/api/auth/google/login", response_model=GoogleAuthURL)
async def google_auth_login(request: Request):
    """
    Initiate Google OAuth login flow.

    Returns:
        Authorization URL and state for CSRF protection
    """
    try:
        # Build redirect URI from the actual request so it works in both local and deployed envs
        # Prefer X-Forwarded-Host/Proto (set by Railway/reverse proxies), fall back to Host header
        forwarded_host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
        forwarded_proto = request.headers.get("x-forwarded-proto", "https" if forwarded_host and "localhost" not in forwarded_host else "http")
        if forwarded_host:
            base_url = f"{forwarded_proto}://{forwarded_host}"
        else:
            # Last resort: try origin/referer headers
            origin = request.headers.get("origin") or request.headers.get("referer", "")
            if origin:
                from urllib.parse import urlparse
                parsed = urlparse(origin)
                base_url = f"{parsed.scheme}://{parsed.netloc}"
            else:
                base_url = f"http://localhost:{settings.port}"
        redirect_uri = f"{base_url}/api/auth/google/callback"
        result = google_oauth_service.create_auth_url(redirect_uri)

        # Store state for verification
        _oauth_states[result['state']] = {
            'redirect_uri': redirect_uri,
            'base_url': base_url,
            'created_at': datetime.now().isoformat()
        }

        return GoogleAuthURL(
            auth_url=result['auth_url'],
            state=result['state']
        )

    except Exception as e:
        logger.error(f"Failed to create auth URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/auth/google/callback")
async def google_auth_callback(
    request: Request,
    code: str = Query(...),
    state: str = Query(...)
):
    """
    Handle Google OAuth callback.

    Args:
        code: Authorization code from Google
        state: State parameter for CSRF verification

    Returns:
        Redirects to frontend with user info
    """
    try:
        # Verify state
        if state not in _oauth_states:
            raise HTTPException(status_code=400, detail="Invalid state parameter")

        stored_state = _oauth_states.pop(state)
        redirect_uri = stored_state['redirect_uri']

        # Exchange code for tokens
        result = google_oauth_service.exchange_code_for_tokens(code, redirect_uri)

        user_info = result.get('user_info', {})
        email = user_info.get('email')

        if not email:
            raise HTTPException(status_code=400, detail="Email not provided by Google")

        # Create or update user
        user_id = db.create_or_update_user(
            email=email,
            name=user_info.get('name'),
            picture_url=user_info.get('picture'),
            google_id=user_info.get('id'),
            access_token=result['access_token'],
            refresh_token=result.get('refresh_token'),
            token_expires_at=result.get('expires_at')
        )

        logger.success(f"User authenticated: {email}")

        # Auto-accept any pending data room invites for this user
        if settings.enable_sharing:
            try:
                accepted_count = db.auto_accept_pending_invites(user_id, email)
                if accepted_count > 0:
                    logger.info(f"Auto-accepted {accepted_count} pending invites for {email}")
            except Exception as invite_err:
                logger.warning(f"Failed to auto-accept invites: {invite_err}")

        # Redirect to frontend — use the base_url stored during login initiation
        base_url = stored_state.get('base_url', f"{request.url.scheme}://{request.url.netloc}")
        frontend_url = f"{base_url}/auth/callback?user_id={user_id}&email={email}"
        return RedirectResponse(url=frontend_url)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OAuth callback failed: {e}")
        # Redirect to frontend with error
        base_url = f"{request.url.scheme}://{request.url.netloc}"
        return RedirectResponse(url=f"{base_url}/auth/callback?error={str(e)}")


@app.get("/api/auth/user/{user_id}", response_model=UserInfo)
async def get_user(user_id: str):
    """
    Get user information.

    Args:
        user_id: User ID

    Returns:
        User information
    """
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return UserInfo(
        id=user['id'],
        email=user['email'],
        name=user.get('name'),
        picture_url=user.get('picture_url'),
        created_at=user.get('created_at')
    )


@app.post("/api/auth/logout/{user_id}")
async def logout_user(user_id: str):
    """
    Logout user and revoke tokens.

    Args:
        user_id: User ID to logout
    """
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Revoke Google token
    if user.get('access_token'):
        google_oauth_service.revoke_token(user['access_token'])

    # Clear tokens in database
    db.update_user_tokens(user_id, access_token="", refresh_token="")

    return {"message": "Logged out successfully"}


# ============================================================================
# Google Drive Browsing Endpoints
# ============================================================================

def get_drive_service(user_id: str) -> GoogleDriveService:
    """Get authenticated Drive service for a user."""
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not user.get('access_token'):
        raise HTTPException(status_code=401, detail="User not authenticated with Google")

    return GoogleDriveService.from_tokens(
        access_token=user['access_token'],
        refresh_token=user.get('refresh_token'),
        expires_at=user.get('token_expires_at')
    )


@app.get("/api/drive/{user_id}/files", response_model=DriveFileList)
async def list_drive_files(
    user_id: str,
    folder_id: Optional[str] = None,
    page_token: Optional[str] = None,
    page_size: int = 50,
    view_mode: str = "my_drive",
    search_query: Optional[str] = None
):
    """
    List files in a Google Drive folder or shared files.

    Args:
        user_id: User ID
        folder_id: Folder ID (None for root/My Drive, ignored for shared_with_me)
        page_token: Pagination token
        page_size: Number of items per page
        view_mode: "my_drive" for user's files, "shared_with_me" for shared files
        search_query: Optional search term to filter by filename

    Returns:
        List of files and folders
    """
    try:
        _drive_rate_limiter.check(user_id)
        drive_service = get_drive_service(user_id)

        result = drive_service.list_files(
            folder_id=folder_id,
            page_size=page_size,
            page_token=page_token,
            include_folders=True,
            view_mode=view_mode,
            search_query=search_query
        )

        # Get folder path for breadcrumb (for both My Drive and shared folder navigation)
        folder_path = None
        if folder_id and not search_query:
            try:
                folder_path = drive_service.get_folder_path(folder_id)
            except Exception as e:
                logger.warning(f"Could not get folder path for {folder_id}: {e}")
                # For shared folders, we may not have access to parent folders
                folder_path = None

        return DriveFileList(
            files=[DriveFile(**f) for f in result['files']],
            nextPageToken=result.get('nextPageToken'),
            totalFiles=result['totalFiles'],
            folderPath=folder_path
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list drive files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/drive/{user_id}/file/{file_id}")
async def get_drive_file_info(user_id: str, file_id: str):
    """
    Get detailed information about a Drive file.

    Args:
        user_id: User ID
        file_id: Google Drive file ID

    Returns:
        File metadata
    """
    try:
        _drive_rate_limiter.check(user_id)
        drive_service = get_drive_service(user_id)
        file_info = drive_service.get_file_info(file_id)
        return file_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get file info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/drive/{user_id}/file/{file_id}/content")
async def get_drive_file_content(
    user_id: str,
    file_id: str,
    download: bool = False
):
    """
    Stream file content from Google Drive.

    Args:
        user_id: User ID
        file_id: Google Drive file ID
        download: If true, triggers browser download instead of inline display

    Returns:
        File content as streaming response
    """
    try:
        import tempfile as _tempfile
        _drive_rate_limiter.check(user_id)
        drive_service = get_drive_service(user_id)

        # Stream to a temp file instead of loading into memory
        tmp = _tempfile.NamedTemporaryFile(delete=False, suffix='.tmp')
        tmp_path = tmp.name
        tmp.close()

        result = drive_service.download_file_to_disk(file_id, tmp_path)
        filename = result['filename']
        mime_type = result['mimeType']

        content_disposition = f'{"attachment" if download else "inline"}; filename="{filename}"'

        return FileResponse(
            tmp_path,
            media_type=mime_type,
            filename=filename,
            headers={"Content-Disposition": content_disposition},
            background=BackgroundTasks()  # FileResponse auto-cleans temp files
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get file content: {e}")
        # Clean up temp file on error
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/drive/{user_id}/file/{file_id}/preview", response_model=DocumentPreview)
async def get_drive_file_preview(
    user_id: str,
    file_id: str,
    sheet: Optional[str] = None,
    max_rows: int = 100
):
    """
    Get preview data for spreadsheet files from Google Drive.

    Args:
        user_id: User ID
        file_id: Google Drive file ID
        sheet: Sheet name for Excel files (optional, defaults to first sheet)
        max_rows: Maximum rows to return (default 100)

    Returns:
        DocumentPreview with headers, rows, and sheet information
    """
    try:
        import pandas as pd
        import tempfile

        _drive_rate_limiter.check(user_id)
        drive_service = get_drive_service(user_id)

        # Get file info first to check type
        file_info = drive_service.get_file_info(file_id)
        filename = file_info['name']
        mime_type = file_info['mimeType']

        # Determine file extension
        extension = file_info.get('extension', '')
        if not extension:
            if 'spreadsheet' in mime_type or 'excel' in mime_type:
                extension = '.xlsx'
            elif 'csv' in mime_type:
                extension = '.csv'
            else:
                # Extract from filename
                extension = Path(filename).suffix.lower()

        # Check if preview is supported
        if extension not in ['.xlsx', '.xls', '.csv']:
            return DocumentPreview(
                file_name=filename,
                file_type=extension.lstrip('.'),
                error=f"Preview not supported for {extension} files. Use the content endpoint to download."
            )

        # Download file to memory
        file_data = drive_service.download_file_to_bytes(file_id)
        content = file_data['content']
        actual_filename = file_data['filename']

        # Determine actual extension from downloaded file
        actual_extension = Path(actual_filename).suffix.lower()
        if not actual_extension:
            actual_extension = extension

        # Write to temp file for pandas to read
        with tempfile.NamedTemporaryFile(suffix=actual_extension, delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            if actual_extension in ['.xlsx', '.xls']:
                # Excel file - get sheet names and data
                excel_file = pd.ExcelFile(tmp_path)
                sheet_names = excel_file.sheet_names

                # Use specified sheet or default to first
                current_sheet = sheet if sheet and sheet in sheet_names else sheet_names[0]

                # Read the sheet
                df = pd.read_excel(tmp_path, sheet_name=current_sheet)

            elif actual_extension == '.csv':
                # CSV file - no sheets
                df = pd.read_csv(tmp_path)
                sheet_names = []
                current_sheet = None

            else:
                return DocumentPreview(
                    file_name=actual_filename,
                    file_type=actual_extension.lstrip('.'),
                    error=f"Preview not supported for {actual_extension} files."
                )

            # Get total rows before limiting
            total_rows = len(df)

            # Limit rows
            df_limited = df.head(max_rows)

            # Convert to list format, handling NaN values
            df_limited = df_limited.fillna('')
            headers = [str(col) for col in df_limited.columns.tolist()]
            rows = df_limited.values.tolist()

            # Convert any remaining non-serializable types to strings
            rows = [[str(cell) if not isinstance(cell, (str, int, float, bool)) else cell for cell in row] for row in rows]

            return DocumentPreview(
                file_name=actual_filename,
                file_type=actual_extension.lstrip('.'),
                sheets=sheet_names if actual_extension in ['.xlsx', '.xls'] else [],
                current_sheet=current_sheet,
                headers=headers,
                rows=rows,
                total_rows=total_rows,
                preview_rows=len(rows),
                has_more=total_rows > max_rows
            )

        finally:
            # Clean up temp file
            import os
            os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get Drive file preview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Connected Folders Endpoints
# ============================================================================

@app.post("/api/drive/{user_id}/connect", response_model=ConnectedFolder)
async def connect_drive_folder(
    user_id: str,
    request: ConnectFolderRequest,
    background_tasks: BackgroundTasks
):
    """
    Connect a Google Drive folder for auto-sync.

    Args:
        user_id: User ID
        request: Folder connection details

    Returns:
        Connected folder record
    """
    try:
        # Verify user exists
        user = db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Create data room if requested
        data_room_id = None
        if request.create_data_room:
            company_name = request.company_name or request.folder_name
            data_room_id = db.create_data_room(
                company_name=company_name,
                analyst_name=user.get('name') or user['email'],
                analyst_email=user['email'],
                security_level="cloud_enabled",
                user_id=user_id
            )
            logger.info(f"Created data room {data_room_id} for connected folder")

        # Create connected folder record
        connection_id = db.create_connected_folder(
            user_id=user_id,
            folder_id=request.folder_id,
            folder_name=request.folder_name,
            folder_path=request.folder_path,
            data_room_id=data_room_id
        )

        # Update data room status IMMEDIATELY to show processing has started
        # This prevents the "Loading data room" issue caused by timing race condition
        if data_room_id:
            db.update_data_room_status(data_room_id, 'parsing', progress=5)

        # Trigger initial sync in background
        background_tasks.add_task(sync_service.trigger_sync, connection_id)

        folder = db.get_connected_folder(connection_id)

        # Include data room status directly in response to avoid race condition
        # Frontend won't need to make an immediate separate status request
        data_room_status = None
        if data_room_id:
            data_room_status = {
                "id": data_room_id,
                "processing_status": "parsing",
                "progress_percent": 5,
                "total_documents": 0
            }

        return ConnectedFolder(
            id=folder['id'],
            folder_id=folder['folder_id'],
            folder_name=folder['folder_name'],
            folder_path=folder.get('folder_path'),
            data_room_id=folder.get('data_room_id'),
            sync_status=folder['sync_status'],
            last_sync_at=folder.get('last_sync_at'),
            total_files=folder.get('total_files', 0),
            processed_files=folder.get('processed_files', 0),
            failed_files=0,  # New connection, no failed files yet
            error_message=folder.get('error_message'),
            data_room_status=data_room_status,
            created_at=folder['created_at']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to connect folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/drive/{user_id}/connect-files")
async def connect_drive_files(
    user_id: str,
    request: ConnectFilesRequest,
    background_tasks: BackgroundTasks
):
    """
    Connect individual Google Drive files for processing.

    Unlike folder connections, individual files are processed directly
    without the discovery phase.

    Args:
        user_id: User ID
        request: Files to connect

    Returns:
        List of connected files and data room ID
    """
    try:
        # Verify user exists
        user = db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Validate request
        if len(request.file_ids) != len(request.file_names):
            raise HTTPException(status_code=400, detail="file_ids and file_names must have same length")

        if len(request.file_ids) == 0:
            raise HTTPException(status_code=400, detail="At least one file must be selected")

        # Step 1: Pre-check ALL files for existing connections BEFORE creating data room
        # This prevents orphaned data rooms when files are already connected
        existing_connections = []
        new_file_indices = []
        existing_data_room_id_from_files = None

        for i, (file_id, file_name) in enumerate(zip(request.file_ids, request.file_names)):
            existing_file = db.get_connected_file_with_data_room(user_id, file_id)
            if existing_file:
                logger.info(f"File already connected: {file_name} (data_room: {existing_file.get('data_room_id')})")
                existing_connections.append(existing_file)
                # Remember the data room ID from existing files for potential reuse
                if not existing_data_room_id_from_files and existing_file.get('data_room_id'):
                    existing_data_room_id_from_files = existing_file.get('data_room_id')
            else:
                new_file_indices.append(i)

        # Step 2: If ALL files are already connected, return existing data without creating new data room
        if len(new_file_indices) == 0:
            logger.info(f"All {len(existing_connections)} files already connected, returning existing data room")
            return {
                "connected_files": [
                    ConnectedFile(
                        id=f['id'],
                        drive_file_id=f['drive_file_id'],
                        file_name=f['file_name'],
                        file_path=f.get('file_path'),
                        mime_type=f.get('mime_type'),
                        file_size=f.get('file_size'),
                        data_room_id=f.get('data_room_id'),
                        document_id=f.get('document_id'),
                        sync_status=f['sync_status'],
                        error_message=f.get('error_message'),
                        created_at=f['created_at']
                    )
                    for f in existing_connections
                ],
                "data_room_id": existing_data_room_id_from_files,
                "total": len(existing_connections),
                "message": "Files already connected"
            }

        # Step 3: Determine target data room (only create if needed)
        if request.existing_data_room_id:
            # User explicitly specified a data room
            data_room = db.get_data_room(request.existing_data_room_id)
            if not data_room:
                raise HTTPException(status_code=404, detail="Data room not found")
            data_room_id = request.existing_data_room_id
        elif existing_data_room_id_from_files:
            # Reuse data room from existing connected files
            data_room_id = existing_data_room_id_from_files
            logger.info(f"Reusing existing data room: {data_room_id}")
        else:
            # Create new data room only if no existing connections
            data_room_name = request.data_room_name or request.file_names[0]
            data_room_id = db.create_data_room(
                company_name=data_room_name,
                analyst_name=user.get('name', 'Unknown'),
                analyst_email=user.get('email'),
                security_level='local_only',
                user_id=user_id
            )
            logger.info(f"Created data room for files: {data_room_name} (ID: {data_room_id})")

        # Update data room status IMMEDIATELY to show processing has started
        # This prevents the "Loading data room" issue caused by timing race condition
        if data_room_id and len(new_file_indices) > 0:
            db.update_data_room_status(data_room_id, 'parsing', progress=5)

        # Step 4: Process only NEW files (not already connected)
        connected_files = list(existing_connections)  # Start with existing
        for i in new_file_indices:
            file_id = request.file_ids[i]
            file_name = request.file_names[i]

            # Get optional metadata
            mime_type = request.mime_types[i] if request.mime_types and i < len(request.mime_types) else None
            file_size = request.file_sizes[i] if request.file_sizes and i < len(request.file_sizes) else None
            file_path = request.file_paths[i] if request.file_paths and i < len(request.file_paths) else None

            # Create connected file record
            connection_id = db.create_connected_file(
                user_id=user_id,
                drive_file_id=file_id,
                file_name=file_name,
                file_path=file_path,
                mime_type=mime_type,
                file_size=file_size,
                data_room_id=data_room_id
            )

            # Process file in background
            background_tasks.add_task(
                _process_connected_file,
                connection_id,
                user_id,
                file_id,
                file_name,
                data_room_id
            )

            file_record = db.get_connected_file(connection_id)
            if file_record:
                connected_files.append(file_record)

        return {
            "connected_files": [
                ConnectedFile(
                    id=f['id'],
                    drive_file_id=f['drive_file_id'],
                    file_name=f['file_name'],
                    file_path=f.get('file_path'),
                    mime_type=f.get('mime_type'),
                    file_size=f.get('file_size'),
                    data_room_id=f.get('data_room_id'),
                    document_id=f.get('document_id'),
                    sync_status=f['sync_status'],
                    error_message=f.get('error_message'),
                    created_at=f['created_at']
                )
                for f in connected_files
            ],
            "data_room_id": data_room_id,
            "total": len(connected_files)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to connect files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _process_connected_file(
    connection_id: str,
    user_id: str,
    drive_file_id: str,
    file_name: str,
    data_room_id: str
):
    """
    Background task to download and process a connected file.
    """
    try:
        # Update status to downloading
        db.update_connected_file_status(connection_id, 'downloading')

        # Ensure data room exists (safeguard for edge cases)
        if data_room_id and not db.get_data_room(data_room_id):
            logger.warning(f"Data room {data_room_id} not found, creating it")
            db.create_data_room(
                company_name=file_name,
                analyst_name='Unknown',
                security_level='local_only',
                data_room_id=data_room_id
            )

        # Update data room status to show downloading progress
        if data_room_id:
            db.update_data_room_status(data_room_id, 'uploading', progress=10)

        # Get user credentials
        user = db.get_user_by_id(user_id)
        if not user or not user.get('access_token'):
            raise Exception("User not authenticated")

        # Create Drive service
        from app.google_oauth import google_oauth_service
        credentials = google_oauth_service.get_credentials_from_tokens(
            access_token=user['access_token'],
            refresh_token=user.get('refresh_token'),
            expires_at=user.get('token_expires_at')
        )
        drive_service = GoogleDriveService(credentials)

        # Get file info
        file_info = drive_service.get_file_info(drive_file_id)
        if not file_info:
            raise Exception(f"File not found in Drive: {drive_file_id}")

        # Download file
        data_room_path = Path(settings.data_rooms_path) / data_room_id
        data_room_path.mkdir(parents=True, exist_ok=True)

        local_path = data_room_path / file_name
        downloaded_path = drive_service.download_file(drive_file_id, str(local_path))

        if not downloaded_path:
            raise Exception(f"Failed to download file: {file_name}")

        # Update data room status - download complete, now processing
        if data_room_id:
            db.update_data_room_status(data_room_id, 'parsing', progress=20)

        # Update status to processing
        db.update_connected_file_status(connection_id, 'processing', local_file_path=str(downloaded_path))

        # Create document record
        file_size = os.path.getsize(downloaded_path) if os.path.exists(downloaded_path) else 0
        file_ext = Path(downloaded_path).suffix.lower()

        document_id = db.create_document(
            data_room_id=data_room_id,
            file_name=file_name,
            file_path=str(downloaded_path),
            file_size=file_size,
            file_type=file_ext
        )

        # Queue document for processing
        from app.job_queue import job_queue, JobType
        job_queue.enqueue(
            job_type=JobType.PROCESS_FILE,
            payload={
                'data_room_id': data_room_id,
                'document_id': document_id,
                'file_path': str(downloaded_path),
                'file_name': file_name
            },
            data_room_id=data_room_id,
            file_name=file_name
        )

        # Update connected file with document ID
        db.update_connected_file_status(
            connection_id,
            'complete',
            document_id=document_id
        )

        logger.info(f"Successfully processed connected file: {file_name}")

    except Exception as e:
        logger.error(f"Failed to process connected file {file_name}: {e}")
        db.update_connected_file_status(
            connection_id,
            'failed',
            error_message=str(e)
        )


@app.get("/api/drive/{user_id}/connected-files")
def list_connected_files(user_id: str):
    """
    List all connected files for a user.

    Args:
        user_id: User ID

    Returns:
        List of connected files
    """
    try:
        files = db.get_connected_files_by_user(user_id)

        return {
            "files": [
                ConnectedFile(
                    id=f['id'],
                    drive_file_id=f['drive_file_id'],
                    file_name=f['file_name'],
                    file_path=f.get('file_path'),
                    mime_type=f.get('mime_type'),
                    file_size=f.get('file_size'),
                    data_room_id=f.get('data_room_id'),
                    document_id=f.get('document_id'),
                    sync_status=f['sync_status'],
                    error_message=f.get('error_message'),
                    created_at=f['created_at']
                )
                for f in files
            ],
            "total": len(files)
        }

    except Exception as e:
        logger.error(f"Failed to list connected files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/drive/file/{connection_id}")
def get_connected_file_status(connection_id: str):
    """
    Get status of a connected file.

    Args:
        connection_id: Connected file ID

    Returns:
        File status and details
    """
    try:
        file = db.get_connected_file(connection_id)
        if not file:
            raise HTTPException(status_code=404, detail="Connected file not found")

        return ConnectedFile(
            id=file['id'],
            drive_file_id=file['drive_file_id'],
            file_name=file['file_name'],
            file_path=file.get('file_path'),
            mime_type=file.get('mime_type'),
            file_size=file.get('file_size'),
            data_room_id=file.get('data_room_id'),
            document_id=file.get('document_id'),
            sync_status=file['sync_status'],
            error_message=file.get('error_message'),
            created_at=file['created_at']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get connected file status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/drive/file/{connection_id}")
async def disconnect_file(connection_id: str):
    """
    Disconnect a Google Drive file.

    Args:
        connection_id: Connected file ID

    Returns:
        Success status
    """
    try:
        file = db.get_connected_file(connection_id)
        if not file:
            raise HTTPException(status_code=404, detail="Connected file not found")

        success = db.delete_connected_file(connection_id)

        return {"success": success, "message": "File disconnected"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disconnect file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/drive/file/{connection_id}/retry")
async def retry_connected_file(connection_id: str, background_tasks: BackgroundTasks):
    """
    Retry processing a failed connected file.

    Args:
        connection_id: Connected file ID

    Returns:
        Updated file status
    """
    try:
        file = db.get_connected_file(connection_id)
        if not file:
            raise HTTPException(status_code=404, detail="Connected file not found")

        if file['sync_status'] != 'failed':
            raise HTTPException(status_code=400, detail="File is not in failed state")

        # Reset status and retry
        db.update_connected_file_status(connection_id, 'pending', error_message=None)

        # Reprocess in background
        user_id = file.get('user_id')
        if not user_id:
            raise HTTPException(status_code=400, detail="File has no associated user")

        background_tasks.add_task(
            _process_connected_file,
            connection_id,
            user_id,
            file['drive_file_id'],
            file['file_name'],
            file.get('data_room_id')
        )

        return {"success": True, "message": "Retry initiated"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retry connected file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/drive/{user_id}/connected")
def list_connected_folders(user_id: str):
    """
    List all connected folders for a user.

    Args:
        user_id: User ID

    Returns:
        List of connected folders
    """
    try:
        folders = db.get_connected_folders_by_user(user_id)

        # Build folder list with failed_files count for each
        folder_list = []
        for f in folders:
            failed_count = db.count_synced_files_by_status(f['id'], 'failed')
            folder_list.append(ConnectedFolder(
                id=f['id'],
                folder_id=f['folder_id'],
                folder_name=f['folder_name'],
                folder_path=f.get('folder_path'),
                data_room_id=f.get('data_room_id'),
                sync_status=f['sync_status'],
                sync_stage=f.get('sync_stage', 'idle'),
                last_sync_at=f.get('last_sync_at'),
                total_files=f.get('total_files', 0),
                processed_files=f.get('processed_files', 0),
                failed_files=failed_count,
                discovered_files=f.get('discovered_files', 0),
                discovered_folders=f.get('discovered_folders', 0),
                current_folder_path=f.get('current_folder_path'),
                error_message=f.get('error_message'),
                created_at=f['created_at']
            ))

        return {
            "folders": folder_list,
            "total": len(folders)
        }

    except Exception as e:
        logger.error(f"Failed to list connected folders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/drive/folder/{connection_id}")
def get_connected_folder_status(connection_id: str):
    """
    Get status of a connected folder.

    Args:
        connection_id: Connected folder ID

    Returns:
        Folder status and synced files
    """
    try:
        folder = db.get_connected_folder(connection_id)
        if not folder:
            raise HTTPException(status_code=404, detail="Connected folder not found")

        synced_files = db.get_synced_files_by_folder(connection_id)

        # Calculate failed files count for accurate progress reporting
        failed_files = db.count_synced_files_by_status(connection_id, 'failed')

        return {
            "folder": ConnectedFolder(
                id=folder['id'],
                folder_id=folder['folder_id'],
                folder_name=folder['folder_name'],
                folder_path=folder.get('folder_path'),
                data_room_id=folder.get('data_room_id'),
                sync_status=folder['sync_status'],
                sync_stage=folder.get('sync_stage', 'idle'),
                last_sync_at=folder.get('last_sync_at'),
                total_files=folder.get('total_files', 0),
                processed_files=folder.get('processed_files', 0),
                failed_files=failed_files,
                discovered_files=folder.get('discovered_files', 0),
                discovered_folders=folder.get('discovered_folders', 0),
                current_folder_path=folder.get('current_folder_path'),
                error_message=folder.get('error_message'),
                created_at=folder['created_at']
            ),
            "files": [
                SyncedFile(
                    id=f['id'],
                    drive_file_id=f['drive_file_id'],
                    file_name=f['file_name'],
                    file_path=f.get('file_path'),
                    mime_type=f.get('mime_type'),
                    file_size=f.get('file_size'),
                    sync_status=f['sync_status'],
                    last_synced_at=f.get('last_synced_at'),
                    error_message=f.get('error_message')
                )
                for f in synced_files
            ]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get folder status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/drive/folder/{connection_id}/progress")
def get_folder_sync_progress(connection_id: str):
    """
    Lightweight sync progress endpoint for the progress UI.
    Returns only the data needed for the animated progress panel,
    avoiding the overhead of returning all synced files.
    """
    try:
        folder = db.get_connected_folder(connection_id)
        if not folder:
            raise HTTPException(status_code=404, detail="Connected folder not found")

        recent = db.get_recently_completed_synced_files(connection_id, limit=5)
        type_counts = db.get_synced_file_type_counts(connection_id)
        current = db.get_current_processing_file(connection_id)

        return {
            "sync_stage": folder.get('sync_stage', 'idle'),
            "sync_status": folder['sync_status'],
            "discovered_files": folder.get('discovered_files', 0),
            "discovered_folders": folder.get('discovered_folders', 0),
            "total_files": folder.get('total_files', 0),
            "processed_files": folder.get('processed_files', 0),
            "file_type_counts": type_counts,
            "recently_completed": [
                {"file_name": f['file_name'], "mime_type": f.get('mime_type')}
                for f in recent
            ],
            "current_file": {
                "file_name": current['file_name'],
                "sync_status": current['sync_status']
            } if current else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get folder sync progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/drive/folder/{connection_id}/sync")
async def trigger_folder_sync(connection_id: str, background_tasks: BackgroundTasks):
    """
    Trigger immediate sync for a connected folder.

    Args:
        connection_id: Connected folder ID
    """
    try:
        folder = db.get_connected_folder(connection_id)
        if not folder:
            raise HTTPException(status_code=404, detail="Connected folder not found")

        background_tasks.add_task(sync_service.trigger_sync, connection_id)

        return {"message": "Sync triggered", "folder_id": connection_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/drive/folder/{connection_id}/pause")
async def pause_folder_sync(connection_id: str):
    """Pause sync for a connected folder."""
    try:
        folder = db.get_connected_folder(connection_id)
        if not folder:
            raise HTTPException(status_code=404, detail="Connected folder not found")

        db.update_connected_folder_status(connection_id, sync_status='paused')

        return {"message": "Sync paused", "folder_id": connection_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/drive/folder/{connection_id}/resume")
async def resume_folder_sync(connection_id: str, background_tasks: BackgroundTasks):
    """Resume sync for a paused connected folder."""
    try:
        folder = db.get_connected_folder(connection_id)
        if not folder:
            raise HTTPException(status_code=404, detail="Connected folder not found")

        db.update_connected_folder_status(connection_id, sync_status='active')
        background_tasks.add_task(sync_service.trigger_sync, connection_id)

        return {"message": "Sync resumed", "folder_id": connection_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/drive/folder/{connection_id}")
async def disconnect_folder(connection_id: str):
    """
    Disconnect a Google Drive folder.

    Args:
        connection_id: Connected folder ID
    """
    try:
        folder = db.get_connected_folder(connection_id)
        if not folder:
            raise HTTPException(status_code=404, detail="Connected folder not found")

        db.delete_connected_folder(connection_id)

        return {"message": "Folder disconnected", "folder_id": connection_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to disconnect folder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/drive/folder/{connection_id}/retry-failed")
async def retry_failed_files(connection_id: str, background_tasks: BackgroundTasks):
    """
    Reset all failed files in a connected folder to retry sync.

    Args:
        connection_id: Connected folder ID

    Returns:
        Number of files reset and triggers a new sync
    """
    try:
        folder = db.get_connected_folder(connection_id)
        if not folder:
            raise HTTPException(status_code=404, detail="Connected folder not found")

        count = db.reset_failed_synced_files(connection_id)

        if count > 0:
            # Trigger immediate sync after reset
            background_tasks.add_task(sync_service.trigger_sync, connection_id)

        return {
            "message": f"Reset {count} failed files for retry",
            "count": count,
            "folder_id": connection_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reset failed files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Financial Analysis Endpoints
# ============================================================================

def run_financial_analysis_background(
    file_path: str,
    data_room_id: str,
    document_id: str
):
    """
    Background task to run financial analysis on an Excel file.
    """
    try:
        logger.info(f"[{data_room_id}] Starting financial analysis for document {document_id}")

        from tools.financial_analysis_agent import analyze_financial_model

        result = analyze_financial_model(
            file_path=file_path,
            data_room_id=data_room_id,
            document_id=document_id,
            analysis_model=settings.financial_analysis_model,
            extraction_model=settings.financial_extraction_model,
            max_cost=settings.financial_analysis_max_cost
        )

        # Save results to database
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

        logger.success(f"[{data_room_id}] Financial analysis complete for {document_id}")

    except Exception as e:
        logger.error(f"[{data_room_id}] Financial analysis failed: {e}", exc_info=True)
        # Save failed analysis
        try:
            db.save_financial_analysis(
                analysis_id=f"fa_{uuid.uuid4().hex[:12]}",
                data_room_id=data_room_id,
                document_id=document_id,
                file_name=Path(file_path).name,
                status="failed",
                error_message=str(e)
            )
        except Exception:
            pass


@app.post("/api/data-room/{data_room_id}/document/{document_id}/analyze-financial")
async def trigger_financial_analysis(
    data_room_id: str,
    document_id: str,
    background_tasks: BackgroundTasks,
    request: FinancialAnalysisTriggerRequest = None
):
    """
    Trigger financial model analysis for an Excel document.

    Args:
        data_room_id: Data room ID
        document_id: Document ID
        request: Optional trigger options

    Returns:
        Analysis ID and status
    """
    try:
        # Verify data room exists
        data_room = db.get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(status_code=404, detail="Data room not found")

        # Get document
        document = db.get_document_by_id(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        if document['data_room_id'] != data_room_id:
            raise HTTPException(status_code=400, detail="Document does not belong to this data room")

        # Check if Excel file
        file_type = document.get('file_type', '').lower()
        if file_type not in ['xlsx', 'xls']:
            raise HTTPException(
                status_code=400,
                detail=f"Financial analysis only supports Excel files (.xlsx, .xls), not .{file_type}"
            )

        # Check for existing analysis
        force_reanalyze = request.force_reanalyze if request else False
        if not force_reanalyze and db.check_financial_analysis_exists(document_id):
            existing = db.get_financial_analysis(document_id, data_room_id)
            return FinancialAnalysisTriggerResponse(
                analysis_id=existing['id'],
                status="exists",
                message="Financial analysis already exists. Use force_reanalyze=true to re-run."
            )

        # Build file path
        file_path = f"{settings.data_rooms_path}/{data_room_id}/raw/{document['file_name']}"

        if not Path(file_path).exists():
            raise HTTPException(status_code=404, detail="Document file not found on disk")

        # Generate analysis ID
        analysis_id = f"fa_{uuid.uuid4().hex[:12]}"

        # Queue background task
        background_tasks.add_task(
            run_financial_analysis_background,
            file_path,
            data_room_id,
            document_id
        )

        return FinancialAnalysisTriggerResponse(
            analysis_id=analysis_id,
            status="queued",
            message="Financial analysis started. Check status endpoint for results."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger financial analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-room/{data_room_id}/document/{document_id}/financial-analysis")
def get_financial_analysis(data_room_id: str, document_id: str):
    """
    Get financial analysis results for a document.

    Args:
        data_room_id: Data room ID
        document_id: Document ID

    Returns:
        Financial analysis results
    """
    try:
        # Verify data room exists
        data_room = db.get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(status_code=404, detail="Data room not found")

        # Get analysis
        analysis = db.get_financial_analysis(document_id, data_room_id)

        if not analysis:
            raise HTTPException(
                status_code=404,
                detail="Financial analysis not found. Trigger analysis first."
            )

        return analysis

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get financial analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-room/{data_room_id}/financial-summary")
def get_financial_summary(data_room_id: str):
    """
    Get aggregated financial summary across all analyzed documents.

    Args:
        data_room_id: Data room ID

    Returns:
        Aggregated financial summary
    """
    try:
        # Verify data room exists
        data_room = db.get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(status_code=404, detail="Data room not found")

        # Get all analyses
        analyses = db.get_financial_analyses_by_data_room(data_room_id)

        if not analyses:
            return {
                "data_room_id": data_room_id,
                "analyzed_documents": 0,
                "message": "No financial analyses found. Upload Excel files to analyze."
            }

        # Aggregate metrics from all analyses
        all_metrics = []
        all_insights = []
        all_issues = []
        all_questions = []
        executive_summaries = []

        for analysis in analyses:
            if analysis.get('status') == 'complete':
                if analysis.get('extracted_metrics'):
                    all_metrics.extend(analysis['extracted_metrics'])
                if analysis.get('insights'):
                    all_insights.extend(analysis['insights'])
                if analysis.get('validation_results', {}).get('red_flags'):
                    all_issues.extend(analysis['validation_results']['red_flags'])
                if analysis.get('follow_up_questions'):
                    all_questions.extend(analysis['follow_up_questions'])
                if analysis.get('executive_summary'):
                    executive_summaries.append(analysis['executive_summary'])

        # Find key metrics
        def find_metric(metrics, name_patterns):
            for m in metrics:
                metric_name = m.get('name', '').lower()
                for pattern in name_patterns:
                    if pattern in metric_name:
                        return m
            return None

        revenue_latest = find_metric(all_metrics, ['revenue', 'arr', 'mrr'])
        gross_margin = find_metric(all_metrics, ['gross margin', 'gross_margin'])
        burn_rate = find_metric(all_metrics, ['burn', 'burn rate', 'burn_rate'])
        runway = find_metric(all_metrics, ['runway'])
        ltv_cac = find_metric(all_metrics, ['ltv:cac', 'ltv_cac', 'ltv cac'])

        return {
            "data_room_id": data_room_id,
            "analyzed_documents": len([a for a in analyses if a.get('status') == 'complete']),
            "total_metrics": len(all_metrics),
            "revenue_latest": revenue_latest,
            "gross_margin": gross_margin.get('value') if gross_margin else None,
            "burn_rate": burn_rate.get('value') if burn_rate else None,
            "runway_months": int(runway.get('value')) if runway else None,
            "ltv_cac_ratio": ltv_cac.get('value') if ltv_cac else None,
            "top_insights": sorted(all_insights, key=lambda x: ['critical', 'high', 'medium', 'low'].index(x.get('importance', 'low')))[:5],
            "critical_issues": [i for i in all_issues if i.get('severity') in ['critical', 'high']][:5],
            "key_questions": sorted(all_questions, key=lambda x: ['must_ask', 'should_ask', 'nice_to_ask'].index(x.get('priority', 'nice_to_ask')))[:5],
            "executive_summary": executive_summaries[0] if executive_summaries else None
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get financial summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data-room/{data_room_id}/financial-metrics")
def get_financial_metrics(
    data_room_id: str,
    category: Optional[str] = None,
    metric_name: Optional[str] = None
):
    """
    Get individual financial metrics for quick querying.

    Args:
        data_room_id: Data room ID
        category: Filter by category (revenue, profitability, cash, saas, etc.)
        metric_name: Filter by metric name (partial match)

    Returns:
        List of financial metrics
    """
    try:
        # Verify data room exists
        data_room = db.get_data_room(data_room_id)
        if not data_room:
            raise HTTPException(status_code=404, detail="Data room not found")

        metrics = db.get_financial_metrics_by_data_room(
            data_room_id,
            category=category,
            metric_name=metric_name
        )

        return {
            "data_room_id": data_room_id,
            "metrics": metrics,
            "total": len(metrics),
            "filter_category": category,
            "filter_name": metric_name
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get financial metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")

    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).model_dump()
    )


# ============================================================================
# Agent-Oriented Endpoints (structured for programmatic AI agent access)
# ============================================================================

@app.get("/api/agent/drive/inventory")
async def agent_drive_inventory(user_id: str):
    """
    Get a summary of all connected Drive folders and their sync status.
    Designed for AI agent discovery — returns structured, token-efficient data.
    """
    try:
        _drive_rate_limiter.check(user_id)
        folders = db.get_connected_folders_by_user(user_id)
        inventory = []
        for f in folders:
            synced_files = db.get_synced_files_by_folder(f['id'])
            inventory.append({
                "folder_id": f['id'],
                "folder_name": f['folder_name'],
                "data_room_id": f.get('data_room_id'),
                "sync_status": f.get('sync_status', 'unknown'),
                "total_files": f.get('total_files', 0),
                "processed_files": f.get('processed_files', 0),
                "files": [
                    {"name": sf['file_name'], "status": sf['sync_status'], "drive_id": sf['drive_file_id']}
                    for sf in synced_files[:50]
                ]
            })
        return {"connected_folders": inventory, "total_folders": len(inventory)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent inventory failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agent/drive/search")
async def agent_drive_search(user_id: str, query: str):
    """
    Search across Google Drive for files matching a query.
    Returns structured results suitable for agent tool use.
    """
    try:
        _drive_rate_limiter.check(user_id)
        drive_service = get_drive_service(user_id)

        result = drive_service.list_files(
            search_query=query, page_size=20, include_folders=False
        )

        return {
            "query": query,
            "results": [
                {
                    "file_id": f["id"],
                    "name": f["name"],
                    "type": f.get("extension") or f.get("mimeType"),
                    "size_bytes": f.get("size"),
                    "modified": f.get("modifiedTime"),
                    "supported": f.get("isSupported", False),
                }
                for f in result["files"]
            ],
            "total": result["totalFiles"],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agent/drive/file-text/{file_id}")
async def agent_get_file_text(user_id: str, file_id: str):
    """
    Download a Drive file and return its extracted text content.
    Useful for agent to read document contents without full processing pipeline.
    """
    try:
        import tempfile as _tempfile
        _drive_rate_limiter.check(user_id)
        drive_service = get_drive_service(user_id)

        # Download to temp file
        tmp = _tempfile.NamedTemporaryFile(delete=False, suffix='.tmp')
        tmp_path = tmp.name
        tmp.close()

        result = drive_service.download_file_to_disk(file_id, tmp_path)
        file_ext = Path(result['filename']).suffix.lower()

        # Extract text based on file type
        text = ""
        try:
            from tools.ingest_data_room import parse_file_by_type
            try:
                parsed = parse_file_by_type(tmp_path)
                text = parsed.get('text', '')
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # Truncate to avoid huge responses
        max_chars = 50000
        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars]

        return {
            "file_id": file_id,
            "filename": result['filename'],
            "file_type": file_ext,
            "text": text,
            "char_count": len(text),
            "truncated": truncated,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent file text extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Static File Serving (Production: serve built frontend)
# ============================================================================

if _serve_frontend:
    from fastapi.staticfiles import StaticFiles

    # Serve static assets (JS, CSS, images)
    if (_frontend_dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(_frontend_dist / "assets")), name="static-assets")

    # SPA catch-all: all non-API routes return index.html
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = _frontend_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_frontend_dist / "index.html"))


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server...")
    logger.info(f"Server: http://{settings.host}:{settings.port}")
    logger.info(f"Docs: http://{settings.host}:{settings.port}/docs")

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
