"""
Database initialization tool for VC Due Diligence system.
Creates SQLite schema and initializes ChromaDB client.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
from loguru import logger

# Configuration (reads from env vars for container deployments, falls back to local .tmp/)
DB_PATH = Path(os.getenv("DATABASE_PATH", ".tmp/due_diligence.db"))
CHROMA_DB_PATH = Path(os.getenv("CHROMA_DB_PATH", ".tmp/chroma_db"))
DATA_ROOMS_PATH = Path(os.getenv("DATA_ROOMS_PATH", ".tmp/data_rooms"))
LOGS_PATH = Path(os.getenv("LOGS_PATH", ".tmp/logs"))


def init_directories():
    """Create necessary directories for the application."""
    directories = [
        DB_PATH.parent,
        CHROMA_DB_PATH,
        DATA_ROOMS_PATH,
        LOGS_PATH
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def init_database():
    """Initialize SQLite database with complete schema."""

    # Ensure parent directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Connect to database
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    logger.info(f"Initializing database at {DB_PATH}")

    # Create data_rooms table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS data_rooms (
        id TEXT PRIMARY KEY,
        company_name TEXT NOT NULL,
        analyst_name TEXT,
        analyst_email TEXT,
        security_level TEXT CHECK(security_level IN ('local_only', 'cloud_enabled')) DEFAULT 'local_only',
        processing_status TEXT CHECK(processing_status IN ('uploading', 'parsing', 'indexing', 'extracting', 'complete', 'failed')) DEFAULT 'uploading',
        progress_percent REAL DEFAULT 0,
        total_documents INTEGER DEFAULT 0,
        total_chunks INTEGER DEFAULT 0,
        estimated_cost REAL,
        actual_cost REAL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP,
        error_message TEXT,
        metadata TEXT,
        user_id TEXT REFERENCES users(id)
    )
    """)
    logger.info("Created table: data_rooms")

    # Migration: Add error_message column if it doesn't exist (for existing databases)
    cursor.execute("PRAGMA table_info(data_rooms)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'error_message' not in columns:
        cursor.execute("ALTER TABLE data_rooms ADD COLUMN error_message TEXT")
        logger.info("Added error_message column to data_rooms (migration)")

    # Create documents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        data_room_id TEXT NOT NULL,
        file_name TEXT NOT NULL,
        file_path TEXT NOT NULL,
        file_size INTEGER,
        file_type TEXT,
        document_category TEXT,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        parse_status TEXT CHECK(parse_status IN ('pending', 'parsing', 'parsed', 'failed')) DEFAULT 'pending',
        parsed_at TIMESTAMP,
        page_count INTEGER,
        token_count INTEGER,
        error_message TEXT,
        metadata TEXT,
        FOREIGN KEY (data_room_id) REFERENCES data_rooms(id) ON DELETE CASCADE,
        UNIQUE(data_room_id, file_name)
    )
    """)
    logger.info("Created table: documents")

    # Create chunks table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id TEXT PRIMARY KEY,
        document_id TEXT NOT NULL,
        data_room_id TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        token_count INTEGER,
        page_number INTEGER,
        section_title TEXT,
        chunk_type TEXT,
        embedding_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        metadata TEXT,
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
        FOREIGN KEY (data_room_id) REFERENCES data_rooms(id) ON DELETE CASCADE
    )
    """)
    logger.info("Created table: chunks")

    # Create queries table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS queries (
        id TEXT PRIMARY KEY,
        data_room_id TEXT NOT NULL,
        analyst_email TEXT,
        user_id TEXT,
        question TEXT NOT NULL,
        answer TEXT,
        sources TEXT,
        confidence_score REAL,
        conversation_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        response_time_ms INTEGER,
        tokens_used INTEGER,
        cost REAL,
        FOREIGN KEY (data_room_id) REFERENCES data_rooms(id) ON DELETE CASCADE,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
    )
    """)
    logger.info("Created table: queries")

    # Create memos table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memos (
        id TEXT PRIMARY KEY,
        data_room_id TEXT NOT NULL,
        version INTEGER DEFAULT 1,
        status TEXT CHECK(status IN ('generating', 'complete', 'partial', 'cancelled', 'failed')) DEFAULT 'generating',
        proposed_investment_terms TEXT,
        executive_summary TEXT,
        market_analysis TEXT,
        team_assessment TEXT,
        product_technology TEXT,
        financial_analysis TEXT,
        valuation_analysis TEXT,
        risks_concerns TEXT,
        outcome_scenario_analysis TEXT,
        investment_recommendation TEXT,
        full_memo TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP,
        tokens_used INTEGER,
        cost REAL,
        ticket_size REAL,
        post_money_valuation REAL,
        valuation_methods TEXT,
        analyst_feedback TEXT,
        metadata TEXT,
        FOREIGN KEY (data_room_id) REFERENCES data_rooms(id) ON DELETE CASCADE
    )
    """)
    logger.info("Created table: memos")

    # Migration: Update memos CHECK constraint to include 'cancelled' (for existing databases)
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='memos'")
    memos_sql_row = cursor.fetchone()
    if memos_sql_row and "'cancelled'" not in (memos_sql_row[0] or ""):
        conn.execute("PRAGMA writable_schema = ON")
        cursor.execute("""
            UPDATE sqlite_master
            SET sql = REPLACE(sql,
                '''generating'', ''complete'', ''partial'', ''failed''',
                '''generating'', ''complete'', ''partial'', ''cancelled'', ''failed''')
            WHERE type='table' AND name='memos'
        """)
        # Also try the variant without 'partial' for older DBs
        cursor.execute("""
            UPDATE sqlite_master
            SET sql = REPLACE(sql,
                '''generating'', ''complete'', ''failed''',
                '''generating'', ''complete'', ''partial'', ''cancelled'', ''failed''')
            WHERE type='table' AND name='memos'
        """)
        conn.execute("PRAGMA writable_schema = OFF")
        conn.execute("PRAGMA integrity_check")
        logger.info("Updated memos CHECK constraint to include 'cancelled' (migration)")

    # Create memo_chat_messages table for persisting memo chat history
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
        cost REAL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (memo_id) REFERENCES memos(id) ON DELETE CASCADE,
        FOREIGN KEY (data_room_id) REFERENCES data_rooms(id) ON DELETE CASCADE
    )
    """)
    logger.info("Created table: memo_chat_messages")

    # Create analysis_cache table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analysis_cache (
        id TEXT PRIMARY KEY,
        data_room_id TEXT NOT NULL,
        analysis_type TEXT NOT NULL,
        extracted_data TEXT NOT NULL,
        extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        version INTEGER DEFAULT 1,
        FOREIGN KEY (data_room_id) REFERENCES data_rooms(id) ON DELETE CASCADE,
        UNIQUE(data_room_id, analysis_type, version)
    )
    """)
    logger.info("Created table: analysis_cache")

    # Create processing_logs table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS processing_logs (
        id TEXT PRIMARY KEY,
        data_room_id TEXT,
        document_id TEXT,
        stage TEXT NOT NULL,
        status TEXT CHECK(status IN ('started', 'completed', 'failed')),
        message TEXT,
        error_details TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        duration_ms INTEGER
    )
    """)
    logger.info("Created table: processing_logs")

    # Create api_usage table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS api_usage (
        id TEXT PRIMARY KEY,
        data_room_id TEXT,
        provider TEXT,
        model TEXT,
        operation TEXT,
        input_tokens INTEGER,
        output_tokens INTEGER,
        cost REAL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)
    logger.info("Created table: api_usage")

    # Create users table for Google OAuth
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE NOT NULL,
        name TEXT,
        picture_url TEXT,
        google_id TEXT UNIQUE,
        access_token TEXT,
        refresh_token TEXT,
        token_expires_at TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login_at TIMESTAMP,
        is_active INTEGER DEFAULT 1
    )
    """)
    logger.info("Created table: users")

    # Create connected_folders table for Google Drive sync
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS connected_folders (
        id TEXT PRIMARY KEY,
        user_id TEXT NOT NULL,
        folder_id TEXT NOT NULL,
        folder_name TEXT NOT NULL,
        folder_path TEXT,
        data_room_id TEXT,
        sync_status TEXT CHECK(sync_status IN ('active', 'paused', 'error', 'syncing')) DEFAULT 'active',
        sync_stage TEXT CHECK(sync_stage IN ('idle', 'discovering', 'discovered', 'queued', 'processing', 'complete', 'error')) DEFAULT 'idle',
        last_sync_at TIMESTAMP,
        sync_page_token TEXT,
        total_files INTEGER DEFAULT 0,
        processed_files INTEGER DEFAULT 0,
        discovered_files INTEGER DEFAULT 0,
        discovered_folders INTEGER DEFAULT 0,
        current_folder_path TEXT,
        error_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY (data_room_id) REFERENCES data_rooms(id) ON DELETE SET NULL,
        UNIQUE(user_id, folder_id)
    )
    """)
    logger.info("Created table: connected_folders")

    # Migration: Add sync_stage columns if they don't exist (for existing databases)
    cursor.execute("PRAGMA table_info(connected_folders)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'sync_stage' not in columns:
        cursor.execute("ALTER TABLE connected_folders ADD COLUMN sync_stage TEXT DEFAULT 'idle'")
        logger.info("Added sync_stage column to connected_folders (migration)")
    if 'discovered_files' not in columns:
        cursor.execute("ALTER TABLE connected_folders ADD COLUMN discovered_files INTEGER DEFAULT 0")
        logger.info("Added discovered_files column to connected_folders (migration)")
    if 'discovered_folders' not in columns:
        cursor.execute("ALTER TABLE connected_folders ADD COLUMN discovered_folders INTEGER DEFAULT 0")
        logger.info("Added discovered_folders column to connected_folders (migration)")
    if 'current_folder_path' not in columns:
        cursor.execute("ALTER TABLE connected_folders ADD COLUMN current_folder_path TEXT")
        logger.info("Added current_folder_path column to connected_folders (migration)")

    # Create synced_files table to track individual synced files
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS synced_files (
        id TEXT PRIMARY KEY,
        connected_folder_id TEXT NOT NULL,
        drive_file_id TEXT NOT NULL,
        file_name TEXT NOT NULL,
        file_path TEXT,
        mime_type TEXT,
        file_size INTEGER,
        drive_modified_time TIMESTAMP,
        local_file_path TEXT,
        document_id TEXT,
        sync_status TEXT CHECK(sync_status IN ('pending', 'downloading', 'queued', 'processing', 'complete', 'failed', 'deleted')) DEFAULT 'pending',
        last_synced_at TIMESTAMP,
        error_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (connected_folder_id) REFERENCES connected_folders(id) ON DELETE CASCADE,
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL,
        UNIQUE(connected_folder_id, drive_file_id)
    )
    """)
    logger.info("Created table: synced_files")

    # Create folder_inventory table to track discovered folders during sync
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS folder_inventory (
        id TEXT PRIMARY KEY,
        connected_folder_id TEXT NOT NULL,
        drive_folder_id TEXT NOT NULL,
        folder_name TEXT NOT NULL,
        folder_path TEXT,
        parent_folder_id TEXT,
        file_count INTEGER DEFAULT 0,
        total_size_bytes INTEGER DEFAULT 0,
        scan_status TEXT CHECK(scan_status IN ('pending', 'scanning', 'scanned', 'error')) DEFAULT 'pending',
        processed_files INTEGER DEFAULT 0,
        discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        error_message TEXT,
        FOREIGN KEY (connected_folder_id) REFERENCES connected_folders(id) ON DELETE CASCADE,
        UNIQUE(connected_folder_id, drive_folder_id)
    )
    """)
    logger.info("Created table: folder_inventory")

    # Create connected_files table for individual Google Drive file connections
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
    logger.info("Created table: connected_files")

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
    logger.info("Created table: data_room_members")

    # Create financial_analyses table for Excel model analysis results
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS financial_analyses (
        id TEXT PRIMARY KEY,
        data_room_id TEXT NOT NULL,
        document_id TEXT NOT NULL,
        file_name TEXT,
        analysis_type TEXT DEFAULT 'excel_model',
        status TEXT CHECK(status IN ('in_progress', 'complete', 'failed')) DEFAULT 'in_progress',
        model_structure TEXT,
        extracted_metrics TEXT,
        time_series TEXT,
        missing_metrics TEXT,
        validation_results TEXT,
        insights TEXT,
        follow_up_questions TEXT,
        key_metrics_summary TEXT,
        risk_assessment TEXT,
        investment_thesis_notes TEXT,
        executive_summary TEXT,
        analysis_cost REAL DEFAULT 0,
        tokens_used INTEGER DEFAULT 0,
        processing_time_ms INTEGER DEFAULT 0,
        error_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (data_room_id) REFERENCES data_rooms(id) ON DELETE CASCADE,
        FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
    )
    """)
    logger.info("Created table: financial_analyses")

    # Create financial_metrics table for individual metrics (for quick querying)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS financial_metrics (
        id TEXT PRIMARY KEY,
        financial_analysis_id TEXT NOT NULL,
        data_room_id TEXT NOT NULL,
        metric_name TEXT NOT NULL,
        category TEXT,
        metric_value REAL,
        metric_unit TEXT,
        period TEXT,
        cell_reference TEXT,
        confidence TEXT CHECK(confidence IN ('high', 'medium', 'low')) DEFAULT 'medium',
        source_sheet TEXT,
        notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (financial_analysis_id) REFERENCES financial_analyses(id) ON DELETE CASCADE,
        FOREIGN KEY (data_room_id) REFERENCES data_rooms(id) ON DELETE CASCADE
    )
    """)
    logger.info("Created table: financial_metrics")

    # Create indexes for better query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_data_room ON documents(data_room_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_data_room ON chunks(data_room_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_queries_data_room ON queries(data_room_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_memos_data_room ON memos(data_room_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_data_room ON api_usage(data_room_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_connected_folders_user ON connected_folders(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_connected_folders_data_room ON connected_folders(data_room_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_synced_files_folder ON synced_files(connected_folder_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_synced_files_drive_id ON synced_files(drive_file_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_folder_inventory_connected ON folder_inventory(connected_folder_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_connected_files_user ON connected_files(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_connected_files_data_room ON connected_files(data_room_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_connected_files_drive_id ON connected_files(drive_file_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_financial_analyses_data_room ON financial_analyses(data_room_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_financial_analyses_document ON financial_analyses(document_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_financial_metrics_analysis ON financial_metrics(financial_analysis_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_financial_metrics_data_room ON financial_metrics(data_room_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_financial_metrics_name ON financial_metrics(metric_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_memo_chat_messages_memo ON memo_chat_messages(memo_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_memo_chat_messages_data_room ON memo_chat_messages(data_room_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_drm_data_room ON data_room_members(data_room_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_drm_user ON data_room_members(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_drm_email ON data_room_members(invited_email)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_data_rooms_user ON data_rooms(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_queries_user ON queries(user_id)")
    logger.info("Created database indexes")

    # Commit changes
    conn.commit()
    conn.close()

    logger.success(f"Database initialized successfully at {DB_PATH}")


def init_chromadb():
    """Initialize ChromaDB client and persistence directory."""
    try:
        import chromadb
        from chromadb.config import Settings

        # Create persistent client
        client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        logger.info(f"ChromaDB initialized at {CHROMA_DB_PATH}")
        logger.success("ChromaDB client created successfully")

        # Test by listing collections
        collections = client.list_collections()
        logger.info(f"Existing collections: {len(collections)}")

        return client

    except ImportError:
        logger.warning("ChromaDB not installed. Run: pip install chromadb")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB: {e}")
        return None


def verify_installation():
    """Verify that all required packages are installed."""
    required_packages = [
        'anthropic',
        'openai',
        'chromadb',
        'PyPDF2',
        'pdfplumber',
        'pandas',
        'openpyxl',
        'docx',
        'fastapi',
        'uvicorn',
        'sqlalchemy',
        'pydantic',
        'loguru'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        logger.warning("Install missing packages: pip install -r requirements.txt")
        return False
    else:
        logger.success("All required packages are installed")
        return True


def main():
    """Main initialization function."""
    logger.info("Starting VC Due Diligence system initialization...")

    # Create directories
    init_directories()

    # Initialize database
    init_database()

    # Initialize ChromaDB
    init_chromadb()

    # Verify installation
    verify_installation()

    logger.success("System initialization complete!")
    logger.info(f"Database location: {DB_PATH.absolute()}")
    logger.info(f"ChromaDB location: {CHROMA_DB_PATH.absolute()}")
    logger.info(f"Data rooms storage: {DATA_ROOMS_PATH.absolute()}")
    logger.info(f"Logs location: {LOGS_PATH.absolute()}")
    logger.info("\nNext steps:")
    logger.info("1. Copy .env.example to .env and add your API keys")
    logger.info("2. Install dependencies: pip install -r requirements.txt")
    logger.info("3. Run tests: pytest tests/")
    logger.info("4. Start building tools (parse_pdf.py, etc.)")


if __name__ == "__main__":
    main()
