"""
Configuration settings for the VC Due Diligence application.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Google API (optional)
    google_client_id: str = os.getenv("GOOGLE_CLIENT_ID", "")
    google_client_secret: str = os.getenv("GOOGLE_CLIENT_SECRET", "")

    # Public URL (set this in production to the app's public-facing URL, e.g. https://myapp.up.railway.app)
    public_url: str = os.getenv("PUBLIC_URL", "")

    # Application Settings
    security_level: str = os.getenv("SECURITY_LEVEL", "local_only")
    max_cost_per_data_room: float = float(os.getenv("MAX_COST_PER_DATA_ROOM", "15.0"))
    monthly_budget_alert: float = float(os.getenv("MONTHLY_BUDGET_ALERT", "45.0"))

    # Database
    database_url: str = os.getenv("DATABASE_URL", "postgresql://noesis:password@localhost:5432/noesis")
    chroma_db_path: str = os.getenv("CHROMA_DB_PATH", ".tmp/chroma_db")
    data_rooms_path: str = os.getenv("DATA_ROOMS_PATH", ".tmp/data_rooms")

    # Model Configuration
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    claude_haiku_model: str = os.getenv("CLAUDE_HAIKU_MODEL", "claude-3-5-haiku-20241022")
    claude_opus_model: str = os.getenv("CLAUDE_OPUS_MODEL", "claude-opus-4-5-20251101")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Financial Analysis Settings
    financial_analysis_model: str = os.getenv("FINANCIAL_ANALYSIS_MODEL", "claude-opus-4-5-20251101")
    financial_extraction_model: str = os.getenv("FINANCIAL_EXTRACTION_MODEL", "claude-sonnet-4-20250514")
    enable_auto_financial_analysis: bool = os.getenv("ENABLE_AUTO_FINANCIAL_ANALYSIS", "False").lower() == "true"
    financial_analysis_max_cost: float = float(os.getenv("FINANCIAL_ANALYSIS_MAX_COST", "5.0"))

    # Processing Settings
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "1500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "4096"))
    streaming_batch_size: int = int(os.getenv("STREAMING_BATCH_SIZE", "512"))
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "100"))

    # Performance & Scaling Settings
    parse_timeout_seconds: int = int(os.getenv("PARSE_TIMEOUT_SECONDS", "180"))
    max_parse_workers: int = int(os.getenv("MAX_PARSE_WORKERS", "8"))
    max_ocr_pages: int = int(os.getenv("MAX_OCR_PAGES", "50"))
    ocr_provider: str = os.getenv("OCR_PROVIDER", "google_document_ai")  # "google_document_ai" | "tesseract"
    google_document_ai_project: str = os.getenv("GOOGLE_DOCUMENT_AI_PROJECT", "")
    google_document_ai_location: str = os.getenv("GOOGLE_DOCUMENT_AI_LOCATION", "us")
    google_document_ai_processor_id: str = os.getenv("GOOGLE_DOCUMENT_AI_PROCESSOR_ID", "")
    max_concurrent_jobs: int = int(os.getenv("MAX_CONCURRENT_JOBS", "10"))
    max_file_memory_mb: int = int(os.getenv("MAX_FILE_MEMORY_MB", "500"))
    job_retry_attempts: int = int(os.getenv("JOB_RETRY_ATTEMPTS", "3"))
    job_poll_interval_seconds: float = float(os.getenv("JOB_POLL_INTERVAL", "0.1"))

    # Rate Limiting for Google Drive Sync
    drive_sync_max_downloads_per_minute: int = int(os.getenv("DRIVE_SYNC_MAX_DOWNLOADS", "600"))
    drive_sync_max_processing_per_minute: int = int(os.getenv("DRIVE_SYNC_MAX_PROCESSING", "600"))
    embedding_max_concurrent: int = int(os.getenv("EMBEDDING_MAX_CONCURRENT", "100"))

    # OpenAI API Rate Limits (Tier 4: 10K RPM, 10M TPM for text-embedding-3-small)
    openai_tier: int = int(os.getenv("OPENAI_TIER", "4"))
    embedding_rpm_limit: int = int(os.getenv("EMBEDDING_RPM_LIMIT", "10000"))
    embedding_tpm_limit: int = int(os.getenv("EMBEDDING_TPM_LIMIT", "10000000"))
    embedding_safety_margin: float = float(os.getenv("EMBEDDING_SAFETY_MARGIN", "0.90"))
    embedding_backoff_multiplier: float = float(os.getenv("EMBEDDING_BACKOFF_MULTIPLIER", "1.5"))

    # Worker Pool Auto-Scaling
    min_workers: int = int(os.getenv("MIN_WORKERS", "8"))
    max_workers: int = int(os.getenv("MAX_WORKERS", "20"))
    scale_up_threshold: int = int(os.getenv("SCALE_UP_THRESHOLD", "3"))
    scale_down_threshold: int = int(os.getenv("SCALE_DOWN_THRESHOLD", "1"))

    # Memory Management
    memory_limit_percent: float = float(os.getenv("MEMORY_LIMIT_PERCENT", "80"))
    gc_threshold_percent: float = float(os.getenv("GC_THRESHOLD_PERCENT", "70"))
    min_free_memory_mb: int = int(os.getenv("MIN_FREE_MEMORY_MB", "500"))

    # Database Optimization
    db_pool_size: int = int(os.getenv("DB_POOL_SIZE", "10"))
    db_batch_size: int = int(os.getenv("DB_BATCH_SIZE", "5000"))

    # Redis Configuration
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_cache_ttl: int = int(os.getenv("REDIS_CACHE_TTL", "3600"))
    redis_enabled: bool = os.getenv("REDIS_ENABLED", "True").lower() == "true"

    # Google Drive Integration
    drive_scan_max_workers: int = int(os.getenv("DRIVE_SCAN_MAX_WORKERS", "20"))
    drive_sync_max_parallel_downloads: int = int(os.getenv("DRIVE_SYNC_MAX_PARALLEL_DOWNLOADS", "40"))
    drive_sync_retry_attempts: int = int(os.getenv("DRIVE_SYNC_RETRY_ATTEMPTS", "3"))
    drive_sync_retry_base_delay: int = int(os.getenv("DRIVE_SYNC_RETRY_BASE_DELAY", "1"))
    drive_api_rate_limit_per_minute: int = int(os.getenv("DRIVE_API_RATE_LIMIT_PER_MINUTE", "600"))
    oauth_state_ttl_seconds: int = int(os.getenv("OAUTH_STATE_TTL_SECONDS", "600"))
    drive_download_chunk_size: int = int(os.getenv("DRIVE_DOWNLOAD_CHUNK_SIZE", "5242880"))

    # Memo Web Search
    memo_web_search_enabled: bool = os.getenv("MEMO_WEB_SEARCH_ENABLED", "True").lower() == "true"

    # Q&A Web Search
    qa_web_search_enabled: bool = os.getenv("QA_WEB_SEARCH_ENABLED", "True").lower() == "true"
    qa_web_search_max_uses: int = int(os.getenv("QA_WEB_SEARCH_MAX_USES", "10"))

    # Sharing Feature
    enable_sharing: bool = os.getenv("ENABLE_SHARING", "True").lower() == "true"

    # Email / SMTP Settings
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")
    frontend_url: str = os.getenv("FRONTEND_URL", "http://localhost:3000")

    # Access Control
    access_password: str = os.getenv("ACCESS_PASSWORD", "")

    # Server Settings
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "False").lower() == "true"

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()


def validate_settings() -> bool:
    """Validate that required settings are configured."""
    errors = []

    if not settings.anthropic_api_key:
        errors.append("ANTHROPIC_API_KEY not set")

    if not settings.openai_api_key:
        errors.append("OPENAI_API_KEY not set")

    # Check database URL is set
    if not settings.database_url:
        errors.append("DATABASE_URL not set")

    # Check chroma db path is writable
    chroma_path = Path(settings.chroma_db_path)
    chroma_path.mkdir(parents=True, exist_ok=True)

    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease update your .env file with required API keys.")
        return False

    return True
