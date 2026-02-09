"""
Pydantic models for API request/response validation.
"""

from typing import List, Optional, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


# Request Models

class DataRoomCreate(BaseModel):
    """Request model for creating a new data room."""
    company_name: str = Field(..., min_length=1, max_length=200)
    analyst_name: str = Field(..., min_length=1, max_length=100)
    analyst_email: Optional[EmailStr] = None
    security_level: Literal["local_only", "cloud_enabled"] = "local_only"


class QuestionRequest(BaseModel):
    """Request model for asking a question."""
    question: str = Field(..., min_length=1, max_length=1000)
    conversation_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None


class MemoGenerateRequest(BaseModel):
    """Request model for generating investment memo."""
    template: Optional[str] = None
    custom_instructions: Optional[str] = None
    ticket_size: Optional[float] = Field(None, description="Investment amount (ticket size) in USD")
    post_money_valuation: Optional[float] = Field(None, description="Post-money valuation in USD")
    valuation_methods: Optional[List[str]] = Field(
        None,
        description="Valuation methods to include: vc_method, revenue_multiple, dcf, comparables"
    )


class MemoChatRequest(BaseModel):
    """Request model for memo follow-up chat."""
    message: str = Field(..., min_length=1, max_length=2000)


class SectionRegenerateRequest(BaseModel):
    """Request model for regenerating a memo section."""
    section_type: Literal[
        "executive_summary",
        "market_analysis",
        "team_assessment",
        "product_technology",
        "financial_analysis",
        "valuation_analysis",
        "risks_concerns",
        "outcome_scenario_analysis",
        "investment_recommendation"
    ]
    feedback: Optional[str] = None


# Response Models

class SourceCitation(BaseModel):
    """Source citation for answers and memos."""
    file_name: str
    page_number: Optional[int] = None
    relevance_score: float
    excerpt: str
    # Excel sheet metadata
    sheet_name: Optional[str] = None
    row_start: Optional[int] = None
    row_end: Optional[int] = None


class DataRoomStatus(BaseModel):
    """Data room processing status."""
    id: str
    company_name: str
    analyst_name: str
    security_level: str
    processing_status: Literal["uploading", "parsing", "indexing", "extracting", "complete", "failed"]
    progress_percent: float
    total_documents: int
    total_chunks: int
    estimated_cost: Optional[float] = None
    actual_cost: float = 0.0
    created_at: datetime
    completed_at: Optional[datetime] = None


class DocumentInfo(BaseModel):
    """Information about a document in the data room."""
    id: str
    file_name: str
    file_type: str
    document_category: Optional[str] = None
    parse_status: str
    page_count: Optional[int] = None
    uploaded_at: datetime
    error_message: Optional[str] = None


class DocumentWithPath(BaseModel):
    """Document with folder path for tree display."""
    id: str
    file_name: str
    file_type: Optional[str] = None
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    parse_status: str
    uploaded_at: Optional[datetime] = None
    file_path: Optional[str] = None


class FolderNode(BaseModel):
    """Folder in document tree."""
    name: str
    path: str
    child_count: int
    has_subfolders: bool = False


class DocumentTreeResponse(BaseModel):
    """Document tree response for folder hierarchy display."""
    data_room_id: str
    current_path: Optional[str] = None
    folders: List[FolderNode] = []
    documents: List[DocumentWithPath] = []
    uploads: List[DocumentWithPath] = []
    total_documents: int = 0


class QuestionAnswer(BaseModel):
    """Answer to an analyst question."""
    id: Optional[str] = None
    question: str
    answer: str
    sources: List[SourceCitation]
    confidence_score: float
    tokens_used: int
    cost: float
    response_time_ms: int
    created_at: Optional[datetime] = None


class MemoSection(BaseModel):
    """Individual memo section."""
    section_type: str
    content: str
    tokens_used: int
    cost: float


class MemoResponse(BaseModel):
    """Investment memo response."""
    id: str
    data_room_id: str
    version: int
    status: Literal["generating", "complete", "failed"]
    executive_summary: Optional[str] = None
    market_analysis: Optional[str] = None
    team_assessment: Optional[str] = None
    product_technology: Optional[str] = None
    financial_analysis: Optional[str] = None
    valuation_analysis: Optional[str] = None
    risks_concerns: Optional[str] = None
    outcome_scenario_analysis: Optional[str] = None
    investment_recommendation: Optional[str] = None
    full_memo: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    tokens_used: int = 0
    cost: float = 0.0
    valuation_methods: Optional[List[str]] = None


class ExtractedFinancials(BaseModel):
    """Extracted financial data."""
    revenue: Optional[float] = None
    revenue_period: Optional[str] = None
    burn_rate: Optional[float] = None
    runway_months: Optional[int] = None
    arr_mrr: Optional[float] = None
    growth_rate: Optional[float] = None
    cac: Optional[float] = None
    ltv: Optional[float] = None
    gross_margin: Optional[float] = None
    employees: Optional[int] = None
    funding_raised: Optional[float] = None
    valuation: Optional[float] = None
    notes: Optional[str] = None


class TeamMember(BaseModel):
    """Team member information."""
    name: str
    role: str
    background: Optional[str] = None
    linkedin: Optional[str] = None


class ExtractedTeam(BaseModel):
    """Extracted team information."""
    founders: List[TeamMember] = []
    executives: List[TeamMember] = []
    advisors: List[TeamMember] = []
    total_employees: Optional[int] = None


class ExtractedMarket(BaseModel):
    """Extracted market information."""
    tam: Optional[float] = None
    sam: Optional[float] = None
    som: Optional[float] = None
    market_description: Optional[str] = None
    competitors: List[str] = []
    competitive_advantages: List[str] = []


class DataRoomSummary(BaseModel):
    """Complete data room summary with extracted data."""
    data_room: DataRoomStatus
    documents: List[DocumentInfo]
    financials: Optional[ExtractedFinancials] = None
    team: Optional[ExtractedTeam] = None
    market: Optional[ExtractedMarket] = None


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    version: str
    database: bool
    vector_db: bool
    api_keys_configured: bool


class CostReport(BaseModel):
    """API cost tracking report."""
    total_cost: float
    cost_by_provider: Dict[str, float]
    cost_by_data_room: Dict[str, float]
    total_tokens: int
    period_start: datetime
    period_end: datetime


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


# Data Room Sharing Models

class InviteMemberRequest(BaseModel):
    """Request to invite a member to a data room."""
    email: EmailStr


class DataRoomMember(BaseModel):
    """Data room member info."""
    id: str
    data_room_id: str
    user_id: Optional[str] = None
    invited_email: str
    role: Literal["owner", "member"]
    status: Literal["pending", "accepted", "revoked"]
    name: Optional[str] = None
    picture_url: Optional[str] = None
    created_at: Optional[datetime] = None
    accepted_at: Optional[datetime] = None


# Google Drive Integration Models

class GoogleAuthURL(BaseModel):
    """Google OAuth authorization URL response."""
    auth_url: str
    state: str


class GoogleAuthCallback(BaseModel):
    """Google OAuth callback request."""
    code: str
    state: str


class UserInfo(BaseModel):
    """User information from Google."""
    id: str
    email: str
    name: Optional[str] = None
    picture_url: Optional[str] = None
    created_at: Optional[datetime] = None


class DriveFile(BaseModel):
    """Google Drive file/folder info."""
    id: str
    name: str
    mimeType: str
    isFolder: bool
    isSupported: bool
    size: Optional[int] = None
    modifiedTime: Optional[str] = None
    webViewLink: Optional[str] = None
    path: Optional[str] = None
    ownerEmail: Optional[str] = None
    sharedByEmail: Optional[str] = None
    shortcutTargetId: Optional[str] = None


class DriveFileList(BaseModel):
    """Google Drive file listing response."""
    files: List[DriveFile]
    nextPageToken: Optional[str] = None
    totalFiles: int
    folderPath: Optional[List[Dict[str, str]]] = None


class ConnectFolderRequest(BaseModel):
    """Request to connect a Google Drive folder."""
    folder_id: str
    folder_name: str
    folder_path: Optional[str] = None
    create_data_room: bool = True
    company_name: Optional[str] = None


class ConnectedFolder(BaseModel):
    """Connected Google Drive folder."""
    id: str
    folder_id: str
    folder_name: str
    folder_path: Optional[str] = None
    data_room_id: Optional[str] = None
    sync_status: str
    sync_stage: str = 'idle'  # idle, discovering, discovered, processing, complete, error
    last_sync_at: Optional[datetime] = None
    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    discovered_files: int = 0
    discovered_folders: int = 0
    current_folder_path: Optional[str] = None
    error_message: Optional[str] = None
    data_room_status: Optional[Dict[str, Any]] = None
    created_at: datetime


class ConnectFilesRequest(BaseModel):
    """Request to connect individual Google Drive files."""
    file_ids: List[str]
    file_names: List[str]
    mime_types: Optional[List[str]] = None
    file_sizes: Optional[List[int]] = None
    file_paths: Optional[List[str]] = None
    create_data_room: bool = True
    data_room_name: Optional[str] = None
    existing_data_room_id: Optional[str] = None


class ConnectedFile(BaseModel):
    """Connected Google Drive file."""
    id: str
    drive_file_id: str
    file_name: str
    file_path: Optional[str] = None
    mime_type: Optional[str] = None
    file_size: Optional[int] = None
    data_room_id: Optional[str] = None
    document_id: Optional[str] = None
    sync_status: str
    error_message: Optional[str] = None
    created_at: datetime


class SyncedFile(BaseModel):
    """Synced file from Google Drive."""
    id: str
    drive_file_id: str
    file_name: str
    file_path: Optional[str] = None
    mime_type: Optional[str] = None
    file_size: Optional[int] = None
    sync_status: str
    last_synced_at: Optional[datetime] = None
    error_message: Optional[str] = None


# Document Preview Models

class DocumentPreview(BaseModel):
    """Preview data for spreadsheet documents."""
    file_name: str
    file_type: str
    sheets: List[str] = []
    current_sheet: Optional[str] = None
    headers: List[str] = []
    rows: List[List[Any]] = []
    total_rows: int = 0
    preview_rows: int = 0
    has_more: bool = False
    error: Optional[str] = None


# Financial Analysis Models

class FinancialMetric(BaseModel):
    """Individual financial metric extracted from Excel."""
    name: str
    category: Literal["revenue", "profitability", "cash", "saas", "unit_economics", "headcount"]
    value: float
    unit: str  # USD, %, months, ratio, count, etc.
    period: str  # "2024", "Q1 2024", "Dec 2024", etc.
    cell_reference: Optional[str] = None
    confidence: Literal["high", "medium", "low"] = "medium"
    source_sheet: Optional[str] = None
    notes: Optional[str] = None


class TimeSeriesDataPoint(BaseModel):
    """Single data point in a time series."""
    period: str
    value: float
    cell_reference: Optional[str] = None


class TimeSeriesMetric(BaseModel):
    """Metric with time series data."""
    metric_name: str
    category: str
    unit: str
    data_points: List[TimeSeriesDataPoint] = []
    growth_rate: Optional[float] = None


class FinancialModelStructure(BaseModel):
    """Structure of the analyzed financial model."""
    has_income_statement: bool = False
    has_balance_sheet: bool = False
    has_cash_flow: bool = False
    has_unit_economics: bool = False
    has_saas_metrics: bool = False
    has_assumptions_sheet: bool = False
    historical_start_year: Optional[int] = None
    historical_end_year: Optional[int] = None
    projection_start_year: Optional[int] = None
    projection_end_year: Optional[int] = None
    granularity: Optional[Literal["annual", "quarterly", "monthly"]] = None
    revenue_model_type: Optional[str] = None
    model_quality_score: Optional[int] = None
    model_quality_notes: List[str] = []


class FinancialValidationIssue(BaseModel):
    """Validation issue found in financial model."""
    issue_type: Literal["consistency", "calculation", "reasonableness", "red_flag", "missing"]
    category: str
    description: str
    cell_references: List[str] = []
    severity: Literal["critical", "high", "medium", "low"]
    expected_value: Optional[float] = None
    actual_value: Optional[float] = None
    recommendation: Optional[str] = None


class FinancialInsight(BaseModel):
    """AI-generated insight about the financials."""
    category: Literal["unit_economics", "growth", "efficiency", "risk", "opportunity"]
    title: str
    insight: str
    supporting_metrics: List[str] = []
    importance: Literal["critical", "high", "medium", "low"]
    sentiment: Literal["positive", "neutral", "negative", "mixed"]


class FollowUpQuestion(BaseModel):
    """Follow-up question for founders."""
    question: str
    reason: str
    priority: Literal["must_ask", "should_ask", "nice_to_ask"]


class KeyMetricSummary(BaseModel):
    """Summary of a key metric with assessment."""
    name: str
    value: str
    assessment: Literal["strong", "acceptable", "concerning", "unknown"]


class RiskAssessment(BaseModel):
    """Risk assessment summary."""
    overall_risk_level: Literal["low", "medium", "high"]
    top_risks: List[str] = []
    mitigating_factors: List[str] = []


class InvestmentThesisNotes(BaseModel):
    """Notes for investment thesis."""
    potential_strengths: List[str] = []
    potential_concerns: List[str] = []
    key_assumptions_to_validate: List[str] = []


class FinancialValidationResults(BaseModel):
    """Complete validation results."""
    overall_score: Optional[int] = None
    passes_basic_checks: bool = True
    consistency_issues: List[FinancialValidationIssue] = []
    reasonableness_flags: List[FinancialValidationIssue] = []
    red_flags: List[FinancialValidationIssue] = []
    missing_elements: List[FinancialValidationIssue] = []
    validation_summary: Optional[str] = None


class MissingMetric(BaseModel):
    """Metric that was expected but not found."""
    name: str
    importance: Literal["critical", "high", "medium", "low"]
    reason: str


class FinancialAnalysisResult(BaseModel):
    """Complete financial analysis result."""
    analysis_id: str
    data_room_id: str
    document_id: str
    file_name: str
    analysis_timestamp: datetime
    status: Literal["in_progress", "complete", "failed"]

    # Model structure
    model_structure: Optional[FinancialModelStructure] = None

    # Extracted metrics
    extracted_metrics: List[FinancialMetric] = []
    time_series: List[TimeSeriesMetric] = []
    missing_metrics: List[MissingMetric] = []

    # Validation
    validation_results: Optional[FinancialValidationResults] = None

    # Insights
    insights: List[FinancialInsight] = []
    follow_up_questions: List[FollowUpQuestion] = []
    key_metrics_summary: Optional[Dict[str, Any]] = None
    risk_assessment: Optional[RiskAssessment] = None
    investment_thesis_notes: Optional[InvestmentThesisNotes] = None
    executive_summary: Optional[str] = None

    # Cost tracking
    analysis_cost: float = 0.0
    tokens_used: int = 0
    processing_time_ms: int = 0
    error: Optional[str] = None


class FinancialAnalysisTriggerRequest(BaseModel):
    """Request to trigger financial analysis."""
    force_reanalyze: bool = False


class FinancialAnalysisTriggerResponse(BaseModel):
    """Response after triggering financial analysis."""
    analysis_id: str
    status: str
    message: str


class FinancialSummary(BaseModel):
    """Aggregated financial summary for a data room."""
    data_room_id: str
    analyzed_documents: int = 0
    total_metrics: int = 0

    # Key metrics across all documents
    revenue_latest: Optional[FinancialMetric] = None
    revenue_growth: Optional[float] = None
    gross_margin: Optional[float] = None
    burn_rate: Optional[float] = None
    runway_months: Optional[int] = None
    arr_mrr: Optional[float] = None
    ltv_cac_ratio: Optional[float] = None

    # Aggregated insights
    top_insights: List[FinancialInsight] = []
    critical_issues: List[FinancialValidationIssue] = []
    key_questions: List[FollowUpQuestion] = []

    # Overall assessment
    overall_model_quality: Optional[int] = None
    overall_risk_level: Optional[str] = None
    executive_summary: Optional[str] = None
