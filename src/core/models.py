"""
Pydantic models for FastAPI request/response validation.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import re


class UserBase(BaseModel):
    """Base user model."""
    username: str = Field(..., min_length=1, max_length=50, description="Username")
    
    @validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_\s\-\.]+$', v):
            raise ValueError('Username can only contain letters, numbers, spaces, hyphens, dots and underscores')
        return v


class DocumentBase(BaseModel):
    """Base document model."""
    filename: str = Field(..., description="Document filename")
    category: str = Field(..., min_length=1, max_length=100, description="Document category")
    size: Optional[int] = Field(None, description="File size in bytes")
    created: Optional[datetime] = Field(None, description="Creation timestamp")
    modified: Optional[datetime] = Field(None, description="Modification timestamp")


class DocumentUploadResponse(BaseModel):
    """Response for document upload."""
    message: str
    filename: str
    category: str
    username: str
    processing: str


class ChatRequest(BaseModel):
    """Chat request model."""
    query: str = Field(..., min_length=1, description="User query")
    category: Optional[str] = Field(None, description="Optional category filter")
    stream: bool = Field(False, description="Whether to stream the response")
    max_tokens: Optional[int] = Field(2000, description="Maximum tokens in response")


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    username: str
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Source documents used")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., min_length=1, description="Search query")
    category: Optional[str] = Field(None, description="Optional category filter")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional search filters")


class SearchResult(BaseModel):
    """Individual search result."""
    title: str
    content: str
    source: str
    category: str
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    """Search response model."""
    results: List[SearchResult]
    total: int
    query: str
    username: str
    processing_time: Optional[float] = None


class UserDocumentsResponse(BaseModel):
    """Response for user documents listing."""
    documents: List[DocumentBase]
    categories: List[str]
    total_documents: int


class UserInfo(BaseModel):
    """User information model."""
    username: str
    document_count: int
    categories: List[str]
    last_activity: Optional[datetime] = None


class UsersListResponse(BaseModel):
    """Response for users listing."""
    users: List[UserInfo]
    total_users: int


class ProcessingStats(BaseModel):
    """Processing statistics model."""
    total_pdfs: int
    processed_pdfs: int
    cached_embeddings: int
    new_embeddings: int
    processing_time: float
    chunks_created: int
    kg_entries_added: int
    vector_entries_added: int


class RebuildIndexResponse(BaseModel):
    """Response for index rebuild."""
    message: str
    username: str
    task_id: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    databases: Dict[str, bool]
    uptime: Optional[float] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str
    error_code: Optional[str] = None
    timestamp: Optional[datetime] = None


class PaperEntity(BaseModel):
    """Paper entity model for knowledge graph."""
    title: str
    summary: Optional[str] = None
    authors: List[str] = Field(default_factory=list)
    publication_date: Optional[str] = None
    source_pdf: str


class KnowledgeGraphStats(BaseModel):
    """Knowledge graph statistics."""
    total_papers: int
    total_authors: int
    total_citations: int
    avg_authors_per_paper: float


class ContentAnalysisResult(BaseModel):
    """Content analysis result."""
    main_topics: List[str]
    key_concepts: List[str]
    research_methods: List[str]
    difficulty_level: str = Field(..., regex="^(beginner|intermediate|advanced)$")
    paper_type: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)


class AdvancedSearchFilters(BaseModel):
    """Advanced search filters."""
    authors: Optional[List[str]] = None
    date_range: Optional[Dict[str, str]] = None  # {"start": "2020-01-01", "end": "2023-12-31"}
    paper_types: Optional[List[str]] = None
    difficulty_levels: Optional[List[str]] = None
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    topics: Optional[List[str]] = None


class BatchProcessRequest(BaseModel):
    """Batch processing request."""
    usernames: List[str] = Field(..., min_items=1)
    force_rebuild: bool = False
    parallel_workers: Optional[int] = Field(None, ge=1, le=16)


class BatchProcessResponse(BaseModel):
    """Batch processing response."""
    message: str
    usernames: List[str]
    total_users: int
    task_id: str


# Request models for specific endpoints
class UploadPDFRequest(BaseModel):
    """PDF upload request (for form validation)."""
    category: str = Field(..., min_length=1, max_length=100)


class DeleteDocumentRequest(BaseModel):
    """Document deletion confirmation."""
    confirm: bool = Field(True, description="Confirmation flag")


# WebSocket models
class WebSocketMessage(BaseModel):
    """WebSocket message model."""
    type: str = Field(..., regex="^(query|response|error|status)$")
    content: str
    username: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class WebSocketResponse(BaseModel):
    """WebSocket response model."""
    type: str
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


# Configuration models
class APIConfig(BaseModel):
    """API configuration model."""
    title: str = "RAG System API"
    description: str = "Multi-user RAG system for academic papers"
    version: str = "1.0.0"
    max_upload_size: int = Field(50 * 1024 * 1024)  # 50MB
    allowed_file_types: List[str] = Field(default_factory=lambda: [".pdf"])
    rate_limit_per_minute: int = 60


# Validation helpers
def validate_pdf_file(filename: str) -> bool:
    """Validate if filename is a PDF."""
    return filename.lower().endswith('.pdf')


def validate_category_name(category: str) -> bool:
    """Validate category name format."""
    return bool(re.match(r'^[a-zA-Z0-9_\s\-]+$', category))