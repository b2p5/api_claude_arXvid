import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

"""
FastAPI application for RAG system with Gemini and ArXiv papers.
Multi-user support with username-based folder structure.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import asyncio
import json
import re
import unicodedata
from typing import List, Optional, AsyncGenerator
import aiofiles

from config import get_config
from logger import get_logger, log_info, log_error
from administration.indexing.rag_bbdd_vector_optimized import OptimizedRAGProcessor
from core.rag.enhanced_rag_processor import EnhancedRAGProcessor
from scripts.chat_with_advanced_search import AdvancedChatRAG
from core.search.advanced_search import AdvancedSearchEngine
from core.analysis.knowledge_graph import create_database
from core.analysis.pdf_validator import validate_pdf
from core.auth import (
    UserCreate, UserLogin, Token, User,
    register_user, login_user, get_current_active_user, get_username_from_user
)
from core.paper_service import (
    PaperService, PaperSearchRequest, PaperDownloadRequest,
    PaperSearchResult, DownloadedPaper, get_paper_service
)
from administration.system.reset_api import router as reset_router
from core.analysis.document_processor import get_processing_service, process_document_background


# Initialize configuration and logger
config = get_config()
logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    log_info("Starting FastAPI RAG application")
    
    # Initialize databases
    create_database()
    log_info("Databases initialized")
    
    yield
    
    log_info("Shutting down FastAPI RAG application")


# Create FastAPI app with lifespan
app = FastAPI(
    title="RAG System API",
    description="Multi-user RAG system for academic papers with Gemini and ArXiv integration",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include reset API router
app.include_router(reset_router)


def normalize_username(username: str) -> str:
    """
    Normalize username for safe filesystem usage.
    
    Args:
        username: Original username
        
    Returns:
        Normalized username safe for filesystem
    """
    # Convert to lowercase
    normalized = username.lower()
    
    # Remove accents
    normalized = unicodedata.normalize('NFD', normalized)
    normalized = ''.join(char for char in normalized if unicodedata.category(char) != 'Mn')
    
    # Replace spaces and special characters with underscores
    normalized = re.sub(r'[^a-z0-9_]', '_', normalized)
    
    # Remove multiple consecutive underscores
    normalized = re.sub(r'_+', '_', normalized)
    
    # Remove leading/trailing underscores
    normalized = normalized.strip('_')
    
    return normalized


def get_user_documents_path(username: str) -> str:
    """
    Get the documents path for a specific user.
    
    Args:
        username: Username (will be normalized)
        
    Returns:
        Full path to user's documents directory
    """
    safe_username = normalize_username(username)
    return os.path.join(config.arxiv.documents_root, safe_username)


def ensure_user_directory_structure(username: str):
    """
    Ensure the user's directory structure exists.
    
    Args:
        username: Username
    """
    user_path = get_user_documents_path(username)
    os.makedirs(user_path, exist_ok=True)
    log_info("User directory created/verified", username=username, path=user_path)


async def get_rag_processor() -> OptimizedRAGProcessor:
    """Dependency to get RAG processor instance."""
    return OptimizedRAGProcessor()


async def get_enhanced_rag_processor() -> EnhancedRAGProcessor:
    """Dependency to get enhanced RAG processor instance."""
    return EnhancedRAGProcessor()


async def get_chat_system() -> AdvancedChatRAG:
    """Dependency to get advanced chat system instance."""
    return AdvancedChatRAG()


async def get_search_engine() -> AdvancedSearchEngine:
    """Dependency to get search engine instance."""
    return AdvancedSearchEngine()


async def get_paper_service_dep() -> PaperService:
    """Dependency to get paper service instance."""
    return get_paper_service()


# Authentication endpoints
@app.post("/auth/register", response_model=dict)
async def register(user_data: UserCreate):
    """Register a new user."""
    return await register_user(user_data)


@app.post("/auth/login", response_model=dict)
async def login(login_data: UserLogin):
    """Login user and get access token."""
    return await login_user(login_data)


@app.get("/auth/me", response_model=dict)
async def get_current_user_info(current_user: dict = Depends(get_current_active_user)):
    """Get current user information."""
    return {"user": current_user}


# Paper Service endpoints
@app.get("/papers/search")
async def search_papers(
    query: str,
    max_results: int = 20,
    sort_by: str = "relevance",
    current_user: dict = Depends(get_current_active_user),
    paper_service: PaperService = Depends(get_paper_service_dep)
):
    """
    Search for papers on arXiv.
    
    Args:
        query: Search query
        max_results: Maximum number of results (default: 20)
        sort_by: Sort criterion (relevance, lastUpdatedDate, submittedDate)
        current_user: Authenticated user
        paper_service: Paper service dependency
    """
    try:
        search_request = PaperSearchRequest(
            query=query,
            max_results=max_results,
            sort_by=sort_by
        )
        
        results = await paper_service.search_papers(search_request)
        
        return {
            "query": query,
            "results": [result.dict() for result in results],
            "total": len(results)
        }
        
    except Exception as e:
        log_error("Paper search failed", error=str(e), query=query, user=current_user.get("email"))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/papers/download")
async def download_paper(
    download_request: PaperDownloadRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_active_user),
    paper_service: PaperService = Depends(get_paper_service_dep)
):
    """
    Download a paper from arXiv.
    
    Args:
        download_request: Download parameters (arxiv_id, category)
        background_tasks: Background task handler
        current_user: Authenticated user
        paper_service: Paper service dependency
    """
    try:
        username = get_username_from_user(current_user)
        
        # Download paper
        downloaded_paper = await paper_service.download_paper(username, download_request)
        
        # Start background processing for RAG
        background_tasks.add_task(
            process_downloaded_paper,
            username,
            downloaded_paper.file_path
        )
        
        return {
            "message": "Paper downloaded successfully",
            "paper": downloaded_paper.dict(),
            "processing": "started_in_background"
        }
        
    except Exception as e:
        log_error("Paper download failed", 
                 error=str(e), 
                 arxiv_id=download_request.arxiv_id, 
                 user=current_user.get("email"))
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@app.get("/papers/downloaded")
async def list_downloaded_papers(
    category: Optional[str] = None,
    current_user: dict = Depends(get_current_active_user),
    paper_service: PaperService = Depends(get_paper_service_dep)
):
    """
    List papers downloaded by the user.
    
    Args:
        category: Optional category filter
        current_user: Authenticated user
        paper_service: Paper service dependency
    """
    try:
        username = get_username_from_user(current_user)
        papers = paper_service.list_downloaded_papers(username, category)
        
        return {
            "papers": [paper.dict() for paper in papers],
            "total": len(papers),
            "username": username
        }
        
    except Exception as e:
        log_error("Failed to list downloaded papers", 
                 error=str(e), 
                 user=current_user.get("email"))
        raise HTTPException(status_code=500, detail=f"Failed to list papers: {str(e)}")


@app.delete("/papers/{arxiv_id}")
async def delete_downloaded_paper(
    arxiv_id: str,
    category: str,
    current_user: dict = Depends(get_current_active_user),
    paper_service: PaperService = Depends(get_paper_service_dep)
):
    """
    Delete a downloaded paper.
    
    Args:
        arxiv_id: ArXiv ID of paper to delete
        category: Category of the paper
        current_user: Authenticated user
        paper_service: Paper service dependency
    """
    try:
        username = get_username_from_user(current_user)
        
        success = paper_service.delete_paper(username, arxiv_id, category)
        
        if success:
            return {"message": "Paper deleted successfully", "arxiv_id": arxiv_id}
        else:
            raise HTTPException(status_code=404, detail="Paper not found")
            
    except HTTPException:
        raise
    except Exception as e:
        log_error("Failed to delete paper", 
                 error=str(e), 
                 arxiv_id=arxiv_id, 
                 user=current_user.get("email"))
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG System API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
        "auth": "JWT required for protected endpoints"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "databases": {
            "vector_db": os.path.exists(config.database.vector_db_path),
            "knowledge_db": os.path.exists(config.database.knowledge_db_path)
        }
    }


@app.post("/upload-pdf")
async def upload_pdf(
    category: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Upload a PDF file for the authenticated user with improved processing tracking.
    
    Args:
        category: Document category/topic
        file: PDF file to upload
        background_tasks: Background task handler
        current_user: Authenticated user
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Get username from authenticated user
        username = get_username_from_user(current_user)
        
        # Ensure user directory structure
        ensure_user_directory_structure(username)
        
        # Create category directory
        user_path = get_user_documents_path(username)
        category_path = os.path.join(user_path, category)
        os.makedirs(category_path, exist_ok=True)
        
        # Save file
        file_path = os.path.join(category_path, file.filename)
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Create processing task
        processing_service = get_processing_service()
        task_id = processing_service.create_processing_task(
            username=username,
            file_path=file_path,
            category=category,
            filename=file.filename
        )
        
        # Start processing in background
        background_tasks.add_task(process_document_background, task_id)
        
        log_info("PDF uploaded successfully", 
                username=username, 
                category=category, 
                filename=file.filename,
                task_id=task_id)
        
        return {
            "message": "PDF uploaded successfully",
            "filename": file.filename,
            "category": category,
            "username": username,
            "task_id": task_id,
            "processing_status": "pending"
        }
        
    except Exception as e:
        log_error("PDF upload failed", username=username, error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/processing-status/{task_id}")
async def get_processing_status(
    task_id: str,
    current_user: dict = Depends(get_current_active_user)
):
    """Get the processing status of a document."""
    try:
        processing_service = get_processing_service()
        status = processing_service.get_task_status(task_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        # Verify user owns this task
        username = get_username_from_user(current_user)
        if status['username'] != username:
            raise HTTPException(status_code=403, detail="Not authorized to view this task")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        log_error("Failed to get processing status", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get status")


@app.get("/my-processing-tasks")
async def get_my_processing_tasks(
    status: Optional[str] = None,
    current_user: dict = Depends(get_current_active_user)
):
    """Get all processing tasks for the current user."""
    try:
        username = get_username_from_user(current_user)
        processing_service = get_processing_service()
        
        from document_processor import ProcessingStatus
        status_filter = None
        if status:
            try:
                status_filter = ProcessingStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
        
        tasks = processing_service.get_user_tasks(username, status_filter)
        
        return {
            "tasks": tasks,
            "total": len(tasks),
            "username": username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        log_error("Failed to get user processing tasks", username=username, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get tasks")


@app.get("/processing-stats")
async def get_processing_stats(
    current_user: dict = Depends(get_current_active_user)
):
    """Get processing statistics for the system."""
    try:
        processing_service = get_processing_service()
        stats = processing_service.get_processing_stats()
        
        return {
            "processing_stats": stats,
            "user": get_username_from_user(current_user)
        }
        
    except Exception as e:
        log_error("Failed to get processing stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get stats")


async def validate_and_process_pdf(file_path: str, username: str, rag_processor: OptimizedRAGProcessor):
    """Background task to validate and process uploaded PDF."""
    try:
        # Validate PDF
        validation_result = validate_pdf(file_path)
        if not validation_result.is_valid:
            log_error("PDF validation failed", file=file_path, errors=validation_result.errors)
            return
        
        # Process PDF into databases
        user_docs_path = get_user_documents_path(username)
        results = rag_processor.update_databases(user_docs_path, config.database.vector_db_path)
        
        log_info("PDF processing completed", 
                file=file_path, 
                username=username, 
                stats=results.get('stats', {}))
        
    except Exception as e:
        log_error("PDF processing failed", file=file_path, username=username, error=str(e))


async def process_downloaded_paper(username: str, file_path: str):
    """Background task to process downloaded paper for RAG."""
    try:
        log_info("Processing downloaded paper", file=file_path, username=username)
        
        # Validate PDF first
        validation_result = validate_pdf(file_path)
        if not validation_result.is_valid:
            log_error("Downloaded PDF validation failed", file=file_path, errors=validation_result.errors)
            return
        
        # Process into RAG system
        rag_processor = OptimizedRAGProcessor()
        user_docs_path = get_user_documents_path(username)
        results = rag_processor.update_databases(user_docs_path, config.database.vector_db_path)
        
        log_info("Downloaded paper processing completed", 
                file=file_path, 
                username=username, 
                stats=results.get('stats', {}))
        
    except Exception as e:
        log_error("Downloaded paper processing failed", file=file_path, username=username, error=str(e))


@app.get("/my-documents")
async def list_user_documents(current_user: dict = Depends(get_current_active_user)):
    """
    List all documents for the authenticated user.
    
    Args:
        current_user: Authenticated user
    """
    username = get_username_from_user(current_user)
    user_path = get_user_documents_path(username)
    
    if not os.path.exists(user_path):
        return {"documents": [], "categories": []}
    
    documents = []
    categories = set()
    
    try:
        for category in os.listdir(user_path):
            category_path = os.path.join(user_path, category)
            if os.path.isdir(category_path):
                categories.add(category)
                
                for filename in os.listdir(category_path):
                    if filename.endswith('.pdf'):
                        file_path = os.path.join(category_path, filename)
                        file_stats = os.stat(file_path)
                        
                        documents.append({
                            "filename": filename,
                            "category": category,
                            "size": file_stats.st_size,
                            "created": file_stats.st_ctime,
                            "modified": file_stats.st_mtime
                        })
        
        return {
            "documents": documents,
            "categories": list(categories),
            "total_documents": len(documents)
        }
        
    except Exception as e:
        log_error("Failed to list user documents", username=username, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.post("/chat")
async def chat_with_documents(
    query: str,
    category: Optional[str] = None,
    stream: bool = False,
    current_user: dict = Depends(get_current_active_user),
    chat_system: AdvancedChatRAG = Depends(get_chat_system)
):
    """
    Chat with user's documents using RAG.
    
    Args:
        query: User question/query
        category: Optional category filter
        stream: Whether to stream the response
        current_user: Authenticated user
        chat_system: Advanced chat system dependency
    """
    try:
        # Get username from authenticated user
        username = get_username_from_user(current_user)
        
        # Filter documents by user and optionally by category
        user_path = get_user_documents_path(username)
        
        if not os.path.exists(user_path):
            raise HTTPException(status_code=404, detail="User has no documents")
        
        # Build search options for filtering
        search_options = {"user_path": user_path}
        if category:
            search_options["category"] = category
        
        # Use the chat system
        result = chat_system.chat(query, search_options)
        
        return {
            "response": result.get("response", "No response generated"),
            "username": username,
            "search_results": result.get("search_results", []),
            "intent": result.get("intent", {})
        }
            
    except Exception as e:
        log_error("Chat processing failed", username=username, query=query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


# Streaming functionality removed - using direct chat response


@app.post("/search")
async def search_documents(
    query: str,
    category: Optional[str] = None,
    limit: int = 10,
    current_user: dict = Depends(get_current_active_user),
    search_engine: AdvancedSearchEngine = Depends(get_search_engine)
):
    """
    Advanced search in user's documents.
    
    Args:
        query: Search query
        category: Optional category filter
        limit: Maximum number of results
        current_user: Authenticated user
        search_engine: Search engine dependency
    """
    try:
        username = get_username_from_user(current_user)
        user_path = get_user_documents_path(username)
        
        if not os.path.exists(user_path):
            return {"results": [], "total": 0}
        
        # Build search filters
        filters = {"user_path": user_path}
        if category:
            filters["category"] = category
        
        results = await search_engine.search_async(query, filters, limit)
        
        return {
            "results": results,
            "total": len(results),
            "query": query,
            "username": username
        }
        
    except Exception as e:
        log_error("Search failed", username=username, query=query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.delete("/documents/{category}/{filename}")
async def delete_document(
    category: str, 
    filename: str,
    current_user: dict = Depends(get_current_active_user)
):
    """
    Delete a specific document.
    
    Args:
        category: Document category
        filename: File name to delete
        current_user: Authenticated user
    """
    try:
        username = get_username_from_user(current_user)
        user_path = get_user_documents_path(username)
        file_path = os.path.join(user_path, category, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Document not found")
        
        os.remove(file_path)
        
        # TODO: Remove from databases as well
        
        log_info("Document deleted", username=username, category=category, filename=filename)
        
        return {"message": "Document deleted successfully"}
        
    except Exception as e:
        log_error("Document deletion failed", 
                 username=username, category=category, filename=filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")


@app.post("/rebuild-index")
async def rebuild_user_index(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_active_user),
    rag_processor: OptimizedRAGProcessor = Depends(get_rag_processor)
):
    """
    Rebuild the index for the authenticated user's documents.
    
    Args:
        background_tasks: Background task handler
        current_user: Authenticated user
        rag_processor: RAG processor dependency
    """
    try:
        username = get_username_from_user(current_user)
        user_path = get_user_documents_path(username)
        
        if not os.path.exists(user_path):
            raise HTTPException(status_code=404, detail="User has no documents")
        
        # Start rebuild in background
        background_tasks.add_task(rebuild_index_task, user_path, username, rag_processor)
        
        return {"message": "Index rebuild started", "username": username}
        
    except Exception as e:
        log_error("Index rebuild failed", username=username, error=str(e))
        raise HTTPException(status_code=500, detail=f"Index rebuild failed: {str(e)}")


async def rebuild_index_task(user_path: str, username: str, rag_processor: OptimizedRAGProcessor):
    """Background task to rebuild user's index."""
    try:
        log_info("Starting index rebuild", username=username, path=user_path)
        
        # Force rebuild for this user's documents
        processor = OptimizedRAGProcessor(force_rebuild=True)
        results = processor.update_databases(user_path, config.database.vector_db_path)
        
        log_info("Index rebuild completed", username=username, results=results.get('stats', {}))
        
    except Exception as e:
        log_error("Index rebuild task failed", username=username, error=str(e))


# Admin endpoint - removed for security
# Only authenticated users can access their own data


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )