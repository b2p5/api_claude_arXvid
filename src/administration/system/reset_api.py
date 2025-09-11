"""
Reset API endpoints for the RAG system.
Provides REST API endpoints for system reset operations with admin authorization.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from enum import Enum
import json

from reset_service import SystemResetService
from auth import get_current_active_user
from logger import get_logger, log_info, log_warning, log_error


# Pydantic models
class ResetType(str, Enum):
    FULL = "full"
    DOCUMENTS = "documents"
    DATABASES = "databases"
    USER = "user"


class ResetRequest(BaseModel):
    reset_type: ResetType
    username: Optional[str] = None
    create_backup: bool = True
    confirmation_code: str  # Required for safety


class ResetResponse(BaseModel):
    success: bool
    message: str
    stats: Dict[str, Any]
    backup_path: Optional[str] = None
    errors: List[str] = []


class BackupRequest(BaseModel):
    backup_name: Optional[str] = None


class BackupResponse(BaseModel):
    success: bool
    backup_path: str
    message: str


# Router
router = APIRouter(prefix="/admin/reset", tags=["admin", "reset"])
logger = get_logger()


def is_admin_user(current_user: dict = Depends(get_current_active_user)) -> dict:
    """
    Check if current user is admin.
    For now, check if email contains 'admin' - in production use proper role system.
    """
    if "admin" not in current_user.get("email", "").lower():
        raise HTTPException(
            status_code=403,
            detail="Admin privileges required for reset operations"
        )
    return current_user


def validate_confirmation_code(reset_type: ResetType, confirmation_code: str) -> bool:
    """
    Validate confirmation code for reset operation.
    
    Args:
        reset_type: Type of reset being performed
        confirmation_code: Confirmation code provided by user
        
    Returns:
        True if valid, False otherwise
    """
    expected_codes = {
        ResetType.FULL: "RESET_EVERYTHING",
        ResetType.DOCUMENTS: "RESET_DOCUMENTS", 
        ResetType.DATABASES: "RESET_DATABASES",
        ResetType.USER: "RESET_USER"
    }
    
    return confirmation_code == expected_codes.get(reset_type)


@router.post("/create-backup", response_model=BackupResponse)
async def create_backup_endpoint(
    request: BackupRequest,
    current_user: dict = Depends(is_admin_user)
):
    """Create a backup of the current system state."""
    try:
        log_info("Admin backup requested", user=current_user.get("email"), 
                backup_name=request.backup_name)
        
        service = SystemResetService()
        backup_path = service.create_backup(request.backup_name)
        
        return BackupResponse(
            success=True,
            backup_path=backup_path,
            message="Backup created successfully"
        )
        
    except Exception as e:
        log_error("Backup creation failed", user=current_user.get("email"), error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Backup creation failed: {str(e)}"
        )


@router.post("/reset", response_model=ResetResponse)
async def reset_system_endpoint(
    request: ResetRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(is_admin_user)
):
    """
    Perform system reset operation.
    
    Requires admin privileges and correct confirmation code.
    """
    try:
        # Validate confirmation code
        if not validate_confirmation_code(request.reset_type, request.confirmation_code):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid confirmation code for {request.reset_type.value} reset"
            )
        
        # Validate user parameter for user-specific reset
        if request.reset_type == ResetType.USER and not request.username:
            raise HTTPException(
                status_code=400,
                detail="Username is required for user-specific reset"
            )
        
        log_info("Admin reset requested", 
                user=current_user.get("email"),
                reset_type=request.reset_type.value,
                username=request.username)
        
        # Perform reset in background
        background_tasks.add_task(
            perform_reset_background,
            request,
            current_user.get("email")
        )
        
        return ResetResponse(
            success=True,
            message=f"{request.reset_type.value} reset initiated in background",
            stats={},
            backup_path=None,
            errors=[]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log_error("Reset request failed", user=current_user.get("email"), error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Reset request failed: {str(e)}"
        )


async def perform_reset_background(request: ResetRequest, admin_email: str):
    """Background task to perform the actual reset."""
    try:
        service = SystemResetService()
        
        log_info("Starting background reset", admin=admin_email, type=request.reset_type.value)
        
        if request.reset_type == ResetType.FULL:
            stats = service.full_system_reset(request.create_backup)
            
        elif request.reset_type == ResetType.DOCUMENTS:
            stats = service.reset_documents(request.username)
            
        elif request.reset_type == ResetType.DATABASES:
            stats = {
                "vector_db": service.reset_vector_database(),
                "knowledge_db": service.reset_knowledge_database(),
                "embeddings_cache": service.reset_embeddings_cache()
            }
            
        elif request.reset_type == ResetType.USER:
            stats = service.reset_documents(request.username)
        
        # Count total errors
        total_errors = 0
        if isinstance(stats, dict):
            for key, value in stats.items():
                if isinstance(value, dict) and "errors" in value:
                    total_errors += len(value["errors"])
        
        if total_errors > 0:
            log_warning("Reset completed with errors", 
                       admin=admin_email, 
                       type=request.reset_type.value,
                       errors=total_errors,
                       stats=stats)
        else:
            log_info("Reset completed successfully", 
                    admin=admin_email, 
                    type=request.reset_type.value,
                    stats=stats)
        
    except Exception as e:
        log_error("Background reset failed", admin=admin_email, error=str(e))


@router.get("/confirmation-codes")
async def get_confirmation_codes(
    current_user: dict = Depends(is_admin_user)
):
    """Get required confirmation codes for each reset type."""
    return {
        "codes": {
            "full": "RESET_EVERYTHING",
            "documents": "RESET_DOCUMENTS",
            "databases": "RESET_DATABASES", 
            "user": "RESET_USER"
        },
        "warning": "These codes are required to confirm dangerous reset operations"
    }


@router.get("/status")
async def get_reset_status(
    current_user: dict = Depends(is_admin_user)
):
    """Get current system status for reset planning."""
    try:
        import os
        from pathlib import Path
        from config import get_config
        
        config = get_config()
        
        # Count documents
        docs_path = Path(config.arxiv.documents_root)
        doc_count = 0
        user_count = 0
        
        if docs_path.exists():
            user_dirs = [d for d in docs_path.iterdir() if d.is_dir()]
            user_count = len(user_dirs)
            doc_count = sum(1 for f in docs_path.rglob("*.pdf"))
        
        # Check database sizes
        vector_db_size = 0
        vector_db_path = Path(config.database.vector_db_path)
        if vector_db_path.exists():
            vector_db_size = sum(f.stat().st_size for f in vector_db_path.rglob("*") if f.is_file())
        
        knowledge_db_size = 0
        knowledge_db_path = Path(config.database.knowledge_db_path)
        if knowledge_db_path.exists():
            knowledge_db_size = knowledge_db_path.stat().st_size
        
        cache_size = 0
        cache_path = Path("cache")
        if cache_path.exists():
            cache_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
        
        users_db_size = 0
        users_db_path = Path("db/users.db")
        if users_db_path.exists():
            users_db_size = users_db_path.stat().st_size
        
        return {
            "documents": {
                "total_files": doc_count,
                "total_users": user_count,
                "path": str(docs_path)
            },
            "databases": {
                "vector_db_size_bytes": vector_db_size,
                "knowledge_db_size_bytes": knowledge_db_size,
                "users_db_size_bytes": users_db_size,
                "cache_size_bytes": cache_size
            },
            "paths": {
                "documents": str(docs_path),
                "vector_db": str(vector_db_path),
                "knowledge_db": str(knowledge_db_path),
                "users_db": str(users_db_path)
            }
        }
        
    except Exception as e:
        log_error("Status check failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Status check failed: {str(e)}"
        )