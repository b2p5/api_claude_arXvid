#!/usr/bin/env python3
"""
Automatic Document Processing Service for RAG System.
Handles PDF processing, embedding generation, and database indexing.
"""

import os
import uuid
import sqlite3
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
from pathlib import Path
import json
from dataclasses import dataclass

from rag_bbdd_vector_optimized import OptimizedRAGProcessor
from pdf_validator import validate_pdf
from config import get_config
from logger import get_logger, log_info, log_error, log_warning


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DocumentProcessingTask:
    """Represents a document processing task."""
    task_id: str
    username: str
    file_path: str
    category: str
    filename: str
    status: ProcessingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    processing_stats: Optional[Dict[str, Any]] = None


class DocumentProcessingService:
    """Service for automatic document processing with queue management."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.processing_db_path = "db/document_processing.db"
        self._init_processing_database()
        
    def _init_processing_database(self):
        """Initialize the processing status database."""
        try:
            conn = sqlite3.connect(self.processing_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS processing_tasks (
                    task_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    category TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TIMESTAMP NOT NULL,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    error_message TEXT,
                    processing_stats TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_processing_username 
                ON processing_tasks(username)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_processing_status 
                ON processing_tasks(status)
            ''')
            
            conn.commit()
            conn.close()
            
            log_info("Processing database initialized", db_path=self.processing_db_path)
            
        except Exception as e:
            log_error("Failed to initialize processing database", error=str(e))
            raise
    
    def create_processing_task(self, username: str, file_path: str, 
                             category: str, filename: str) -> str:
        """
        Create a new document processing task.
        
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        task = DocumentProcessingTask(
            task_id=task_id,
            username=username,
            file_path=file_path,
            category=category,
            filename=filename,
            status=ProcessingStatus.PENDING,
            created_at=datetime.now()
        )
        
        try:
            conn = sqlite3.connect(self.processing_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO processing_tasks 
                (task_id, username, file_path, category, filename, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.task_id, task.username, task.file_path, 
                task.category, task.filename, task.status.value, 
                task.created_at
            ))
            
            conn.commit()
            conn.close()
            
            log_info("Processing task created", 
                    task_id=task_id, 
                    username=username, 
                    filename=filename)
            
            return task_id
            
        except Exception as e:
            log_error("Failed to create processing task", 
                     username=username, 
                     filename=filename, 
                     error=str(e))
            raise
    
    def update_task_status(self, task_id: str, status: ProcessingStatus, 
                          error_message: Optional[str] = None,
                          processing_stats: Optional[Dict[str, Any]] = None):
        """Update the status of a processing task."""
        try:
            conn = sqlite3.connect(self.processing_db_path)
            cursor = conn.cursor()
            
            # Prepare update data
            now = datetime.now()
            stats_json = json.dumps(processing_stats) if processing_stats else None
            
            if status == ProcessingStatus.PROCESSING:
                cursor.execute('''
                    UPDATE processing_tasks 
                    SET status = ?, started_at = ?, error_message = NULL
                    WHERE task_id = ?
                ''', (status.value, now, task_id))
                
            elif status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]:
                cursor.execute('''
                    UPDATE processing_tasks 
                    SET status = ?, completed_at = ?, error_message = ?, processing_stats = ?
                    WHERE task_id = ?
                ''', (status.value, now, error_message, stats_json, task_id))
            
            conn.commit()
            conn.close()
            
            log_info("Task status updated", 
                    task_id=task_id, 
                    status=status.value, 
                    has_error=error_message is not None)
            
        except Exception as e:
            log_error("Failed to update task status", 
                     task_id=task_id, 
                     error=str(e))
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a processing task."""
        try:
            conn = sqlite3.connect(self.processing_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM processing_tasks WHERE task_id = ?
            ''', (task_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                stats = json.loads(row['processing_stats']) if row['processing_stats'] else None
                return {
                    'task_id': row['task_id'],
                    'username': row['username'],
                    'filename': row['filename'],
                    'category': row['category'],
                    'status': row['status'],
                    'created_at': row['created_at'],
                    'started_at': row['started_at'],
                    'completed_at': row['completed_at'],
                    'error_message': row['error_message'],
                    'processing_stats': stats
                }
            
            return None
            
        except Exception as e:
            log_error("Failed to get task status", task_id=task_id, error=str(e))
            return None
    
    def get_user_tasks(self, username: str, status: Optional[ProcessingStatus] = None) -> List[Dict[str, Any]]:
        """Get all processing tasks for a user."""
        try:
            conn = sqlite3.connect(self.processing_db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if status:
                cursor.execute('''
                    SELECT * FROM processing_tasks 
                    WHERE username = ? AND status = ?
                    ORDER BY created_at DESC
                ''', (username, status.value))
            else:
                cursor.execute('''
                    SELECT * FROM processing_tasks 
                    WHERE username = ?
                    ORDER BY created_at DESC
                ''', (username,))
            
            rows = cursor.fetchall()
            conn.close()
            
            tasks = []
            for row in rows:
                stats = json.loads(row['processing_stats']) if row['processing_stats'] else None
                tasks.append({
                    'task_id': row['task_id'],
                    'filename': row['filename'],
                    'category': row['category'],
                    'status': row['status'],
                    'created_at': row['created_at'],
                    'started_at': row['started_at'],
                    'completed_at': row['completed_at'],
                    'error_message': row['error_message'],
                    'processing_stats': stats
                })
            
            return tasks
            
        except Exception as e:
            log_error("Failed to get user tasks", username=username, error=str(e))
            return []
    
    async def process_document(self, task_id: str):
        """Process a document asynchronously."""
        task_info = self.get_task_status(task_id)
        if not task_info:
            log_error("Task not found", task_id=task_id)
            return
        
        # Update status to processing
        self.update_task_status(task_id, ProcessingStatus.PROCESSING)
        
        try:
            log_info("Starting document processing", 
                    task_id=task_id, 
                    filename=task_info['filename'],
                    username=task_info['username'])
            
            # Validate PDF
            file_path = task_info.get('file_path')  # This should be in processing_tasks table
            if not file_path:
                # Reconstruct file path from stored info
                from main import get_user_documents_path
                user_path = get_user_documents_path(task_info['username'])
                file_path = os.path.join(user_path, task_info['category'], task_info['filename'])
            
            validation_result = validate_pdf(file_path)
            if not validation_result.is_valid:
                error_msg = f"PDF validation failed: {'; '.join(validation_result.errors)}"
                self.update_task_status(task_id, ProcessingStatus.FAILED, error_msg)
                return
            
            # Process with RAG processor
            rag_processor = OptimizedRAGProcessor()
            
            # Get user documents path
            from main import get_user_documents_path
            user_docs_path = get_user_documents_path(task_info['username'])
            
            # Process the document
            results = rag_processor.update_databases(
                user_docs_path, 
                self.config.database.vector_db_path
            )
            
            # Extract processing statistics
            processing_stats = {
                'processing_time': results.get('stats', {}).get('processing_time', 0),
                'documents_processed': results.get('stats', {}).get('documents_processed', 0),
                'embeddings_generated': results.get('stats', {}).get('embeddings_generated', 0),
                'db_size_mb': results.get('stats', {}).get('db_size_mb', 0),
                'processed_at': datetime.now().isoformat()
            }
            
            # Mark as completed
            self.update_task_status(
                task_id, 
                ProcessingStatus.COMPLETED, 
                processing_stats=processing_stats
            )
            
            log_info("Document processing completed successfully", 
                    task_id=task_id,
                    filename=task_info['filename'],
                    stats=processing_stats)
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            self.update_task_status(task_id, ProcessingStatus.FAILED, error_msg)
            log_error("Document processing failed", 
                     task_id=task_id,
                     filename=task_info.get('filename', 'unknown'),
                     error=str(e))
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get overall processing statistics."""
        try:
            conn = sqlite3.connect(self.processing_db_path)
            cursor = conn.cursor()
            
            # Get counts by status
            cursor.execute('''
                SELECT status, COUNT(*) as count 
                FROM processing_tasks 
                GROUP BY status
            ''')
            status_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Get recent activity (last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) FROM processing_tasks 
                WHERE created_at > datetime('now', '-1 day')
            ''')
            recent_tasks = cursor.fetchone()[0]
            
            # Get average processing time for completed tasks
            cursor.execute('''
                SELECT AVG(
                    (julianday(completed_at) - julianday(started_at)) * 24 * 60 * 60
                ) as avg_processing_time_seconds
                FROM processing_tasks 
                WHERE status = 'completed' 
                AND started_at IS NOT NULL 
                AND completed_at IS NOT NULL
            ''')
            avg_processing_time = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'status_counts': status_counts,
                'recent_tasks_24h': recent_tasks,
                'average_processing_time_seconds': round(avg_processing_time, 2),
                'queue_size': status_counts.get('pending', 0) + status_counts.get('processing', 0)
            }
            
        except Exception as e:
            log_error("Failed to get processing stats", error=str(e))
            return {}


# Global service instance
_processing_service = None


def get_processing_service() -> DocumentProcessingService:
    """Get the global processing service instance."""
    global _processing_service
    if _processing_service is None:
        _processing_service = DocumentProcessingService()
    return _processing_service


# Background processing function for integration with FastAPI
async def process_document_background(task_id: str):
    """Background task wrapper for document processing."""
    service = get_processing_service()
    await service.process_document(task_id)