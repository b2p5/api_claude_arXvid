"""
FastAPI utility functions and adapters for existing RAG system components.
"""

import asyncio
import os
from typing import List, Dict, Any, Optional, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
import json

from core.rag.enhanced_rag_processor import EnhancedRAGProcessor
from core.search.advanced_search import AdvancedSearchEngine
from core.analysis.content_analysis import ContentAnalysisEngine
from core.analysis.knowledge_graph import get_db_connection
from logger import get_logger, log_info, log_error

logger = get_logger()


class AsyncRAGAdapter:
    """Adapter to make existing RAG components async-compatible."""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.enhanced_processor = None
        self.search_engine = None
        self.content_engine = None
    
    async def get_enhanced_processor(self) -> EnhancedRAGProcessor:
        """Get or create enhanced processor instance."""
        if not self.enhanced_processor:
            loop = asyncio.get_event_loop()
            self.enhanced_processor = await loop.run_in_executor(
                self.executor, 
                lambda: EnhancedRAGProcessor(enable_content_analysis=True)
            )
        return self.enhanced_processor
    
    async def get_search_engine(self) -> AdvancedSearchEngine:
        """Get or create search engine instance."""
        if not self.search_engine:
            loop = asyncio.get_event_loop()
            self.search_engine = await loop.run_in_executor(
                self.executor,
                lambda: AdvancedSearchEngine()
            )
        return self.search_engine
    
    async def get_content_engine(self) -> ContentAnalysisEngine:
        """Get or create content analysis engine."""
        if not self.content_engine:
            loop = asyncio.get_event_loop()
            self.content_engine = await loop.run_in_executor(
                self.executor,
                lambda: ContentAnalysisEngine()
            )
        return self.content_engine


# Global adapter instance
_adapter = AsyncRAGAdapter()


async def process_query_async(
    query: str, 
    context_filter: Optional[Dict[str, Any]] = None
) -> str:
    """
    Process a query asynchronously using enhanced RAG processor.
    
    Args:
        query: User query
        context_filter: Optional context filters (user_path, category, etc.)
        
    Returns:
        Processed response
    """
    try:
        processor = await _adapter.get_enhanced_processor()
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            _adapter.executor,
            lambda: _process_query_sync(processor, query, context_filter)
        )
        
        return response
        
    except Exception as e:
        log_error("Async query processing failed", query=query, error=str(e))
        return f"Error processing query: {str(e)}"


def _process_query_sync(
    processor: EnhancedRAGProcessor, 
    query: str, 
    context_filter: Optional[Dict[str, Any]]
) -> str:
    """Synchronous query processing wrapper."""
    try:
        # This would be implemented based on your existing enhanced processor
        # For now, return a placeholder response
        log_info("Processing query", query=query, context_filter=context_filter)
        return f"Processed query: {query} with context: {context_filter}"
        
    except Exception as e:
        log_error("Sync query processing failed", query=query, error=str(e))
        raise


async def stream_query_response(
    query: str, 
    context_filter: Optional[Dict[str, Any]] = None
) -> AsyncGenerator[str, None]:
    """
    Stream query response for real-time interaction.
    
    Args:
        query: User query
        context_filter: Optional context filters
        
    Yields:
        Response chunks
    """
    try:
        processor = await _adapter.get_enhanced_processor()
        
        # Simulate streaming response (replace with actual streaming logic)
        response = await process_query_async(query, context_filter)
        
        # Split response into chunks for streaming
        words = response.split()
        for i in range(0, len(words), 5):  # 5 words per chunk
            chunk = " ".join(words[i:i+5])
            yield chunk + " "
            await asyncio.sleep(0.1)  # Small delay for streaming effect
            
    except Exception as e:
        log_error("Streaming response failed", query=query, error=str(e))
        yield f"Error: {str(e)}"


async def search_documents_async(
    query: str, 
    filters: Optional[Dict[str, Any]] = None, 
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Search documents asynchronously.
    
    Args:
        query: Search query
        filters: Search filters
        limit: Maximum number of results
        
    Returns:
        Search results
    """
    try:
        search_engine = await _adapter.get_search_engine()
        
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            _adapter.executor,
            lambda: _search_documents_sync(search_engine, query, filters, limit)
        )
        
        return results
        
    except Exception as e:
        log_error("Async search failed", query=query, error=str(e))
        return []


def _search_documents_sync(
    search_engine: AdvancedSearchEngine,
    query: str,
    filters: Optional[Dict[str, Any]],
    limit: int
) -> List[Dict[str, Any]]:
    """Synchronous search wrapper."""
    try:
        # This would use your existing advanced search engine
        # For now, return placeholder results
        log_info("Searching documents", query=query, filters=filters, limit=limit)
        
        return [
            {
                "title": f"Result {i+1} for: {query}",
                "content": f"Content preview for result {i+1}",
                "source": f"document_{i+1}.pdf",
                "category": filters.get("category", "general") if filters else "general",
                "score": 1.0 - (i * 0.1),
                "metadata": {"page": i+1}
            }
            for i in range(min(limit, 3))  # Return up to 3 placeholder results
        ]
        
    except Exception as e:
        log_error("Sync search failed", query=query, error=str(e))
        raise


async def analyze_content_async(
    file_path: str
) -> Optional[Dict[str, Any]]:
    """
    Analyze document content asynchronously.
    
    Args:
        file_path: Path to document
        
    Returns:
        Content analysis results
    """
    try:
        content_engine = await _adapter.get_content_engine()
        
        loop = asyncio.get_event_loop()
        analysis = await loop.run_in_executor(
            _adapter.executor,
            lambda: _analyze_content_sync(content_engine, file_path)
        )
        
        return analysis
        
    except Exception as e:
        log_error("Async content analysis failed", file_path=file_path, error=str(e))
        return None


def _analyze_content_sync(
    content_engine: ContentAnalysisEngine,
    file_path: str
) -> Dict[str, Any]:
    """Synchronous content analysis wrapper."""
    try:
        # This would use your existing content analysis engine
        log_info("Analyzing content", file_path=file_path)
        
        return {
            "main_topics": ["AI", "Machine Learning"],
            "key_concepts": ["Neural Networks", "Deep Learning"],
            "research_methods": ["Experimental", "Comparative"],
            "difficulty_level": "intermediate",
            "paper_type": "research",
            "confidence_score": 0.85
        }
        
    except Exception as e:
        log_error("Sync content analysis failed", file_path=file_path, error=str(e))
        raise


async def get_user_statistics_async(username: str) -> Dict[str, Any]:
    """
    Get user statistics asynchronously.
    
    Args:
        username: Username
        
    Returns:
        User statistics
    """
    try:
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(
            _adapter.executor,
            lambda: _get_user_statistics_sync(username)
        )
        
        return stats
        
    except Exception as e:
        log_error("Failed to get user statistics", username=username, error=str(e))
        return {"error": str(e)}


def _get_user_statistics_sync(username: str) -> Dict[str, Any]:
    """Get user statistics synchronously."""
    try:
        # Query knowledge graph for user's papers
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Count papers by user (this would need user_id integration)
            cursor.execute("SELECT COUNT(*) FROM papers")
            total_papers = cursor.fetchone()[0]
            
            # Count authors
            cursor.execute("SELECT COUNT(*) FROM authors")
            total_authors = cursor.fetchone()[0]
            
            return {
                "total_papers": total_papers,
                "total_authors": total_authors,
                "last_updated": "2024-01-01",  # Placeholder
                "categories": [],  # Would be populated from file system
                "recent_activity": []
            }
            
    except Exception as e:
        log_error("Failed to get user statistics sync", username=username, error=str(e))
        raise


async def validate_and_process_pdf_async(
    file_path: str,
    username: str,
    category: str
) -> Dict[str, Any]:
    """
    Validate and process PDF asynchronously.
    
    Args:
        file_path: Path to PDF file
        username: Username
        category: Document category
        
    Returns:
        Processing results
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _adapter.executor,
            lambda: _validate_and_process_pdf_sync(file_path, username, category)
        )
        
        return result
        
    except Exception as e:
        log_error("Async PDF processing failed", 
                 file_path=file_path, username=username, error=str(e))
        return {"success": False, "error": str(e)}


def _validate_and_process_pdf_sync(
    file_path: str,
    username: str, 
    category: str
) -> Dict[str, Any]:
    """Validate and process PDF synchronously."""
    try:
        from pdf_validator import validate_pdf
        
        # Validate PDF
        is_valid, error_msg = validate_pdf(file_path)
        if not is_valid:
            return {"success": False, "error": f"PDF validation failed: {error_msg}"}
        
        # Process PDF (this would integrate with your existing processors)
        log_info("Processing PDF", file_path=file_path, username=username, category=category)
        
        return {
            "success": True,
            "file_path": file_path,
            "username": username,
            "category": category,
            "processed_at": "2024-01-01T00:00:00Z"  # Placeholder
        }
        
    except Exception as e:
        log_error("Sync PDF processing failed", 
                 file_path=file_path, username=username, error=str(e))
        raise


def cleanup_adapter():
    """Cleanup adapter resources."""
    global _adapter
    if _adapter and _adapter.executor:
        _adapter.executor.shutdown(wait=True)
        log_info("FastAPI adapter cleaned up")