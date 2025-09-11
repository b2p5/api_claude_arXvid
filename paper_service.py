"""
Paper Service for arXiv integration.
Provides search, download, and management of academic papers.
"""

import os
import asyncio
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import arxiv
from pydantic import BaseModel

from config import get_config
from logger import get_logger, log_info, log_error, log_warning
from get_arxiv import search_arxiv, save_and_download_results
from pdf_validator import validate_pdf
from auth import get_username_from_user


# Pydantic models
class PaperSearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = None
    sort_by: Optional[str] = "relevance"  # relevance, lastUpdatedDate, submittedDate


class PaperDownloadRequest(BaseModel):
    arxiv_id: str
    category: Optional[str] = "arxiv_papers"


class PaperSearchResult(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str]
    summary: str
    published: str
    updated: str
    pdf_url: str
    categories: List[str]
    journal_ref: Optional[str] = None


class DownloadedPaper(BaseModel):
    arxiv_id: str
    title: str
    authors: List[str]
    category: str
    file_path: str
    downloaded_at: str
    file_size: int
    is_processed: bool = False


@dataclass
class PaperServiceConfig:
    """Configuration for Paper Service."""
    base_documents_path: str
    max_search_results: int = 50
    download_timeout: int = 300
    auto_process: bool = True


class PaperService:
    """Service for managing arXiv papers."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.service_config = PaperServiceConfig(
            base_documents_path=self.config.arxiv.documents_root,
            max_search_results=self.config.arxiv.max_results,
            auto_process=True
        )
    
    def get_user_papers_path(self, username: str) -> str:
        """Get the papers directory for a user."""
        return os.path.join(self.service_config.base_documents_path, username)
    
    def ensure_user_papers_directory(self, username: str, category: str = "arxiv_papers"):
        """Ensure user's papers directory exists."""
        user_path = self.get_user_papers_path(username)
        category_path = os.path.join(user_path, category)
        os.makedirs(category_path, exist_ok=True)
        return category_path
    
    def _download_single_paper(self, paper_info, file_path: str) -> bool:
        """
        Download a single paper PDF.
        
        Args:
            paper_info: ArXiv paper info object
            file_path: Full path where to save the PDF
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            import urllib.request
            from retry_utils import download_with_retry
            
            log_info("Downloading paper PDF", arxiv_id=paper_info.entry_id.split('/')[-1]) 
            
            # Use retry utility for robust download
            success = download_with_retry(paper_info.pdf_url, file_path)
            
            if success:
                log_info("Paper PDF downloaded successfully", file_path=file_path)
                return True
            else:
                log_error("Paper PDF download failed", url=paper_info.pdf_url)
                return False
                
        except Exception as e:
            log_error("Error downloading paper PDF", error=str(e), url=paper_info.pdf_url)
            return False
    
    async def search_papers(self, search_request: PaperSearchRequest) -> List[PaperSearchResult]:
        """
        Search for papers on arXiv.
        
        Args:
            search_request: Search parameters
            
        Returns:
            List of paper search results
        """
        try:
            log_info("Searching arXiv papers", query=search_request.query)
            
            # Use the existing search_arxiv function
            results = search_arxiv(search_request.query)
            
            papers = []
            count = 0
            max_results = search_request.max_results or self.service_config.max_search_results
            
            for result in results:
                if count >= max_results:
                    break
                
                paper = PaperSearchResult(
                    arxiv_id=result.entry_id.split('/')[-1],
                    title=result.title,
                    authors=[str(author) for author in result.authors],
                    summary=result.summary,
                    published=result.published.isoformat(),
                    updated=result.updated.isoformat(),
                    pdf_url=result.pdf_url,
                    categories=result.categories,
                    journal_ref=result.journal_ref
                )
                papers.append(paper)
                count += 1
            
            log_info("ArXiv search completed", query=search_request.query, results_count=len(papers))
            return papers
            
        except Exception as e:
            log_error("ArXiv search failed", error=str(e), query=search_request.query)
            raise
    
    async def download_paper(self, username: str, download_request: PaperDownloadRequest) -> DownloadedPaper:
        """
        Download a paper from arXiv.
        
        Args:
            username: User identifier
            download_request: Download parameters
            
        Returns:
            Information about the downloaded paper
        """
        try:
            log_info("Downloading paper", arxiv_id=download_request.arxiv_id, user=username)
            
            # Ensure directory exists
            category_path = self.ensure_user_papers_directory(username, download_request.category)
            
            # Get paper info first
            client = arxiv.Client()
            search = arxiv.Search(id_list=[download_request.arxiv_id])
            results = list(client.results(search))
            
            if not results:
                raise ValueError(f"Paper with ID {download_request.arxiv_id} not found")
            
            paper_info = results[0]
            
            # Generate safe filename
            safe_title = "".join(c for c in paper_info.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title[:100]  # Limit length
            filename = f"{download_request.arxiv_id}_{safe_title}.pdf"
            file_path = os.path.join(category_path, filename)
            
            # Download the paper
            success = await asyncio.get_event_loop().run_in_executor(
                None, 
                self._download_single_paper, 
                paper_info, 
                file_path
            )
            
            if not success:
                raise Exception("Failed to download paper")
            
            # Validate PDF
            validation_result = validate_pdf(file_path)
            if not validation_result.is_valid:
                os.remove(file_path)  # Clean up invalid file
                raise Exception(f"Downloaded PDF is invalid: {', '.join(validation_result.errors)}")
            
            # Get file size
            file_size = os.path.getsize(file_path)
            
            downloaded_paper = DownloadedPaper(
                arxiv_id=download_request.arxiv_id,
                title=paper_info.title,
                authors=[str(author) for author in paper_info.authors],
                category=download_request.category,
                file_path=file_path,
                downloaded_at=datetime.now().isoformat(),
                file_size=file_size,
                is_processed=False
            )
            
            log_info("Paper downloaded successfully", 
                    arxiv_id=download_request.arxiv_id, 
                    user=username,
                    file_size=file_size)
            
            return downloaded_paper
            
        except Exception as e:
            log_error("Paper download failed", 
                     error=str(e), 
                     arxiv_id=download_request.arxiv_id, 
                     user=username)
            raise
    
    def list_downloaded_papers(self, username: str, category: Optional[str] = None) -> List[DownloadedPaper]:
        """
        List papers downloaded by a user.
        
        Args:
            username: User identifier
            category: Optional category filter
            
        Returns:
            List of downloaded papers
        """
        try:
            user_path = self.get_user_papers_path(username)
            
            if not os.path.exists(user_path):
                return []
            
            papers = []
            
            # Search in all categories or specific category
            if category:
                categories_to_search = [category]
            else:
                categories_to_search = [d for d in os.listdir(user_path) 
                                      if os.path.isdir(os.path.join(user_path, d))]
            
            for cat in categories_to_search:
                category_path = os.path.join(user_path, cat)
                if not os.path.exists(category_path):
                    continue
                
                for filename in os.listdir(category_path):
                    if filename.endswith('.pdf'):
                        file_path = os.path.join(category_path, filename)
                        file_stats = os.stat(file_path)
                        
                        # Extract arXiv ID from filename
                        arxiv_id = filename.split('_')[0]
                        
                        # Try to get paper info (might be expensive for many papers)
                        title = filename.replace('.pdf', '').replace(f'{arxiv_id}_', '')
                        
                        paper = DownloadedPaper(
                            arxiv_id=arxiv_id,
                            title=title,
                            authors=[],  # Could be populated from paper metadata
                            category=cat,
                            file_path=file_path,
                            downloaded_at=datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                            file_size=file_stats.st_size,
                            is_processed=False  # TODO: Check if processed in RAG system
                        )
                        papers.append(paper)
            
            return papers
            
        except Exception as e:
            log_error("Failed to list downloaded papers", error=str(e), user=username)
            return []
    
    def delete_paper(self, username: str, arxiv_id: str, category: str) -> bool:
        """
        Delete a downloaded paper.
        
        Args:
            username: User identifier
            arxiv_id: ArXiv ID of paper to delete
            category: Category of the paper
            
        Returns:
            True if deleted successfully
        """
        try:
            user_path = self.get_user_papers_path(username)
            category_path = os.path.join(user_path, category)
            
            # Find file with this arXiv ID
            if os.path.exists(category_path):
                for filename in os.listdir(category_path):
                    if filename.startswith(f"{arxiv_id}_") and filename.endswith('.pdf'):
                        file_path = os.path.join(category_path, filename)
                        os.remove(file_path)
                        log_info("Paper deleted", arxiv_id=arxiv_id, user=username)
                        return True
            
            return False
            
        except Exception as e:
            log_error("Failed to delete paper", error=str(e), arxiv_id=arxiv_id, user=username)
            return False


# Global service instance
paper_service = PaperService()


def get_paper_service() -> PaperService:
    """Dependency to get paper service instance."""
    return paper_service