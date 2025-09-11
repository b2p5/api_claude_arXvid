"""
Advanced Search Filters for arXiv Papers
Extends the basic filtering system with arXiv-specific filters and smart filtering.
"""

import re
import sqlite3
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

from logger import get_logger, log_info, log_warning, log_error
import knowledge_graph


class DateRangePreset(Enum):
    """Predefined date range presets."""
    LAST_WEEK = "last_week"
    LAST_MONTH = "last_month"
    LAST_3_MONTHS = "last_3_months"
    LAST_6_MONTHS = "last_6_months"
    LAST_YEAR = "last_year"
    LAST_2_YEARS = "last_2_years"
    ALL_TIME = "all_time"


class ArxivCategory(Enum):
    """Major arXiv categories with subcategories."""
    # Computer Science
    CS_AI = "cs.AI"           # Artificial Intelligence
    CS_CL = "cs.CL"           # Computation and Language
    CS_CV = "cs.CV"           # Computer Vision and Pattern Recognition
    CS_LG = "cs.LG"           # Machine Learning
    CS_NE = "cs.NE"           # Neural and Evolutionary Computing
    CS_RO = "cs.RO"           # Robotics
    CS_SE = "cs.SE"           # Software Engineering
    
    # Physics
    PHYSICS_HEP_PH = "hep-ph"  # High Energy Physics - Phenomenology
    PHYSICS_COND_MAT = "cond-mat"  # Condensed Matter
    PHYSICS_ASTRO_PH = "astro-ph"  # Astrophysics
    PHYSICS_GR_QC = "gr-qc"    # General Relativity and Quantum Cosmology
    
    # Mathematics
    MATH_CO = "math.CO"        # Combinatorics
    MATH_DS = "math.DS"        # Dynamical Systems
    MATH_OC = "math.OC"        # Optimization and Control
    MATH_ST = "math.ST"        # Statistics Theory
    
    # Statistics
    STAT_ML = "stat.ML"        # Machine Learning
    STAT_AP = "stat.AP"        # Applications
    
    # Economics
    ECON_EM = "econ.EM"        # Econometrics
    
    # Quantitative Biology
    Q_BIO_NC = "q-bio.NC"      # Neurons and Cognition
    
    # Quantitative Finance
    Q_FIN_ST = "q-fin.ST"      # Statistical Finance


@dataclass 
class AdvancedSearchFilters:
    """Advanced search filters with arXiv-specific options."""
    
    # Basic filters
    authors: Optional[List[str]] = None
    author_mode: str = "any"  # "any", "all", "exact"
    
    # Date filters
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    date_preset: Optional[DateRangePreset] = None
    
    # ArXiv specific filters
    arxiv_categories: Optional[List[str]] = None
    arxiv_id_pattern: Optional[str] = None  # e.g., "2023.*", "cs.AI.*"
    
    # Content filters
    title_keywords: Optional[List[str]] = None
    abstract_keywords: Optional[List[str]] = None
    exclude_keywords: Optional[List[str]] = None
    
    # Quality/relevance filters
    min_relevance_score: float = 0.0
    max_results: int = 50
    min_content_length: int = 0
    
    # Advanced filters
    has_code: Optional[bool] = None
    has_dataset: Optional[bool] = None
    language: str = "en"
    
    # Sorting and ranking
    sort_by: str = "relevance"  # "relevance", "date", "title", "authors"
    sort_ascending: bool = False
    
    def __post_init__(self):
        """Validate and normalize filter parameters."""
        # Normalize author names
        if self.authors:
            self.authors = [author.strip() for author in self.authors if author.strip()]
        
        # Normalize keywords
        for keyword_list in [self.title_keywords, self.abstract_keywords, self.exclude_keywords]:
            if keyword_list:
                keyword_list[:] = [kw.strip().lower() for kw in keyword_list if kw.strip()]
        
        # Normalize arXiv categories
        if self.arxiv_categories:
            self.arxiv_categories = [cat.strip().lower() for cat in self.arxiv_categories if cat.strip()]
        
        # Apply date preset if specified
        if self.date_preset and not (self.date_from or self.date_to):
            self._apply_date_preset()
        
        # Validate date range
        if self.date_from and self.date_to and self.date_from > self.date_to:
            self.date_from, self.date_to = self.date_to, self.date_from
    
    def _apply_date_preset(self):
        """Apply predefined date range preset."""
        today = date.today()
        
        if self.date_preset == DateRangePreset.LAST_WEEK:
            self.date_from = today - timedelta(weeks=1)
        elif self.date_preset == DateRangePreset.LAST_MONTH:
            self.date_from = today - timedelta(days=30)
        elif self.date_preset == DateRangePreset.LAST_3_MONTHS:
            self.date_from = today - timedelta(days=90)
        elif self.date_preset == DateRangePreset.LAST_6_MONTHS:
            self.date_from = today - timedelta(days=180)
        elif self.date_preset == DateRangePreset.LAST_YEAR:
            self.date_from = today - timedelta(days=365)
        elif self.date_preset == DateRangePreset.LAST_2_YEARS:
            self.date_from = today - timedelta(days=730)
        
        self.date_to = today
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert filters to dictionary for serialization."""
        return {
            'authors': self.authors,
            'author_mode': self.author_mode,
            'date_from': self.date_from.isoformat() if self.date_from else None,
            'date_to': self.date_to.isoformat() if self.date_to else None,
            'arxiv_categories': self.arxiv_categories,
            'arxiv_id_pattern': self.arxiv_id_pattern,
            'title_keywords': self.title_keywords,
            'abstract_keywords': self.abstract_keywords,
            'exclude_keywords': self.exclude_keywords,
            'min_relevance_score': self.min_relevance_score,
            'max_results': self.max_results,
            'has_code': self.has_code,
            'has_dataset': self.has_dataset,
            'language': self.language,
            'sort_by': self.sort_by,
            'sort_ascending': self.sort_ascending
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdvancedSearchFilters':
        """Create filters from dictionary."""
        # Parse dates
        if data.get('date_from'):
            data['date_from'] = datetime.fromisoformat(data['date_from']).date()
        if data.get('date_to'):
            data['date_to'] = datetime.fromisoformat(data['date_to']).date()
        
        return cls(**{k: v for k, v in data.items() if v is not None})


class SmartFilterEngine:
    """Smart filtering engine with optimization and suggestions."""
    
    def __init__(self):
        """Initialize the smart filter engine."""
        self.logger = get_logger()
        self._category_cache = {}
        self._author_cache = {}
        
    def apply_filters(
        self, 
        papers: List[Dict[str, Any]], 
        filters: AdvancedSearchFilters
    ) -> List[Dict[str, Any]]:
        """
        Apply advanced filters to a list of papers.
        
        Args:
            papers: List of paper dictionaries with metadata
            filters: AdvancedSearchFilters to apply
            
        Returns:
            Filtered list of papers
        """
        if not papers:
            return papers
        
        log_info("Applying advanced filters", 
                paper_count=len(papers), 
                filter_count=self._count_active_filters(filters))
        
        filtered_papers = papers.copy()
        
        # Apply each filter type
        if filters.authors:
            filtered_papers = self._filter_by_authors(filtered_papers, filters)
        
        if filters.date_from or filters.date_to:
            filtered_papers = self._filter_by_date_range(filtered_papers, filters)
        
        if filters.arxiv_categories:
            filtered_papers = self._filter_by_arxiv_categories(filtered_papers, filters)
        
        if filters.arxiv_id_pattern:
            filtered_papers = self._filter_by_arxiv_id_pattern(filtered_papers, filters)
        
        if filters.title_keywords:
            filtered_papers = self._filter_by_title_keywords(filtered_papers, filters)
        
        if filters.abstract_keywords:
            filtered_papers = self._filter_by_abstract_keywords(filtered_papers, filters)
        
        if filters.exclude_keywords:
            filtered_papers = self._filter_by_exclude_keywords(filtered_papers, filters)
        
        if filters.min_relevance_score > 0:
            filtered_papers = self._filter_by_relevance_score(filtered_papers, filters)
        
        if filters.has_code is not None:
            filtered_papers = self._filter_by_has_code(filtered_papers, filters)
        
        if filters.has_dataset is not None:
            filtered_papers = self._filter_by_has_dataset(filtered_papers, filters)
        
        # Apply sorting
        filtered_papers = self._apply_sorting(filtered_papers, filters)
        
        # Apply result limit
        if filters.max_results > 0:
            filtered_papers = filtered_papers[:filters.max_results]
        
        log_info("Filters applied successfully", 
                before=len(papers), 
                after=len(filtered_papers))
        
        return filtered_papers
    
    def _count_active_filters(self, filters: AdvancedSearchFilters) -> int:
        """Count number of active filters."""
        active_filters = 0
        
        if filters.authors: active_filters += 1
        if filters.date_from or filters.date_to: active_filters += 1
        if filters.arxiv_categories: active_filters += 1
        if filters.arxiv_id_pattern: active_filters += 1
        if filters.title_keywords: active_filters += 1
        if filters.abstract_keywords: active_filters += 1
        if filters.exclude_keywords: active_filters += 1
        if filters.min_relevance_score > 0: active_filters += 1
        if filters.has_code is not None: active_filters += 1
        if filters.has_dataset is not None: active_filters += 1
        
        return active_filters
    
    def _filter_by_authors(
        self, 
        papers: List[Dict[str, Any]], 
        filters: AdvancedSearchFilters
    ) -> List[Dict[str, Any]]:
        """Filter papers by author names."""
        filtered = []
        
        for paper in papers:
            paper_authors = paper.get('authors', [])
            if not paper_authors:
                continue
            
            # Convert to lowercase for comparison
            paper_authors_lower = [author.lower() for author in paper_authors]
            filter_authors_lower = [author.lower() for author in filters.authors]
            
            if filters.author_mode == "all":
                # All filter authors must be present
                if all(any(filter_author in paper_author 
                          for paper_author in paper_authors_lower)
                      for filter_author in filter_authors_lower):
                    filtered.append(paper)
            
            elif filters.author_mode == "exact":
                # Exact match for at least one author
                if any(filter_author == paper_author
                      for filter_author in filter_authors_lower
                      for paper_author in paper_authors_lower):
                    filtered.append(paper)
            
            else:  # "any" mode (default)
                # At least one filter author must be present
                if any(any(filter_author in paper_author 
                          for paper_author in paper_authors_lower)
                      for filter_author in filter_authors_lower):
                    filtered.append(paper)
        
        return filtered
    
    def _filter_by_date_range(
        self, 
        papers: List[Dict[str, Any]], 
        filters: AdvancedSearchFilters
    ) -> List[Dict[str, Any]]:
        """Filter papers by publication date range."""
        filtered = []
        
        for paper in papers:
            pub_date_str = paper.get('publication_date')
            if not pub_date_str:
                continue
            
            try:
                pub_date = datetime.fromisoformat(pub_date_str).date()
                
                # Check date range
                if filters.date_from and pub_date < filters.date_from:
                    continue
                if filters.date_to and pub_date > filters.date_to:
                    continue
                
                filtered.append(paper)
                
            except (ValueError, TypeError):
                # Skip papers with invalid dates
                continue
        
        return filtered
    
    def _filter_by_arxiv_categories(
        self, 
        papers: List[Dict[str, Any]], 
        filters: AdvancedSearchFilters
    ) -> List[Dict[str, Any]]:
        """Filter papers by arXiv categories."""
        filtered = []
        
        for paper in papers:
            arxiv_id = paper.get('arxiv_id', '')
            if not arxiv_id:
                continue
            
            # Extract category from arXiv ID (format: category/number or category.subcategory/number)
            # Handle both old format (cs.CV/2306.12345) and new format (2306.12345)
            category_match = re.match(r'^([a-z-]+(?:\.[A-Z]{2})?)', arxiv_id.lower())
            if not category_match:
                continue
            
            paper_category = category_match.group(1)
            
            # Check if paper category matches any filter category (case insensitive)
            if any(filter_cat.lower() in paper_category.lower() or paper_category.lower().startswith(filter_cat.lower())
                  for filter_cat in filters.arxiv_categories):
                filtered.append(paper)
        
        return filtered
    
    def _filter_by_arxiv_id_pattern(
        self, 
        papers: List[Dict[str, Any]], 
        filters: AdvancedSearchFilters
    ) -> List[Dict[str, Any]]:
        """Filter papers by arXiv ID pattern."""
        filtered = []
        
        try:
            # Convert pattern to regex
            pattern = filters.arxiv_id_pattern.replace('*', '.*')
            regex = re.compile(pattern, re.IGNORECASE)
            
            for paper in papers:
                arxiv_id = paper.get('arxiv_id', '')
                if arxiv_id and regex.match(arxiv_id):
                    filtered.append(paper)
        
        except re.error:
            log_warning("Invalid arXiv ID pattern", pattern=filters.arxiv_id_pattern)
            return papers
        
        return filtered
    
    def _filter_by_title_keywords(
        self, 
        papers: List[Dict[str, Any]], 
        filters: AdvancedSearchFilters
    ) -> List[Dict[str, Any]]:
        """Filter papers by title keywords."""
        filtered = []
        
        for paper in papers:
            title = paper.get('title', '').lower()
            if not title:
                continue
            
            # Check if any keyword is in the title
            if any(keyword in title for keyword in filters.title_keywords):
                filtered.append(paper)
        
        return filtered
    
    def _filter_by_abstract_keywords(
        self, 
        papers: List[Dict[str, Any]], 
        filters: AdvancedSearchFilters
    ) -> List[Dict[str, Any]]:
        """Filter papers by abstract keywords."""
        filtered = []
        
        for paper in papers:
            summary = paper.get('summary', '').lower()
            if not summary:
                continue
            
            # Check if any keyword is in the abstract
            if any(keyword in summary for keyword in filters.abstract_keywords):
                filtered.append(paper)
        
        return filtered
    
    def _filter_by_exclude_keywords(
        self, 
        papers: List[Dict[str, Any]], 
        filters: AdvancedSearchFilters
    ) -> List[Dict[str, Any]]:
        """Filter out papers containing excluded keywords."""
        filtered = []
        
        for paper in papers:
            title = paper.get('title', '').lower()
            summary = paper.get('summary', '').lower()
            content = f"{title} {summary}"
            
            # Exclude if any excluded keyword is found
            if not any(keyword in content for keyword in filters.exclude_keywords):
                filtered.append(paper)
        
        return filtered
    
    def _filter_by_relevance_score(
        self, 
        papers: List[Dict[str, Any]], 
        filters: AdvancedSearchFilters
    ) -> List[Dict[str, Any]]:
        """Filter papers by minimum relevance score."""
        filtered = []
        
        for paper in papers:
            score = paper.get('relevance_score', 0.0)
            if score >= filters.min_relevance_score:
                filtered.append(paper)
        
        return filtered
    
    def _filter_by_has_code(
        self, 
        papers: List[Dict[str, Any]], 
        filters: AdvancedSearchFilters
    ) -> List[Dict[str, Any]]:
        """Filter papers by code availability."""
        filtered = []
        
        code_indicators = ['github', 'code', 'implementation', 'repository', 'available']
        
        for paper in papers:
            summary = paper.get('summary', '').lower()
            has_code_mentioned = any(indicator in summary for indicator in code_indicators)
            
            if filters.has_code == has_code_mentioned:
                filtered.append(paper)
        
        return filtered
    
    def _filter_by_has_dataset(
        self, 
        papers: List[Dict[str, Any]], 
        filters: AdvancedSearchFilters
    ) -> List[Dict[str, Any]]:
        """Filter papers by dataset availability."""
        filtered = []
        
        dataset_indicators = ['dataset', 'data', 'benchmark', 'corpus', 'collection']
        
        for paper in papers:
            summary = paper.get('summary', '').lower()
            has_dataset_mentioned = any(indicator in summary for indicator in dataset_indicators)
            
            if filters.has_dataset == has_dataset_mentioned:
                filtered.append(paper)
        
        return filtered
    
    def _apply_sorting(
        self, 
        papers: List[Dict[str, Any]], 
        filters: AdvancedSearchFilters
    ) -> List[Dict[str, Any]]:
        """Apply sorting to filtered papers."""
        if not papers:
            return papers
        
        try:
            if filters.sort_by == "date":
                papers.sort(
                    key=lambda p: p.get('publication_date', ''), 
                    reverse=not filters.sort_ascending
                )
            elif filters.sort_by == "title":
                papers.sort(
                    key=lambda p: p.get('title', '').lower(), 
                    reverse=not filters.sort_ascending
                )
            elif filters.sort_by == "authors":
                papers.sort(
                    key=lambda p: ', '.join(p.get('authors', [])).lower(), 
                    reverse=not filters.sort_ascending
                )
            else:  # Default to relevance
                papers.sort(
                    key=lambda p: p.get('relevance_score', 0.0), 
                    reverse=not filters.sort_ascending
                )
        
        except Exception as e:
            log_warning("Sorting failed", error=str(e), sort_by=filters.sort_by)
        
        return papers
    
    def get_filter_suggestions(self, query: str) -> Dict[str, List[str]]:
        """Get filter suggestions based on query and existing data."""
        suggestions = {
            'authors': [],
            'categories': [],
            'keywords': [],
            'years': []
        }
        
        try:
            with knowledge_graph.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Author suggestions
                cursor.execute("""
                    SELECT DISTINCT name FROM authors 
                    WHERE name LIKE ? 
                    ORDER BY name ASC 
                    LIMIT 10
                """, (f"%{query}%",))
                suggestions['authors'] = [row[0] for row in cursor.fetchall()]
                
                # Year suggestions from recent papers
                cursor.execute("""
                    SELECT DISTINCT strftime('%Y', publication_date) as year
                    FROM papers 
                    WHERE publication_date IS NOT NULL
                    ORDER BY year DESC 
                    LIMIT 10
                """)
                suggestions['years'] = [row[0] for row in cursor.fetchall() if row[0]]
        
        except Exception as e:
            log_error("Failed to get filter suggestions", error=str(e))
        
        # Category suggestions
        suggestions['categories'] = [cat.value for cat in ArxivCategory 
                                   if query.lower() in cat.value.lower()][:10]
        
        return suggestions
    
    def optimize_filters(self, filters: AdvancedSearchFilters) -> AdvancedSearchFilters:
        """Optimize filters for better performance and results."""
        optimized = filters
        
        # Remove empty or ineffective filters
        if optimized.authors:
            optimized.authors = [a for a in optimized.authors if len(a) > 1]
        
        if optimized.title_keywords:
            optimized.title_keywords = [k for k in optimized.title_keywords if len(k) > 2]
        
        if optimized.abstract_keywords:
            optimized.abstract_keywords = [k for k in optimized.abstract_keywords if len(k) > 2]
        
        # Optimize date ranges
        if optimized.date_from and optimized.date_to:
            delta = optimized.date_to - optimized.date_from
            if delta.days > 3650:  # More than 10 years
                log_warning("Very broad date range might affect performance", 
                           days=delta.days)
        
        return optimized


def create_smart_filter_engine() -> SmartFilterEngine:
    """Create and return configured smart filter engine instance."""
    return SmartFilterEngine()