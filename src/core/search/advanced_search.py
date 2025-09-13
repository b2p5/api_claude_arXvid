"""
Advanced Search System for arXiv Papers RAG
Implements hybrid search (semantic + keyword), filters, and ranking.
"""

import os
import re
import sqlite3
import numpy as np
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import get_config
from logger import get_logger, log_info, log_warning, log_error
from core.analysis import knowledge_graph


class SearchMode(Enum):
    """Search modes available."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword" 
    HYBRID = "hybrid"


class RankingStrategy(Enum):
    """Ranking strategies for search results."""
    RELEVANCE = "relevance"
    RECENCY = "recency"
    CITATION_COUNT = "citation_count"
    HYBRID_SCORE = "hybrid_score"
    CUSTOM = "custom"


@dataclass
class SearchFilters:
    """Search filters for advanced filtering."""
    authors: Optional[List[str]] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    arxiv_categories: Optional[List[str]] = None
    min_relevance_score: float = 0.0
    max_results: int = 50
    
    def __post_init__(self):
        """Validate filter parameters."""
        if self.authors:
            self.authors = [author.lower().strip() for author in self.authors]
        if self.arxiv_categories:
            self.arxiv_categories = [cat.lower().strip() for cat in self.arxiv_categories]


@dataclass
class SearchResult:
    """Single search result with metadata."""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    final_score: float = 0.0
    paper_id: Optional[int] = None
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    publication_date: Optional[date] = None
    arxiv_id: Optional[str] = None


@dataclass
class SearchConfig:
    """Configuration for search behavior."""
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    boost_title_matches: float = 1.5
    boost_author_matches: float = 1.2
    boost_recent_papers: float = 1.1  # Papers from last year
    enable_fuzzy_matching: bool = True
    fuzzy_threshold: float = 0.8
    ranking_strategy: RankingStrategy = RankingStrategy.HYBRID_SCORE


class AdvancedSearchEngine:
    """Advanced search engine with hybrid search, filters, and ranking."""
    
    def __init__(self, db_path: str = None, embedding_model: str = None):
        """Initialize the advanced search engine."""
        self.config = get_config()
        self.logger = get_logger()
        self.search_config = SearchConfig()
        
        # Database paths
        self.vector_db_path = db_path or self.config.database.vector_db_path
        self.kg_db_path = self.config.database.knowledge_db_path
        
        # Initialize components
        self._initialize_vector_db()
        self._initialize_embeddings(embedding_model)
        self._initialize_tfidf()
        
        log_info("Advanced search engine initialized",
                vector_db_path=self.vector_db_path,
                kg_db_path=self.kg_db_path)
    
    def _initialize_vector_db(self):
        """Initialize ChromaDB vector database."""
        try:
            if not os.path.exists(self.vector_db_path):
                raise FileNotFoundError(f"Vector database not found: {self.vector_db_path}")
            
            self.vector_db = chromadb.PersistentClient(path=self.vector_db_path)
            self.collection = self.vector_db.get_collection(
                name=self.config.database.vector_collection_name
            )
            log_info("Vector database initialized successfully")
            
        except Exception as e:
            log_error("Failed to initialize vector database", error=str(e))
            raise
    
    def _initialize_embeddings(self, model_name: str = None):
        """Initialize embedding model."""
        try:
            model_name = model_name or self.config.models.embedding_model_name
            self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)
            log_info("Embedding model initialized", model=model_name)
            
        except Exception as e:
            log_error("Failed to initialize embedding model", error=str(e))
            raise
    
    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer for keyword search."""
        try:
            # Get all documents from vector database for TF-IDF corpus
            all_docs = self.collection.get()
            corpus = all_docs['documents'] if all_docs['documents'] else [""]
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            if len(corpus) > 1:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
                self.doc_ids = all_docs['ids']
                log_info("TF-IDF initialized", corpus_size=len(corpus))
            else:
                log_warning("Empty corpus for TF-IDF initialization")
                self.tfidf_matrix = None
                self.doc_ids = []
                
        except Exception as e:
            log_error("Failed to initialize TF-IDF", error=str(e))
            self.tfidf_matrix = None
            self.doc_ids = []
    
    def search(
        self, 
        query: str, 
        mode: SearchMode = SearchMode.HYBRID,
        filters: SearchFilters = None,
        ranking: RankingStrategy = None,
        search_config: SearchConfig = None
    ) -> List[SearchResult]:
        """
        Perform advanced search with multiple modes and filters.
        
        Args:
            query: Search query string
            mode: Search mode (semantic, keyword, hybrid)
            filters: Search filters to apply
            ranking: Ranking strategy for results
            search_config: Custom search configuration
            
        Returns:
            List of search results sorted by relevance
        """
        log_info("Starting advanced search", 
                query=query[:100], 
                mode=mode.value, 
                has_filters=filters is not None)
        
        # Use provided config or default
        config = search_config or self.search_config
        filters = filters or SearchFilters()
        ranking = ranking or config.ranking_strategy
        
        # Perform search based on mode
        if mode == SearchMode.SEMANTIC:
            results = self._semantic_search(query, filters, config)
        elif mode == SearchMode.KEYWORD:
            results = self._keyword_search(query, filters, config)
        elif mode == SearchMode.HYBRID:
            results = self._hybrid_search(query, filters, config)
        else:
            raise ValueError(f"Unknown search mode: {mode}")
        
        # Enrich results with knowledge graph data
        results = self._enrich_with_kg_data(results)
        
        # Apply ranking strategy
        results = self._apply_ranking(results, ranking, config)
        
        # Apply final filters and limits
        results = self._apply_final_filters(results, filters)
        
        log_info("Search completed", 
                results_count=len(results),
                top_score=results[0].final_score if results else 0)
        
        return results
    
    def _semantic_search(
        self, 
        query: str, 
        filters: SearchFilters, 
        config: SearchConfig
    ) -> List[SearchResult]:
        """Perform semantic search using vector embeddings."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Search in vector database
            vector_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(filters.max_results * 2, 200),  # Get more for filtering
                include=['metadatas', 'documents', 'distances']
            )
            
            results = []
            for i, (doc_id, content, metadata, distance) in enumerate(zip(
                vector_results['ids'][0],
                vector_results['documents'][0],
                vector_results['metadatas'][0],
                vector_results['distances'][0]
            )):
                # Convert distance to similarity score (assuming cosine distance)
                semantic_score = 1.0 - distance if distance <= 1.0 else 0.0
                
                result = SearchResult(
                    doc_id=doc_id,
                    content=content,
                    metadata=metadata,
                    relevance_score=semantic_score,
                    semantic_score=semantic_score,
                    final_score=semantic_score
                )
                results.append(result)
            
            log_info("Semantic search completed", results_count=len(results))
            return results
            
        except Exception as e:
            log_error("Semantic search failed", error=str(e))
            return []
    
    def _keyword_search(
        self, 
        query: str, 
        filters: SearchFilters, 
        config: SearchConfig
    ) -> List[SearchResult]:
        """Perform keyword search using TF-IDF."""
        if self.tfidf_matrix is None:
            log_warning("TF-IDF not initialized, returning empty results")
            return []
        
        try:
            # Vectorize query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarity scores
            similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = np.argsort(similarity_scores)[::-1][:filters.max_results * 2]
            
            results = []
            for idx in top_indices:
                if similarity_scores[idx] < filters.min_relevance_score:
                    continue
                
                doc_id = self.doc_ids[idx]
                
                # Get document content and metadata from vector DB
                doc_data = self.collection.get(ids=[doc_id])
                if not doc_data['documents']:
                    continue
                
                result = SearchResult(
                    doc_id=doc_id,
                    content=doc_data['documents'][0],
                    metadata=doc_data['metadatas'][0] if doc_data['metadatas'] else {},
                    relevance_score=similarity_scores[idx],
                    keyword_score=similarity_scores[idx],
                    final_score=similarity_scores[idx]
                )
                results.append(result)
            
            log_info("Keyword search completed", results_count=len(results))
            return results
            
        except Exception as e:
            log_error("Keyword search failed", error=str(e))
            return []
    
    def _hybrid_search(
        self, 
        query: str, 
        filters: SearchFilters, 
        config: SearchConfig
    ) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword search."""
        # Get results from both search methods
        semantic_results = self._semantic_search(query, filters, config)
        keyword_results = self._keyword_search(query, filters, config)
        
        # Combine results by document ID
        combined_results = {}
        
        # Add semantic results
        for result in semantic_results:
            combined_results[result.doc_id] = result
        
        # Add/merge keyword results
        for result in keyword_results:
            if result.doc_id in combined_results:
                # Combine scores
                existing = combined_results[result.doc_id]
                existing.keyword_score = result.keyword_score
                existing.final_score = (
                    config.semantic_weight * existing.semantic_score +
                    config.keyword_weight * result.keyword_score
                )
            else:
                # New result from keyword search only
                result.final_score = config.keyword_weight * result.keyword_score
                combined_results[result.doc_id] = result
        
        # Convert to list and sort by final score
        results = list(combined_results.values())
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        log_info("Hybrid search completed", 
                results_count=len(results),
                semantic_only=len(semantic_results) - len([r for r in results if r.keyword_score > 0]),
                keyword_only=len(keyword_results) - len([r for r in results if r.semantic_score > 0]),
                combined=len([r for r in results if r.semantic_score > 0 and r.keyword_score > 0]))
        
        return results
    
    def _enrich_with_kg_data(self, results: List[SearchResult]) -> List[SearchResult]:
        """Enrich search results with knowledge graph data."""
        try:
            with knowledge_graph.get_db_connection() as conn:
                cursor = conn.cursor()
                
                for result in results:
                    # Try to find paper by source PDF path
                    source_path = result.metadata.get('source', '')
                    if source_path:
                        cursor.execute("""
                            SELECT p.id, p.title, p.summary, p.publication_date,
                                   GROUP_CONCAT(a.name, ', ') as authors
                            FROM papers p
                            LEFT JOIN paper_authors pa ON p.id = pa.paper_id
                            LEFT JOIN authors a ON pa.author_id = a.id
                            WHERE p.source_pdf = ?
                            GROUP BY p.id
                        """, (source_path,))
                        
                        row = cursor.fetchone()
                        if row:
                            result.paper_id = row[0]
                            result.title = row[1]
                            result.authors = row[4].split(', ') if row[4] else []
                            
                            # Parse publication date
                            if row[3]:
                                try:
                                    result.publication_date = datetime.fromisoformat(row[3]).date()
                                except ValueError:
                                    pass
            
            log_info("Results enriched with knowledge graph data", 
                    enriched=len([r for r in results if r.paper_id is not None]))
            
        except Exception as e:
            log_error("Failed to enrich results with KG data", error=str(e))
        
        return results
    
    def _apply_ranking(
        self, 
        results: List[SearchResult], 
        strategy: RankingStrategy, 
        config: SearchConfig
    ) -> List[SearchResult]:
        """Apply ranking strategy to search results."""
        if not results:
            return results
        
        current_date = date.today()
        
        for result in results:
            if strategy == RankingStrategy.RELEVANCE:
                # Use existing relevance score
                pass
            elif strategy == RankingStrategy.RECENCY:
                # Boost recent papers
                if result.publication_date:
                    days_old = (current_date - result.publication_date).days
                    recency_boost = max(0, 1.0 - (days_old / 365.0))  # Linear decay over 1 year
                    result.final_score = result.relevance_score * (1.0 + recency_boost)
            elif strategy == RankingStrategy.HYBRID_SCORE:
                # Apply various boosts
                boost_factor = 1.0
                
                # Title matching boost
                if result.title and any(term.lower() in result.title.lower() 
                                     for term in result.content.split()[:10]):
                    boost_factor *= config.boost_title_matches
                
                # Recent papers boost
                if result.publication_date:
                    days_old = (current_date - result.publication_date).days
                    if days_old < 365:  # Papers from last year
                        boost_factor *= config.boost_recent_papers
                
                result.final_score = result.relevance_score * boost_factor
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        log_info("Ranking applied", strategy=strategy.value, top_score=results[0].final_score)
        return results
    
    def _apply_final_filters(
        self, 
        results: List[SearchResult], 
        filters: SearchFilters
    ) -> List[SearchResult]:
        """Apply final filters to search results."""
        filtered_results = []
        
        for result in results:
            # Author filter
            if filters.authors:
                if not result.authors or not any(
                    filter_author in ' '.join(result.authors).lower()
                    for filter_author in filters.authors
                ):
                    continue
            
            # Date range filter
            if filters.date_from and result.publication_date:
                if result.publication_date < filters.date_from:
                    continue
            
            if filters.date_to and result.publication_date:
                if result.publication_date > filters.date_to:
                    continue
            
            # Relevance score filter
            if result.final_score < filters.min_relevance_score:
                continue
            
            filtered_results.append(result)
            
            # Max results limit
            if len(filtered_results) >= filters.max_results:
                break
        
        log_info("Final filters applied", 
                before=len(results), 
                after=len(filtered_results))
        
        return filtered_results
    
    def get_search_suggestions(self, partial_query: str, limit: int = 10) -> List[str]:
        """Get search suggestions based on partial query."""
        if not partial_query or len(partial_query) < 2:
            return []
        
        suggestions = []
        
        try:
            with knowledge_graph.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Search in paper titles
                cursor.execute("""
                    SELECT DISTINCT title FROM papers 
                    WHERE title LIKE ? 
                    ORDER BY LENGTH(title) ASC
                    LIMIT ?
                """, (f"%{partial_query}%", limit))
                
                suggestions.extend([row[0] for row in cursor.fetchall()])
                
                # Search in author names
                if len(suggestions) < limit:
                    cursor.execute("""
                        SELECT DISTINCT name FROM authors 
                        WHERE name LIKE ? 
                        ORDER BY LENGTH(name) ASC
                        LIMIT ?
                    """, (f"%{partial_query}%", limit - len(suggestions)))
                    
                    suggestions.extend([row[0] for row in cursor.fetchall()])
        
        except Exception as e:
            log_error("Failed to get search suggestions", error=str(e))
        
        return suggestions[:limit]
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        stats = {
            'vector_db_size': 0,
            'kg_papers': 0,
            'kg_authors': 0,
            'tfidf_vocab_size': 0,
            'last_updated': datetime.now().isoformat()
        }
        
        try:
            # Vector database stats
            all_docs = self.collection.get()
            stats['vector_db_size'] = len(all_docs['ids']) if all_docs['ids'] else 0
            
            # Knowledge graph stats
            with knowledge_graph.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM papers")
                stats['kg_papers'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM authors")
                stats['kg_authors'] = cursor.fetchone()[0]
            
            # TF-IDF stats
            if hasattr(self.tfidf_vectorizer, 'vocabulary_'):
                stats['tfidf_vocab_size'] = len(self.tfidf_vectorizer.vocabulary_)
        
        except Exception as e:
            log_error("Failed to get search statistics", error=str(e))
        
        return stats


def create_search_engine() -> AdvancedSearchEngine:
    """Create and return configured search engine instance."""
    return AdvancedSearchEngine()