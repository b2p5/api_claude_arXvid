"""
Advanced Search Ranking System for arXiv Papers
Implements multiple ranking strategies with customizable weights and scoring.
"""

import math
import re
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum

from logger import get_logger, log_info, log_warning, log_error


class RankingFactor(Enum):
    """Individual ranking factors that can be combined."""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    KEYWORD_MATCH = "keyword_match"
    TITLE_MATCH = "title_match"
    AUTHOR_RELEVANCE = "author_relevance"
    RECENCY = "recency"
    CITATION_COUNT = "citation_count"
    CONTENT_QUALITY = "content_quality"
    CATEGORY_RELEVANCE = "category_relevance"
    LENGTH_SCORE = "length_score"
    KEYWORD_DENSITY = "keyword_density"


class RankingStrategy(Enum):
    """Ranking strategies for search results."""
    RELEVANCE = "relevance"
    RECENCY = "recency"
    CITATION_COUNT = "citation_count"
    HYBRID_SCORE = "hybrid_score"
    CUSTOM = "custom"


@dataclass
class RankingWeights:
    """Weights for different ranking factors."""
    semantic_similarity: float = 0.4
    keyword_match: float = 0.2
    title_match: float = 0.15
    author_relevance: float = 0.05
    recency: float = 0.1
    citation_count: float = 0.05
    content_quality: float = 0.03
    category_relevance: float = 0.02
    
    def __post_init__(self):
        """Validate and normalize weights."""
        total = sum(self.__dict__.values())
        if abs(total - 1.0) > 0.01:
            log_warning("Ranking weights don't sum to 1.0", total=total)
            # Normalize weights
            for key, value in self.__dict__.items():
                setattr(self, key, value / total)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return self.__dict__.copy()
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'RankingWeights':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RankingConfig:
    """Configuration for ranking behavior."""
    weights: RankingWeights = field(default_factory=RankingWeights)
    boost_exact_matches: bool = True
    boost_recent_papers: bool = True
    recency_decay_days: int = 365  # Days for full recency decay
    min_score_threshold: float = 0.01
    normalize_scores: bool = True
    use_logarithmic_scaling: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.recency_decay_days <= 0:
            self.recency_decay_days = 365


class AdvancedRankingEngine:
    """Advanced ranking engine with multiple strategies and customizable scoring."""
    
    def __init__(self, config: RankingConfig = None):
        """Initialize the ranking engine."""
        self.config = config or RankingConfig()
        self.logger = get_logger()
        
        # Cache for expensive computations
        self._author_cache = {}
        self._category_cache = {}
        
        log_info("Advanced ranking engine initialized", 
                weights=self.config.weights.to_dict())
    
    def rank_results(
        self, 
        results: List[Dict[str, Any]], 
        query: str,
        query_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Rank search results using advanced scoring.
        
        Args:
            results: List of search results with metadata
            query: Original search query
            query_context: Additional context for ranking (user preferences, etc.)
            
        Returns:
            Ranked list of results with computed scores
        """
        if not results:
            return results
        
        log_info("Starting advanced ranking", 
                result_count=len(results), 
                query_length=len(query))
        
        # Compute individual ranking scores
        scored_results = []
        for result in results:
            scores = self._compute_ranking_scores(result, query, query_context or {})
            final_score = self._combine_scores(scores, self.config.weights)
            
            result_with_score = result.copy()
            result_with_score.update({
                'individual_scores': scores,
                'final_ranking_score': final_score,
                'ranking_explanation': self._generate_explanation(scores, self.config.weights)
            })
            
            scored_results.append(result_with_score)
        
        # Sort by final score
        scored_results.sort(key=lambda x: x['final_ranking_score'], reverse=True)
        
        # Normalize scores if requested
        if self.config.normalize_scores:
            scored_results = self._normalize_final_scores(scored_results)
        
        log_info("Ranking completed", 
                top_score=scored_results[0]['final_ranking_score'] if scored_results else 0,
                bottom_score=scored_results[-1]['final_ranking_score'] if scored_results else 0)
        
        return scored_results
    
    def _compute_ranking_scores(
        self, 
        result: Dict[str, Any], 
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Compute individual ranking scores for a result."""
        scores = {}
        
        # Semantic similarity (from vector search)
        scores[RankingFactor.SEMANTIC_SIMILARITY.value] = result.get('semantic_score', result.get('relevance_score', 0.0))
        
        # Keyword match score
        scores[RankingFactor.KEYWORD_MATCH.value] = self._compute_keyword_match_score(result, query)
        
        # Title match score
        scores[RankingFactor.TITLE_MATCH.value] = self._compute_title_match_score(result, query)
        
        # Author relevance
        scores[RankingFactor.AUTHOR_RELEVANCE.value] = self._compute_author_relevance_score(result, query, context)
        
        # Recency score
        scores[RankingFactor.RECENCY.value] = self._compute_recency_score(result)
        
        # Citation count (if available)
        scores[RankingFactor.CITATION_COUNT.value] = self._compute_citation_score(result)
        
        # Content quality
        scores[RankingFactor.CONTENT_QUALITY.value] = self._compute_content_quality_score(result)
        
        # Category relevance
        scores[RankingFactor.CATEGORY_RELEVANCE.value] = self._compute_category_relevance_score(result, query, context)
        
        return scores
    
    def _compute_keyword_match_score(self, result: Dict[str, Any], query: str) -> float:
        """Compute keyword match score."""
        content = result.get('content', '')
        if not content or not query:
            return 0.0
        
        query_terms = set(query.lower().split())
        content_lower = content.lower()
        
        # Count exact matches
        exact_matches = sum(1 for term in query_terms if term in content_lower)
        
        # Compute match ratio
        match_ratio = exact_matches / len(query_terms) if query_terms else 0.0
        
        # Bonus for exact phrase matches
        phrase_bonus = 1.0
        if len(query.split()) > 1 and query.lower() in content_lower:
            phrase_bonus = 1.2  # Smaller bonus to ensure score differences
        
        final_score = match_ratio * phrase_bonus
        return min(final_score, 1.0)
    
    def _compute_title_match_score(self, result: Dict[str, Any], query: str) -> float:
        """Compute title match score with higher weight."""
        title = result.get('title', '')
        if not title or not query:
            return 0.0
        
        title_lower = title.lower()
        query_lower = query.lower()
        
        # Exact phrase match in title
        if query_lower in title_lower:
            return 1.0
        
        # Individual word matches
        query_words = query_lower.split()
        title_words = title_lower.split()
        
        matches = sum(1 for word in query_words if any(word in title_word for title_word in title_words))
        match_ratio = matches / len(query_words) if query_words else 0.0
        
        # Bonus for matches at the beginning of the title
        if title_words and query_words and any(title_words[0].startswith(word) for word in query_words):
            match_ratio *= 1.1  # Smaller bonus to ensure score differences
        
        return min(match_ratio, 1.0)
    
    def _compute_author_relevance_score(
        self, 
        result: Dict[str, Any], 
        query: str, 
        context: Dict[str, Any]
    ) -> float:
        """Compute author relevance score."""
        authors = result.get('authors', [])
        if not authors:
            return 0.0
        
        # Check if query contains author names
        query_lower = query.lower()
        author_score = 0.0
        
        for author in authors:
            author_lower = author.lower()
            if any(part in query_lower for part in author_lower.split()):
                author_score = 1.0
                break
            
            # Partial name matching
            author_parts = author_lower.split()
            if any(part in query_lower for part in author_parts if len(part) > 2):
                author_score = max(author_score, 0.5)
        
        # Bonus for well-known authors (if we have this information)
        preferred_authors = context.get('preferred_authors', [])
        if preferred_authors:
            for author in authors:
                if any(preferred in author.lower() for preferred in preferred_authors):
                    author_score = max(author_score, 0.8)
        
        return author_score
    
    def _compute_recency_score(self, result: Dict[str, Any]) -> float:
        """Compute recency score with configurable decay."""
        pub_date_str = result.get('publication_date')
        if not pub_date_str:
            return 0.5  # Neutral score for unknown dates
        
        try:
            pub_date = datetime.fromisoformat(pub_date_str).date()
            today = date.today()
            days_old = (today - pub_date).days
            
            if days_old < 0:  # Future dates (shouldn't happen)
                return 1.0
            
            # Exponential decay over the specified period
            decay_factor = math.exp(-days_old / self.config.recency_decay_days)
            return max(decay_factor, 0.1)  # Minimum score for very old papers
            
        except (ValueError, TypeError):
            return 0.5
    
    def _compute_citation_score(self, result: Dict[str, Any]) -> float:
        """Compute citation count score (if available)."""
        citation_count = result.get('citation_count', 0)
        
        if citation_count <= 0:
            return 0.0
        
        # Use logarithmic scaling for citation counts
        if self.config.use_logarithmic_scaling:
            return min(math.log10(citation_count + 1) / math.log10(1000), 1.0)  # Cap at 1000 citations
        else:
            return min(citation_count / 100.0, 1.0)  # Linear scaling capped at 100
    
    def _compute_content_quality_score(self, result: Dict[str, Any]) -> float:
        """Compute content quality score based on various heuristics."""
        content = result.get('content', '')
        if not content:
            return 0.0
        
        quality_score = 0.0
        
        # Length score (prefer moderate length)
        length = len(content)
        if 500 <= length <= 5000:  # Optimal length range
            quality_score += 0.3
        elif length > 200:  # At least some content
            quality_score += 0.1
        
        # Structure indicators
        structure_indicators = [
            'abstract', 'introduction', 'method', 'results', 'conclusion',
            'figure', 'table', 'equation', 'references'
        ]
        
        structure_matches = sum(1 for indicator in structure_indicators 
                              if indicator in content.lower())
        quality_score += min(structure_matches / len(structure_indicators), 0.4)
        
        # Technical depth indicators
        technical_indicators = [
            'algorithm', 'model', 'experiment', 'analysis', 'evaluation',
            'performance', 'accuracy', 'precision', 'recall'
        ]
        
        technical_matches = sum(1 for indicator in technical_indicators 
                               if indicator in content.lower())
        quality_score += min(technical_matches / len(technical_indicators), 0.3)
        
        return min(quality_score, 1.0)
    
    def _compute_category_relevance_score(
        self, 
        result: Dict[str, Any], 
        query: str, 
        context: Dict[str, Any]
    ) -> float:
        """Compute category/domain relevance score."""
        arxiv_id = result.get('arxiv_id', '')
        if not arxiv_id:
            return 0.5  # Neutral score
        
        # Extract category from arXiv ID
        category_match = re.match(r'^([a-z-]+(?:\.[A-Z]{2})?)', arxiv_id.lower())
        if not category_match:
            return 0.5
        
        paper_category = category_match.group(1)
        
        # Check if query or context indicates preferred categories
        preferred_categories = context.get('preferred_categories', [])
        
        # Category relevance based on query keywords
        category_keywords = {
            'cs.ai': ['artificial intelligence', 'ai', 'machine learning', 'neural'],
            'cs.cv': ['computer vision', 'image', 'visual', 'object detection'],
            'cs.cl': ['natural language', 'nlp', 'text', 'language model'],
            'cs.lg': ['machine learning', 'deep learning', 'neural network'],
            'stat.ml': ['statistics', 'statistical', 'bayesian', 'probability']
        }
        
        query_lower = query.lower()
        category_score = 0.5  # Base score
        
        for category, keywords in category_keywords.items():
            if paper_category.startswith(category.replace('.', '')) or paper_category == category:
                if any(keyword in query_lower for keyword in keywords):
                    category_score = 1.0
                    break
                elif category in preferred_categories:
                    category_score = 0.8
        
        return category_score
    
    def _combine_scores(self, scores: Dict[str, float], weights: RankingWeights) -> float:
        """Combine individual scores using weighted average."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        weight_dict = weights.to_dict()
        
        for factor, score in scores.items():
            if factor in weight_dict:
                weight = weight_dict[factor]
                weighted_sum += score * weight
                total_weight += weight
        
        # Handle missing factors by using available weights
        if total_weight == 0:
            return 0.0
        
        final_score = weighted_sum / total_weight
        
        # Apply threshold
        if final_score < self.config.min_score_threshold:
            final_score = 0.0
        
        return final_score
    
    def _normalize_final_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize final scores to [0, 1] range."""
        if not results:
            return results
        
        scores = [r['final_ranking_score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:  # All scores are equal
            for result in results:
                result['final_ranking_score'] = 1.0
        else:
            score_range = max_score - min_score
            for result in results:
                normalized_score = (result['final_ranking_score'] - min_score) / score_range
                result['final_ranking_score'] = normalized_score
        
        return results
    
    def _generate_explanation(self, scores: Dict[str, float], weights: RankingWeights) -> str:
        """Generate human-readable explanation for ranking."""
        explanations = []
        weight_dict = weights.to_dict()
        
        # Find top contributing factors
        contributions = {}
        for factor, score in scores.items():
            if factor in weight_dict:
                contribution = score * weight_dict[factor]
                contributions[factor] = contribution
        
        # Sort by contribution
        top_factors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]
        
        factor_names = {
            'semantic_similarity': 'semantic relevance',
            'keyword_match': 'keyword matching',
            'title_match': 'title relevance',
            'author_relevance': 'author match',
            'recency': 'publication recency',
            'citation_count': 'citation impact',
            'content_quality': 'content quality',
            'category_relevance': 'category match'
        }
        
        for factor, contribution in top_factors:
            factor_name = factor_names.get(factor, factor.replace('_', ' '))
            score = scores[factor]
            explanations.append(f"{factor_name}: {score:.2f}")
        
        return ", ".join(explanations)
    
    def create_custom_ranking_config(
        self, 
        user_preferences: Dict[str, Any]
    ) -> RankingConfig:
        """Create custom ranking configuration based on user preferences."""
        config = RankingConfig()
        
        # Adjust weights based on preferences
        search_focus = user_preferences.get('search_focus', 'balanced')
        
        if search_focus == 'semantic':
            config.weights.semantic_similarity = 0.6
            config.weights.keyword_match = 0.15
            config.weights.title_match = 0.1
        elif search_focus == 'keyword':
            config.weights.semantic_similarity = 0.2
            config.weights.keyword_match = 0.4
            config.weights.title_match = 0.2
        elif search_focus == 'recent':
            config.weights.recency = 0.3
            config.weights.semantic_similarity = 0.4
        elif search_focus == 'authoritative':
            config.weights.citation_count = 0.2
            config.weights.author_relevance = 0.15
            config.weights.semantic_similarity = 0.35
        
        # Other preferences
        if user_preferences.get('prefer_recent', False):
            config.boost_recent_papers = True
            config.recency_decay_days = 180  # Shorter decay period
        
        if user_preferences.get('strict_matching', False):
            config.boost_exact_matches = True
            config.min_score_threshold = 0.1
        
        return config
    
    def benchmark_ranking(
        self, 
        test_results: List[Dict[str, Any]], 
        queries: List[str],
        ground_truth: Optional[List[List[int]]] = None
    ) -> Dict[str, float]:
        """Benchmark ranking performance."""
        if not test_results or not queries:
            return {}
        
        benchmark_results = {
            'avg_processing_time': 0.0,
            'score_distribution': {},
            'top_k_accuracy': {}
        }
        
        try:
            import time
            
            processing_times = []
            
            for query in queries[:10]:  # Limit for benchmarking
                start_time = time.time()
                ranked_results = self.rank_results(test_results, query)
                end_time = time.time()
                
                processing_times.append(end_time - start_time)
            
            benchmark_results['avg_processing_time'] = np.mean(processing_times)
            
            # Score distribution analysis
            if len(queries) > 0:
                sample_ranked = self.rank_results(test_results, queries[0])
                scores = [r['final_ranking_score'] for r in sample_ranked]
                benchmark_results['score_distribution'] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
        
        except Exception as e:
            log_error("Ranking benchmark failed", error=str(e))
        
        return benchmark_results


def create_ranking_engine(config: RankingConfig = None) -> AdvancedRankingEngine:
    """Create and return configured ranking engine instance."""
    return AdvancedRankingEngine(config)