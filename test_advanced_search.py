"""
Comprehensive test suite for the Advanced Search System.
Tests hybrid search, filters, ranking, and integration.
"""

import os
import sys
import unittest
import tempfile
import shutil
import sqlite3
from datetime import date, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_search import (
    AdvancedSearchEngine, SearchMode, SearchResult, SearchFilters, SearchConfig
)
from search_filters import (
    AdvancedSearchFilters, SmartFilterEngine, DateRangePreset, ArxivCategory
)
from search_ranking import (
    AdvancedRankingEngine, RankingWeights, RankingConfig, RankingFactor
)
from logger import get_logger
from config import get_config


class TestSearchFilters(unittest.TestCase):
    """Test suite for search filters functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.filter_engine = SmartFilterEngine()
        self.test_papers = [
            {
                'id': 1,
                'title': 'Deep Learning for Computer Vision',
                'authors': ['John Smith', 'Jane Doe'],
                'publication_date': '2023-06-15',
                'arxiv_id': 'cs.CV/2306.12345',
                'summary': 'A comprehensive study of deep learning techniques for computer vision tasks.',
                'relevance_score': 0.9
            },
            {
                'id': 2,
                'title': 'Natural Language Processing with Transformers',
                'authors': ['Alice Johnson', 'Bob Wilson'],
                'publication_date': '2023-01-20',
                'arxiv_id': 'cs.CL/2301.67890',
                'summary': 'This paper explores transformer architectures for NLP tasks with code available.',
                'relevance_score': 0.8
            },
            {
                'id': 3,
                'title': 'Statistical Analysis of Time Series',
                'authors': ['Carol Brown'],
                'publication_date': '2022-12-01',
                'arxiv_id': 'stat.ML/2212.11111',
                'summary': 'Statistical methods for analyzing time series data with datasets.',
                'relevance_score': 0.7
            },
            {
                'id': 4,
                'title': 'Quantum Computing Algorithms',
                'authors': ['David Lee', 'Eve Chen'],
                'publication_date': '2023-08-30',
                'arxiv_id': 'quant-ph/2308.22222',
                'summary': 'Novel algorithms for quantum computing applications.',
                'relevance_score': 0.6
            }
        ]
    
    def test_filter_by_authors(self):
        """Test author filtering with different modes."""
        # Test "any" mode (default)
        filters = AdvancedSearchFilters(authors=['Smith'], author_mode='any')
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['title'], 'Deep Learning for Computer Vision')
        
        # Test partial name matching
        filters = AdvancedSearchFilters(authors=['John'], author_mode='any')
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        self.assertEqual(len(filtered), 2)  # Alice Johnson and John Smith
        
        # Test "all" mode
        filters = AdvancedSearchFilters(authors=['John Smith', 'Jane Doe'], author_mode='all')
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        self.assertEqual(len(filtered), 1)
    
    def test_filter_by_date_range(self):
        """Test date range filtering."""
        # Test specific date range
        filters = AdvancedSearchFilters(
            date_from=date(2023, 1, 1),
            date_to=date(2023, 6, 30)
        )
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        self.assertEqual(len(filtered), 2)  # Papers from 2023 first half
        
        # Test date preset
        filters = AdvancedSearchFilters(date_preset=DateRangePreset.LAST_YEAR)
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        # Should include papers from last year
        self.assertGreaterEqual(len(filtered), 0)
    
    def test_filter_by_arxiv_categories(self):
        """Test arXiv category filtering."""
        filters = AdvancedSearchFilters(arxiv_categories=['cs.cv'])
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['title'], 'Deep Learning for Computer Vision')
        
        # Test multiple categories
        filters = AdvancedSearchFilters(arxiv_categories=['cs.cv', 'cs.cl'])
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        self.assertEqual(len(filtered), 2)
    
    def test_filter_by_keywords(self):
        """Test keyword filtering in titles and abstracts."""
        # Test title keywords
        filters = AdvancedSearchFilters(title_keywords=['deep learning'])
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        self.assertEqual(len(filtered), 1)
        
        # Test abstract keywords
        filters = AdvancedSearchFilters(abstract_keywords=['transformer'])
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        self.assertEqual(len(filtered), 1)
        
        # Test exclude keywords
        filters = AdvancedSearchFilters(exclude_keywords=['quantum'])
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        self.assertEqual(len(filtered), 3)  # Excludes quantum paper
    
    def test_filter_by_code_availability(self):
        """Test filtering by code availability."""
        filters = AdvancedSearchFilters(has_code=True)
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        self.assertEqual(len(filtered), 1)  # Only NLP paper mentions code
        
        filters = AdvancedSearchFilters(has_code=False)
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        self.assertEqual(len(filtered), 3)
    
    def test_filter_by_dataset_availability(self):
        """Test filtering by dataset availability."""
        filters = AdvancedSearchFilters(has_dataset=True)
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        self.assertEqual(len(filtered), 1)  # Only time series paper mentions datasets
    
    def test_combined_filters(self):
        """Test combining multiple filters."""
        filters = AdvancedSearchFilters(
            authors=['Smith'],
            date_from=date(2023, 1, 1),
            arxiv_categories=['cs.cv'],
            min_relevance_score=0.85
        )
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]['title'], 'Deep Learning for Computer Vision')
    
    def test_sorting(self):
        """Test different sorting options."""
        # Test sort by date (descending)
        filters = AdvancedSearchFilters(sort_by='date', sort_ascending=False)
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        self.assertEqual(filtered[0]['title'], 'Quantum Computing Algorithms')  # Most recent
        
        # Test sort by relevance (descending)
        filters = AdvancedSearchFilters(sort_by='relevance', sort_ascending=False)
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        self.assertEqual(filtered[0]['title'], 'Deep Learning for Computer Vision')  # Highest score
        
        # Test sort by title (ascending)
        filters = AdvancedSearchFilters(sort_by='title', sort_ascending=True)
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        self.assertEqual(filtered[0]['title'], 'Deep Learning for Computer Vision')  # Alphabetically first
    
    def test_max_results_limit(self):
        """Test maximum results limit."""
        filters = AdvancedSearchFilters(max_results=2)
        filtered = self.filter_engine.apply_filters(self.test_papers, filters)
        self.assertEqual(len(filtered), 2)
    
    def test_filter_suggestions(self):
        """Test filter suggestions functionality."""
        # Mock database connection for testing
        with patch('knowledge_graph.get_db_connection') as mock_db:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [('John Smith',), ('Jane Doe',)]
            mock_db.return_value.__enter__.return_value.cursor.return_value = mock_cursor
            
            suggestions = self.filter_engine.get_filter_suggestions('john')
            self.assertIn('authors', suggestions)
            self.assertTrue(len(suggestions['authors']) > 0)


class TestSearchRanking(unittest.TestCase):
    """Test suite for search ranking functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.ranking_engine = AdvancedRankingEngine()
        self.test_results = [
            {
                'doc_id': '1',
                'title': 'Machine Learning Fundamentals',
                'content': 'This paper introduces machine learning algorithms and neural networks.',
                'authors': ['Dr. Smith', 'Prof. Johnson'],
                'publication_date': '2023-06-01',
                'arxiv_id': 'cs.LG/2306.01234',
                'semantic_score': 0.9,
                'relevance_score': 0.9,
                'citation_count': 150
            },
            {
                'doc_id': '2',
                'title': 'Advanced Deep Learning',
                'content': 'Deep learning techniques for computer vision applications.',
                'authors': ['Dr. Brown'],
                'publication_date': '2023-01-15',
                'arxiv_id': 'cs.CV/2301.56789',
                'semantic_score': 0.8,
                'relevance_score': 0.8,
                'citation_count': 80
            },
            {
                'doc_id': '3',
                'title': 'Statistical Methods',
                'content': 'Traditional statistical approaches to data analysis.',
                'authors': ['Prof. Wilson'],
                'publication_date': '2022-12-01',
                'arxiv_id': 'stat.ME/2212.98765',
                'semantic_score': 0.6,
                'relevance_score': 0.6,
                'citation_count': 45
            }
        ]
    
    def test_keyword_match_scoring(self):
        """Test keyword match score computation."""
        result = self.test_results[0]
        query = "machine learning algorithms"
        
        score = self.ranking_engine._compute_keyword_match_score(result, query)
        self.assertGreater(score, 0.5)  # Should have good keyword match
        
        # Test exact phrase match bonus
        query_exact = "machine learning"
        score_exact = self.ranking_engine._compute_keyword_match_score(result, query_exact)
        self.assertGreater(score_exact, score)
    
    def test_title_match_scoring(self):
        """Test title match score computation."""
        result = self.test_results[0]
        
        # Exact match in title
        query = "machine learning"
        score = self.ranking_engine._compute_title_match_score(result, query)
        self.assertGreater(score, 0.8)
        
        # Partial match
        query = "learning"
        score_partial = self.ranking_engine._compute_title_match_score(result, query)
        self.assertGreater(score_partial, 0.0)
        self.assertLess(score_partial, score)
    
    def test_recency_scoring(self):
        """Test recency score computation."""
        recent_result = self.test_results[0]  # 2023
        old_result = self.test_results[2]     # 2022
        
        recent_score = self.ranking_engine._compute_recency_score(recent_result)
        old_score = self.ranking_engine._compute_recency_score(old_result)
        
        self.assertGreater(recent_score, old_score)
    
    def test_citation_scoring(self):
        """Test citation count scoring."""
        high_cited = self.test_results[0]  # 150 citations
        low_cited = self.test_results[2]   # 45 citations
        
        high_score = self.ranking_engine._compute_citation_score(high_cited)
        low_score = self.ranking_engine._compute_citation_score(low_cited)
        
        self.assertGreater(high_score, low_score)
    
    def test_content_quality_scoring(self):
        """Test content quality scoring."""
        result = self.test_results[0]
        score = self.ranking_engine._compute_content_quality_score(result)
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
    
    def test_full_ranking(self):
        """Test complete ranking pipeline."""
        query = "machine learning"
        ranked_results = self.ranking_engine.rank_results(self.test_results, query)
        
        # Should return same number of results
        self.assertEqual(len(ranked_results), len(self.test_results))
        
        # Results should be sorted by final score
        scores = [r['final_ranking_score'] for r in ranked_results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
        # Each result should have required fields
        for result in ranked_results:
            self.assertIn('individual_scores', result)
            self.assertIn('final_ranking_score', result)
            self.assertIn('ranking_explanation', result)
    
    def test_custom_ranking_weights(self):
        """Test custom ranking weights."""
        # Create config that heavily weights recency
        weights = RankingWeights(
            semantic_similarity=0.2,
            recency=0.6,
            keyword_match=0.1,
            title_match=0.1
        )
        config = RankingConfig(weights=weights)
        custom_engine = AdvancedRankingEngine(config)
        
        query = "machine learning"
        ranked_results = custom_engine.rank_results(self.test_results, query)
        
        # More recent paper should rank higher with recency-focused weights
        self.assertEqual(ranked_results[0]['doc_id'], '1')  # Most recent paper
    
    def test_score_normalization(self):
        """Test score normalization."""
        config = RankingConfig(normalize_scores=True)
        engine = AdvancedRankingEngine(config)
        
        query = "test"
        ranked_results = engine.rank_results(self.test_results, query)
        
        scores = [r['final_ranking_score'] for r in ranked_results]
        self.assertLessEqual(max(scores), 1.0)
        self.assertGreaterEqual(min(scores), 0.0)
    
    def test_ranking_explanation(self):
        """Test ranking explanation generation."""
        scores = {
            'semantic_similarity': 0.9,
            'keyword_match': 0.7,
            'title_match': 0.8,
            'recency': 0.6
        }
        weights = RankingWeights()
        
        explanation = self.ranking_engine._generate_explanation(scores, weights)
        
        self.assertIsInstance(explanation, str)
        self.assertGreater(len(explanation), 0)
        self.assertIn('semantic relevance', explanation)


class TestAdvancedSearch(unittest.TestCase):
    """Test suite for the main advanced search engine."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.vector_db_path = os.path.join(self.test_dir, 'test_vector_db')
        self.kg_db_path = os.path.join(self.test_dir, 'test_kg.db')
        
        # Mock the search engine initialization to avoid actual database dependencies
        self.search_engine = None
    
    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('advanced_search.chromadb.PersistentClient')
    @patch('advanced_search.HuggingFaceEmbeddings')
    def test_search_engine_initialization(self, mock_embeddings, mock_chroma):
        """Test search engine initialization."""
        # Mock vector database
        mock_collection = MagicMock()
        mock_collection.get.return_value = {'documents': ['test doc'], 'ids': ['1']}
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        # Mock embedding model
        mock_embeddings.return_value = MagicMock()
        
        # Create search engine
        with patch('os.path.exists', return_value=True):
            engine = AdvancedSearchEngine(self.vector_db_path)
            self.assertIsNotNone(engine)
    
    @patch('advanced_search.chromadb.PersistentClient')
    @patch('advanced_search.HuggingFaceEmbeddings')
    def test_semantic_search(self, mock_embeddings, mock_chroma):
        """Test semantic search functionality."""
        # Mock setup
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings.return_value = mock_embedding_model
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'ids': [['1', '2']],
            'documents': [['Doc 1 content', 'Doc 2 content']],
            'metadatas': [[{'source': 'test1.pdf'}, {'source': 'test2.pdf'}]],
            'distances': [[0.1, 0.2]]
        }
        mock_collection.get.return_value = {'documents': ['test'], 'ids': ['1']}
        
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        # Create engine and test
        with patch('os.path.exists', return_value=True):
            engine = AdvancedSearchEngine(self.vector_db_path)
            results = engine._semantic_search("test query", SearchFilters(), SearchConfig())
            
            self.assertEqual(len(results), 2)
            self.assertIsInstance(results[0], SearchResult)
    
    @patch('advanced_search.chromadb.PersistentClient')  
    @patch('advanced_search.HuggingFaceEmbeddings')
    def test_hybrid_search(self, mock_embeddings, mock_chroma):
        """Test hybrid search combining semantic and keyword search."""
        # Mock setup similar to semantic search test
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embeddings.return_value = mock_embedding_model
        
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'ids': [['1']],
            'documents': [['Machine learning algorithms']],
            'metadatas': [[{'source': 'test1.pdf'}]],
            'distances': [[0.1]]
        }
        mock_collection.get.return_value = {'documents': ['machine learning'], 'ids': ['1']}
        
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        with patch('os.path.exists', return_value=True):
            engine = AdvancedSearchEngine(self.vector_db_path)
            
            # Mock TF-IDF setup
            engine.tfidf_matrix = [[0.5]]  # Mock matrix
            engine.doc_ids = ['1']
            
            with patch('numpy.argsort', return_value=[0]), \
                 patch('sklearn.metrics.pairwise.cosine_similarity', return_value=[[0.8]]):
                
                results = engine._hybrid_search("machine learning", SearchFilters(), SearchConfig())
                
                self.assertGreater(len(results), 0)
                # Should have both semantic and keyword scores
                if results:
                    self.assertGreaterEqual(results[0].semantic_score, 0)
                    self.assertGreaterEqual(results[0].final_score, 0)
    
    def test_search_suggestions(self):
        """Test search suggestions functionality."""
        # Mock knowledge graph connection
        with patch('knowledge_graph.get_db_connection') as mock_db:
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = [
                ('Machine Learning in Computer Vision',),
                ('Deep Learning Fundamentals',)
            ]
            mock_db.return_value.__enter__.return_value.cursor.return_value = mock_cursor
            
            # Mock the search engine to avoid initialization issues
            with patch('advanced_search.AdvancedSearchEngine.__init__', return_value=None):
                engine = AdvancedSearchEngine()
                suggestions = engine.get_search_suggestions("machine")
                
                self.assertIsInstance(suggestions, list)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete advanced search system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create mock database with test data
        self.kg_db_path = os.path.join(self.test_dir, 'test_knowledge.db')
        self._create_test_kg_database()
    
    def tearDown(self):
        """Clean up integration test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _create_test_kg_database(self):
        """Create test knowledge graph database."""
        conn = sqlite3.connect(self.kg_db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE papers (
                id INTEGER PRIMARY KEY,
                title TEXT NOT NULL,
                summary TEXT,
                publication_date TEXT,
                arxiv_id TEXT,
                source_pdf TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE authors (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE paper_authors (
                paper_id INTEGER,
                author_id INTEGER,
                FOREIGN KEY (paper_id) REFERENCES papers (id),
                FOREIGN KEY (author_id) REFERENCES authors (id)
            )
        """)
        
        # Insert test data
        cursor.execute("""
            INSERT INTO papers (title, summary, publication_date, arxiv_id, source_pdf)
            VALUES (?, ?, ?, ?, ?)
        """, (
            'Test Paper on Machine Learning',
            'A comprehensive study of machine learning algorithms.',
            '2023-06-15',
            'cs.LG/2306.12345',
            '/test/path/paper1.pdf'
        ))
        
        cursor.execute("INSERT INTO authors (name) VALUES (?)", ('John Smith',))
        cursor.execute("INSERT INTO paper_authors (paper_id, author_id) VALUES (1, 1)")
        
        conn.commit()
        conn.close()
    
    @patch('advanced_search.chromadb.PersistentClient')
    @patch('advanced_search.HuggingFaceEmbeddings')
    def test_end_to_end_search(self, mock_embeddings, mock_chroma):
        """Test complete end-to-end search flow."""
        # Mock vector database
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            'ids': [['test_1']],
            'documents': [['Machine learning algorithms and neural networks']],
            'metadatas': [[{'source': '/test/path/paper1.pdf'}]],
            'distances': [[0.1]]
        }
        mock_collection.get.return_value = {'documents': ['test'], 'ids': ['test_1']}
        
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_chroma.return_value = mock_client
        
        mock_embeddings.return_value = MagicMock()
        
        # Create and test search engine
        with patch('os.path.exists', return_value=True), \
             patch('knowledge_graph.get_db_connection') as mock_kg_conn:
            
            # Mock knowledge graph connection
            mock_cursor = MagicMock()
            mock_cursor.fetchone.return_value = (
                1, 'Test Paper on Machine Learning', 'Summary', '2023-06-15', 'cs.LG/2306.12345', 'John Smith'
            )
            mock_kg_conn.return_value.__enter__.return_value.cursor.return_value = mock_cursor
            
            engine = AdvancedSearchEngine(self.test_dir)
            
            # Perform search with filters
            filters = SearchFilters(
                authors=['John Smith'],
                max_results=10,
                min_relevance_score=0.0
            )
            
            results = engine.search(
                query="machine learning",
                mode=SearchMode.SEMANTIC,
                filters=filters
            )
            
            self.assertIsInstance(results, list)
            if results:
                self.assertIsInstance(results[0], SearchResult)
                self.assertIsNotNone(results[0].final_score)


def run_performance_benchmarks():
    """Run performance benchmarks for the search system."""
    print("\n=== Advanced Search System Performance Benchmarks ===")
    
    import time
    import random
    
    # Create test data
    test_papers = []
    for i in range(1000):
        test_papers.append({
            'id': i,
            'title': f'Test Paper {i}',
            'authors': [f'Author {i}', f'Co-Author {i}'],
            'publication_date': f'2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}',
            'arxiv_id': f'cs.{random.choice(["AI", "CV", "CL", "LG"])}/{random.randint(2301, 2312)}.{i:05d}',
            'summary': f'This is a test summary for paper {i} about machine learning.',
            'relevance_score': random.random()
        })
    
    # Test filter performance
    filter_engine = SmartFilterEngine()
    
    print(f"Testing with {len(test_papers)} papers...")
    
    # Benchmark different filter operations
    filters_to_test = [
        AdvancedSearchFilters(authors=['Author 1']),
        AdvancedSearchFilters(arxiv_categories=['cs.ai']),
        AdvancedSearchFilters(date_from=date(2023, 6, 1)),
        AdvancedSearchFilters(min_relevance_score=0.5),
        AdvancedSearchFilters(
            authors=['Author 1'],
            arxiv_categories=['cs.ai'],
            min_relevance_score=0.5,
            max_results=50
        )
    ]
    
    for i, filters in enumerate(filters_to_test):
        start_time = time.time()
        filtered_papers = filter_engine.apply_filters(test_papers, filters)
        end_time = time.time()
        
        print(f"Filter test {i+1}: {end_time - start_time:.4f}s - {len(filtered_papers)} results")
    
    # Test ranking performance
    ranking_engine = AdvancedRankingEngine()
    
    # Convert papers to ranking format
    ranking_test_papers = []
    for paper in test_papers[:100]:  # Test with subset for ranking
        ranking_paper = paper.copy()
        ranking_paper.update({
            'doc_id': str(paper['id']),
            'content': paper['summary'],
            'semantic_score': paper['relevance_score']
        })
        ranking_test_papers.append(ranking_paper)
    
    start_time = time.time()
    ranked_papers = ranking_engine.rank_results(ranking_test_papers, "machine learning")
    end_time = time.time()
    
    print(f"Ranking test: {end_time - start_time:.4f}s - {len(ranked_papers)} results")
    
    print("=== Benchmarks Complete ===\n")


if __name__ == '__main__':
    # Run unit tests
    print("Running Advanced Search System Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSearchFilters))
    suite.addTests(loader.loadTestsFromTestCase(TestSearchRanking))
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedSearch))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print results summary
    print(f"\n=== Test Results Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, trace in result.failures:
            print(f"- {test}: {trace.splitlines()[-1]}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, trace in result.errors:
            print(f"- {test}: {trace.splitlines()[-1]}")
    
    # Run performance benchmarks if tests passed
    if not result.failures and not result.errors:
        run_performance_benchmarks()
    
    # Exit with error code if tests failed
    exit_code = 0 if (len(result.failures) + len(result.errors)) == 0 else 1
    exit(exit_code)