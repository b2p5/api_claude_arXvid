"""
Example usage of the Advanced Search System for arXiv Papers.
Demonstrates various search modes, filters, and ranking options.
"""

import os
import sys
from datetime import date, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_search import AdvancedSearchEngine, SearchMode
from search_filters import AdvancedSearchFilters, DateRangePreset, ArxivCategory
from search_ranking import RankingStrategy, RankingWeights, RankingConfig
from chat_with_advanced_search import AdvancedChatRAG
from config import get_config


def demonstrate_basic_search():
    """Demonstrate basic search functionality."""
    print("=== Basic Search Examples ===\n")
    
    try:
        # Initialize search engine
        print("🔍 Initializing search engine...")
        search_engine = AdvancedSearchEngine()
        
        # Example 1: Simple semantic search
        print("\n1. Simple Semantic Search:")
        results = search_engine.search(
            query="machine learning algorithms",
            mode=SearchMode.SEMANTIC
        )
        
        print(f"Found {len(results)} results")
        if results:
            print(f"Top result: {results[0].title}")
            print(f"Score: {results[0].final_score:.3f}")
        
        # Example 2: Keyword search
        print("\n2. Keyword Search:")
        results = search_engine.search(
            query="neural networks deep learning",
            mode=SearchMode.KEYWORD
        )
        
        print(f"Found {len(results)} results")
        if results:
            print(f"Top result: {results[0].title}")
        
        # Example 3: Hybrid search (default)
        print("\n3. Hybrid Search:")
        results = search_engine.search(
            query="computer vision transformers",
            mode=SearchMode.HYBRID
        )
        
        print(f"Found {len(results)} results")
        if results:
            print(f"Top result: {results[0].title}")
            print(f"Semantic score: {results[0].semantic_score:.3f}")
            print(f"Keyword score: {results[0].keyword_score:.3f}")
        
    except FileNotFoundError:
        print("❌ Vector database not found. Please run rag_bbdd_vector_optimized.py first.")
    except Exception as e:
        print(f"❌ Error: {e}")


def demonstrate_advanced_filters():
    """Demonstrate advanced filtering capabilities."""
    print("\n=== Advanced Filter Examples ===\n")
    
    try:
        search_engine = AdvancedSearchEngine()
        
        # Example 1: Author filter
        print("1. Search by specific author:")
        filters = AdvancedSearchFilters(
            authors=["Yann LeCun"],
            max_results=5
        )
        results = search_engine.search(
            query="deep learning",
            filters=filters
        )
        print(f"Found {len(results)} papers by Yann LeCun about deep learning")
        
        # Example 2: Date range filter
        print("\n2. Recent papers only:")
        filters = AdvancedSearchFilters(
            date_preset=DateRangePreset.LAST_6_MONTHS,
            max_results=5
        )
        results = search_engine.search(
            query="large language models",
            filters=filters
        )
        print(f"Found {len(results)} recent papers about LLMs")
        
        # Example 3: arXiv category filter
        print("\n3. Computer Vision papers only:")
        filters = AdvancedSearchFilters(
            arxiv_categories=["cs.CV"],
            max_results=5
        )
        results = search_engine.search(
            query="object detection",
            filters=filters
        )
        print(f"Found {len(results)} CV papers about object detection")
        
        # Example 4: Combined filters
        print("\n4. Combined filters:")
        filters = AdvancedSearchFilters(
            authors=["Geoffrey Hinton", "Yoshua Bengio"],
            date_from=date(2020, 1, 1),
            arxiv_categories=["cs.LG", "cs.AI"],
            min_relevance_score=0.5,
            max_results=3
        )
        results = search_engine.search(
            query="representation learning",
            filters=filters
        )
        print(f"Found {len(results)} highly relevant papers by Hinton/Bengio since 2020")
        
        # Example 5: Content-based filters
        print("\n5. Papers with code/datasets:")
        filters = AdvancedSearchFilters(
            has_code=True,
            has_dataset=True,
            max_results=5
        )
        results = search_engine.search(
            query="benchmark evaluation",
            filters=filters
        )
        print(f"Found {len(results)} papers with both code and datasets")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demonstrate_custom_ranking():
    """Demonstrate custom ranking strategies."""
    print("\n=== Custom Ranking Examples ===\n")
    
    try:
        search_engine = AdvancedSearchEngine()
        
        # Example 1: Recency-focused ranking
        print("1. Ranking by recency:")
        results_recency = search_engine.search(
            query="transformer architecture",
            ranking=RankingStrategy.RECENCY,
            filters=AdvancedSearchFilters(max_results=3)
        )
        print("Top 3 most recent papers:")
        for i, result in enumerate(results_recency, 1):
            print(f"  {i}. {result.title} ({result.publication_date})")
        
        # Example 2: Custom ranking weights
        print("\n2. Custom ranking weights (emphasize semantics):")
        custom_weights = RankingWeights(
            semantic_similarity=0.6,  # Higher weight on semantics
            keyword_match=0.15,
            title_match=0.1,
            recency=0.1,
            citation_count=0.05
        )
        custom_config = RankingConfig(weights=custom_weights)
        
        # Create new engine with custom config
        custom_engine = AdvancedSearchEngine()
        custom_engine.search_config = custom_config
        
        results_custom = search_engine.search(
            query="attention mechanism",
            ranking=RankingStrategy.CUSTOM,
            search_config=custom_config,
            filters=AdvancedSearchFilters(max_results=3)
        )
        print("Top 3 semantically relevant papers:")
        for i, result in enumerate(results_custom, 1):
            if hasattr(result, 'ranking_explanation'):
                print(f"  {i}. {result.title} - {result.ranking_explanation}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demonstrate_chat_integration():
    """Demonstrate chat system with advanced search."""
    print("\n=== Chat Integration Examples ===\n")
    
    try:
        # Initialize chat system
        print("🤖 Initializing advanced chat system...")
        chat_system = AdvancedChatRAG()
        
        # Example queries
        example_queries = [
            "What are the latest advances in transformer architectures?",
            "Can you compare different attention mechanisms?",
            "Show me papers by Geoffrey Hinton about deep learning",
            "What are the best computer vision papers from 2023?",
            "Recommend papers about large language model training"
        ]
        
        for i, query in enumerate(example_queries, 1):
            print(f"\n{i}. Query: {query}")
            try:
                result = chat_system.chat(query)
                print(f"Response: {result['response'][:200]}...")
                print(f"Found {result['total_results_found']} relevant papers")
                print(f"Search strategy: {result.get('search_strategy', 'N/A')}")
            except Exception as e:
                print(f"Error processing query: {e}")
        
        # Show chat statistics
        print(f"\n📊 Chat Statistics:")
        stats = chat_system.get_chat_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except FileNotFoundError:
        print("❌ Database not found. Please create the vector database first.")
    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")


def demonstrate_search_suggestions():
    """Demonstrate search suggestion functionality."""
    print("\n=== Search Suggestions Examples ===\n")
    
    try:
        search_engine = AdvancedSearchEngine()
        
        # Test partial queries
        partial_queries = ["machine", "deep", "neural", "computer"]
        
        for partial in partial_queries:
            suggestions = search_engine.get_search_suggestions(partial)
            print(f"Suggestions for '{partial}': {suggestions[:3]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def demonstrate_performance_features():
    """Demonstrate performance and optimization features."""
    print("\n=== Performance Features ===\n")
    
    try:
        search_engine = AdvancedSearchEngine()
        
        # Get search statistics
        stats = search_engine.get_search_statistics()
        print("📈 Search Engine Statistics:")
        print(f"  Vector database size: {stats.get('vector_db_size', 0)} documents")
        print(f"  Knowledge graph papers: {stats.get('kg_papers', 0)}")
        print(f"  Knowledge graph authors: {stats.get('kg_authors', 0)}")
        print(f"  TF-IDF vocabulary size: {stats.get('tfidf_vocab_size', 0)}")
        
        # Demonstrate batch search (for performance)
        print("\n⚡ Batch Search Performance Test:")
        import time
        
        test_queries = [
            "machine learning",
            "deep learning",
            "computer vision",
            "natural language processing",
            "reinforcement learning"
        ]
        
        start_time = time.time()
        for query in test_queries:
            results = search_engine.search(
                query=query,
                filters=AdvancedSearchFilters(max_results=5)
            )
        end_time = time.time()
        
        avg_time = (end_time - start_time) / len(test_queries)
        print(f"Average search time: {avg_time:.3f} seconds per query")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all demonstration examples."""
    print("🚀 Advanced Search System Demonstration\n")
    
    # Check if vector database exists
    config = get_config()
    if not os.path.exists(config.database.vector_db_path):
        print("❌ Vector database not found!")
        print("Please run the following command first:")
        print("python rag_bbdd_vector_optimized.py")
        print("\nThis will create the necessary databases for the search system.")
        return
    
    # Run demonstrations
    try:
        demonstrate_basic_search()
        demonstrate_advanced_filters()
        demonstrate_custom_ranking()
        demonstrate_search_suggestions()
        demonstrate_performance_features()
        demonstrate_chat_integration()
        
        print("\n✅ All demonstrations completed successfully!")
        print("\n🎯 Next Steps:")
        print("1. Try the interactive chat: python chat_with_advanced_search.py --interactive")
        print("2. Run performance tests: python test_advanced_search.py")
        print("3. Explore the API documentation in the source files")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demonstration interrupted by user.")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")


if __name__ == "__main__":
    main()