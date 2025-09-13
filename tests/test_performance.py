#!/usr/bin/env python3
"""
Performance testing script for the optimized arXiv papers analysis system.
Compares performance between original and optimized implementations.
"""

import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
import statistics

from config import get_config
from logger import get_logger, log_info, log_warning
from core.parallel_processing import process_pdfs_parallel, get_optimal_worker_count
from core.rag.embedding_cache import get_embedding_cache, CachedEmbeddingModel
from core.analysis.intelligent_chunking import create_intelligent_splitter, analyze_chunking_quality
from administration.indexing.database_optimizer import SQLiteOptimizer, optimize_all_databases


class PerformanceTester:
    """Performance testing suite for optimization features."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        self.test_results = {}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all performance tests."""
        self.logger.log_operation_start("Performance testing suite")
        
        print("Performance Testing Suite")
        print("=" * 50)
        
        # Test parallel processing
        self.test_results['parallel_processing'] = self.test_parallel_processing()
        
        # Test embedding caching
        self.test_results['embedding_caching'] = self.test_embedding_caching()
        
        # Test intelligent chunking
        self.test_results['intelligent_chunking'] = self.test_intelligent_chunking()
        
        # Test database optimization
        self.test_results['database_optimization'] = self.test_database_optimization()
        
        # Generate summary
        self.test_results['summary'] = self.generate_performance_summary()
        
        self.logger.log_operation_success("Performance testing suite")
        
        return self.test_results
    
    def test_parallel_processing(self) -> Dict[str, Any]:
        """Test parallel processing performance."""
        print("\n1. Testing Parallel Processing Performance")
        print("-" * 40)
        
        # Find test PDFs
        test_pdfs = self._find_test_pdfs()
        
        if len(test_pdfs) < 2:
            print("   [SKIP] Need at least 2 PDFs for parallel testing")
            return {'status': 'skipped', 'reason': 'insufficient_test_data'}
        
        results = {}
        
        # Test different worker counts
        worker_counts = [1, 2, min(4, len(test_pdfs))]
        
        for workers in worker_counts:
            print(f"   Testing with {workers} worker(s)...")
            
            start_time = time.time()
            processing_results, summary = process_pdfs_parallel(
                test_pdfs[:8],  # Limit to 8 PDFs for testing
                operations=['validate'],
                max_workers=workers
            )
            processing_time = time.time() - start_time
            
            successful = sum(1 for r in processing_results if r.success)
            
            results[f'{workers}_workers'] = {
                'processing_time': processing_time,
                'successful_pdfs': successful,
                'total_pdfs': len(test_pdfs[:8]),
                'pdfs_per_second': successful / processing_time if processing_time > 0 else 0,
                'summary': summary
            }
            
            print(f"     Time: {processing_time:.2f}s, Success rate: {successful}/{len(test_pdfs[:8])}, "
                  f"Speed: {successful / processing_time:.2f} PDFs/s")
        
        # Calculate speedup
        if '1_workers' in results and '2_workers' in results:
            speedup = results['1_workers']['processing_time'] / results['2_workers']['processing_time']
            results['speedup_2x'] = speedup
            print(f"   2x worker speedup: {speedup:.2f}x")
        
        return results
    
    def test_embedding_caching(self) -> Dict[str, Any]:
        """Test embedding caching performance."""
        print("\n2. Testing Embedding Caching Performance")
        print("-" * 40)
        
        cache = get_embedding_cache()
        cached_model = CachedEmbeddingModel()
        
        # Test texts
        test_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning uses neural networks with multiple layers.",
            "Computer vision allows machines to interpret visual information.",
            "Machine learning is a subset of artificial intelligence.",  # Duplicate for cache test
        ]
        
        results = {}
        
        # First run (cold cache)
        print("   Testing cold cache performance...")
        start_time = time.time()
        embeddings_cold = cached_model.embed_documents(test_texts)
        cold_time = time.time() - start_time
        
        cache_stats_after_cold = cache.get_cache_stats()
        
        results['cold_cache'] = {
            'time': cold_time,
            'embeddings_computed': len(embeddings_cold),
            'cache_hits': cache_stats_after_cold['hits'],
            'cache_misses': cache_stats_after_cold['misses']
        }
        
        print(f"     Cold cache time: {cold_time:.4f}s, Cache hits: {cache_stats_after_cold['hits']}")
        
        # Second run (warm cache)
        print("   Testing warm cache performance...")
        start_time = time.time()
        embeddings_warm = cached_model.embed_documents(test_texts)
        warm_time = time.time() - start_time
        
        cache_stats_after_warm = cache.get_cache_stats()
        
        results['warm_cache'] = {
            'time': warm_time,
            'embeddings_computed': len(embeddings_warm),
            'cache_hits': cache_stats_after_warm['hits'] - cache_stats_after_cold['hits'],
            'cache_misses': cache_stats_after_warm['misses'] - cache_stats_after_cold['misses']
        }
        
        # Calculate speedup
        if warm_time > 0:
            speedup = cold_time / warm_time
            results['cache_speedup'] = speedup
            print(f"     Warm cache time: {warm_time:.4f}s, Speedup: {speedup:.2f}x")
        
        # Test cache hit rate
        hit_rate = (results['warm_cache']['cache_hits'] / 
                   max(results['warm_cache']['cache_hits'] + results['warm_cache']['cache_misses'], 1) * 100)
        results['hit_rate_percent'] = hit_rate
        print(f"     Cache hit rate: {hit_rate:.1f}%")
        
        return results
    
    def test_intelligent_chunking(self) -> Dict[str, Any]:
        """Test intelligent chunking vs. simple chunking."""
        print("\n3. Testing Intelligent Chunking Performance")
        print("-" * 40)
        
        # Get sample text from a PDF if available
        test_pdfs = self._find_test_pdfs()
        
        if not test_pdfs:
            print("   [SKIP] No test PDFs available")
            return {'status': 'skipped', 'reason': 'no_test_pdfs'}
        
        try:
            from langchain_community.document_loaders import PyPDFLoader
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            # Load a test PDF
            loader = PyPDFLoader(test_pdfs[0])
            documents = loader.load()
            
            if not documents:
                print("   [SKIP] Could not load test PDF content")
                return {'status': 'skipped', 'reason': 'pdf_load_failed'}
            
            sample_text = " ".join([doc.page_content for doc in documents])
            
            results = {}
            
            # Test simple chunking
            print("   Testing simple chunking...")
            simple_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.models.chunk_size,
                chunk_overlap=self.config.models.chunk_overlap
            )
            
            start_time = time.time()
            simple_chunks = simple_splitter.split_documents(documents)
            simple_time = time.time() - start_time
            
            results['simple_chunking'] = {
                'time': simple_time,
                'chunk_count': len(simple_chunks),
                'avg_chunk_size': statistics.mean(len(chunk.page_content) for chunk in simple_chunks) if simple_chunks else 0
            }
            
            print(f"     Simple chunking: {simple_time:.4f}s, {len(simple_chunks)} chunks")
            
            # Test intelligent chunking
            print("   Testing intelligent chunking...")
            intelligent_splitter = create_intelligent_splitter()
            
            start_time = time.time()
            intelligent_chunks = intelligent_splitter.split_documents(documents)
            intelligent_time = time.time() - start_time
            
            results['intelligent_chunking'] = {
                'time': intelligent_time,
                'chunk_count': len(intelligent_chunks),
                'avg_chunk_size': statistics.mean(len(chunk.page_content) for chunk in intelligent_chunks) if intelligent_chunks else 0
            }
            
            print(f"     Intelligent chunking: {intelligent_time:.4f}s, {len(intelligent_chunks)} chunks")
            
            # Analyze chunking quality
            from intelligent_chunking import TextChunk, ChunkType
            
            # Convert to TextChunk objects for analysis (simplified)
            intelligent_text_chunks = []
            for chunk in intelligent_chunks:
                chunk_type = ChunkType(chunk.metadata.get('chunk_type', 'paragraph'))
                text_chunk = TextChunk(
                    content=chunk.page_content,
                    chunk_type=chunk_type,
                    section_title=chunk.metadata.get('section_title', ''),
                    confidence=chunk.metadata.get('chunk_confidence', 0.5)
                )
                intelligent_text_chunks.append(text_chunk)
            
            quality_analysis = analyze_chunking_quality(intelligent_text_chunks)
            results['quality_analysis'] = quality_analysis
            
            print(f"     Quality: {quality_analysis.get('sections_identified', 0)} sections identified, "
                  f"avg confidence: {quality_analysis.get('avg_confidence', 0):.2f}")
            
            return results
            
        except Exception as e:
            print(f"   [ERROR] Chunking test failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def test_database_optimization(self) -> Dict[str, Any]:
        """Test database optimization performance."""
        print("\n4. Testing Database Optimization Performance")
        print("-" * 40)
        
        results = {}
        
        # Test knowledge graph optimization
        kg_db_path = self.config.database.knowledge_db_path
        
        if os.path.exists(kg_db_path):
            print("   Testing knowledge graph optimization...")
            
            try:
                optimizer = SQLiteOptimizer(kg_db_path)
                
                # Benchmark queries before optimization
                print("     Benchmarking queries before optimization...")
                benchmark_before = optimizer.benchmark_queries()
                
                # Optimize database
                print("     Optimizing database...")
                start_time = time.time()
                optimization_results = optimizer.optimize_knowledge_graph()
                optimization_time = time.time() - start_time
                
                # Benchmark queries after optimization
                print("     Benchmarking queries after optimization...")
                benchmark_after = optimizer.benchmark_queries()
                
                # Calculate improvement
                time_improvement = 0
                if benchmark_before['average_time'] > 0:
                    time_improvement = (
                        (benchmark_before['average_time'] - benchmark_after['average_time']) /
                        benchmark_before['average_time'] * 100
                    )
                
                results['knowledge_graph'] = {
                    'optimization_time': optimization_time,
                    'indices_created': len(optimization_results['indices_created']),
                    'before_avg_query_time': benchmark_before['average_time'],
                    'after_avg_query_time': benchmark_after['average_time'],
                    'performance_improvement_percent': time_improvement
                }
                
                print(f"     Optimization time: {optimization_time:.2f}s")
                print(f"     Indices created: {len(optimization_results['indices_created'])}")
                print(f"     Query performance improvement: {time_improvement:.1f}%")
                
            except Exception as e:
                print(f"     [ERROR] Database optimization test failed: {e}")
                results['knowledge_graph'] = {'status': 'error', 'error': str(e)}
        else:
            print("   [SKIP] Knowledge graph database not found")
            results['knowledge_graph'] = {'status': 'skipped', 'reason': 'database_not_found'}
        
        return results
    
    def _find_test_pdfs(self) -> List[str]:
        """Find available PDF files for testing."""
        test_pdfs = []
        
        docs_root = self.config.arxiv.documents_root
        if not os.path.exists(docs_root):
            return test_pdfs
        
        for concept_dir in os.listdir(docs_root):
            concept_path = os.path.join(docs_root, concept_dir)
            if os.path.isdir(concept_path):
                for file_name in os.listdir(concept_path):
                    if file_name.endswith('.pdf'):
                        pdf_path = os.path.join(concept_path, file_name)
                        test_pdfs.append(pdf_path)
                        
                        # Limit to 10 PDFs for testing
                        if len(test_pdfs) >= 10:
                            return test_pdfs
        
        return test_pdfs
    
    def generate_performance_summary(self) -> Dict[str, Any]:
        """Generate overall performance summary."""
        summary = {
            'tests_run': 0,
            'tests_successful': 0,
            'tests_skipped': 0,
            'tests_failed': 0,
            'key_improvements': [],
            'recommendations': []
        }
        
        for test_name, test_result in self.test_results.items():
            if test_name == 'summary':
                continue
                
            summary['tests_run'] += 1
            
            if isinstance(test_result, dict):
                if test_result.get('status') == 'skipped':
                    summary['tests_skipped'] += 1
                elif test_result.get('status') == 'error':
                    summary['tests_failed'] += 1
                else:
                    summary['tests_successful'] += 1
                    
                    # Extract key improvements
                    if test_name == 'parallel_processing':
                        if 'speedup_2x' in test_result:
                            speedup = test_result['speedup_2x']
                            summary['key_improvements'].append(
                                f"Parallel processing: {speedup:.1f}x speedup with 2 workers"
                            )
                    
                    elif test_name == 'embedding_caching':
                        if 'cache_speedup' in test_result:
                            speedup = test_result['cache_speedup']
                            hit_rate = test_result.get('hit_rate_percent', 0)
                            summary['key_improvements'].append(
                                f"Embedding cache: {speedup:.1f}x speedup, {hit_rate:.1f}% hit rate"
                            )
                    
                    elif test_name == 'database_optimization':
                        if 'knowledge_graph' in test_result:
                            kg_result = test_result['knowledge_graph']
                            if 'performance_improvement_percent' in kg_result:
                                improvement = kg_result['performance_improvement_percent']
                                summary['key_improvements'].append(
                                    f"Database optimization: {improvement:.1f}% query performance improvement"
                                )
        
        # Generate recommendations
        if summary['tests_successful'] > 0:
            summary['recommendations'].append("Performance optimizations are working correctly")
            
        if summary['tests_skipped'] > 0:
            summary['recommendations'].append("Some tests were skipped - consider adding more test data")
            
        if summary['tests_failed'] > 0:
            summary['recommendations'].append("Some tests failed - check error logs for details")
        
        return summary


def main():
    """Run performance tests."""
    print("arXiv Papers Analysis - Performance Testing")
    print("=" * 50)
    
    logger = get_logger()
    
    try:
        tester = PerformanceTester()
        results = tester.run_all_tests()
        
        # Print summary
        print("\n" + "=" * 50)
        print("PERFORMANCE TEST SUMMARY")
        print("=" * 50)
        
        summary = results['summary']
        
        print(f"Tests run: {summary['tests_run']}")
        print(f"Successful: {summary['tests_successful']}")
        print(f"Skipped: {summary['tests_skipped']}")
        print(f"Failed: {summary['tests_failed']}")
        
        if summary['key_improvements']:
            print(f"\nKey Performance Improvements:")
            for improvement in summary['key_improvements']:
                print(f"  • {improvement}")
        
        if summary['recommendations']:
            print(f"\nRecommendations:")
            for recommendation in summary['recommendations']:
                print(f"  • {recommendation}")
        
        # Save detailed results
        import json
        results_file = "performance_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        log_info("Performance testing completed successfully")
        
        return 0
        
    except Exception as e:
        logger.log_operation_failure("Performance testing", e)
        print(f"\nPerformance testing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())