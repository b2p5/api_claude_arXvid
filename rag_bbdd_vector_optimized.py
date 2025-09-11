"""
Optimized version of the RAG database vector script with performance improvements.
Includes parallel processing, caching, intelligent chunking, and database optimization.
"""

import os
import shutil
import argparse
import chromadb
import dotenv
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

from config import get_config
from logger import get_logger, log_info, log_error, log_warning
from pdf_validator import validate_pdf
from llm_utils import extract_paper_entities_safe
from retry_utils import safe_execute
from parallel_processing import process_pdfs_parallel, ParallelEmbeddingProcessor
from embedding_cache import get_embedding_cache, CachedEmbeddingModel
from intelligent_chunking import create_intelligent_splitter, analyze_chunking_quality
from database_optimizer import optimize_all_databases

# Import and initialize the knowledge graph
import knowledge_graph


class OptimizedRAGProcessor:
    """Optimized RAG processor with all performance enhancements."""
    
    def __init__(self, force_rebuild: bool = False):
        self.config = get_config()
        self.logger = get_logger()
        self.force_rebuild = force_rebuild
        
        # Initialize components
        self.embedding_cache = get_embedding_cache()
        self.cached_embedding_model = CachedEmbeddingModel()
        self.intelligent_splitter = create_intelligent_splitter()
        
        # Statistics
        self.stats = {
            'total_pdfs': 0,
            'processed_pdfs': 0,
            'cached_embeddings': 0,
            'new_embeddings': 0,
            'processing_time': 0,
            'chunks_created': 0,
            'kg_entries_added': 0,
            'vector_entries_added': 0
        }
        
        log_info("Initialized optimized RAG processor", 
               force_rebuild=force_rebuild)
    
    def update_databases(self, root_docs_path: str, db_path: str) -> Dict[str, Any]:
        """
        Update both vector and knowledge graph databases with optimizations.
        
        Args:
            root_docs_path: Root directory containing PDF documents
            db_path: Path to vector database
            
        Returns:
            Processing statistics and results
        """
        start_time = time.time()
        
        self.logger.log_operation_start("Optimized database update", 
                                      root_docs_path=root_docs_path,
                                      db_path=db_path)
        
        # Clean existing databases if force rebuild
        if self.force_rebuild:
            self._clean_existing_databases(db_path)
        
        # Collect all PDF files
        pdf_files = self._collect_pdf_files(root_docs_path)
        self.stats['total_pdfs'] = len(pdf_files)
        
        if not pdf_files:
            log_warning("No PDF files found", root_docs_path=root_docs_path)
            return self.stats
        
        # Initialize databases
        vector_db = chromadb.PersistentClient(path=db_path)
        collection = vector_db.get_or_create_collection(
            name=self.config.database.vector_collection_name
        )
        knowledge_graph.create_database()
        
        # Optimize databases
        optimization_results = optimize_all_databases()
        log_info("Database optimization completed", results=optimization_results)
        
        # Process PDFs in parallel
        processing_results = self._parallel_pdf_processing(pdf_files, collection)
        
        # Update statistics
        self.stats['processing_time'] = time.time() - start_time
        
        self.logger.log_operation_success("Optimized database update", **self.stats)
        
        return {
            'stats': self.stats,
            'optimization_results': optimization_results,
            'processing_results': processing_results
        }
    
    def _clean_existing_databases(self, db_path: str):
        """Clean existing databases for fresh rebuild."""
        log_info("Force rebuild - cleaning existing databases")
        
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
            log_info("Removed vector database", path=db_path)
        
        if os.path.exists(knowledge_graph.DB_FILE):
            os.remove(knowledge_graph.DB_FILE)
            log_info("Removed knowledge graph database", path=knowledge_graph.DB_FILE)
        
        # Clean embedding cache if requested
        cache_stats = self.embedding_cache.get_cache_stats()
        if cache_stats['embeddings_cached'] > 0:
            log_info("Cleaning embedding cache", cached_embeddings=cache_stats['embeddings_cached'])
            # Optionally clean cache - for now just log
    
    def _collect_pdf_files(self, root_docs_path: str) -> List[Tuple[str, str]]:
        """
        Collect all PDF files with their concept directories.
        
        Returns:
            List of (pdf_path, concept) tuples
        """
        pdf_files = []
        
        if not os.path.exists(root_docs_path):
            log_error("Root documents directory not found", path=root_docs_path)
            return pdf_files
        
        for concept_dir in os.listdir(root_docs_path):
            concept_path = os.path.join(root_docs_path, concept_dir)
            if not os.path.isdir(concept_path):
                continue
            
            for file_name in os.listdir(concept_path):
                if file_name.endswith(".pdf"):
                    pdf_path = os.path.join(concept_path, file_name)
                    pdf_files.append((pdf_path, concept_dir))
        
        log_info("Collected PDF files", total_files=len(pdf_files))
        return pdf_files
    
    def _parallel_pdf_processing(self, pdf_files: List[Tuple[str, str]], collection) -> Dict[str, Any]:
        """Process PDFs in parallel with all optimizations."""
        
        # Filter PDFs that need processing
        pdfs_to_process = self._filter_pdfs_for_processing(pdf_files, collection)
        
        if not pdfs_to_process:
            log_info("All PDFs already processed - nothing to do")
            return {'message': 'All PDFs already processed'}
        
        log_info("Starting parallel processing", 
               total_pdfs=len(pdf_files),
               to_process=len(pdfs_to_process))
        
        # Process in parallel using custom processor
        results, summary = process_pdfs_parallel(
            [pdf_info[0] for pdf_info in pdfs_to_process],
            operations=['validate', 'extract', 'chunk'],
            max_workers=min(8, len(pdfs_to_process))
        )
        
        # Process results and update databases
        self._process_parallel_results(results, pdfs_to_process, collection)
        
        return summary
    
    def _filter_pdfs_for_processing(self, pdf_files: List[Tuple[str, str]], collection) -> List[Tuple[str, str]]:
        """Filter PDFs that need processing (not already in databases)."""
        pdfs_to_process = []
        
        with knowledge_graph.get_db_connection() as kg_conn:
            for pdf_path, concept in pdf_files:
                # Check if already in knowledge graph
                cursor = kg_conn.cursor()
                cursor.execute("SELECT id FROM papers WHERE source_pdf = ?", (pdf_path,))
                kg_exists = cursor.fetchone() is not None
                
                # Check if already in vector database
                existing_docs = collection.get(where={"source": pdf_path})
                vector_exists = len(existing_docs['ids']) > 0
                
                if not (kg_exists and vector_exists) or self.force_rebuild:
                    pdfs_to_process.append((pdf_path, concept))
                else:
                    log_info("PDF already processed - skipping", 
                           file=os.path.basename(pdf_path))
        
        return pdfs_to_process
    
    def _process_parallel_results(
        self, 
        results: List[Any], 
        pdf_info_list: List[Tuple[str, str]], 
        collection
    ):
        """Process results from parallel PDF processing and update databases."""
        
        # Separate successful and failed results
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        log_info("Processing parallel results", 
               successful=len(successful_results),
               failed=len(failed_results))
        
        if failed_results:
            log_warning("Some PDF processing failed", 
                      failed_count=len(failed_results))
            for result in failed_results:
                log_error("PDF processing failed", 
                        file=os.path.basename(result.input_data),
                        error=result.error)
        
        # Process successful results
        for result in successful_results:
            try:
                self._update_databases_from_result(result, collection)
                self.stats['processed_pdfs'] += 1
            except Exception as e:
                log_error("Database update failed for processed PDF", 
                        file=os.path.basename(result.input_data),
                        error=str(e))
        
        log_info("Database updates completed", 
               successful_updates=self.stats['processed_pdfs'])
    
    def _update_databases_from_result(self, result: Any, collection):
        """Update both knowledge graph and vector database from processing result."""
        pdf_path = result.input_data
        processing_data = result.result
        
        # Update knowledge graph
        if 'extraction' in processing_data:
            extraction_data = processing_data['extraction']
            entities = extraction_data['entities']
            
            if entities and entities.get('title') != 'Unknown':
                try:
                    paper_id = knowledge_graph.add_paper_with_authors(
                        title=entities['title'],
                        summary=entities['summary'],
                        source_pdf=pdf_path,
                        author_names=entities['authors'],
                        publication_date=entities.get('publication_date')
                    )
                    
                    if paper_id:
                        self.stats['kg_entries_added'] += 1
                        log_info("Added to knowledge graph", 
                               file=os.path.basename(pdf_path))
                    
                except Exception as e:
                    log_error("Knowledge graph update failed", 
                            file=os.path.basename(pdf_path),
                            error=str(e))
        
        # Update vector database with intelligent chunking
        if 'chunking' in processing_data:
            try:
                # Load the PDF again for vector processing
                # (In a more optimized version, we'd cache this)
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                if documents:
                    # Use intelligent chunking
                    chunks = self.intelligent_splitter.split_documents(documents)
                    
                    if chunks:
                        # Generate embeddings with caching
                        texts = [chunk.page_content for chunk in chunks]
                        embeddings = self.cached_embedding_model.embed_documents(texts)
                        
                        # Prepare for vector database insertion
                        chunk_ids = []
                        chunk_texts = []
                        chunk_metadatas = []
                        
                        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                            chunk_id = f"{os.path.basename(pdf_path)}_{i}_{hash(chunk.page_content)}"
                            chunk_ids.append(str(abs(hash(chunk_id))))
                            chunk_texts.append(chunk.page_content)
                            chunk_metadatas.append(chunk.metadata)
                        
                        # Add to vector database
                        collection.add(
                            ids=chunk_ids,
                            documents=chunk_texts,
                            metadatas=chunk_metadatas
                        )
                        
                        self.stats['vector_entries_added'] += len(chunks)
                        self.stats['chunks_created'] += len(chunks)
                        
                        log_info("Added to vector database", 
                               file=os.path.basename(pdf_path),
                               chunks=len(chunks))
                        
                        # Get cache statistics
                        cache_stats = self.embedding_cache.get_cache_stats()
                        self.stats['cached_embeddings'] = cache_stats['hits']
                        self.stats['new_embeddings'] = cache_stats['misses']
            
            except Exception as e:
                log_error("Vector database update failed", 
                        file=os.path.basename(pdf_path),
                        error=str(e))
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        # Add cache statistics
        cache_stats = self.embedding_cache.get_cache_stats()
        
        comprehensive_stats = {
            'processing': self.stats,
            'cache': cache_stats,
            'performance': {
                'pdfs_per_second': self.stats['processed_pdfs'] / max(self.stats['processing_time'], 1),
                'avg_chunks_per_pdf': self.stats['chunks_created'] / max(self.stats['processed_pdfs'], 1),
                'cache_hit_rate': cache_stats.get('hit_rate_percent', 0)
            }
        }
        
        return comprehensive_stats


def main():
    """Main function with optimized processing."""
    parser = argparse.ArgumentParser(
        description="Create or update the vector DB and knowledge graph from PDFs (OPTIMIZED VERSION)."
    )
    parser.add_argument("--force", action="store_true", 
                       help="Force deletion of existing databases before updating.")
    parser.add_argument("--parallel-workers", type=int, default=None,
                       help="Number of parallel workers (auto-detect if not specified)")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmarking after processing")
    parser.add_argument("--cache-stats", action="store_true", 
                       help="Show detailed cache statistics")
    
    args = parser.parse_args()
    
    logger = get_logger()
    
    try:
        logger.log_operation_start("Optimized RAG processing script")
        
        # Initialize optimized processor
        processor = OptimizedRAGProcessor(force_rebuild=args.force)
        
        # Get configuration
        config = get_config()
        
        # Process databases
        results = processor.update_databases(
            config.arxiv.documents_root, 
            config.database.vector_db_path
        )
        
        # Show results
        log_info("Processing completed successfully")
        
        stats = processor.get_processing_statistics()
        
        print(f"\n=== PROCESSING RESULTS ===")
        print(f"PDFs processed: {stats['processing']['processed_pdfs']}/{stats['processing']['total_pdfs']}")
        print(f"Total processing time: {stats['processing']['processing_time']:.2f} seconds")
        print(f"Knowledge graph entries: {stats['processing']['kg_entries_added']}")
        print(f"Vector database entries: {stats['processing']['vector_entries_added']}")
        print(f"Chunks created: {stats['processing']['chunks_created']}")
        print(f"Performance: {stats['performance']['pdfs_per_second']:.2f} PDFs/second")
        print(f"Cache hit rate: {stats['performance']['cache_hit_rate']:.1f}%")
        
        # Show cache statistics if requested
        if args.cache_stats:
            print(f"\n=== CACHE STATISTICS ===")
            cache_stats = stats['cache']
            print(f"Cache size: {cache_stats['cache_size_mb']:.2f} MB")
            print(f"Cached embeddings: {cache_stats['embeddings_cached']}")
            print(f"Cache hits: {cache_stats['hits']}")
            print(f"Cache misses: {cache_stats['misses']}")
            print(f"Hit rate: {cache_stats.get('hit_rate_percent', 0):.1f}%")
        
        # Run benchmarks if requested
        if args.benchmark:
            print(f"\n=== RUNNING BENCHMARKS ===")
            from database_optimizer import SQLiteOptimizer
            
            # Benchmark knowledge graph
            try:
                kg_optimizer = SQLiteOptimizer(config.database.knowledge_db_path)
                benchmark_results = kg_optimizer.benchmark_queries()
                print(f"Knowledge graph avg query time: {benchmark_results['average_time']:.4f}s")
            except Exception as e:
                print(f"Knowledge graph benchmark failed: {e}")
        
        logger.log_operation_success("Optimized RAG processing script", **stats['processing'])
        
    except KeyboardInterrupt:
        log_warning("Processing interrupted by user")
        print("\nProcessing interrupted by user.")
    except Exception as e:
        logger.log_operation_failure("Optimized RAG processing script", e)
        print(f"Processing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())