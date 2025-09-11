"""
Parallel processing utilities for PDF processing and other CPU-intensive tasks.
Provides multiprocessing support with progress tracking and error handling.
"""

import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Callable, Any, Optional, Dict, Tuple
from functools import partial
import threading
import queue

from config import get_config
from logger import get_logger, log_info, log_warning, log_error


@dataclass
class ProcessingResult:
    """Result of a parallel processing operation."""
    success: bool = False
    result: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    worker_id: str = ""
    input_data: Any = None


class ProgressTracker:
    """Thread-safe progress tracker for parallel operations."""
    
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.lock = threading.Lock()
        self.start_time = time.time()
        
    def update(self, success: bool = True):
        """Update progress counters."""
        with self.lock:
            self.completed += 1
            if not success:
                self.failed += 1
    
    def get_progress(self) -> Tuple[int, int, int, float]:
        """Get current progress: (completed, failed, total, elapsed_time)."""
        with self.lock:
            elapsed = time.time() - self.start_time
            return self.completed, self.failed, self.total, elapsed
    
    def get_progress_string(self) -> str:
        """Get formatted progress string."""
        completed, failed, total, elapsed = self.get_progress()
        remaining = total - completed
        success_rate = ((completed - failed) / completed * 100) if completed > 0 else 0
        
        return (f"Progress: {completed}/{total} ({remaining} remaining) | "
                f"Success: {success_rate:.1f}% | Elapsed: {elapsed:.1f}s")


def process_pdf_worker(pdf_path: str, worker_id: str, operations: List[str]) -> ProcessingResult:
    """
    Worker function to process a single PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        worker_id: Unique identifier for this worker
        operations: List of operations to perform ['validate', 'extract', 'chunk']
    
    Returns:
        ProcessingResult with the results of processing
    """
    start_time = time.time()
    result = ProcessingResult(worker_id=worker_id, input_data=pdf_path)
    
    try:
        from pdf_validator import validate_pdf
        from llm_utils import extract_paper_entities_safe
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        config = get_config()
        processing_data = {}
        
        # Validation
        if 'validate' in operations:
            validation_result = validate_pdf(pdf_path)
            processing_data['validation'] = {
                'is_valid': validation_result.is_valid,
                'size_mb': validation_result.size_mb,
                'page_count': validation_result.page_count,
                'has_text': validation_result.has_text,
                'errors': validation_result.errors,
                'warnings': validation_result.warnings
            }
            
            if not validation_result.is_valid:
                result.error = f"PDF validation failed: {len(validation_result.errors)} errors"
                return result
        
        # Text extraction and entity extraction
        if 'extract' in operations:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            if documents:
                paper_text = " ".join([doc.page_content for doc in documents])
                entities, errors, warnings = extract_paper_entities_safe(paper_text)
                
                processing_data['extraction'] = {
                    'entities': entities,
                    'text_length': len(paper_text),
                    'document_count': len(documents),
                    'errors': errors,
                    'warnings': warnings
                }
            else:
                result.error = "Could not extract text from PDF"
                return result
        
        # Text chunking
        if 'chunk' in operations:
            if 'extract' not in operations:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
            
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.models.chunk_size,
                    chunk_overlap=config.models.chunk_overlap
                )
                chunks = text_splitter.split_documents(documents)
                
                processing_data['chunking'] = {
                    'chunk_count': len(chunks),
                    'total_chars': sum(len(chunk.page_content) for chunk in chunks),
                    'avg_chunk_size': sum(len(chunk.page_content) for chunk in chunks) / len(chunks) if chunks else 0
                }
            else:
                result.error = "Could not load documents for chunking"
                return result
        
        result.success = True
        result.result = processing_data
        result.processing_time = time.time() - start_time
        
    except Exception as e:
        result.error = f"Processing error: {str(e)}"
        result.processing_time = time.time() - start_time
    
    return result


def process_pdfs_parallel(
    pdf_paths: List[str], 
    operations: List[str] = None,
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable] = None
) -> Tuple[List[ProcessingResult], Dict[str, Any]]:
    """
    Process multiple PDFs in parallel.
    
    Args:
        pdf_paths: List of PDF file paths to process
        operations: Operations to perform ['validate', 'extract', 'chunk']
        max_workers: Maximum number of worker processes (None for auto)
        progress_callback: Optional callback function for progress updates
    
    Returns:
        Tuple of (results_list, summary_stats)
    """
    if operations is None:
        operations = ['validate', 'extract', 'chunk']
    
    logger = get_logger()
    
    # Determine optimal number of workers
    if max_workers is None:
        cpu_count = mp.cpu_count()
        max_workers = max(1, min(cpu_count - 1, len(pdf_paths), 8))  # Leave 1 core free, max 8 workers
    
    logger.log_operation_start(
        "Parallel PDF processing",
        file_count=len(pdf_paths),
        operations=operations,
        max_workers=max_workers
    )
    
    start_time = time.time()
    progress = ProgressTracker(len(pdf_paths))
    results = []
    
    # Create worker function with bound operations
    worker_func = partial(process_pdf_worker, operations=operations)
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_path = {}
            for i, pdf_path in enumerate(pdf_paths):
                worker_id = f"worker_{i:03d}"
                future = executor.submit(worker_func, pdf_path, worker_id)
                future_to_path[future] = pdf_path
            
            # Process completed tasks
            for future in as_completed(future_to_path):
                pdf_path = future_to_path[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    progress.update(result.success)
                    
                    if progress_callback:
                        progress_callback(progress.get_progress_string())
                    
                    if result.success:
                        log_info("PDF processed successfully", 
                               file=os.path.basename(pdf_path),
                               worker=result.worker_id,
                               time=f"{result.processing_time:.2f}s")
                    else:
                        log_warning("PDF processing failed",
                                  file=os.path.basename(pdf_path),
                                  worker=result.worker_id,
                                  error=result.error)
                
                except Exception as e:
                    error_result = ProcessingResult(
                        success=False,
                        error=f"Future execution error: {str(e)}",
                        input_data=pdf_path
                    )
                    results.append(error_result)
                    progress.update(False)
                    
                    log_error("PDF processing exception", 
                            file=os.path.basename(pdf_path),
                            error=e)
    
    except Exception as e:
        logger.log_operation_failure("Parallel PDF processing", e)
        raise
    
    # Calculate summary statistics
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r.success)
    failed = len(results) - successful
    
    avg_time = sum(r.processing_time for r in results if r.success) / max(successful, 1)
    
    summary_stats = {
        'total_files': len(pdf_paths),
        'successful': successful,
        'failed': failed,
        'success_rate': (successful / len(pdf_paths) * 100) if pdf_paths else 0,
        'total_time': total_time,
        'average_processing_time': avg_time,
        'speedup_estimate': len(pdf_paths) * avg_time / total_time if total_time > 0 else 1,
        'operations_performed': operations,
        'workers_used': max_workers
    }
    
    logger.log_operation_success(
        "Parallel PDF processing",
        **summary_stats
    )
    
    return results, summary_stats


def batch_process_with_memory_management(
    items: List[Any], 
    processor_func: Callable,
    batch_size: int = 50,
    max_workers: int = None
) -> List[Any]:
    """
    Process items in batches to manage memory usage.
    
    Args:
        items: List of items to process
        processor_func: Function to process each item
        batch_size: Number of items per batch
        max_workers: Maximum number of workers per batch
    
    Returns:
        List of processing results
    """
    logger = get_logger()
    
    if max_workers is None:
        max_workers = min(mp.cpu_count() - 1, 4)
    
    total_items = len(items)
    all_results = []
    
    logger.log_operation_start(
        "Batch processing",
        total_items=total_items,
        batch_size=batch_size,
        estimated_batches=(total_items + batch_size - 1) // batch_size
    )
    
    for batch_num, start_idx in enumerate(range(0, total_items, batch_size)):
        end_idx = min(start_idx + batch_size, total_items)
        batch = items[start_idx:end_idx]
        
        log_info(f"Processing batch {batch_num + 1}",
               items=len(batch),
               progress=f"{end_idx}/{total_items}")
        
        # Process batch with ThreadPoolExecutor for I/O bound tasks
        # or ProcessPoolExecutor for CPU bound tasks
        batch_results = []
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(processor_func, item) for item in batch]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        log_error("Batch item processing failed", error=e)
                        batch_results.append(None)
        
        except Exception as e:
            log_error("Batch processing failed", batch_num=batch_num, error=e)
            batch_results = [None] * len(batch)
        
        all_results.extend(batch_results)
        
        # Optional: Force garbage collection between batches
        import gc
        gc.collect()
    
    logger.log_operation_success(
        "Batch processing",
        total_processed=len(all_results),
        successful=sum(1 for r in all_results if r is not None)
    )
    
    return all_results


class ParallelEmbeddingProcessor:
    """Specialized processor for computing embeddings in parallel."""
    
    def __init__(self, model_name: str = None, batch_size: int = 32):
        config = get_config()
        self.model_name = model_name or config.models.embedding_model_name
        self.batch_size = batch_size
        self.logger = get_logger()
    
    def compute_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for a batch of texts."""
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            
            # Initialize embeddings model in worker process
            embeddings_model = HuggingFaceEmbeddings(model_name=self.model_name)
            
            # Compute embeddings
            embeddings = embeddings_model.embed_documents(texts)
            
            return embeddings
        
        except Exception as e:
            self.logger.get_logger().error(f"Embedding computation failed: {e}")
            return [[0.0] * 384] * len(texts)  # Return zero embeddings as fallback
    
    def process_texts_parallel(
        self, 
        texts: List[str], 
        max_workers: int = None
    ) -> List[List[float]]:
        """Process texts in parallel to compute embeddings."""
        
        if max_workers is None:
            max_workers = min(mp.cpu_count() // 2, 4)  # Conservative for memory
        
        self.logger.log_operation_start(
            "Parallel embedding computation",
            text_count=len(texts),
            batch_size=self.batch_size,
            max_workers=max_workers
        )
        
        # Split texts into batches
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batches.append(batch)
        
        all_embeddings = []
        
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all batches
                futures = [
                    executor.submit(self.compute_embeddings_batch, batch) 
                    for batch in batches
                ]
                
                # Collect results
                for i, future in enumerate(as_completed(futures)):
                    try:
                        batch_embeddings = future.result()
                        all_embeddings.extend(batch_embeddings)
                        
                        log_info(f"Completed embedding batch {i+1}/{len(batches)}")
                        
                    except Exception as e:
                        log_error("Embedding batch failed", batch_idx=i, error=e)
                        # Add zero embeddings for failed batch
                        batch_size = len(batches[i])
                        all_embeddings.extend([[0.0] * 384] * batch_size)
            
            self.logger.log_operation_success(
                "Parallel embedding computation",
                embeddings_computed=len(all_embeddings)
            )
            
            return all_embeddings
        
        except Exception as e:
            self.logger.log_operation_failure("Parallel embedding computation", e)
            # Return zero embeddings as complete fallback
            return [[0.0] * 384] * len(texts)


# Utility functions for common parallel operations
def get_optimal_worker_count(task_type: str = "cpu") -> int:
    """Get optimal worker count based on task type and system resources."""
    cpu_count = mp.cpu_count()
    
    if task_type == "cpu":
        # CPU-intensive tasks: leave 1-2 cores free
        return max(1, cpu_count - 1)
    elif task_type == "io":
        # I/O intensive tasks: can use more workers
        return min(cpu_count * 2, 16)
    elif task_type == "memory":
        # Memory-intensive tasks: conservative worker count
        return max(1, cpu_count // 2)
    else:
        # Default: balanced approach
        return max(1, cpu_count - 1)


def progress_printer(progress_string: str):
    """Simple progress callback that prints to console."""
    print(f"\r{progress_string}", end="", flush=True)


# Example usage functions
def parallel_pdf_validation(pdf_paths: List[str]) -> Tuple[List[ProcessingResult], Dict]:
    """Validate multiple PDFs in parallel."""
    return process_pdfs_parallel(pdf_paths, operations=['validate'])


def parallel_pdf_extraction(pdf_paths: List[str]) -> Tuple[List[ProcessingResult], Dict]:
    """Extract entities from multiple PDFs in parallel."""
    return process_pdfs_parallel(pdf_paths, operations=['validate', 'extract'])


def parallel_pdf_full_processing(pdf_paths: List[str]) -> Tuple[List[ProcessingResult], Dict]:
    """Full processing of multiple PDFs in parallel."""
    return process_pdfs_parallel(
        pdf_paths, 
        operations=['validate', 'extract', 'chunk'],
        progress_callback=progress_printer
    )