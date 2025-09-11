"""
Database optimization utilities for SQLite knowledge graph and vector databases.
Provides indexing, query optimization, and performance monitoring.
"""

import sqlite3
import os
import time
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from config import get_config
from logger import get_logger, log_info, log_warning, log_error


class SQLiteOptimizer:
    """SQLite database optimization and performance monitoring."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = get_logger()
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
    
    def analyze_database(self) -> Dict[str, Any]:
        """Analyze database structure and performance characteristics."""
        self.logger.log_operation_start("Database analysis", db_path=self.db_path)
        
        analysis = {
            'db_path': self.db_path,
            'file_size_mb': 0,
            'tables': {},
            'indices': {},
            'performance_stats': {},
            'optimization_suggestions': []
        }
        
        try:
            # Get file size
            file_size = os.path.getsize(self.db_path)
            analysis['file_size_mb'] = file_size / (1024 * 1024)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get table information
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    # Table row count and size
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    row_count = cursor.fetchone()[0]
                    
                    # Get table schema
                    cursor.execute(f"PRAGMA table_info({table})")
                    columns = cursor.fetchall()
                    
                    analysis['tables'][table] = {
                        'row_count': row_count,
                        'columns': len(columns),
                        'column_info': [
                            {
                                'name': col[1], 
                                'type': col[2], 
                                'not_null': bool(col[3]),
                                'primary_key': bool(col[5])
                            } 
                            for col in columns
                        ]
                    }
                
                # Get index information
                cursor.execute("SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index'")
                indices = cursor.fetchall()
                
                for index in indices:
                    index_name, table_name, sql = index
                    analysis['indices'][index_name] = {
                        'table': table_name,
                        'sql': sql,
                        'is_auto': index_name.startswith('sqlite_autoindex_')
                    }
                
                # Performance statistics
                cursor.execute("PRAGMA compile_options")
                compile_options = [row[0] for row in cursor.fetchall()]
                
                analysis['performance_stats'] = {
                    'page_size': self._get_pragma_value(cursor, 'page_size'),
                    'page_count': self._get_pragma_value(cursor, 'page_count'),
                    'cache_size': self._get_pragma_value(cursor, 'cache_size'),
                    'journal_mode': self._get_pragma_value(cursor, 'journal_mode'),
                    'synchronous': self._get_pragma_value(cursor, 'synchronous'),
                    'temp_store': self._get_pragma_value(cursor, 'temp_store'),
                    'compile_options': compile_options
                }
        
        except Exception as e:
            self.logger.log_operation_failure("Database analysis", e, db_path=self.db_path)
            raise
        
        # Generate optimization suggestions
        analysis['optimization_suggestions'] = self._generate_optimization_suggestions(analysis)
        
        self.logger.log_operation_success("Database analysis", 
                                        tables=len(analysis['tables']),
                                        indices=len(analysis['indices']),
                                        size_mb=analysis['file_size_mb'])
        
        return analysis
    
    def _get_pragma_value(self, cursor: sqlite3.Cursor, pragma_name: str) -> Any:
        """Get a single pragma value."""
        try:
            cursor.execute(f"PRAGMA {pragma_name}")
            result = cursor.fetchone()
            return result[0] if result else None
        except:
            return None
    
    def _generate_optimization_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on analysis."""
        suggestions = []
        
        # Check for missing indices on frequently queried columns
        tables = analysis['tables']
        indices = analysis['indices']
        
        # Papers table optimizations
        if 'papers' in tables:
            papers_indices = [idx for idx in indices.values() if idx['table'] == 'papers']
            papers_index_names = [list(indices.keys())[i] for i, idx in enumerate(indices.values()) if idx['table'] == 'papers']
            
            # Check for common query patterns
            if not any('source_pdf' in str(idx.get('sql', '')) for idx in papers_indices):
                suggestions.append("Add index on papers.source_pdf for faster lookups")
            
            if not any('title' in str(idx.get('sql', '')) for idx in papers_indices):
                suggestions.append("Add index on papers.title for text searches")
        
        # Authors table optimizations
        if 'authors' in tables:
            authors_indices = [idx for idx in indices.values() if idx['table'] == 'authors']
            
            if not any('name' in str(idx.get('sql', '')) for idx in authors_indices):
                suggestions.append("Add index on authors.name for author lookups")
        
        # Paper-authors relationship optimizations
        if 'paper_authors' in tables:
            pa_indices = [idx for idx in indices.values() if idx['table'] == 'paper_authors']
            
            if not any('paper_id' in str(idx.get('sql', '')) for idx in pa_indices):
                suggestions.append("Add index on paper_authors.paper_id")
            
            if not any('author_id' in str(idx.get('sql', '')) for idx in pa_indices):
                suggestions.append("Add index on paper_authors.author_id")
        
        # Performance settings
        perf_stats = analysis.get('performance_stats', {})
        
        if perf_stats.get('cache_size', 0) < 10000:
            suggestions.append("Increase cache_size to at least 10000 pages")
        
        if perf_stats.get('journal_mode') != 'WAL':
            suggestions.append("Consider using WAL journal mode for better concurrency")
        
        # Large table warnings
        for table_name, table_info in tables.items():
            if table_info['row_count'] > 100000:
                suggestions.append(f"Table {table_name} has {table_info['row_count']} rows - consider partitioning or archiving")
        
        return suggestions
    
    def optimize_knowledge_graph(self) -> Dict[str, Any]:
        """Optimize the knowledge graph database with appropriate indices."""
        self.logger.log_operation_start("Knowledge graph optimization", db_path=self.db_path)
        
        optimization_results = {
            'indices_created': [],
            'indices_failed': [],
            'performance_settings': {},
            'vacuum_performed': False,
            'analyze_performed': False
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create optimized indices
                indices_to_create = [
                    # Papers table indices
                    ("idx_papers_source_pdf", "CREATE INDEX IF NOT EXISTS idx_papers_source_pdf ON papers(source_pdf)"),
                    ("idx_papers_title_fts", "CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title COLLATE NOCASE)"),
                    ("idx_papers_publication_date", "CREATE INDEX IF NOT EXISTS idx_papers_publication_date ON papers(publication_date)"),
                    
                    # Authors table indices
                    ("idx_authors_name_fts", "CREATE INDEX IF NOT EXISTS idx_authors_name ON authors(name COLLATE NOCASE)"),
                    
                    # Paper-authors relationship indices
                    ("idx_paper_authors_paper_id", "CREATE INDEX IF NOT EXISTS idx_paper_authors_paper_id ON paper_authors(paper_id)"),
                    ("idx_paper_authors_author_id", "CREATE INDEX IF NOT EXISTS idx_paper_authors_author_id ON paper_authors(author_id)"),
                    ("idx_paper_authors_composite", "CREATE INDEX IF NOT EXISTS idx_paper_authors_composite ON paper_authors(paper_id, author_id)"),
                    
                    # Citations table indices (if exists)
                    ("idx_citations_citing", "CREATE INDEX IF NOT EXISTS idx_citations_citing ON citations(citing_paper_id)"),
                    ("idx_citations_cited", "CREATE INDEX IF NOT EXISTS idx_citations_cited ON citations(cited_paper_id)"),
                ]
                
                for index_name, sql in indices_to_create:
                    try:
                        start_time = time.time()
                        cursor.execute(sql)
                        creation_time = time.time() - start_time
                        
                        optimization_results['indices_created'].append({
                            'name': index_name,
                            'creation_time_seconds': creation_time
                        })
                        
                        log_info("Index created successfully", 
                               index=index_name, 
                               time=f"{creation_time:.2f}s")
                    
                    except sqlite3.Error as e:
                        optimization_results['indices_failed'].append({
                            'name': index_name,
                            'error': str(e)
                        })
                        log_warning("Index creation failed", index=index_name, error=str(e))
                
                # Optimize performance settings
                performance_settings = [
                    ("PRAGMA cache_size = 20000", "Increase cache size to 20MB"),
                    ("PRAGMA temp_store = MEMORY", "Store temporary tables in memory"),
                    ("PRAGMA synchronous = NORMAL", "Balance durability and performance"),
                    ("PRAGMA journal_mode = WAL", "Enable Write-Ahead Logging"),
                ]
                
                for pragma_sql, description in performance_settings:
                    try:
                        cursor.execute(pragma_sql)
                        setting_name = pragma_sql.split('=')[0].strip().replace('PRAGMA ', '')
                        optimization_results['performance_settings'][setting_name] = description
                        log_info("Performance setting applied", setting=setting_name)
                    except sqlite3.Error as e:
                        log_warning("Performance setting failed", setting=pragma_sql, error=str(e))
                
                # Update table statistics
                try:
                    cursor.execute("ANALYZE")
                    optimization_results['analyze_performed'] = True
                    log_info("Database statistics updated")
                except sqlite3.Error as e:
                    log_warning("ANALYZE command failed", error=str(e))
                
                conn.commit()
        
        except Exception as e:
            self.logger.log_operation_failure("Knowledge graph optimization", e, db_path=self.db_path)
            raise
        
        # Perform VACUUM in a separate connection (can't be in transaction)
        try:
            with sqlite3.connect(self.db_path) as vacuum_conn:
                vacuum_conn.execute("VACUUM")
                optimization_results['vacuum_performed'] = True
                log_info("Database vacuum completed")
        except sqlite3.Error as e:
            log_warning("VACUUM operation failed", error=str(e))
        
        self.logger.log_operation_success("Knowledge graph optimization",
                                        indices_created=len(optimization_results['indices_created']),
                                        indices_failed=len(optimization_results['indices_failed']))
        
        return optimization_results
    
    def benchmark_queries(self, test_queries: List[str] = None) -> Dict[str, Any]:
        """Benchmark common queries to measure performance."""
        if test_queries is None:
            test_queries = [
                "SELECT COUNT(*) FROM papers",
                "SELECT COUNT(*) FROM authors", 
                "SELECT COUNT(*) FROM paper_authors",
                "SELECT p.title FROM papers p ORDER BY p.id LIMIT 10",
                "SELECT a.name FROM authors a ORDER BY a.name LIMIT 10",
                "SELECT p.title, a.name FROM papers p JOIN paper_authors pa ON p.id = pa.paper_id JOIN authors a ON a.id = pa.author_id LIMIT 10",
            ]
        
        self.logger.log_operation_start("Query benchmarking", query_count=len(test_queries))
        
        benchmark_results = {
            'queries': [],
            'total_time': 0,
            'average_time': 0,
            'fastest_query': None,
            'slowest_query': None
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for i, query in enumerate(test_queries):
                    start_time = time.time()
                    
                    try:
                        cursor.execute(query)
                        results = cursor.fetchall()
                        execution_time = time.time() - start_time
                        
                        query_result = {
                            'query': query,
                            'execution_time': execution_time,
                            'result_count': len(results),
                            'success': True
                        }
                        
                    except sqlite3.Error as e:
                        execution_time = time.time() - start_time
                        query_result = {
                            'query': query,
                            'execution_time': execution_time,
                            'error': str(e),
                            'success': False
                        }
                    
                    benchmark_results['queries'].append(query_result)
        
        except Exception as e:
            self.logger.log_operation_failure("Query benchmarking", e)
            raise
        
        # Calculate statistics
        successful_queries = [q for q in benchmark_results['queries'] if q['success']]
        
        if successful_queries:
            times = [q['execution_time'] for q in successful_queries]
            benchmark_results['total_time'] = sum(times)
            benchmark_results['average_time'] = sum(times) / len(times)
            
            fastest = min(successful_queries, key=lambda x: x['execution_time'])
            slowest = max(successful_queries, key=lambda x: x['execution_time'])
            
            benchmark_results['fastest_query'] = {
                'query': fastest['query'],
                'time': fastest['execution_time']
            }
            benchmark_results['slowest_query'] = {
                'query': slowest['query'],
                'time': slowest['execution_time']
            }
        
        self.logger.log_operation_success("Query benchmarking",
                                        total_queries=len(test_queries),
                                        successful=len(successful_queries),
                                        avg_time=benchmark_results['average_time'])
        
        return benchmark_results
    
    def get_query_plan(self, query: str) -> List[Dict[str, Any]]:
        """Get the query execution plan for a specific query."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get query plan
                cursor.execute(f"EXPLAIN QUERY PLAN {query}")
                plan_rows = cursor.fetchall()
                
                query_plan = []
                for row in plan_rows:
                    query_plan.append({
                        'selectid': row[0],
                        'order': row[1], 
                        'from': row[2],
                        'detail': row[3]
                    })
                
                return query_plan
        
        except sqlite3.Error as e:
            log_error("Query plan analysis failed", query=query[:100], error=str(e))
            return []


def optimize_all_databases():
    """Optimize all known databases in the system."""
    config = get_config()
    logger = get_logger()
    
    logger.log_operation_start("System-wide database optimization")
    
    results = {
        'knowledge_graph': None,
        'vector_cache': None,
        'embedding_cache': None
    }
    
    # Optimize knowledge graph
    kg_db_path = config.database.knowledge_db_path
    if os.path.exists(kg_db_path):
        try:
            optimizer = SQLiteOptimizer(kg_db_path)
            results['knowledge_graph'] = optimizer.optimize_knowledge_graph()
            log_info("Knowledge graph optimized", db_path=kg_db_path)
        except Exception as e:
            log_error("Knowledge graph optimization failed", error=e, db_path=kg_db_path)
    
    # Optimize embedding cache if it exists
    cache_db_path = os.path.join("cache", "embeddings.db")
    if os.path.exists(cache_db_path):
        try:
            optimizer = SQLiteOptimizer(cache_db_path)
            # Simple optimization for cache database
            with sqlite3.connect(cache_db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA optimize")
                cursor.execute("VACUUM")
            results['embedding_cache'] = {'optimized': True}
            log_info("Embedding cache optimized", db_path=cache_db_path)
        except Exception as e:
            log_error("Embedding cache optimization failed", error=e, db_path=cache_db_path)
    
    logger.log_operation_success("System-wide database optimization", 
                               databases_optimized=sum(1 for r in results.values() if r is not None))
    
    return results


def create_performance_report(db_path: str) -> str:
    """Create a comprehensive performance report for a database."""
    try:
        optimizer = SQLiteOptimizer(db_path)
        
        # Analyze database
        analysis = optimizer.analyze_database()
        
        # Benchmark queries
        benchmark = optimizer.benchmark_queries()
        
        # Generate report
        report_lines = [
            "DATABASE PERFORMANCE REPORT",
            "=" * 50,
            f"Database: {db_path}",
            f"File Size: {analysis['file_size_mb']:.2f} MB",
            f"Tables: {len(analysis['tables'])}",
            f"Indices: {len(analysis['indices'])}",
            "",
            "TABLE STATISTICS:",
            "-" * 20
        ]
        
        for table_name, table_info in analysis['tables'].items():
            report_lines.append(f"{table_name}: {table_info['row_count']:,} rows, {table_info['columns']} columns")
        
        report_lines.extend([
            "",
            "QUERY PERFORMANCE:",
            "-" * 20,
            f"Average Query Time: {benchmark['average_time']:.4f} seconds",
            f"Fastest Query: {benchmark['fastest_query']['time']:.4f}s" if benchmark['fastest_query'] else "N/A",
            f"Slowest Query: {benchmark['slowest_query']['time']:.4f}s" if benchmark['slowest_query'] else "N/A",
            "",
            "OPTIMIZATION SUGGESTIONS:",
            "-" * 30
        ])
        
        for suggestion in analysis['optimization_suggestions']:
            report_lines.append(f"• {suggestion}")
        
        return "\n".join(report_lines)
    
    except Exception as e:
        return f"Error generating performance report: {e}"