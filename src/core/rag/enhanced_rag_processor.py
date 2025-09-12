"""
Enhanced RAG Processor with Content Analysis Integration.
Extends the existing RAG system with advanced content analysis capabilities.
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import chromadb
from langchain_community.document_loaders import PyPDFLoader

from config import get_config
from logger import get_logger, log_info, log_warning, log_error
from core.analysis.pdf_validator import validate_pdf
from core.parallel_processing import ProcessingResult
from core.analysis.content_analysis import ContentAnalysisEngine, ContentAnalysis
from core.analysis.content_analysis_db import ContentAnalysisDatabase
from core.analysis import knowledge_graph


class EnhancedRAGProcessor:
    """Enhanced RAG processor with content analysis integration."""
    
    def __init__(self, enable_content_analysis: bool = True):
        self.config = get_config()
        self.logger = get_logger()
        self.enable_content_analysis = enable_content_analysis
        
        # Initialize components
        if self.enable_content_analysis:
            self.content_engine = ContentAnalysisEngine()
            self.content_db = ContentAnalysisDatabase()
        
        self.stats = {
            'total_papers': 0,
            'processed_papers': 0,
            'analyses_created': 0,
            'references_extracted': 0,
            'concepts_extracted': 0,
            'processing_time': 0
        }
        
        log_info("Enhanced RAG processor initialized", 
                content_analysis_enabled=self.enable_content_analysis)
    
    def process_papers_with_analysis(
        self, 
        pdf_files: List[str], 
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """Process papers with full content analysis."""
        start_time = time.time()
        self.stats['total_papers'] = len(pdf_files)
        
        log_info("Starting enhanced paper processing", 
                total_papers=len(pdf_files),
                content_analysis=self.enable_content_analysis)
        
        try:
            # Process papers in parallel
            results = self._parallel_process_papers(pdf_files, max_workers)
            
            # Process successful results
            for result in results:
                if result.success:
                    self._process_paper_result(result)
                    self.stats['processed_papers'] += 1
                else:
                    log_warning("Paper processing failed", 
                              file=result.input_data,
                              error=result.error)
            
            # Generate corpus-level analysis if enabled
            if self.enable_content_analysis and self.stats['processed_papers'] > 1:
                self._generate_corpus_analysis()
            
            self.stats['processing_time'] = time.time() - start_time
            
            log_info("Enhanced processing completed", **self.stats)
            
            return {
                'stats': self.stats,
                'success': True,
                'message': f"Processed {self.stats['processed_papers']}/{self.stats['total_papers']} papers"
            }
            
        except Exception as e:
            log_error("Enhanced processing failed", error=str(e))
            return {
                'stats': self.stats,
                'success': False,
                'error': str(e)
            }
    
    def _parallel_process_papers(self, pdf_files: List[str], max_workers: int) -> List[ProcessingResult]:
        """Process papers in parallel with content extraction."""
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_file = {
                executor.submit(self._process_single_paper, pdf_file): pdf_file
                for pdf_file in pdf_files
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                pdf_file = future_to_file[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    error_result = ProcessingResult(
                        input_data=pdf_file,
                        success=False,
                        result=None,
                        error=str(e),
                        processing_time=0.0
                    )
                    results.append(error_result)
        
        return results
    
    def _process_single_paper(self, pdf_file: str) -> ProcessingResult:
        """Process a single paper with content analysis."""
        start_time = time.time()
        
        try:
            # Validate PDF
            if not validate_pdf(pdf_file):
                return ProcessingResult(
                    input_data=pdf_file,
                    success=False,
                    result=None,
                    error="PDF validation failed",
                    processing_time=time.time() - start_time
                )
            
            # Extract content from PDF
            loader = PyPDFLoader(pdf_file)
            documents = loader.load()
            
            if not documents:
                return ProcessingResult(
                    input_data=pdf_file,
                    success=False,
                    result=None,
                    error="No content extracted from PDF",
                    processing_time=time.time() - start_time
                )
            
            # Combine all pages
            full_content = "\n".join(doc.page_content for doc in documents)
            
            # Extract metadata
            metadata = documents[0].metadata if documents else {}
            title = metadata.get('title', os.path.basename(pdf_file))
            
            result_data = {
                'pdf_file': pdf_file,
                'title': title,
                'content': full_content,
                'metadata': metadata,
                'page_count': len(documents)
            }
            
            # Perform content analysis if enabled
            if self.enable_content_analysis:
                try:
                    paper_id = self._generate_paper_id(pdf_file, title)
                    analysis = self.content_engine.analyze_paper(paper_id, title, full_content)
                    result_data['content_analysis'] = analysis
                except Exception as e:
                    log_warning("Content analysis failed", 
                              file=pdf_file, error=str(e))
                    result_data['content_analysis'] = None
            
            return ParallelResult(
                input_data=pdf_file,
                success=True,
                result=result_data,
                error=None,
                processing_time=time.time() - start_time
            )
            
        except Exception as e:
            return ParallelResult(
                input_data=pdf_file,
                success=False,
                result=None,
                error=str(e),
                processing_time=time.time() - start_time
            )
    
    def _process_paper_result(self, result: ProcessingResult):
        """Process a successful paper result."""
        try:
            data = result.result
            pdf_file = data['pdf_file']
            title = data['title']
            content = data['content']
            metadata = data['metadata']
            analysis = data.get('content_analysis')
            
            # Store in knowledge graph (basic info)
            authors = self._extract_authors_from_metadata(metadata)
            publication_date = self._extract_date_from_metadata(metadata)
            
            try:
                with knowledge_graph.get_db_connection() as conn:
                    # Check if paper already exists
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM papers WHERE source_pdf = ?", (pdf_file,))
                    existing = cursor.fetchone()
                    
                    if not existing:
                        paper_id = knowledge_graph.add_paper_with_authors(
                            title=title,
                            summary=analysis.overall_summary if analysis else "No summary available",
                            source_pdf=pdf_file,
                            author_names=authors,
                            publication_date=publication_date
                        )
                        
                        if paper_id and analysis:
                            # Store detailed content analysis
                            analysis.paper_id = str(paper_id)
                            stored_analysis_id = self.content_db.store_analysis(analysis)
                            
                            if stored_analysis_id:
                                self.stats['analyses_created'] += 1
                                self.stats['references_extracted'] += len(analysis.references)
                                self.stats['concepts_extracted'] += len(analysis.concepts)
                                
                                log_info("Paper with analysis stored", 
                                        title=title[:50], 
                                        analysis_id=stored_analysis_id)
            
            except Exception as e:
                log_error("Failed to store paper analysis", 
                         file=pdf_file, error=str(e))
        
        except Exception as e:
            log_error("Failed to process paper result", error=str(e))
    
    def _generate_corpus_analysis(self):
        """Generate corpus-level analysis and insights."""
        try:
            log_info("Generating corpus-level analysis")
            
            # Get all papers for corpus analysis
            papers_data = []
            
            with knowledge_graph.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT p.id, p.title, p.summary, ca.overall_summary
                    FROM papers p
                    LEFT JOIN content_analyses ca ON p.id = ca.paper_id
                    WHERE ca.id IS NOT NULL
                """)
                
                for row in cursor.fetchall():
                    paper_id, title, summary, overall_summary = row
                    papers_data.append({
                        'id': str(paper_id),
                        'title': title,
                        'summary': summary,
                        'content': overall_summary or summary or title
                    })
            
            if len(papers_data) > 1:
                # Perform cross-paper analysis
                corpus_analyses = self.content_engine.analyze_corpus(papers_data)
                
                # Update database with cross-references
                self._update_cross_references(corpus_analyses)
                
                log_info("Corpus analysis completed", 
                        papers_analyzed=len(corpus_analyses))
            
        except Exception as e:
            log_error("Corpus analysis failed", error=str(e))
    
    def _update_cross_references(self, analyses: Dict[str, ContentAnalysis]):
        """Update cross-references between papers."""
        try:
            # Update related terms in concepts based on corpus analysis
            for paper_id, analysis in analyses.items():
                for concept in analysis.concepts:
                    if concept.related_terms:
                        # Update concept in database with related terms
                        # This would require extending the content_analysis_db
                        pass
            
        except Exception as e:
            log_error("Failed to update cross-references", error=str(e))
    
    def _generate_paper_id(self, pdf_file: str, title: str) -> str:
        """Generate a unique paper ID."""
        # Use filename without extension as base ID
        base_id = os.path.splitext(os.path.basename(pdf_file))[0]
        
        # Clean and normalize
        import re
        clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', base_id)
        return clean_id[:50]  # Limit length
    
    def _extract_authors_from_metadata(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract author names from PDF metadata."""
        authors = []
        
        # Try different metadata fields
        author_field = metadata.get('author', '') or metadata.get('Author', '')
        
        if author_field:
            # Handle different author formats
            if ';' in author_field:
                authors = [a.strip() for a in author_field.split(';')]
            elif ',' in author_field and ' and ' not in author_field:
                authors = [a.strip() for a in author_field.split(',')]
            elif ' and ' in author_field:
                authors = [a.strip() for a in author_field.split(' and ')]
            else:
                authors = [author_field.strip()]
        
        return [a for a in authors if len(a) > 2]  # Filter out too short names
    
    def _extract_date_from_metadata(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract publication date from PDF metadata."""
        date_fields = ['creationdate', 'CreationDate', 'date', 'Date']
        
        for field in date_fields:
            date_value = metadata.get(field)
            if date_value:
                try:
                    # Try to parse and normalize date
                    import re
                    date_match = re.search(r'(\d{4})', str(date_value))
                    if date_match:
                        year = date_match.group(1)
                        return f"{year}-01-01"  # Default to January 1st
                except Exception:
                    continue
        
        return None
    
    def get_enhanced_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics including content analysis."""
        basic_stats = self.stats.copy()
        
        if self.enable_content_analysis:
            # Get content analysis statistics
            content_stats = self.content_db.get_analysis_statistics()
            basic_stats.update(content_stats)
        
        return basic_stats
    
    def search_by_concepts(self, concepts: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search papers by concepts."""
        if not self.enable_content_analysis:
            return []
        
        try:
            with knowledge_graph.get_db_connection() as conn:
                cursor = conn.cursor()
                
                # Build query for concept search
                concept_conditions = []
                params = []
                
                for concept in concepts:
                    concept_conditions.append("pc.term LIKE ?")
                    params.append(f"%{concept}%")
                
                query = f"""
                    SELECT DISTINCT p.id, p.title, p.summary, 
                           GROUP_CONCAT(pc.term) as matching_concepts,
                           AVG(pc.importance_score) as avg_importance
                    FROM papers p
                    JOIN content_analyses ca ON p.id = ca.paper_id
                    JOIN paper_concepts pc ON ca.id = pc.analysis_id
                    WHERE ({' OR '.join(concept_conditions)})
                    GROUP BY p.id, p.title, p.summary
                    ORDER BY avg_importance DESC
                    LIMIT ?
                """
                
                params.append(limit)
                cursor.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'paper_id': row[0],
                        'title': row[1],
                        'summary': row[2],
                        'matching_concepts': row[3].split(',') if row[3] else [],
                        'importance_score': row[4]
                    })
                
                return results
                
        except Exception as e:
            log_error("Concept search failed", error=str(e))
            return []
    
    def get_citation_network(self) -> Dict[str, List[str]]:
        """Get citation network from content analysis."""
        if not self.enable_content_analysis:
            return {}
        
        return self.content_db.get_citation_network()
    
    def get_concept_cooccurrence(self, min_frequency: int = 2) -> Dict[str, List[Tuple[str, int]]]:
        """Get concept co-occurrence patterns."""
        if not self.enable_content_analysis:
            return {}
        
        return self.content_db.get_concept_co_occurrence(min_frequency)


def create_enhanced_processor(enable_content_analysis: bool = True) -> EnhancedRAGProcessor:
    """Create enhanced RAG processor instance."""
    return EnhancedRAGProcessor(enable_content_analysis=enable_content_analysis)
