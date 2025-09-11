"""
Database integration for Content Analysis System.
Extends the knowledge graph to store analysis results.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from content_analysis import ContentAnalysis, Reference, Concept, Topic, Section, SectionType
from logger import get_logger, log_info, log_warning, log_error
import knowledge_graph


class ContentAnalysisDatabase:
    """Database manager for content analysis results."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or knowledge_graph.DB_FILE
        self.logger = get_logger()
        self._create_content_analysis_tables()
    
    def _create_content_analysis_tables(self):
        """Create tables for storing content analysis results."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Content analysis metadata table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS content_analyses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        paper_id INTEGER NOT NULL,
                        analysis_date TEXT NOT NULL,
                        overall_summary TEXT,
                        technical_level TEXT,
                        analysis_version TEXT DEFAULT '1.0',
                        FOREIGN KEY (paper_id) REFERENCES papers (id),
                        UNIQUE(paper_id, analysis_version)
                    )
                """)
                
                # References table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS paper_references (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analysis_id INTEGER NOT NULL,
                        raw_text TEXT NOT NULL,
                        paper_title TEXT,
                        authors_json TEXT,
                        publication_year INTEGER,
                        venue TEXT,
                        doi TEXT,
                        arxiv_id TEXT,
                        confidence REAL,
                        context TEXT,
                        FOREIGN KEY (analysis_id) REFERENCES content_analyses (id)
                    )
                """)
                
                # Concepts table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS paper_concepts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analysis_id INTEGER NOT NULL,
                        term TEXT NOT NULL,
                        frequency INTEGER,
                        importance_score REAL,
                        context_examples_json TEXT,
                        related_terms_json TEXT,
                        definition TEXT,
                        FOREIGN KEY (analysis_id) REFERENCES content_analyses (id)
                    )
                """)
                
                # Topics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS paper_topics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analysis_id INTEGER NOT NULL,
                        topic_name TEXT NOT NULL,
                        keywords_json TEXT,
                        weight REAL,
                        description TEXT,
                        topic_type TEXT DEFAULT 'auto',
                        FOREIGN KEY (analysis_id) REFERENCES content_analyses (id)
                    )
                """)
                
                # Sections table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS paper_sections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analysis_id INTEGER NOT NULL,
                        section_type TEXT NOT NULL,
                        title TEXT,
                        content TEXT,
                        summary TEXT,
                        key_points_json TEXT,
                        section_order INTEGER DEFAULT 0,
                        FOREIGN KEY (analysis_id) REFERENCES content_analyses (id)
                    )
                """)
                
                # Main contributions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS paper_contributions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analysis_id INTEGER NOT NULL,
                        contribution_text TEXT NOT NULL,
                        contribution_order INTEGER DEFAULT 0,
                        FOREIGN KEY (analysis_id) REFERENCES content_analyses (id)
                    )
                """)
                
                # Citation relationships table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS citation_relationships (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        citing_paper_id INTEGER NOT NULL,
                        cited_paper_id INTEGER,
                        cited_external_title TEXT,
                        cited_arxiv_id TEXT,
                        context TEXT,
                        relationship_type TEXT DEFAULT 'cites',
                        FOREIGN KEY (citing_paper_id) REFERENCES papers (id),
                        FOREIGN KEY (cited_paper_id) REFERENCES papers (id)
                    )
                """)
                
                # Create indexes for better performance
                self._create_indexes(cursor)
                
                conn.commit()
                
            log_info("Content analysis database tables created successfully")
            
        except Exception as e:
            log_error("Failed to create content analysis tables", error=str(e))
            raise
    
    def _create_indexes(self, cursor):
        """Create indexes for better query performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_content_analyses_paper_id ON content_analyses(paper_id)",
            "CREATE INDEX IF NOT EXISTS idx_paper_references_analysis_id ON paper_references(analysis_id)",
            "CREATE INDEX IF NOT EXISTS idx_paper_references_arxiv_id ON paper_references(arxiv_id)",
            "CREATE INDEX IF NOT EXISTS idx_paper_concepts_analysis_id ON paper_concepts(analysis_id)",
            "CREATE INDEX IF NOT EXISTS idx_paper_concepts_term ON paper_concepts(term)",
            "CREATE INDEX IF NOT EXISTS idx_paper_topics_analysis_id ON paper_topics(analysis_id)",
            "CREATE INDEX IF NOT EXISTS idx_paper_sections_analysis_id ON paper_sections(analysis_id)",
            "CREATE INDEX IF NOT EXISTS idx_paper_sections_type ON paper_sections(section_type)",
            "CREATE INDEX IF NOT EXISTS idx_citation_relationships_citing ON citation_relationships(citing_paper_id)",
            "CREATE INDEX IF NOT EXISTS idx_citation_relationships_cited ON citation_relationships(cited_paper_id)"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
            except sqlite3.Error as e:
                log_warning("Failed to create index", sql=index_sql, error=str(e))
    
    def store_analysis(self, analysis: ContentAnalysis) -> int:
        """Store complete content analysis results in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get paper_id from knowledge graph
                cursor.execute("SELECT id FROM papers WHERE id = ? OR title = ?", 
                              (analysis.paper_id, analysis.title))
                paper_row = cursor.fetchone()
                
                if not paper_row:
                    log_warning("Paper not found in knowledge graph", 
                              paper_id=analysis.paper_id, title=analysis.title)
                    return None
                
                paper_id = paper_row[0]
                
                # Insert main analysis record
                cursor.execute("""
                    INSERT OR REPLACE INTO content_analyses 
                    (paper_id, analysis_date, overall_summary, technical_level)
                    VALUES (?, ?, ?, ?)
                """, (
                    paper_id,
                    datetime.now().isoformat(),
                    analysis.overall_summary,
                    analysis.technical_level
                ))
                
                analysis_id = cursor.lastrowid
                
                # Store references
                self._store_references(cursor, analysis_id, analysis.references)
                
                # Store concepts
                self._store_concepts(cursor, analysis_id, analysis.concepts)
                
                # Store topics
                self._store_topics(cursor, analysis_id, analysis.topics)
                
                # Store sections
                self._store_sections(cursor, analysis_id, analysis.sections)
                
                # Store contributions
                self._store_contributions(cursor, analysis_id, analysis.main_contributions)
                
                # Store citation relationships
                self._store_citation_relationships(cursor, paper_id, analysis.references)
                
                conn.commit()
                
                log_info("Content analysis stored successfully", 
                        analysis_id=analysis_id, paper_id=paper_id)
                
                return analysis_id
                
        except Exception as e:
            log_error("Failed to store content analysis", error=str(e))
            raise
    
    def _store_references(self, cursor, analysis_id: int, references: List[Reference]):
        """Store references in database."""
        for ref in references:
            cursor.execute("""
                INSERT INTO paper_references 
                (analysis_id, raw_text, paper_title, authors_json, publication_year, 
                 venue, doi, arxiv_id, confidence, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_id,
                ref.raw_text,
                ref.paper_title,
                json.dumps(ref.authors) if ref.authors else None,
                ref.year,
                ref.venue,
                ref.doi,
                ref.arxiv_id,
                ref.confidence,
                ref.context
            ))
    
    def _store_concepts(self, cursor, analysis_id: int, concepts: List[Concept]):
        """Store concepts in database."""
        for concept in concepts:
            cursor.execute("""
                INSERT INTO paper_concepts 
                (analysis_id, term, frequency, importance_score, 
                 context_examples_json, related_terms_json, definition)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_id,
                concept.term,
                concept.frequency,
                concept.importance_score,
                json.dumps(concept.context_examples) if concept.context_examples else None,
                json.dumps(concept.related_terms) if concept.related_terms else None,
                concept.definition
            ))
    
    def _store_topics(self, cursor, analysis_id: int, topics: List[Topic]):
        """Store topics in database."""
        for topic in topics:
            cursor.execute("""
                INSERT INTO paper_topics 
                (analysis_id, topic_name, keywords_json, weight, description)
                VALUES (?, ?, ?, ?, ?)
            """, (
                analysis_id,
                topic.name,
                json.dumps(topic.keywords) if topic.keywords else None,
                topic.weight,
                topic.description
            ))
    
    def _store_sections(self, cursor, analysis_id: int, sections: List[Section]):
        """Store sections in database."""
        for i, section in enumerate(sections):
            cursor.execute("""
                INSERT INTO paper_sections 
                (analysis_id, section_type, title, content, summary, 
                 key_points_json, section_order)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                analysis_id,
                section.section_type.value,
                section.title,
                section.content,
                section.summary,
                json.dumps(section.key_points) if section.key_points else None,
                i
            ))
    
    def _store_contributions(self, cursor, analysis_id: int, contributions: List[str]):
        """Store main contributions in database."""
        for i, contribution in enumerate(contributions):
            cursor.execute("""
                INSERT INTO paper_contributions 
                (analysis_id, contribution_text, contribution_order)
                VALUES (?, ?, ?)
            """, (analysis_id, contribution, i))
    
    def _store_citation_relationships(self, cursor, citing_paper_id: int, references: List[Reference]):
        """Store citation relationships in database."""
        for ref in references:
            # Try to find cited paper in our database
            cited_paper_id = None
            
            if ref.arxiv_id:
                # Look for paper with matching arXiv ID (would need to extend papers table)
                pass
            
            if ref.paper_title:
                # Look for paper with similar title
                cursor.execute("""
                    SELECT id FROM papers WHERE title LIKE ?
                """, (f"%{ref.paper_title}%",))
                match = cursor.fetchone()
                if match:
                    cited_paper_id = match[0]
            
            # Insert citation relationship
            cursor.execute("""
                INSERT INTO citation_relationships 
                (citing_paper_id, cited_paper_id, cited_external_title, 
                 cited_arxiv_id, context)
                VALUES (?, ?, ?, ?, ?)
            """, (
                citing_paper_id,
                cited_paper_id,
                ref.paper_title,
                ref.arxiv_id,
                ref.context
            ))
    
    def get_analysis(self, paper_id: str) -> Optional[ContentAnalysis]:
        """Retrieve content analysis for a paper."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get main analysis record
                cursor.execute("""
                    SELECT ca.id, ca.overall_summary, ca.technical_level, p.title
                    FROM content_analyses ca
                    JOIN papers p ON ca.paper_id = p.id
                    WHERE ca.paper_id = ? OR p.title = ?
                    ORDER BY ca.analysis_date DESC
                    LIMIT 1
                """, (paper_id, paper_id))
                
                analysis_row = cursor.fetchone()
                if not analysis_row:
                    return None
                
                analysis_id, overall_summary, technical_level, title = analysis_row
                
                # Create analysis object
                analysis = ContentAnalysis(
                    paper_id=paper_id,
                    title=title,
                    overall_summary=overall_summary,
                    technical_level=technical_level
                )
                
                # Load references
                analysis.references = self._load_references(cursor, analysis_id)
                
                # Load concepts
                analysis.concepts = self._load_concepts(cursor, analysis_id)
                
                # Load topics
                analysis.topics = self._load_topics(cursor, analysis_id)
                
                # Load sections
                analysis.sections = self._load_sections(cursor, analysis_id)
                
                # Load contributions
                analysis.main_contributions = self._load_contributions(cursor, analysis_id)
                
                return analysis
                
        except Exception as e:
            log_error("Failed to retrieve content analysis", error=str(e))
            return None
    
    def _load_references(self, cursor, analysis_id: int) -> List[Reference]:
        """Load references from database."""
        cursor.execute("""
            SELECT raw_text, paper_title, authors_json, publication_year,
                   venue, doi, arxiv_id, confidence, context
            FROM paper_references
            WHERE analysis_id = ?
        """, (analysis_id,))
        
        references = []
        for row in cursor.fetchall():
            ref = Reference(
                raw_text=row[0],
                paper_title=row[1],
                authors=json.loads(row[2]) if row[2] else [],
                year=row[3],
                venue=row[4],
                doi=row[5],
                arxiv_id=row[6],
                confidence=row[7],
                context=row[8]
            )
            references.append(ref)
        
        return references
    
    def _load_concepts(self, cursor, analysis_id: int) -> List[Concept]:
        """Load concepts from database."""
        cursor.execute("""
            SELECT term, frequency, importance_score, context_examples_json,
                   related_terms_json, definition
            FROM paper_concepts
            WHERE analysis_id = ?
            ORDER BY importance_score DESC
        """, (analysis_id,))
        
        concepts = []
        for row in cursor.fetchall():
            concept = Concept(
                term=row[0],
                frequency=row[1],
                importance_score=row[2],
                context_examples=json.loads(row[3]) if row[3] else [],
                related_terms=json.loads(row[4]) if row[4] else [],
                definition=row[5]
            )
            concepts.append(concept)
        
        return concepts
    
    def _load_topics(self, cursor, analysis_id: int) -> List[Topic]:
        """Load topics from database."""
        cursor.execute("""
            SELECT topic_name, keywords_json, weight, description
            FROM paper_topics
            WHERE analysis_id = ?
            ORDER BY weight DESC
        """, (analysis_id,))
        
        topics = []
        for row in cursor.fetchall():
            topic = Topic(
                name=row[0],
                keywords=json.loads(row[1]) if row[1] else [],
                weight=row[2],
                description=row[3]
            )
            topics.append(topic)
        
        return topics
    
    def _load_sections(self, cursor, analysis_id: int) -> List[Section]:
        """Load sections from database."""
        cursor.execute("""
            SELECT section_type, title, content, summary, key_points_json
            FROM paper_sections
            WHERE analysis_id = ?
            ORDER BY section_order
        """, (analysis_id,))
        
        sections = []
        for row in cursor.fetchall():
            section = Section(
                section_type=SectionType(row[0]),
                title=row[1],
                content=row[2],
                summary=row[3],
                key_points=json.loads(row[4]) if row[4] else []
            )
            sections.append(section)
        
        return sections
    
    def _load_contributions(self, cursor, analysis_id: int) -> List[str]:
        """Load contributions from database."""
        cursor.execute("""
            SELECT contribution_text
            FROM paper_contributions
            WHERE analysis_id = ?
            ORDER BY contribution_order
        """, (analysis_id,))
        
        return [row[0] for row in cursor.fetchall()]
    
    def get_citation_network(self) -> Dict[str, List[str]]:
        """Get citation network from stored relationships."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT p1.title as citing, p2.title as cited
                    FROM citation_relationships cr
                    JOIN papers p1 ON cr.citing_paper_id = p1.id
                    LEFT JOIN papers p2 ON cr.cited_paper_id = p2.id
                    WHERE p2.title IS NOT NULL
                """)
                
                network = {}
                for citing, cited in cursor.fetchall():
                    if citing not in network:
                        network[citing] = []
                    network[citing].append(cited)
                
                return network
                
        except Exception as e:
            log_error("Failed to get citation network", error=str(e))
            return {}
    
    def get_concept_co_occurrence(self, min_frequency: int = 2) -> Dict[str, List[str]]:
        """Get concept co-occurrence patterns."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get concepts that appear together in papers
                cursor.execute("""
                    SELECT c1.term, c2.term, COUNT(*) as frequency
                    FROM paper_concepts c1
                    JOIN paper_concepts c2 ON c1.analysis_id = c2.analysis_id
                    WHERE c1.term != c2.term
                    GROUP BY c1.term, c2.term
                    HAVING frequency >= ?
                    ORDER BY frequency DESC
                """, (min_frequency,))
                
                co_occurrence = {}
                for term1, term2, freq in cursor.fetchall():
                    if term1 not in co_occurrence:
                        co_occurrence[term1] = []
                    co_occurrence[term1].append((term2, freq))
                
                return co_occurrence
                
        except Exception as e:
            log_error("Failed to get concept co-occurrence", error=str(e))
            return {}
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get statistics about content analyses."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Basic counts
                cursor.execute("SELECT COUNT(*) FROM content_analyses")
                stats['total_analyses'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM paper_references")
                stats['total_references'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM paper_concepts")
                stats['total_concepts'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM paper_topics")
                stats['total_topics'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM citation_relationships")
                stats['total_citations'] = cursor.fetchone()[0]
                
                # Top concepts
                cursor.execute("""
                    SELECT term, SUM(frequency) as total_freq
                    FROM paper_concepts
                    GROUP BY term
                    ORDER BY total_freq DESC
                    LIMIT 10
                """)
                stats['top_concepts'] = cursor.fetchall()
                
                # Technical level distribution
                cursor.execute("""
                    SELECT technical_level, COUNT(*) 
                    FROM content_analyses
                    GROUP BY technical_level
                """)
                stats['technical_levels'] = dict(cursor.fetchall())
                
                return stats
                
        except Exception as e:
            log_error("Failed to get analysis statistics", error=str(e))
            return {}