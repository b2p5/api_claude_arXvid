"""
Intelligent chunking system for research papers.
Provides structure-aware text splitting based on sections, paragraphs, and semantic boundaries.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from langchain.schema import Document
from langchain.text_splitter import TextSplitter
from config import get_config
from logger import get_logger, log_info, log_warning


class ChunkType(Enum):
    """Types of text chunks based on document structure."""
    TITLE = "title"
    ABSTRACT = "abstract" 
    INTRODUCTION = "introduction"
    METHODOLOGY = "methodology"
    RESULTS = "results"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    SECTION_HEADER = "section_header"
    PARAGRAPH = "paragraph"
    FIGURE_CAPTION = "figure_caption"
    TABLE_CAPTION = "table_caption"
    EQUATION = "equation"
    UNKNOWN = "unknown"


@dataclass
class TextChunk:
    """Structured text chunk with metadata."""
    content: str
    chunk_type: ChunkType
    section_title: str = ""
    start_position: int = 0
    end_position: int = 0
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_langchain_document(self) -> Document:
        """Convert to LangChain Document format."""
        full_metadata = {
            'chunk_type': self.chunk_type.value,
            'section_title': self.section_title,
            'start_position': self.start_position,
            'end_position': self.end_position,
            'confidence': self.confidence,
            **self.metadata
        }
        
        return Document(
            page_content=self.content,
            metadata=full_metadata
        )


class PaperStructureDetector:
    """Detect structure in research papers using pattern matching."""
    
    def __init__(self):
        self.logger = get_logger()
        
        # Common section patterns for academic papers
        self.section_patterns = {
            ChunkType.ABSTRACT: [
                r'^abstract\s*$',
                r'^\s*abstract\s*[:\-]',
                r'^\d+\.?\s*abstract\s*$'
            ],
            ChunkType.INTRODUCTION: [
                r'^introduction\s*$',
                r'^\s*introduction\s*[:\-]',
                r'^\d+\.?\s*introduction\s*$',
                r'^1\.?\s*introduction',
                r'^i\.?\s*introduction'
            ],
            ChunkType.METHODOLOGY: [
                r'^methodology\s*$',
                r'^methods?\s*$',
                r'^materials?\s+and\s+methods?',
                r'^experimental\s+setup',
                r'^\d+\.?\s*(methodology|methods?)\s*$'
            ],
            ChunkType.RESULTS: [
                r'^results?\s*$',
                r'^findings\s*$',
                r'^\d+\.?\s*results?\s*$',
                r'^experimental\s+results?'
            ],
            ChunkType.DISCUSSION: [
                r'^discussion\s*$',
                r'^\d+\.?\s*discussion\s*$',
                r'^results?\s+and\s+discussion',
                r'^discussion\s+of\s+results?'
            ],
            ChunkType.CONCLUSION: [
                r'^conclusions?\s*$',
                r'^concluding\s+remarks',
                r'^\d+\.?\s*conclusions?\s*$',
                r'^summary\s+and\s+conclusions?'
            ],
            ChunkType.REFERENCES: [
                r'^references?\s*$',
                r'^bibliography\s*$',
                r'^\d+\.?\s*references?\s*$'
            ]
        }
        
        # General section header patterns
        self.general_section_patterns = [
            r'^\d+\.?\d*\.?\s+[A-Z][A-Za-z\s]+$',  # "1. Section Title" or "1.1 Subsection"
            r'^[IVX]+\.?\s+[A-Z][A-Za-z\s]+$',     # Roman numerals
            r'^[A-Z][A-Z\s]{5,}$',                 # ALL CAPS headers
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*$' # Title Case
        ]
        
        # Figure and table caption patterns
        self.caption_patterns = {
            ChunkType.FIGURE_CAPTION: [
                r'^fig(?:ure)?\s*\d+[:\.\-]',
                r'^figure\s+\d+[:\.\-]'
            ],
            ChunkType.TABLE_CAPTION: [
                r'^table\s*\d+[:\.\-]',
                r'^tab\.\s*\d+[:\.\-]'
            ]
        }
    
    def detect_chunk_type(self, text: str, context: Dict[str, Any] = None) -> Tuple[ChunkType, float]:
        """
        Detect the type of a text chunk.
        
        Args:
            text: Text to classify
            context: Additional context (position, surrounding text, etc.)
            
        Returns:
            Tuple of (chunk_type, confidence)
        """
        text_lower = text.lower().strip()
        text_clean = re.sub(r'\s+', ' ', text_lower)
        
        # Check for specific section types
        for chunk_type, patterns in self.section_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text_clean, re.IGNORECASE | re.MULTILINE):
                    return chunk_type, 0.9
        
        # Check for captions
        for chunk_type, patterns in self.caption_patterns.items():
            for pattern in patterns:
                if re.match(pattern, text_clean, re.IGNORECASE):
                    return chunk_type, 0.8
        
        # Check for general section headers
        for pattern in self.general_section_patterns:
            if re.match(pattern, text.strip()):
                return ChunkType.SECTION_HEADER, 0.7
        
        # Check for equations (LaTeX-style or numbered)
        if re.search(r'\$.*\$|\\begin\{equation\}|^\s*\(\d+\)\s*$', text):
            return ChunkType.EQUATION, 0.8
        
        # Default to paragraph
        return ChunkType.PARAGRAPH, 0.5
    
    def detect_paper_structure(self, text: str) -> Dict[str, List[Tuple[int, int, str]]]:
        """
        Detect the overall structure of a research paper.
        
        Args:
            text: Full paper text
            
        Returns:
            Dictionary mapping section types to list of (start, end, title) tuples
        """
        structure = {chunk_type.value: [] for chunk_type in ChunkType}
        
        # Split into lines for analysis
        lines = text.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            line_start = current_pos
            line_end = current_pos + len(line) + 1  # +1 for newline
            current_pos = line_end
            
            if line.strip():  # Skip empty lines
                chunk_type, confidence = self.detect_chunk_type(line)
                
                if confidence > 0.6:  # Only include confident detections
                    structure[chunk_type.value].append((line_start, line_end, line.strip()))
        
        return structure


class IntelligentTextSplitter(TextSplitter):
    """Intelligent text splitter that respects document structure."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        respect_structure: bool = True,
        min_chunk_size: int = 100,
        preserve_sections: bool = True
    ):
        super().__init__()
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_structure = respect_structure
        self.min_chunk_size = min_chunk_size
        self.preserve_sections = preserve_sections
        
        self.structure_detector = PaperStructureDetector()
        self.logger = get_logger()
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks respecting document structure."""
        if not self.respect_structure:
            # Fall back to simple splitting
            return self._simple_split(text)
        
        chunks = self._structure_aware_split(text)
        return [chunk.content for chunk in chunks]
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into structured chunks."""
        all_chunks = []
        
        for doc in documents:
            text_chunks = self._structure_aware_split(doc.page_content)
            
            for chunk in text_chunks:
                # Merge original metadata with chunk metadata
                merged_metadata = {**doc.metadata, **chunk.metadata}
                merged_metadata.update({
                    'chunk_type': chunk.chunk_type.value,
                    'section_title': chunk.section_title,
                    'chunk_confidence': chunk.confidence
                })
                
                chunk_doc = Document(
                    page_content=chunk.content,
                    metadata=merged_metadata
                )
                all_chunks.append(chunk_doc)
        
        return all_chunks
    
    def _structure_aware_split(self, text: str) -> List[TextChunk]:
        """Split text using structure awareness."""
        self.logger.log_operation_start("Intelligent chunking", text_length=len(text))
        
        # Detect document structure
        structure = self.structure_detector.detect_paper_structure(text)
        
        # Find section boundaries
        section_boundaries = []
        for chunk_type_str, sections in structure.items():
            for start, end, title in sections:
                section_boundaries.append((start, end, title, ChunkType(chunk_type_str)))
        
        # Sort by position
        section_boundaries.sort(key=lambda x: x[0])
        
        # Split text into structure-aware chunks
        chunks = []
        current_section = "Unknown"
        current_section_type = ChunkType.UNKNOWN
        
        if not section_boundaries:
            # No structure detected, fall back to paragraph-based splitting
            chunks = self._paragraph_based_split(text)
        else:
            # Process text between section boundaries
            text_sections = self._extract_text_sections(text, section_boundaries)
            
            for section_text, section_info in text_sections:
                if section_info:
                    current_section = section_info[2]  # title
                    current_section_type = section_info[3]  # ChunkType
                
                # Split section into appropriately sized chunks
                section_chunks = self._split_section(
                    section_text, 
                    current_section, 
                    current_section_type
                )
                chunks.extend(section_chunks)
        
        # Post-process chunks
        chunks = self._post_process_chunks(chunks)
        
        self.logger.log_operation_success("Intelligent chunking", 
                                        chunks_created=len(chunks),
                                        avg_chunk_size=sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0)
        
        return chunks
    
    def _extract_text_sections(self, text: str, boundaries: List[Tuple[int, int, str, ChunkType]]) -> List[Tuple[str, Optional[Tuple]]]:
        """Extract text sections based on detected boundaries."""
        sections = []
        last_end = 0
        
        for boundary in boundaries:
            start, end, title, chunk_type = boundary
            
            # Add text before this boundary (if any)
            if start > last_end:
                section_text = text[last_end:start].strip()
                if section_text:
                    sections.append((section_text, None))
            
            # Add the boundary section itself
            boundary_text = text[start:end].strip()
            if boundary_text:
                sections.append((boundary_text, boundary))
            
            last_end = end
        
        # Add remaining text after last boundary
        if last_end < len(text):
            remaining_text = text[last_end:].strip()
            if remaining_text:
                sections.append((remaining_text, None))
        
        return sections
    
    def _split_section(self, text: str, section_title: str, section_type: ChunkType) -> List[TextChunk]:
        """Split a section into appropriately sized chunks."""
        if len(text) <= self.chunk_size:
            # Section fits in one chunk
            return [TextChunk(
                content=text,
                chunk_type=section_type,
                section_title=section_title,
                confidence=0.8
            )]
        
        # Section needs to be split
        chunks = []
        
        # Try paragraph-based splitting first
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk = ""
        current_chunks = []
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) + 1 > self.chunk_size and current_chunk:
                # Save current chunk and start a new one
                if len(current_chunk.strip()) >= self.min_chunk_size:
                    chunks.append(TextChunk(
                        content=current_chunk.strip(),
                        chunk_type=section_type,
                        section_title=section_title,
                        confidence=0.7
                    ))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + "\n" + paragraph if overlap_text else paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip() and len(current_chunk.strip()) >= self.min_chunk_size:
            chunks.append(TextChunk(
                content=current_chunk.strip(),
                chunk_type=section_type,
                section_title=section_title,
                confidence=0.7
            ))
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newlines (typical paragraph separator)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean up paragraphs
        clean_paragraphs = []
        for para in paragraphs:
            cleaned = re.sub(r'\s+', ' ', para.strip())
            if cleaned and len(cleaned) > 20:  # Filter very short paragraphs
                clean_paragraphs.append(cleaned)
        
        return clean_paragraphs
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get the last part of text for overlap."""
        if len(text) <= overlap_size:
            return text
        
        # Try to break at sentence boundary
        sentences = re.split(r'[.!?]+', text)
        
        overlap_text = ""
        for sentence in reversed(sentences):
            if len(overlap_text) + len(sentence) <= overlap_size:
                if overlap_text:
                    overlap_text = sentence + ". " + overlap_text
                else:
                    overlap_text = sentence
            else:
                break
        
        # If no good sentence break, just take the last N characters
        if not overlap_text:
            overlap_text = text[-overlap_size:]
        
        return overlap_text.strip()
    
    def _paragraph_based_split(self, text: str) -> List[TextChunk]:
        """Fall back to paragraph-based splitting when no structure is detected."""
        paragraphs = self._split_into_paragraphs(text)
        chunks = []
        
        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 1 > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_type, confidence = self.structure_detector.detect_chunk_type(current_chunk)
                chunks.append(TextChunk(
                    content=current_chunk.strip(),
                    chunk_type=chunk_type,
                    confidence=confidence
                ))
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk, self.chunk_overlap)
                current_chunk = overlap_text + "\n" + paragraph if overlap_text else paragraph
            else:
                if current_chunk:
                    current_chunk += "\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunk_type, confidence = self.structure_detector.detect_chunk_type(current_chunk)
            chunks.append(TextChunk(
                content=current_chunk.strip(),
                chunk_type=chunk_type,
                confidence=confidence
            ))
        
        return chunks
    
    def _post_process_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Post-process chunks to improve quality."""
        processed_chunks = []
        
        for chunk in chunks:
            # Skip chunks that are too small
            if len(chunk.content.strip()) < self.min_chunk_size:
                continue
            
            # Clean up chunk content
            cleaned_content = re.sub(r'\s+', ' ', chunk.content.strip())
            chunk.content = cleaned_content
            
            # Set positions (simplified for now)
            chunk.start_position = 0  # Would need full text to calculate
            chunk.end_position = len(cleaned_content)
            
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _simple_split(self, text: str) -> List[str]:
        """Simple text splitting as fallback."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        simple_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        return simple_splitter.split_text(text)


class SemanticChunkMerger:
    """Merge related chunks based on semantic similarity."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.logger = get_logger()
    
    def merge_similar_chunks(self, chunks: List[TextChunk], max_merged_size: int = 1500) -> List[TextChunk]:
        """Merge semantically similar chunks."""
        if len(chunks) <= 1:
            return chunks
        
        self.logger.log_operation_start("Semantic chunk merging", chunk_count=len(chunks))
        
        # Simple heuristic merging based on section titles and chunk types
        merged_chunks = []
        current_group = [chunks[0]]
        
        for i in range(1, len(chunks)):
            current_chunk = chunks[i]
            last_chunk = current_group[-1]
            
            # Check if chunks should be merged
            should_merge = (
                current_chunk.section_title == last_chunk.section_title and
                current_chunk.chunk_type == last_chunk.chunk_type and
                sum(len(c.content) for c in current_group) + len(current_chunk.content) <= max_merged_size
            )
            
            if should_merge:
                current_group.append(current_chunk)
            else:
                # Finalize current group
                if len(current_group) == 1:
                    merged_chunks.append(current_group[0])
                else:
                    merged_chunk = self._merge_chunk_group(current_group)
                    merged_chunks.append(merged_chunk)
                
                # Start new group
                current_group = [current_chunk]
        
        # Handle final group
        if len(current_group) == 1:
            merged_chunks.append(current_group[0])
        else:
            merged_chunk = self._merge_chunk_group(current_group)
            merged_chunks.append(merged_chunk)
        
        self.logger.log_operation_success("Semantic chunk merging", 
                                        original_count=len(chunks),
                                        merged_count=len(merged_chunks))
        
        return merged_chunks
    
    def _merge_chunk_group(self, chunks: List[TextChunk]) -> TextChunk:
        """Merge a group of chunks into a single chunk."""
        merged_content = "\n\n".join(chunk.content for chunk in chunks)
        
        # Use properties from the first chunk
        first_chunk = chunks[0]
        
        return TextChunk(
            content=merged_content,
            chunk_type=first_chunk.chunk_type,
            section_title=first_chunk.section_title,
            confidence=min(chunk.confidence for chunk in chunks),  # Use lowest confidence
            metadata={
                'merged_from': len(chunks),
                'original_chunks': [chunk.metadata for chunk in chunks]
            }
        )


# Factory function for creating optimized text splitter
def create_intelligent_splitter(chunk_size: int = None, chunk_overlap: int = None) -> IntelligentTextSplitter:
    """Create an intelligent text splitter with optimal settings."""
    config = get_config()
    
    chunk_size = chunk_size or config.models.chunk_size
    chunk_overlap = chunk_overlap or config.models.chunk_overlap
    
    return IntelligentTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        respect_structure=True,
        min_chunk_size=max(100, chunk_size // 10),  # At least 10% of chunk size
        preserve_sections=True
    )


# Utility functions
def analyze_chunking_quality(chunks: List[TextChunk]) -> Dict[str, Any]:
    """Analyze the quality of chunking results."""
    if not chunks:
        return {'error': 'No chunks to analyze'}
    
    chunk_lengths = [len(chunk.content) for chunk in chunks]
    chunk_types = [chunk.chunk_type.value for chunk in chunks]
    
    # Count chunk types
    type_counts = {}
    for chunk_type in chunk_types:
        type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1
    
    analysis = {
        'total_chunks': len(chunks),
        'avg_chunk_length': sum(chunk_lengths) / len(chunks),
        'min_chunk_length': min(chunk_lengths),
        'max_chunk_length': max(chunk_lengths),
        'chunk_type_distribution': type_counts,
        'avg_confidence': sum(chunk.confidence for chunk in chunks) / len(chunks),
        'sections_identified': len(set(chunk.section_title for chunk in chunks if chunk.section_title))
    }
    
    return analysis