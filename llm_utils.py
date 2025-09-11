"""
LLM utilities for robust entity extraction with graceful error handling.
Provides fallback strategies, validation, and recovery mechanisms for LLM operations.
"""

import json
import re
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from langchain_deepseek import ChatDeepSeek
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

from config import get_config
from logger import get_logger
from retry_utils import api_call_with_retry, safe_execute


class ExtractionStrategy(Enum):
    """Different strategies for entity extraction."""
    FULL_TEXT = "full_text"           # Use full text for extraction
    CHUNKED = "chunked"               # Process text in chunks
    FIRST_PAGES = "first_pages"       # Only process first few pages
    FALLBACK_SIMPLE = "fallback_simple"  # Simple regex-based extraction


@dataclass
class ExtractionResult:
    """Result of LLM entity extraction."""
    success: bool = False
    data: Optional[Dict[str, Any]] = None
    strategy_used: Optional[ExtractionStrategy] = None
    confidence: float = 0.0
    errors: List[str] = None
    warnings: List[str] = None
    raw_response: str = ""
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class LLMEntityExtractor:
    """Robust LLM entity extractor with fallback strategies."""
    
    def __init__(self, model_name: str = None, max_retries: int = 3):
        self.config = get_config()
        self.model_name = model_name or self.config.models.llm_model
        self.max_retries = max_retries
        self.logger = get_logger()
        
        # Initialize LLM
        self.llm = ChatDeepSeek(model=self.model_name)
        
        # Extraction prompt template
        self.extraction_template = """
From the following research paper text, extract the title, authors, publication date, and a brief summary.

IMPORTANT INSTRUCTIONS:
1. Provide the output ONLY as a clean JSON object
2. Use these exact keys: "title", "authors", "publication_date", "summary"  
3. "authors" should be a list of strings (individual author names)
4. "publication_date" should be in YYYY-MM-DD format if found, or null if not found
5. "summary" should be 2-3 sentences maximum
6. If any information is not found, use null for that field
7. Do not include any text before or after the JSON

PAPER_TEXT:
---
{text}
---

JSON_OUTPUT:
"""
        
        self.prompt = PromptTemplate.from_template(self.extraction_template)
        self.extraction_chain = self.prompt | self.llm | StrOutputParser()
    
    def _clean_json_response(self, response: str) -> Tuple[str, List[str]]:
        """Clean and extract JSON from LLM response."""
        warnings = []
        
        # Remove any markdown code blocks
        json_match = re.search(r'```json\s*\n?(.*?)\n?```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
            warnings.append("Response contained markdown code blocks")
        else:
            # Try to find JSON object in the response
            json_match = re.search(r'({.*?})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1).strip()
                warnings.append("Extracted JSON from mixed response")
            else:
                json_str = response.strip()
        
        return json_str, warnings
    
    def _validate_extracted_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str], float]:
        """Validate extracted data and calculate confidence score."""
        errors = []
        confidence = 0.0
        
        required_fields = ["title", "authors", "summary", "publication_date"]
        
        # Check required fields exist
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if errors:
            return False, errors, 0.0
        
        # Validate field types and content
        if not isinstance(data.get("authors"), list):
            errors.append("'authors' must be a list")
        elif len(data["authors"]) == 0:
            errors.append("'authors' list cannot be empty")
        else:
            confidence += 0.25  # Authors present and valid
        
        if data.get("title") and isinstance(data["title"], str) and len(data["title"].strip()) > 5:
            confidence += 0.35  # Title present and reasonable
        else:
            errors.append("Title missing or too short")
        
        if data.get("summary") and isinstance(data["summary"], str) and len(data["summary"].strip()) > 20:
            confidence += 0.25  # Summary present and reasonable
        else:
            errors.append("Summary missing or too short")
        
        # Publication date is optional but adds to confidence
        if data.get("publication_date") and data["publication_date"] is not None:
            # Basic date format validation
            if re.match(r'\d{4}-\d{2}-\d{2}', str(data["publication_date"])):
                confidence += 0.15
            else:
                confidence += 0.05  # Date present but wrong format
        
        is_valid = len(errors) == 0
        return is_valid, errors, confidence
    
    def _extract_with_full_text(self, text: str) -> ExtractionResult:
        """Extract entities using full text."""
        start_time = time.time()
        result = ExtractionResult(strategy_used=ExtractionStrategy.FULL_TEXT)
        
        try:
            # Limit text size
            text_to_use = text[:self.config.models.extraction_text_limit]
            if len(text) > self.config.models.extraction_text_limit:
                result.warnings.append(f"Text truncated to {self.config.models.extraction_text_limit} characters")
            
            # Make API call with retry
            response = api_call_with_retry(self.extraction_chain.invoke, {"text": text_to_use})
            result.raw_response = response
            
            # Clean and parse JSON
            json_str, warnings = self._clean_json_response(response)
            result.warnings.extend(warnings)
            
            # Parse JSON
            try:
                data = json.loads(json_str)
                result.data = data
                
                # Validate data
                is_valid, validation_errors, confidence = self._validate_extracted_data(data)
                result.success = is_valid
                result.confidence = confidence
                result.errors.extend(validation_errors)
                
            except json.JSONDecodeError as e:
                result.errors.append(f"JSON parsing failed: {e}")
                result.success = False
        
        except Exception as e:
            result.errors.append(f"LLM extraction failed: {e}")
            result.success = False
        
        result.processing_time = time.time() - start_time
        return result
    
    def _extract_with_chunked_strategy(self, text: str) -> ExtractionResult:
        """Extract entities by processing text in chunks and combining results."""
        start_time = time.time()
        result = ExtractionResult(strategy_used=ExtractionStrategy.CHUNKED)
        
        try:
            # Split text into overlapping chunks
            chunk_size = self.config.models.extraction_text_limit // 2
            overlap = 200
            
            chunks = []
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                chunks.append(chunk)
                if len(chunk) < chunk_size:
                    break
            
            result.warnings.append(f"Processing {len(chunks)} text chunks")
            
            # Process each chunk
            best_result = None
            best_confidence = 0.0
            
            for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks
                chunk_result = self._extract_with_full_text(chunk)
                if chunk_result.success and chunk_result.confidence > best_confidence:
                    best_result = chunk_result
                    best_confidence = chunk_result.confidence
            
            if best_result and best_result.success:
                result.data = best_result.data
                result.success = True
                result.confidence = best_confidence
                result.raw_response = best_result.raw_response
                result.warnings.extend(best_result.warnings)
            else:
                result.errors.append("No chunk produced valid extraction")
                result.success = False
        
        except Exception as e:
            result.errors.append(f"Chunked extraction failed: {e}")
            result.success = False
        
        result.processing_time = time.time() - start_time
        return result
    
    def _extract_with_first_pages_strategy(self, text: str) -> ExtractionResult:
        """Extract entities using only the first portion of the text."""
        start_time = time.time()
        result = ExtractionResult(strategy_used=ExtractionStrategy.FIRST_PAGES)
        
        # Use first 2000 characters (typically covers title, authors, abstract)
        first_portion = text[:2000]
        result.warnings.append("Using only first 2000 characters for extraction")
        
        extraction_result = self._extract_with_full_text(first_portion)
        
        # Copy results
        result.success = extraction_result.success
        result.data = extraction_result.data
        result.confidence = extraction_result.confidence * 0.8  # Slight penalty for limited text
        result.errors = extraction_result.errors
        result.warnings.extend(extraction_result.warnings)
        result.raw_response = extraction_result.raw_response
        result.processing_time = time.time() - start_time
        
        return result
    
    def _extract_with_fallback_strategy(self, text: str) -> ExtractionResult:
        """Simple regex-based fallback extraction."""
        start_time = time.time()
        result = ExtractionResult(strategy_used=ExtractionStrategy.FALLBACK_SIMPLE)
        
        data = {
            "title": None,
            "authors": [],
            "publication_date": None,
            "summary": None
        }
        
        # Simple regex patterns for common paper formats
        text_start = text[:3000]  # Check first 3000 characters
        
        # Try to find title (often the first significant line)
        title_patterns = [
            r'^(.{10,200})\n',  # First line if reasonable length
            r'Title[:\s]*(.+?)(?:\n|Author)',
            r'^([A-Z][^.\n]{10,150}?)(?:\n|\s{2,})',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text_start, re.MULTILINE | re.IGNORECASE)
            if match:
                potential_title = match.group(1).strip()
                if len(potential_title) > 10 and not potential_title.isupper():
                    data["title"] = potential_title
                    break
        
        # Try to find authors
        author_patterns = [
            r'Authors?[:\s]*(.+?)(?:\n\n|\nabstract|\ndate)',
            r'By[:\s]+(.+?)(?:\n|\s{2,})',
            r'(?:^|\n)([A-Z][a-z]+ [A-Z][a-z]+(?:,\s*[A-Z][a-z]+ [A-Z][a-z]+)*)',
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, text_start, re.MULTILINE | re.IGNORECASE)
            if match:
                authors_text = match.group(1).strip()
                # Split by comma or 'and'
                authors = re.split(r',|\sand\s', authors_text)
                authors = [author.strip() for author in authors if author.strip()]
                if authors:
                    data["authors"] = authors[:10]  # Limit to 10 authors
                    break
        
        # Try to find abstract as summary
        abstract_patterns = [
            r'Abstract[:\s]*(.+?)(?:\n\n|\nKeywords|\nIntroduction)',
            r'Summary[:\s]*(.+?)(?:\n\n|\nKeywords)',
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Take first few sentences
                sentences = re.split(r'[.!?]+', abstract)
                summary_sentences = []
                char_count = 0
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and char_count + len(sentence) < 300:
                        summary_sentences.append(sentence)
                        char_count += len(sentence)
                    else:
                        break
                
                if summary_sentences:
                    data["summary"] = '. '.join(summary_sentences) + '.'
                    break
        
        # Calculate simple confidence based on what we found
        confidence = 0.0
        if data["title"]:
            confidence += 0.4
        if data["authors"]:
            confidence += 0.3
        if data["summary"]:
            confidence += 0.3
        
        result.data = data
        result.confidence = confidence
        result.success = confidence > 0.5
        result.warnings.append("Used simple regex-based extraction")
        result.processing_time = time.time() - start_time
        
        return result
    
    def extract_entities(self, text: str, strategies: List[ExtractionStrategy] = None) -> ExtractionResult:
        """
        Extract entities using multiple strategies with fallbacks.
        
        Args:
            text: Text to extract from
            strategies: List of strategies to try (in order)
            
        Returns:
            ExtractionResult with best extraction found
        """
        if strategies is None:
            strategies = [
                ExtractionStrategy.FULL_TEXT,
                ExtractionStrategy.CHUNKED,
                ExtractionStrategy.FIRST_PAGES,
                ExtractionStrategy.FALLBACK_SIMPLE
            ]
        
        self.logger.log_operation_start("LLM entity extraction", text_length=len(text))
        
        best_result = None
        best_confidence = 0.0
        
        for strategy in strategies:
            try:
                if strategy == ExtractionStrategy.FULL_TEXT:
                    result = self._extract_with_full_text(text)
                elif strategy == ExtractionStrategy.CHUNKED:
                    result = self._extract_with_chunked_strategy(text)
                elif strategy == ExtractionStrategy.FIRST_PAGES:
                    result = self._extract_with_first_pages_strategy(text)
                elif strategy == ExtractionStrategy.FALLBACK_SIMPLE:
                    result = self._extract_with_fallback_strategy(text)
                else:
                    continue
                
                self.logger.get_logger().info(
                    f"Strategy {strategy.value}: success={result.success}, confidence={result.confidence:.2f}"
                )
                
                if result.success and result.confidence > best_confidence:
                    best_result = result
                    best_confidence = result.confidence
                    
                    # If we got a very good result, stop trying other strategies
                    if result.confidence >= 0.9:
                        break
                
                # If we got a decent result with full text, don't try more complex strategies
                if strategy == ExtractionStrategy.FULL_TEXT and result.success and result.confidence >= 0.7:
                    break
            
            except Exception as e:
                self.logger.get_logger().error(f"Strategy {strategy.value} failed: {e}")
                continue
        
        # Use the best result found, or create a failure result
        if best_result is None:
            best_result = ExtractionResult(
                success=False,
                errors=["All extraction strategies failed"],
                strategy_used=strategies[0] if strategies else None
            )
        
        # Log the final result
        self.logger.log_llm_extraction(
            "text_extraction",
            best_result.success,
            {
                'strategy': best_result.strategy_used.value if best_result.strategy_used else 'none',
                'confidence': best_result.confidence,
                'processing_time': best_result.processing_time
            }
        )
        
        return best_result


# Global extractor instance
_extractor_instance = None

def get_entity_extractor() -> LLMEntityExtractor:
    """Get the global entity extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = LLMEntityExtractor()
    return _extractor_instance


def extract_paper_entities(text: str) -> Dict[str, Any]:
    """
    Convenience function for extracting paper entities.
    
    Args:
        text: Paper text to extract from
        
    Returns:
        Dictionary with extracted entities or None if failed
    """
    extractor = get_entity_extractor()
    result = extractor.extract_entities(text)
    
    if result.success:
        return result.data
    else:
        return None


def extract_paper_entities_safe(text: str) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Safe extraction that always returns something.
    
    Args:
        text: Paper text to extract from
        
    Returns:
        Tuple of (data, errors, warnings)
    """
    extractor = get_entity_extractor()
    result = extractor.extract_entities(text)
    
    data = result.data if result.success else {
        "title": "Unknown",
        "authors": ["Unknown"],
        "publication_date": None,
        "summary": "Could not extract summary"
    }
    
    return data, result.errors, result.warnings