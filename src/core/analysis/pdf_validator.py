"""
PDF validation utilities for ensuring PDF files are valid before processing.
Provides comprehensive validation including file integrity, readability, and content extraction.
"""

import os
import mimetypes
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import tempfile

from logger import get_logger, log_info, log_warning, log_error


class PDFValidationResult:
    """Result object for PDF validation."""
    
    def __init__(
        self, 
        is_valid: bool = False,
        file_path: str = "",
        size_mb: float = 0.0,
        page_count: int = 0,
        has_text: bool = False,
        errors: list = None,
        warnings: list = None,
        metadata: dict = None
    ):
        self.is_valid = is_valid
        self.file_path = file_path
        self.size_mb = size_mb
        self.page_count = page_count
        self.has_text = has_text
        self.errors = errors or []
        self.warnings = warnings or []
        self.metadata = metadata or {}
    
    def __str__(self):
        status = "VALID" if self.is_valid else "INVALID"
        return f"PDF Validation [{status}]: {self.file_path} | {self.size_mb:.2f}MB | {self.page_count} pages"


class PDFValidator:
    """Comprehensive PDF validator."""
    
    def __init__(
        self, 
        min_size_mb: float = 0.01,  # 10KB minimum
        max_size_mb: float = 100.0,  # 100MB maximum
        min_pages: int = 1,
        max_pages: int = 1000,
        require_text: bool = True
    ):
        self.min_size_mb = min_size_mb
        self.max_size_mb = max_size_mb
        self.min_pages = min_pages
        self.max_pages = max_pages
        self.require_text = require_text
        self.logger = get_logger()
    
    def validate_file_existence(self, file_path: str) -> Tuple[bool, list, list]:
        """Validate that the file exists and is accessible."""
        errors = []
        warnings = []
        
        if not os.path.exists(file_path):
            errors.append(f"File does not exist: {file_path}")
            return False, errors, warnings
        
        if not os.path.isfile(file_path):
            errors.append(f"Path is not a file: {file_path}")
            return False, errors, warnings
        
        if not os.access(file_path, os.R_OK):
            errors.append(f"File is not readable: {file_path}")
            return False, errors, warnings
        
        return True, errors, warnings
    
    def validate_file_properties(self, file_path: str) -> Tuple[bool, list, list, float]:
        """Validate file size and MIME type."""
        errors = []
        warnings = []
        
        # Check file size
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        
        if size_mb < self.min_size_mb:
            errors.append(f"File too small: {size_mb:.2f}MB (minimum: {self.min_size_mb}MB)")
        
        if size_mb > self.max_size_mb:
            errors.append(f"File too large: {size_mb:.2f}MB (maximum: {self.max_size_mb}MB)")
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type != 'application/pdf' and not file_path.lower().endswith('.pdf'):
            errors.append(f"File does not appear to be a PDF: MIME type = {mime_type}")
        
        # Check file extension
        if not file_path.lower().endswith('.pdf'):
            warnings.append("File does not have .pdf extension")
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings, size_mb
    
    def validate_pdf_structure(self, file_path: str) -> Tuple[bool, list, list, int, bool, dict]:
        """Validate PDF structure and extract metadata."""
        errors = []
        warnings = []
        page_count = 0
        has_text = False
        metadata = {}
        
        try:
            # Import PyPDF here to handle cases where it's not available
            from pypdf import PdfReader
            
            # Try to open and read the PDF
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                
                # Check if PDF is encrypted
                if reader.is_encrypted:
                    warnings.append("PDF is encrypted (may affect text extraction)")
                
                # Get page count
                page_count = len(reader.pages)
                
                if page_count < self.min_pages:
                    errors.append(f"Too few pages: {page_count} (minimum: {self.min_pages})")
                
                if page_count > self.max_pages:
                    errors.append(f"Too many pages: {page_count} (maximum: {self.max_pages})")
                
                # Try to extract text from first few pages
                text_sample = ""
                pages_to_check = min(3, page_count)  # Check first 3 pages
                
                for i in range(pages_to_check):
                    try:
                        page_text = reader.pages[i].extract_text()
                        if page_text and page_text.strip():
                            text_sample += page_text
                            has_text = True
                    except Exception as e:
                        warnings.append(f"Could not extract text from page {i+1}: {str(e)}")
                
                if self.require_text and not has_text:
                    errors.append("PDF contains no extractable text")
                elif not has_text:
                    warnings.append("PDF contains no extractable text")
                
                # Extract metadata if available
                if reader.metadata:
                    try:
                        metadata = {
                            'title': reader.metadata.get('/Title', ''),
                            'author': reader.metadata.get('/Author', ''),
                            'subject': reader.metadata.get('/Subject', ''),
                            'creator': reader.metadata.get('/Creator', ''),
                            'producer': reader.metadata.get('/Producer', ''),
                            'creation_date': str(reader.metadata.get('/CreationDate', '')),
                            'modification_date': str(reader.metadata.get('/ModDate', ''))
                        }
                        # Remove empty values
                        metadata = {k: v for k, v in metadata.items() if v}
                    except Exception as e:
                        warnings.append(f"Could not extract metadata: {str(e)}")
                
        except ImportError:
            errors.append("PyPDF library not available for PDF structure validation")
        except Exception as e:
            errors.append(f"Could not read PDF structure: {str(e)}")
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings, page_count, has_text, metadata
    
    def validate_pdf_content(self, file_path: str) -> Tuple[bool, list, list]:
        """Validate PDF content using langchain loader."""
        errors = []
        warnings = []
        
        try:
            from langchain_community.document_loaders import PyPDFLoader
            
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            if not documents:
                errors.append("PDF loader returned no documents")
                return False, errors, warnings
            
            # Check if we can extract meaningful content
            total_content_length = sum(len(doc.page_content.strip()) for doc in documents)
            
            if total_content_length == 0:
                if self.require_text:
                    errors.append("No text content could be extracted by PDF loader")
                else:
                    warnings.append("No text content could be extracted by PDF loader")
            elif total_content_length < 100:
                warnings.append(f"Very little text content extracted: {total_content_length} characters")
            
        except ImportError:
            warnings.append("Langchain PyPDFLoader not available for content validation")
        except Exception as e:
            errors.append(f"PDF content validation failed: {str(e)}")
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
    
    def validate(self, file_path: str) -> PDFValidationResult:
        """
        Perform comprehensive PDF validation.
        
        Args:
            file_path: Path to the PDF file to validate
            
        Returns:
            PDFValidationResult with validation details
        """
        self.logger.log_operation_start("PDF validation", file_path=file_path)
        
        result = PDFValidationResult(file_path=file_path)
        
        # Step 1: File existence and accessibility
        exists_ok, exist_errors, exist_warnings = self.validate_file_existence(file_path)
        result.errors.extend(exist_errors)
        result.warnings.extend(exist_warnings)
        
        if not exists_ok:
            result.is_valid = False
            self.logger.log_pdf_validation(file_path, False, 0.0)
            return result
        
        # Step 2: File properties (size, type)
        props_ok, prop_errors, prop_warnings, size_mb = self.validate_file_properties(file_path)
        result.errors.extend(prop_errors)
        result.warnings.extend(prop_warnings)
        result.size_mb = size_mb
        
        # Step 3: PDF structure
        struct_ok, struct_errors, struct_warnings, page_count, has_text, metadata = self.validate_pdf_structure(file_path)
        result.errors.extend(struct_errors)
        result.warnings.extend(struct_warnings)
        result.page_count = page_count
        result.has_text = has_text
        result.metadata = metadata
        
        # Step 4: Content validation (using langchain loader)
        content_ok, content_errors, content_warnings = self.validate_pdf_content(file_path)
        result.errors.extend(content_errors)
        result.warnings.extend(content_warnings)
        
        # Final validation result
        result.is_valid = exists_ok and props_ok and struct_ok and content_ok
        
        # Log the result
        self.logger.log_pdf_validation(
            file_path, 
            result.is_valid, 
            result.size_mb, 
            result.page_count
        )
        
        if result.errors:
            log_error(f"PDF validation failed", file_path=file_path, errors=len(result.errors))
        elif result.warnings:
            log_warning(f"PDF validation passed with warnings", file_path=file_path, warnings=len(result.warnings))
        else:
            log_info(f"PDF validation passed", file_path=file_path)
        
        return result


# Default validator instance
default_validator = PDFValidator()

def validate_pdf(file_path: str, validator: PDFValidator = None) -> PDFValidationResult:
    """
    Quick validation function using default or provided validator.
    
    Args:
        file_path: Path to PDF file
        validator: Optional custom validator
        
    Returns:
        PDFValidationResult
    """
    if validator is None:
        validator = default_validator
    
    return validator.validate(file_path)


def is_valid_pdf(file_path: str, validator: PDFValidator = None) -> bool:
    """
    Quick check if PDF is valid.
    
    Args:
        file_path: Path to PDF file
        validator: Optional custom validator
        
    Returns:
        True if valid, False otherwise
    """
    result = validate_pdf(file_path, validator)
    return result.is_valid


def get_pdf_info(file_path: str, validator: PDFValidator = None) -> Dict[str, Any]:
    """
    Get PDF information as a dictionary.
    
    Args:
        file_path: Path to PDF file
        validator: Optional custom validator
        
    Returns:
        Dictionary with PDF information
    """
    result = validate_pdf(file_path, validator)
    
    return {
        'is_valid': result.is_valid,
        'size_mb': result.size_mb,
        'page_count': result.page_count,
        'has_text': result.has_text,
        'errors': result.errors,
        'warnings': result.warnings,
        'metadata': result.metadata
    }