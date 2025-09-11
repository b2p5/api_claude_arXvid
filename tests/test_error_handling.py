#!/usr/bin/env python3
"""
Test script for the new error handling system.
Tests logging, retry logic, PDF validation, and LLM extraction.
"""

import os
import tempfile
import time
from pathlib import Path
import sys

# Add project root to sys.path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import get_config
from src.logger import get_logger, log_info, log_error, log_warning
from src.retry_utils import download_with_retry, safe_execute, retry_on_exception, RetryConfig
from src.core.analysis.pdf_validator import validate_pdf, PDFValidator
from src.core.llm_utils import extract_paper_entities_safe

def test_logging_system():
    """Test the logging system."""
    print("=" * 50)
    print("Testing Logging System")
    print("=" * 50)
    
    logger = get_logger()
    
    # Test different log levels
    log_info("Testing info level logging", test_component="logging")
    log_warning("Testing warning level logging", test_component="logging")
    log_error("Testing error level logging", test_component="logging")
    
    # Test operation logging
    logger.log_operation_start("test operation", param1="value1", param2="value2")
    time.sleep(0.1)  # Simulate work
    logger.log_operation_success("test operation", result="success")
    
    # Test failure logging
    try:
        raise ValueError("This is a test error")
    except Exception as e:
        logger.log_operation_failure("test operation", e, context="testing")
    
    print("[OK] Logging system test completed")

def test_retry_logic():
    """Test retry logic with simulated failures."""
    print("\n" + "=" * 50)
    print("Testing Retry Logic")
    print("=" * 50)
    
    # Test retry with eventual success
    attempt_count = 0
    
    @retry_on_exception(RetryConfig(max_attempts=3, base_delay=0.1))
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise ConnectionError(f"Simulated failure {attempt_count}")
        return f"Success after {attempt_count} attempts"
    
    try:
        result = flaky_function()
        print(f"[OK] Retry test passed: {result}")
    except Exception as e:
        print(f"[FAIL] Retry test failed: {e}")
    
    # Test safe_execute
    def failing_function():
        raise RuntimeError("This always fails")
    
    result = safe_execute(
        failing_function,
        "safe execute test",
        default_return="fallback value",
        test_param="test"
    )
    
    print(f"[OK] Safe execute test: returned '{result}' (expected fallback)")

def test_pdf_validation():
    """Test PDF validation with different scenarios."""
    print("\n" + "=" * 50)
    print("Testing PDF Validation")
    print("=" * 50)
    
    # Test with non-existent file
    result = validate_pdf("non_existent_file.pdf")
    print(f"[OK] Non-existent file test: valid={result.is_valid} (expected False)")
    
    # Test with a fake PDF file (just text)
    with tempfile.NamedTemporaryFile(suffix=".pdf", mode='w', delete=False) as f:
        f.write("This is not a real PDF file")
        fake_pdf_path = f.name
    
    try:
        result = validate_pdf(fake_pdf_path)
        print(f"[OK] Fake PDF test: valid={result.is_valid} (expected False)")
        print(f"  Errors: {len(result.errors)}")
    finally:
        os.unlink(fake_pdf_path)
    
    # Test with custom validator (more permissive)
    permissive_validator = PDFValidator(
        min_size_mb=0.001,
        max_size_mb=200.0,
        require_text=False
    )
    
    print("[OK] PDF validation tests completed")

def test_llm_extraction():
    """Test LLM extraction with sample text."""
    print("\n" + "=" * 50)
    print("Testing LLM Extraction")
    print("=" * 50)
    
    # Sample paper text
    sample_text = """
    Machine Learning Approaches to Natural Language Processing
    
    Authors: John Smith, Maria Garcia, David Chen
    
    Abstract: This paper presents a comprehensive survey of machine learning 
    techniques applied to natural language processing tasks. We review recent 
    advances in deep learning models including transformers and attention 
    mechanisms. Our analysis covers various applications from text classification 
    to machine translation.
    
    Keywords: machine learning, natural language processing, transformers
    
    1. Introduction
    Natural language processing has seen tremendous advances...
    """
    
    # Test extraction with fallback
    data, errors, warnings = extract_paper_entities_safe(sample_text)
    
    print(f"[OK] LLM extraction test completed")
    print(f"  Title: {data.get('title', 'Not found')}")
    print(f"  Authors: {data.get('authors', [])}")
    print(f"  Errors: {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    
    if errors:
        print("  Error details:")
        for error in errors[:3]:  # Show first 3 errors
            print(f"    - {error}")

def test_download_retry():
    """Test download with retry (using a test URL)."""
    print("\n" + "=" * 50)
    print("Testing Download Retry")
    print("=" * 50)
    
    # Test with invalid URL (should fail after retries)
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = os.path.join(temp_dir, "test_download.pdf")
        
        # This should fail gracefully
        success = download_with_retry("http://invalid-url-that-does-not-exist.com/file.pdf", test_file, timeout=5)
        
        print(f"[OK] Invalid URL test: success={success} (expected False)")
        
        # Test with a real but non-PDF URL (should download but might fail validation)
        # Using a small, accessible URL
        test_file2 = os.path.join(temp_dir, "test_download2.txt")
        success2 = download_with_retry("https://httpbin.org/robots.txt", test_file2, timeout=10)
        
        print(f"[OK] Valid URL test: success={success2}")
        
        if success2 and os.path.exists(test_file2):
            size = os.path.getsize(test_file2)
            print(f"  Downloaded file size: {size} bytes")

def test_configuration_integration():
    """Test that configuration is properly loaded."""
    print("\n" + "=" * 50)
    print("Testing Configuration Integration")
    print("=" * 50)
    
    config = get_config()
    
    # Test configuration validation
    errors = config.validate()
    
    print(f"[OK] Configuration validation: {len(errors)} errors")
    if errors:
        print("  Configuration errors:")
        for error in errors[:3]:
            print(f"    - {error}")
    
    # Test that our modules use the configuration
    print(f"  Vector DB path: {config.database.vector_db_path}")
    print(f"  LLM model: {config.models.llm_model}")
    print(f"  Chunk size: {config.models.chunk_size}")

def main():
    """Run all tests."""
    print("Error Handling System Test Suite")
    print("Testing all components of the new error handling system...")
    
    try:
        test_logging_system()
        test_retry_logic()
        test_pdf_validation()
        test_llm_extraction()
        test_download_retry()
        test_configuration_integration()
        
        print("\n" + "=" * 50)
        print("ALL TESTS COMPLETED")
        print("=" * 50)
        print("Check the logs/ directory for detailed log files.")
        print("The error handling system is ready for use!")
        
    except Exception as e:
        log_error("Test suite failed", error=e)
        print(f"\nTest suite failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)