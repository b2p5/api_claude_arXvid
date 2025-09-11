"""
Retry utilities for handling network operations, API calls, and file operations.
Provides decorators and functions for robust error handling with exponential backoff.
"""

import time
import random
import functools
from typing import Union, Tuple, Callable, Any, Type, List
from urllib.error import URLError, HTTPError
import urllib.request
import json

from logger import get_logger


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Tuple[Type[Exception], ...] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or (
            URLError, HTTPError, ConnectionError, TimeoutError, 
            json.JSONDecodeError, OSError, IOError
        )


def exponential_backoff(
    attempt: int, 
    base_delay: float = 1.0, 
    max_delay: float = 60.0, 
    backoff_factor: float = 2.0, 
    jitter: bool = True
) -> float:
    """Calculate exponential backoff delay with optional jitter."""
    delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)
    
    if jitter:
        # Add random jitter (±25% of calculated delay)
        jitter_range = delay * 0.25
        delay += random.uniform(-jitter_range, jitter_range)
        delay = max(0, delay)  # Ensure non-negative
    
    return delay


def retry_on_exception(config: RetryConfig = None):
    """
    Decorator for retrying functions that might fail due to transient errors.
    
    Args:
        config: RetryConfig instance with retry parameters
    
    Returns:
        Decorated function with retry logic
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            logger = get_logger()
            
            last_exception = None
            for attempt in range(1, config.max_attempts + 1):
                try:
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        logger.log_operation_success(
                            f"{func.__name__} (after {attempt} attempts)",
                            function=func.__name__
                        )
                    return result
                
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt < config.max_attempts:
                        delay = exponential_backoff(
                            attempt, 
                            config.base_delay, 
                            config.max_delay, 
                            config.backoff_factor, 
                            config.jitter
                        )
                        
                        logger.log_retry(
                            func.__name__, 
                            attempt, 
                            config.max_attempts, 
                            e
                        )
                        
                        logger.get_logger().debug(f"Waiting {delay:.2f} seconds before retry...")
                        time.sleep(delay)
                    else:
                        logger.log_operation_failure(
                            f"{func.__name__} (all {config.max_attempts} attempts failed)",
                            e,
                            function=func.__name__
                        )
                
                except Exception as e:
                    # Non-retryable exception
                    logger.log_operation_failure(
                        f"{func.__name__} (non-retryable error)",
                        e,
                        function=func.__name__
                    )
                    raise
            
            # All attempts failed
            raise last_exception
        
        return wrapper
    return decorator


# Specific retry configurations for different operations
PDF_DOWNLOAD_RETRY = RetryConfig(
    max_attempts=3,
    base_delay=2.0,
    max_delay=30.0,
    backoff_factor=2.0,
    jitter=True,
    retryable_exceptions=(URLError, HTTPError, ConnectionError, TimeoutError, OSError)
)

API_CALL_RETRY = RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    max_delay=20.0,
    backoff_factor=1.5,
    jitter=True,
    retryable_exceptions=(
        URLError, HTTPError, ConnectionError, TimeoutError, 
        json.JSONDecodeError, Exception  # API-specific exceptions
    )
)

FILE_OPERATION_RETRY = RetryConfig(
    max_attempts=2,
    base_delay=0.5,
    max_delay=5.0,
    backoff_factor=2.0,
    jitter=False,
    retryable_exceptions=(OSError, IOError, PermissionError)
)


def download_with_retry(url: str, filepath: str, timeout: int = 30) -> bool:
    """
    Download a file with retry logic.
    
    Args:
        url: URL to download from
        filepath: Local filepath to save to
        timeout: Request timeout in seconds
    
    Returns:
        True if successful, False otherwise
    """
    logger = get_logger()
    
    @retry_on_exception(PDF_DOWNLOAD_RETRY)
    def _download():
        logger.log_operation_start("PDF download", url=url, filepath=filepath)
        
        # Set a reasonable timeout and add headers to appear more like a browser
        req = urllib.request.Request(
            url, 
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
        )
        
        with urllib.request.urlopen(req, timeout=timeout) as response:
            with open(filepath, 'wb') as f:
                # Download in chunks to handle large files
                chunk_size = 8192
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
        
        logger.log_operation_success("PDF download", url=url, filepath=filepath)
        return True
    
    try:
        return _download()
    except Exception as e:
        logger.log_operation_failure("PDF download", e, url=url, filepath=filepath)
        return False


def api_call_with_retry(api_function: Callable, *args, **kwargs) -> Any:
    """
    Execute an API call with retry logic.
    
    Args:
        api_function: Function to call (e.g., LLM API call)
        *args, **kwargs: Arguments to pass to the function
    
    Returns:
        Result of the API call
    """
    logger = get_logger()
    
    @retry_on_exception(API_CALL_RETRY)
    def _api_call():
        logger.log_operation_start("API call", function=api_function.__name__)
        result = api_function(*args, **kwargs)
        logger.log_operation_success("API call", function=api_function.__name__)
        return result
    
    try:
        return _api_call()
    except Exception as e:
        logger.log_operation_failure("API call", e, function=api_function.__name__)
        raise


def file_operation_with_retry(file_function: Callable, *args, **kwargs) -> Any:
    """
    Execute a file operation with retry logic.
    
    Args:
        file_function: Function to call (e.g., file read/write)
        *args, **kwargs: Arguments to pass to the function
    
    Returns:
        Result of the file operation
    """
    logger = get_logger()
    
    @retry_on_exception(FILE_OPERATION_RETRY)
    def _file_operation():
        logger.log_operation_start("File operation", function=file_function.__name__)
        result = file_function(*args, **kwargs)
        logger.log_operation_success("File operation", function=file_function.__name__)
        return result
    
    try:
        return _file_operation()
    except Exception as e:
        logger.log_operation_failure("File operation", e, function=file_function.__name__)
        raise


def safe_execute(
    operation: Callable, 
    operation_name: str, 
    default_return=None,
    reraise: bool = False,
    **kwargs
) -> Any:
    """
    Safely execute an operation with logging and optional default return.
    
    Args:
        operation: Function to execute
        operation_name: Name for logging purposes
        default_return: Value to return if operation fails
        reraise: Whether to reraise the exception
        **kwargs: Additional context for logging
    
    Returns:
        Result of operation or default_return if failed
    """
    logger = get_logger()
    
    try:
        logger.log_operation_start(operation_name, **kwargs)
        result = operation()
        logger.log_operation_success(operation_name, **kwargs)
        return result
    except Exception as e:
        logger.log_operation_failure(operation_name, e, **kwargs)
        if reraise:
            raise
        return default_return


# Convenience decorators for common patterns
def retry_pdf_download(func):
    """Decorator specifically for PDF download functions."""
    return retry_on_exception(PDF_DOWNLOAD_RETRY)(func)

def retry_api_call(func):
    """Decorator specifically for API call functions."""
    return retry_on_exception(API_CALL_RETRY)(func)

def retry_file_operation(func):
    """Decorator specifically for file operation functions."""
    return retry_on_exception(FILE_OPERATION_RETRY)(func)