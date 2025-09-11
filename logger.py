"""
Structured logging system for arXiv papers analysis project.
Provides centralized logging with different levels and formatters.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import get_config


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to the log level
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


class ArxivLogger:
    """Centralized logger for the arXiv papers analysis system."""
    
    def __init__(self, name: str = "arxiv_papers", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.logger = None
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup the logger with appropriate handlers and formatters."""
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        
        # Avoid adding handlers multiple times
        if self.logger.handlers:
            return
        
        # Create logs directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler for all logs
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Error file handler (only errors and critical)
        error_file = self.log_dir / f"{self.name}_errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self.logger
    
    def log_operation_start(self, operation: str, **kwargs):
        """Log the start of an operation with context."""
        context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        msg = f"Starting {operation}"
        if context:
            msg += f" | {context}"
        self.logger.info(msg)
    
    def log_operation_success(self, operation: str, **kwargs):
        """Log successful completion of an operation."""
        context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        msg = f"Completed {operation}"
        if context:
            msg += f" | {context}"
        self.logger.info(msg)
    
    def log_operation_failure(self, operation: str, error: Exception, **kwargs):
        """Log failure of an operation with error details."""
        context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
        msg = f"Failed {operation}: {str(error)}"
        if context:
            msg += f" | {context}"
        self.logger.error(msg, exc_info=True)
    
    def log_retry(self, operation: str, attempt: int, max_attempts: int, error: Exception):
        """Log retry attempts."""
        self.logger.warning(
            f"Retry {attempt}/{max_attempts} for {operation}: {str(error)}"
        )
    
    def log_pdf_validation(self, pdf_path: str, is_valid: bool, size_mb: float, pages: int = None):
        """Log PDF validation results."""
        status = "VALID" if is_valid else "INVALID"
        msg = f"PDF validation: {pdf_path} | Status: {status} | Size: {size_mb:.2f}MB"
        if pages:
            msg += f" | Pages: {pages}"
        
        if is_valid:
            self.logger.info(msg)
        else:
            self.logger.warning(msg)
    
    def log_llm_extraction(self, pdf_path: str, success: bool, details: Optional[dict] = None):
        """Log LLM extraction results."""
        status = "SUCCESS" if success else "FAILED"
        msg = f"LLM extraction: {pdf_path} | Status: {status}"
        
        if details:
            context = " | ".join([f"{k}={v}" for k, v in details.items()])
            msg += f" | {context}"
        
        if success:
            self.logger.info(msg)
        else:
            self.logger.error(msg)


# Global logger instance
_logger_instance = None

def get_logger(name: str = "arxiv_papers") -> ArxivLogger:
    """Get the global logger instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = ArxivLogger(name)
    return _logger_instance

def setup_logging(log_level: str = "INFO"):
    """Setup logging for the entire application."""
    level = getattr(logging, log_level.upper())
    logger = get_logger()
    logger.logger.setLevel(level)
    return logger


# Convenience functions for quick logging
def log_info(msg: str, **kwargs):
    """Quick info logging."""
    logger = get_logger()
    context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    if context:
        msg += f" | {context}"
    logger.logger.info(msg)

def log_error(msg: str, error: Exception = None, **kwargs):
    """Quick error logging."""
    logger = get_logger()
    context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    if context:
        msg += f" | {context}"
    if error:
        msg += f": {str(error)}"
    logger.logger.error(msg, exc_info=error is not None)

def log_warning(msg: str, **kwargs):
    """Quick warning logging."""
    logger = get_logger()
    context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    if context:
        msg += f" | {context}"
    logger.logger.warning(msg)

def log_debug(msg: str, **kwargs):
    """Quick debug logging."""
    logger = get_logger()
    context = " | ".join([f"{k}={v}" for k, v in kwargs.items()])
    if context:
        msg += f" | {context}"
    logger.logger.debug(msg)