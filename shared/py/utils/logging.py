"""Logging configuration."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Configure structured logging for all modules.
    
    Args:
        log_level: DEBUG, INFO, WARNING, ERROR
        log_file: Optional file path (default: logs/{module}.log)
    """
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Console handler
    existing_console = None
    for handler in root_logger.handlers:
        if getattr(handler, "_ehc_console_handler", False):
            existing_console = handler
            break

    if existing_console is None:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler._ehc_console_handler = True
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    else:
        existing_console.setLevel(log_level)
    
    # File handler (if specified)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        abs_log_file = os.path.abspath(log_file)
        existing_file = None
        for handler in root_logger.handlers:
            if getattr(handler, "_ehc_log_file", None) == abs_log_file:
                existing_file = handler
                break

        if existing_file is None:
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5
            )
            file_handler._ehc_log_file = abs_log_file
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
        else:
            existing_file.setLevel(log_level)
