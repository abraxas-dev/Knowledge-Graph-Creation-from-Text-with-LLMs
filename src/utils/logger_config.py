"""
Logging Configuration Module
Author: @abraxas-dev
"""

import logging
import os
from datetime import datetime
from typing import Dict

_loggers: Dict[str, logging.Logger] = {}

def setup_logger(name: str, log_dir: str = "./data/logs") -> logging.Logger:
    """
    Configure and retrieve a logger instance with file and console handlers.
    Notes:
        - File logs include timestamp, level, module, and message
        - Console logs only show the message for cleaner output
        - Log files are named with timestamp: kg_pipeline_YYYYMMDD_HHMMSS.log
    """
    # Return existing logger if already configured for this directory
    if log_dir in _loggers:
        return _loggers[log_dir]
    
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("KnowledgeGraphPipeline")
    logger.setLevel(logging.INFO)
    
    # Prevent handler duplication
    if logger.handlers:
        return logger
    
    # Configure formatters for different outputs
    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter('%(message)s')
    
    # Setup file handler with detailed formatting
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"kg_pipeline_{timestamp}.log")
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Capture all levels in file
    file_handler.setFormatter(file_formatter)
    
    # Setup console handler with minimal formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Only INFO and above in console
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    _loggers[log_dir] = logger
    
    return logger 