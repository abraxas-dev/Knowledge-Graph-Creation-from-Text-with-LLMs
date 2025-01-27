"""
Project : Knowledge Graph Creation from Text
Author : @abraxas-dev
"""
import logging
import os
from datetime import datetime

_loggers = {}

def setup_logger(name: str, log_dir: str = "./data/logs") -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    Returns the same logger instance for the same log directory.
    
    Args:
        name: Name of the logger (usually __name__)
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    # If we already have a logger for this directory, return it
    if log_dir in _loggers:
        return _loggers[log_dir]
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("KnowledgeGraphPipeline")  # Use same name for all components
    logger.setLevel(logging.INFO)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(module)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(message)s'
    )
    
    # File handler
    log_file = os.path.join(
        log_dir, 
        f"kg_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Store logger in global dictionary
    _loggers[log_dir] = logger
    
    return logger 