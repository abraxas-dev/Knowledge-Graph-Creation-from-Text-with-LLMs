import logging
import os
from datetime import datetime

def setup_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Name of the logger (usually __name__)
        log_dir: Directory to store log files
        
    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s'
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
    
    return logger 