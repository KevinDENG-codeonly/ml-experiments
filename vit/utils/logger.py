import os
import logging
import sys
from typing import Optional

def setup_logging(log_dir: str, log_level: int = logging.INFO) -> None:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save logs
        log_level: Logging level
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(log_dir, 'run.log'))
        ]
    )
    
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (module name)
        
    Returns:
        Logger instance
    """
    if name is None:
        return logging.getLogger()
    return logging.getLogger(name) 