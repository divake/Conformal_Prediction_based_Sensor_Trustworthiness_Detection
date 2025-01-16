# src_vision/utils/logging_utils.py

import logging
from pathlib import Path
from datetime import datetime

def setup_logging(experiment_name: str = 'vision_experiment') -> logging.Logger:
    """
    Set up logging configuration for the vision experiments.
    
    Args:
        experiment_name (str): Name for the experiment/logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamp for unique log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    
    logger = logging.getLogger(experiment_name)
    
    # Log initial information
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Log file created at: {log_file}")
    
    return logger