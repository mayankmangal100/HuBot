"""Utility functions for logging and performance tracking."""

import time
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json

# Configure logging
def setup_logging(log_file: str = "rag_system.log", level: int = logging.INFO) -> logging.Logger:
    """Set up logging configuration"""
    logger = logging.getLogger("rag_system")
    logger.setLevel(level)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

@dataclass
class LatencyTracker:
    """Track execution time of operations"""
    start_time: float = 0
    operation_name: Optional[str] = None
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        self.metrics = {}
    
    def start(self, operation_name: Optional[str] = None) -> 'LatencyTracker':
        """Start tracking an operation"""
        self.start_time = time.time()
        self.operation_name = operation_name
        return self
    
    def end(self, operation_name: Optional[str] = None) -> float:
        """End tracking and return duration"""
        if operation_name is None:
            operation_name = self.operation_name
            
        duration = time.time() - self.start_time
        self.metrics[operation_name] = duration
        logger.info(f"LATENCY: {operation_name} took {duration:.4f} seconds")
        return duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return self.metrics

class Config:
    """Configuration management"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {str(e)}")
            raise
