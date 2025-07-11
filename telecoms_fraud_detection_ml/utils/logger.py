"""
Logger utility for telecoms fraud detection.
"""

import logging
import os
from datetime import datetime
from typing import Optional

class Logger:
    """Enhanced logging utility for telecoms fraud detection."""
    
    def __init__(self, name: str = 'telecoms_fraud_detection', log_dir: str = 'logs'):
        """
        Initialize the logger.
        
        Args:
            name: Name of the logger
            log_dir: Directory to store log files
        """
        self.name = name
        self.log_dir = log_dir
        self.logger = None
        self._setup_logger()
    
    def _setup_logger(self):
        """Set up the logger with file and console handlers."""
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if self.logger.handlers:
            return
        
        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_filename = f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(os.path.join(self.log_dir, log_filename))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
    
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)
    
    def log_data_processing_step(self, step_name: str, input_shape: tuple, output_shape: tuple):
        """Log data processing steps with shapes."""
        self.logger.info(f"Data Processing - {step_name}: {input_shape} → {output_shape}")
    
    def log_model_training(self, model_type: str, n_features: int, n_samples: int, metrics: dict):
        """Log model training information."""
        self.logger.info(f"Model Training - {model_type}: Features={n_features}, "
                        f"Samples={n_samples}, Metrics={metrics}")
    
    def log_fraud_detection(self, n_cases: int, fraud_rate: float, high_risk_cases: int):
        """Log fraud detection results."""
        self.logger.info(f"Fraud Detection - Total Cases: {n_cases}, "
                        f"Fraud Rate: {fraud_rate:.4f}, High Risk: {high_risk_cases}")
    
    def get_logger(self) -> logging.Logger:
        """Get the underlying logger object."""
        return self.logger
