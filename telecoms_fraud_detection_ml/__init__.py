"""
Telecoms Fraud Detection ML Package

A comprehensive machine learning package for detecting fraudulent activities
in telecommunications data using multiple data sources including billing,
CRM, social, and customer care data.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .data import DataProcessor, DataUnifier
from .features import FeatureEngineer
from .models import FraudDetectionModel, ModelTrainer
from .utils import Logger, ConfigManager, DateUtils

__all__ = [
    "DataProcessor",
    "DataUnifier", 
    "FeatureEngineer",
    "FraudDetectionModel",
    "ModelTrainer",
    "Logger",
    "ConfigManager",
    "DateUtils"
]
