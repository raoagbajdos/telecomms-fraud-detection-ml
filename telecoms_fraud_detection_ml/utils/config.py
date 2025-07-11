"""
Configuration manager for telecoms fraud detection.
"""

import yaml
import json
import os
from typing import Dict, Any, Optional

class ConfigManager:
    """Manages configuration for telecoms fraud detection pipeline."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = {}
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self._set_default_config()
    
    def load_config(self, config_path: str) -> bool:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if loading was successful
        """
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                    self.config = yaml.safe_load(f)
                elif config_path.endswith('.json'):
                    self.config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path}")
            
            self.config_path = config_path
            return True
            
        except Exception as e:
            print(f"Error loading config from {config_path}: {str(e)}")
            self._set_default_config()
            return False
    
    def save_config(self, config_path: Optional[str] = None) -> bool:
        """
        Save configuration to file.
        
        Args:
            config_path: Path to save configuration (optional)
            
        Returns:
            True if saving was successful
        """
        save_path = config_path or self.config_path
        if not save_path:
            print("No config path specified")
            return False
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w') as f:
                if save_path.endswith('.yaml') or save_path.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                elif save_path.endswith('.json'):
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {save_path}")
            
            return True
            
        except Exception as e:
            print(f"Error saving config to {save_path}: {str(e)}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def _set_default_config(self):
        """Set default configuration values."""
        self.config = {
            'data_processing': {
                'missing_value_threshold': 0.5,
                'outlier_method': 'iqr',
                'encoding_method': 'label',
                'scaling_method': 'standard',
                'customer_id_column': 'customer_id',
                'target_column': 'is_fraud'
            },
            'feature_engineering': {
                'create_interaction_features': True,
                'create_polynomial_features': False,
                'polynomial_degree': 2,
                'create_aggregated_features': True,
                'apply_log_transform': True
            },
            'model_training': {
                'auto_select_best': True,
                'available_models': [
                    'random_forest',
                    'xgboost', 
                    'logistic_regression'
                ],
                'test_size': 0.2,
                'random_state': 42,
                'cross_validation_folds': 5,
                'tune_hyperparameters': False
            },
            'model_parameters': {
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt'
                },
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8
                },
                'logistic_regression': {
                    'C': 1.0,
                    'penalty': 'l2',
                    'solver': 'liblinear',
                    'max_iter': 1000
                }
            },
            'evaluation': {
                'metrics': [
                    'accuracy',
                    'precision',
                    'recall',
                    'f1_score',
                    'roc_auc'
                ],
                'generate_plots': True,
                'save_confusion_matrix': True,
                'calculate_feature_importance': True
            },
            'output': {
                'save_models': True,
                'model_format': 'pickle',
                'include_date_in_filename': True,
                'save_predictions': True,
                'save_evaluation_reports': True
            },
            'logging': {
                'level': 'INFO',
                'log_to_file': True,
                'log_dir': 'logs',
                'log_data_processing': True,
                'log_model_training': True
            }
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data processing configuration."""
        return self.get('data_processing', {})
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model training configuration."""
        return self.get('model_training', {})
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self.get('feature_engineering', {})
    
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.get('evaluation', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.get('output', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get('logging', {})
    
    def get_model_parameters(self, model_type: str) -> Dict[str, Any]:
        """
        Get model-specific parameters.
        
        Args:
            model_type: Type of model
            
        Returns:
            Dictionary of model parameters
        """
        return self.get(f'model_parameters.{model_type}', {})
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate the current configuration.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check required sections
        required_sections = [
            'data_processing', 'model_training', 
            'feature_engineering', 'evaluation'
        ]
        
        for section in required_sections:
            if section not in self.config:
                validation_results['errors'].append(f"Missing required section: {section}")
                validation_results['valid'] = False
        
        # Check model parameters
        model_config = self.get_model_config()
        available_models = model_config.get('available_models', [])
        
        for model_type in available_models:
            if not self.get_model_parameters(model_type):
                validation_results['warnings'].append(f"No parameters defined for model: {model_type}")
        
        # Check test_size value
        test_size = model_config.get('test_size', 0.2)
        if not (0 < test_size < 1):
            validation_results['errors'].append("test_size must be between 0 and 1")
            validation_results['valid'] = False
        
        return validation_results
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary."""
        return self.config.copy()
