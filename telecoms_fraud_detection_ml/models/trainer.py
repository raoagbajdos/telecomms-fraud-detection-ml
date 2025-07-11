"""
Model trainer for telecoms fraud detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

class ModelTrainer:
    """Handles training and evaluation of fraud detection models."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the model trainer.
        
        Args:
            config: Configuration dictionary for training parameters
        """
        self.config = config or {}
        self.models = {}
        self.training_history = []
        self.logger = logging.getLogger(__name__)
        
    def train_single_model(self, 
                          model_type: str,
                          training_data: pd.DataFrame,
                          target_column: str,
                          test_size: float = 0.2,
                          random_state: int = 42) -> Dict[str, Any]:
        """
        Train a single fraud detection model.
        
        Args:
            model_type: Type of model to train
            training_data: DataFrame with features and target
            target_column: Name of the target column
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing training results
        """
        try:
            from ..models.fraud_detector import FraudDetectionModel
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
            
            # Prepare data
            features = training_data.drop(columns=[target_column, 'customer_id'], errors='ignore')
            target = training_data[target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, target, test_size=test_size, random_state=random_state,
                stratify=target
            )
            
            # Initialize and train model
            model = FraudDetectionModel(model_type=model_type)
            training_result = model.train(X_train, y_train, random_state=random_state)
            
            if not training_result['success']:
                return training_result
            
            # Evaluate model
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': float((y_pred == y_test).mean()),
                'roc_auc': float(roc_auc_score(y_test, y_proba[:, 1])),
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            # Store model and results
            training_record = {
                'model_type': model_type,
                'training_date': datetime.now().isoformat(),
                'metrics': metrics,
                'model_object': model,
                'feature_names': list(features.columns),
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
            self.models[model_type] = model
            self.training_history.append(training_record)
            
            self.logger.info(f"Successfully trained {model_type} model. "
                           f"ROC-AUC: {metrics['roc_auc']:.4f}, "
                           f"Accuracy: {metrics['accuracy']:.4f}")
            
            return {
                'success': True,
                'model_type': model_type,
                'metrics': metrics,
                'model': model
            }
            
        except Exception as e:
            self.logger.error(f"Error training {model_type} model: {str(e)}")
            return {'success': False, 'model_type': model_type, 'error': str(e)}
    
    def train_multiple_models(self, 
                             training_data: pd.DataFrame,
                             target_column: str,
                             model_types: Optional[List[str]] = None,
                             **kwargs) -> Dict[str, Any]:
        """
        Train multiple fraud detection models and compare performance.
        
        Args:
            training_data: DataFrame with features and target
            target_column: Name of the target column
            model_types: List of model types to train
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing results for all models
        """
        if model_types is None:
            model_types = ['random_forest', 'xgboost', 'logistic_regression']
        
        results = {
            'training_summary': {
                'total_models': len(model_types),
                'successful_models': 0,
                'failed_models': 0,
                'training_date': datetime.now().isoformat()
            },
            'model_results': {},
            'best_model': None
        }
        
        self.logger.info(f"Starting training for {len(model_types)} models...")
        
        best_roc_auc = 0
        best_model_type = None
        
        for model_type in model_types:
            self.logger.info(f"Training {model_type} model...")
            
            model_result = self.train_single_model(
                model_type=model_type,
                training_data=training_data,
                target_column=target_column,
                **kwargs
            )
            
            results['model_results'][model_type] = model_result
            
            if model_result['success']:
                results['training_summary']['successful_models'] += 1
                
                # Track best model
                current_roc_auc = model_result['metrics']['roc_auc']
                if current_roc_auc > best_roc_auc:
                    best_roc_auc = current_roc_auc
                    best_model_type = model_type
                    results['best_model'] = {
                        'model_type': model_type,
                        'roc_auc': current_roc_auc,
                        'model_object': model_result['model']
                    }
            else:
                results['training_summary']['failed_models'] += 1
        
        self.logger.info(f"Training completed. Best model: {best_model_type} "
                        f"(ROC-AUC: {best_roc_auc:.4f})")
        
        return results
    
    def hyperparameter_tuning(self, 
                             model_type: str,
                             training_data: pd.DataFrame,
                             target_column: str,
                             param_grid: Optional[Dict] = None,
                             cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            model_type: Type of model to tune
            training_data: DataFrame with features and target
            target_column: Name of the target column
            param_grid: Parameter grid for tuning
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing tuning results
        """
        try:
            from sklearn.model_selection import GridSearchCV
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            
            # Prepare data
            features = training_data.drop(columns=[target_column, 'customer_id'], errors='ignore')
            target = training_data[target_column]
            
            # Define default parameter grids
            default_param_grids = {
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'logistic_regression': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
            
            if param_grid is None:
                param_grid = default_param_grids.get(model_type, {})
            
            # Initialize base model
            if model_type == 'random_forest':
                base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
            elif model_type == 'logistic_regression':
                base_model = LogisticRegression(random_state=42, max_iter=1000)
            else:
                raise ValueError(f"Hyperparameter tuning not implemented for {model_type}")
            
            # Perform grid search
            self.logger.info(f"Starting hyperparameter tuning for {model_type}...")
            
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv_folds,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(features, target)
            
            # Compile results
            tuning_results = {
                'model_type': model_type,
                'best_params': grid_search.best_params_,
                'best_score': float(grid_search.best_score_),
                'cv_results': {
                    'mean_scores': grid_search.cv_results_['mean_test_score'].tolist(),
                    'std_scores': grid_search.cv_results_['std_test_score'].tolist(),
                    'params': grid_search.cv_results_['params']
                },
                'tuning_date': datetime.now().isoformat()
            }
            
            self.logger.info(f"Hyperparameter tuning completed for {model_type}. "
                           f"Best ROC-AUC: {tuning_results['best_score']:.4f}")
            
            return tuning_results
            
        except Exception as e:
            self.logger.error(f"Error during hyperparameter tuning: {str(e)}")
            return {'error': str(e), 'model_type': model_type}
    
    def cross_validation_score(self, 
                              model_type: str,
                              training_data: pd.DataFrame,
                              target_column: str,
                              cv_folds: int = 5) -> Dict[str, Any]:
        """
        Perform cross-validation scoring for a model.
        
        Args:
            model_type: Type of model to evaluate
            training_data: DataFrame with features and target
            target_column: Name of the target column
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary containing cross-validation results
        """
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            
            # Prepare data
            features = training_data.drop(columns=[target_column, 'customer_id'], errors='ignore')
            target = training_data[target_column]
            
            # Initialize model
            if model_type == 'random_forest':
                model = RandomForestClassifier(random_state=42, n_jobs=-1)
            elif model_type == 'logistic_regression':
                model = LogisticRegression(random_state=42, max_iter=1000)
            else:
                raise ValueError(f"Cross-validation not implemented for {model_type}")
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model, features, target, cv=cv_folds, scoring='roc_auc', n_jobs=-1
            )
            
            results = {
                'model_type': model_type,
                'cv_scores': cv_scores.tolist(),
                'mean_score': float(cv_scores.mean()),
                'std_score': float(cv_scores.std()),
                'cv_folds': cv_folds,
                'evaluation_date': datetime.now().isoformat()
            }
            
            self.logger.info(f"Cross-validation completed for {model_type}. "
                           f"Mean ROC-AUC: {results['mean_score']:.4f} "
                           f"(±{results['std_score']:.4f})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during cross-validation: {str(e)}")
            return {'error': str(e), 'model_type': model_type}
    
    def get_training_summary(self) -> pd.DataFrame:
        """
        Get summary of all training sessions.
        
        Returns:
            DataFrame containing training summary
        """
        if not self.training_history:
            return pd.DataFrame()
        
        summary_data = []
        for record in self.training_history:
            summary = {
                'model_type': record['model_type'],
                'training_date': record['training_date'],
                'accuracy': record['metrics']['accuracy'],
                'roc_auc': record['metrics']['roc_auc'],
                'training_samples': record['training_samples'],
                'test_samples': record['test_samples'],
                'n_features': len(record['feature_names'])
            }
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def save_best_model(self, output_dir: str = 'models') -> str:
        """
        Save the best performing model.
        
        Args:
            output_dir: Directory to save the model
            
        Returns:
            Path to the saved model
        """
        if not self.training_history:
            raise ValueError("No trained models to save")
        
        # Find best model by ROC-AUC
        best_record = max(self.training_history, key=lambda x: x['metrics']['roc_auc'])
        best_model = best_record['model_object']
        
        # Create filename with date and model type
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fraud_detection_{best_record['model_type']}_{date_str}.pkl"
        filepath = f"{output_dir}/{filename}"
        
        # Save the model
        saved_path = best_model.save_model(filepath, include_date=False)
        
        self.logger.info(f"Best model ({best_record['model_type']}) saved to: {saved_path}")
        return saved_path
    
    def get_trained_models(self) -> Dict[str, Any]:
        """
        Get all trained models.
        
        Returns:
            Dictionary of trained models
        """
        return self.models.copy()
