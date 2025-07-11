"""
Fraud detection model for telecoms data.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import os

class FraudDetectionModel:
    """Main fraud detection model class for telecoms data."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the fraud detection model.
        
        Args:
            model_type: Type of ML model to use
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = []
        self.preprocessors = {}
        self.metadata = {}
        self.logger = logging.getLogger(__name__)
        
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict[str, Any]:
        """
        Train the fraud detection model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training results
        """
        try:
            # Store feature names
            self.feature_names = list(X.columns)
            
            # Import and initialize model based on type
            if self.model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    n_estimators=kwargs.get('n_estimators', 100),
                    max_depth=kwargs.get('max_depth', 10),
                    random_state=kwargs.get('random_state', 42),
                    n_jobs=-1
                )
            elif self.model_type == 'xgboost':
                import xgboost as xgb
                self.model = xgb.XGBClassifier(
                    n_estimators=kwargs.get('n_estimators', 100),
                    max_depth=kwargs.get('max_depth', 6),
                    learning_rate=kwargs.get('learning_rate', 0.1),
                    random_state=kwargs.get('random_state', 42)
                )
            elif self.model_type == 'logistic_regression':
                from sklearn.linear_model import LogisticRegression
                self.model = LogisticRegression(
                    random_state=kwargs.get('random_state', 42),
                    max_iter=1000
                )
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Train the model
            self.logger.info(f"Training {self.model_type} model...")
            self.model.fit(X, y)
            
            # Store metadata
            self.metadata = {
                'model_type': self.model_type,
                'training_date': datetime.now().isoformat(),
                'n_features': len(self.feature_names),
                'n_samples': len(X),
                'feature_names': self.feature_names,
                'class_distribution': y.value_counts().to_dict()
            }
            
            self.logger.info(f"Model training completed. Features: {len(self.feature_names)}, Samples: {len(X)}")
            
            return {
                'success': True,
                'model_type': self.model_type,
                'n_features': len(self.feature_names),
                'n_samples': len(X)
            }
            
        except Exception as e:
            self.logger.error(f"Error during model training: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make fraud predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Ensure feature consistency
        if list(X.columns) != self.feature_names:
            self.logger.warning("Feature names don't match training features")
            # Reorder columns to match training
            X = X[self.feature_names]
        
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get fraud prediction probabilities.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Ensure feature consistency
        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        importance_scores = None
        
        if hasattr(self.model, 'feature_importances_'):
            importance_scores = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance_scores = np.abs(self.model.coef_[0])
        else:
            self.logger.warning("Model does not support feature importance")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str, include_date: bool = True) -> str:
        """
        Save the trained model to disk with optional date in filename.
        
        Args:
            filepath: Base filepath for the model
            include_date: Whether to include date in filename
            
        Returns:
            Actual filepath where model was saved
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Add date to filename if requested
        if include_date:
            base_path, ext = os.path.splitext(filepath)
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            dated_filepath = f"{base_path}_{date_str}{ext}"
        else:
            dated_filepath = filepath
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'metadata': self.metadata,
            'preprocessors': self.preprocessors
        }
        
        try:
            # Use joblib for sklearn models, pickle for others
            if self.model_type in ['random_forest', 'logistic_regression']:
                joblib.dump(model_data, dated_filepath)
            else:
                with open(dated_filepath, 'wb') as f:
                    pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to: {dated_filepath}")
            return dated_filepath
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            True if loading was successful
        """
        try:
            # Try joblib first, then pickle
            try:
                model_data = joblib.load(filepath)
            except:
                with open(filepath, 'rb') as f:
                    model_data = pickle.load(f)
            
            # Restore model components
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.metadata = model_data.get('metadata', {})
            self.preprocessors = model_data.get('preprocessors', {})
            self.model_type = self.metadata.get('model_type', 'unknown')
            
            self.logger.info(f"Model loaded from: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {'status': 'No model loaded'}
        
        info = {
            'model_type': self.model_type,
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'metadata': self.metadata,
            'model_parameters': {}
        }
        
        # Get model-specific parameters
        if hasattr(self.model, 'get_params'):
            info['model_parameters'] = self.model.get_params()
        
        return info
    
    def validate_input(self, X: pd.DataFrame) -> bool:
        """
        Validate input data for prediction.
        
        Args:
            X: Input DataFrame
            
        Returns:
            True if input is valid
        """
        if self.model is None:
            self.logger.error("No model loaded")
            return False
        
        # Check if all required features are present
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            self.logger.error(f"Missing features: {missing_features}")
            return False
        
        # Check for null values
        null_counts = X[self.feature_names].isnull().sum()
        if null_counts.sum() > 0:
            self.logger.warning(f"Null values found in features: {null_counts[null_counts > 0].to_dict()}")
        
        return True
    
    def create_prediction_report(self, X: pd.DataFrame, y_true: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Create a comprehensive prediction report.
        
        Args:
            X: Feature DataFrame
            y_true: True labels (optional)
            
        Returns:
            Dictionary containing prediction report
        """
        if not self.validate_input(X):
            return {'error': 'Invalid input data'}
        
        # Make predictions
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        report = {
            'n_samples': len(X),
            'n_predicted_fraud': int(predictions.sum()),
            'fraud_rate': float(predictions.mean()),
            'prediction_distribution': {
                'normal': int((predictions == 0).sum()),
                'fraud': int((predictions == 1).sum())
            },
            'confidence_stats': {
                'mean_fraud_probability': float(probabilities[:, 1].mean()),
                'std_fraud_probability': float(probabilities[:, 1].std()),
                'max_fraud_probability': float(probabilities[:, 1].max()),
                'min_fraud_probability': float(probabilities[:, 1].min())
            }
        }
        
        # Add performance metrics if true labels are provided
        if y_true is not None:
            from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
            
            report['performance'] = {
                'accuracy': float((predictions == y_true).mean()),
                'roc_auc': float(roc_auc_score(y_true, probabilities[:, 1])),
                'classification_report': classification_report(y_true, predictions, output_dict=True),
                'confusion_matrix': confusion_matrix(y_true, predictions).tolist()
            }
        
        return report
