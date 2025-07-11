"""
Feature selector for telecoms fraud detection.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
import logging

class FeatureSelector:
    """Selects optimal features for fraud detection model."""
    
    def __init__(self, max_features: Optional[int] = None):
        """
        Initialize the feature selector.
        
        Args:
            max_features: Maximum number of features to select
        """
        self.max_features = max_features
        self.selected_features = []
        self.feature_scores = {}
        self.logger = logging.getLogger(__name__)
    
    def select_features_by_correlation(self, 
                                     df: pd.DataFrame, 
                                     target_col: str,
                                     threshold: float = 0.05) -> List[str]:
        """
        Select features based on correlation with target variable.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of the target column
            threshold: Minimum correlation threshold
            
        Returns:
            List of selected feature names
        """
        if target_col not in df.columns:
            self.logger.error(f"Target column '{target_col}' not found")
            return []
        
        # Get numerical columns only
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numerical_cols if col != target_col]
        
        # Calculate correlations
        correlations = {}
        for col in feature_cols:
            corr = abs(df[col].corr(df[target_col]))
            if not pd.isna(corr):
                correlations[col] = corr
        
        # Filter by threshold
        selected = [col for col, corr in correlations.items() if corr >= threshold]
        
        # Sort by correlation and limit if max_features is set
        selected = sorted(selected, key=lambda x: correlations[x], reverse=True)
        if self.max_features:
            selected = selected[:self.max_features]
        
        self.selected_features = selected
        self.feature_scores = correlations
        
        self.logger.info(f"Selected {len(selected)} features based on correlation (threshold: {threshold})")
        return selected
    
    def select_features_by_variance(self, 
                                   df: pd.DataFrame, 
                                   threshold: float = 0.01) -> List[str]:
        """
        Select features based on variance threshold.
        
        Args:
            df: DataFrame with features
            threshold: Minimum variance threshold
            
        Returns:
            List of selected feature names
        """
        # Get numerical columns only
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        variances = {}
        selected = []
        
        for col in numerical_cols:
            if col != 'customer_id':  # Exclude ID columns
                var = df[col].var()
                variances[col] = var
                
                if var >= threshold:
                    selected.append(col)
        
        self.feature_scores.update(variances)
        
        self.logger.info(f"Selected {len(selected)} features based on variance (threshold: {threshold})")
        return selected
    
    def remove_highly_correlated_features(self, 
                                        df: pd.DataFrame, 
                                        features: List[str],
                                        threshold: float = 0.95) -> List[str]:
        """
        Remove highly correlated features to reduce multicollinearity.
        
        Args:
            df: DataFrame with features
            features: List of feature names to consider
            threshold: Correlation threshold for removal
            
        Returns:
            List of remaining feature names
        """
        # Calculate correlation matrix
        corr_matrix = df[features].corr().abs()
        
        # Find pairs of highly correlated features
        to_remove = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] >= threshold:
                    # Remove the feature with lower variance
                    col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                    var1, var2 = df[col1].var(), df[col2].var()
                    
                    if var1 < var2:
                        to_remove.add(col1)
                    else:
                        to_remove.add(col2)
        
        remaining_features = [f for f in features if f not in to_remove]
        
        self.logger.info(f"Removed {len(to_remove)} highly correlated features. "
                        f"Remaining: {len(remaining_features)}")
        
        return remaining_features
    
    def select_top_features(self, 
                           feature_scores: Dict[str, float], 
                           n_features: int) -> List[str]:
        """
        Select top N features based on scores.
        
        Args:
            feature_scores: Dictionary of feature names and scores
            n_features: Number of top features to select
            
        Returns:
            List of top feature names
        """
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = [feature for feature, score in sorted_features[:n_features]]
        
        self.logger.info(f"Selected top {len(top_features)} features")
        return top_features
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get summary statistics for features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with feature summary
        """
        summary_data = []
        
        for col in df.columns:
            if col == 'customer_id':
                continue
                
            summary = {
                'feature_name': col,
                'data_type': str(df[col].dtype),
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_values': df[col].nunique(),
                'variance': df[col].var() if df[col].dtype in ['int64', 'float64'] else None,
                'correlation_score': self.feature_scores.get(col, None)
            }
            
            if df[col].dtype in ['int64', 'float64']:
                summary.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                })
            
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def automatic_feature_selection(self, 
                                   df: pd.DataFrame, 
                                   target_col: str,
                                   correlation_threshold: float = 0.05,
                                   variance_threshold: float = 0.01,
                                   multicollinearity_threshold: float = 0.95) -> List[str]:
        """
        Perform automatic feature selection using multiple criteria.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of the target column
            correlation_threshold: Minimum correlation with target
            variance_threshold: Minimum variance threshold
            multicollinearity_threshold: Maximum correlation between features
            
        Returns:
            List of selected feature names
        """
        self.logger.info("Starting automatic feature selection...")
        
        # Step 1: Remove low variance features
        variance_features = self.select_features_by_variance(df, variance_threshold)
        
        # Step 2: Select features correlated with target
        corr_features = self.select_features_by_correlation(
            df[variance_features + [target_col]], target_col, correlation_threshold
        )
        
        # Step 3: Remove highly correlated features
        final_features = self.remove_highly_correlated_features(
            df, corr_features, multicollinearity_threshold
        )
        
        # Step 4: Limit to max_features if specified
        if self.max_features and len(final_features) > self.max_features:
            # Use correlation scores to select top features
            target_correlations = {
                col: abs(df[col].corr(df[target_col])) 
                for col in final_features
            }
            final_features = self.select_top_features(target_correlations, self.max_features)
        
        self.selected_features = final_features
        
        self.logger.info(f"Automatic feature selection completed. "
                        f"Selected {len(final_features)} features from {len(df.columns)-1} candidates")
        
        return final_features
    
    def get_selected_features(self) -> List[str]:
        """
        Get the list of selected features.
        
        Returns:
            List of selected feature names
        """
        return self.selected_features.copy()
    
    def get_feature_scores(self) -> Dict[str, float]:
        """
        Get the feature scores dictionary.
        
        Returns:
            Dictionary of feature names and scores
        """
        return self.feature_scores.copy()
