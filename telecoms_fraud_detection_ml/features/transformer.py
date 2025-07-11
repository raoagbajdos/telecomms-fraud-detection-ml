"""
Feature transformer for telecoms fraud detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging

class FeatureTransformer:
    """Transforms features for optimal model performance."""
    
    def __init__(self):
        """Initialize the feature transformer."""
        self.logger = logging.getLogger(__name__)
        self.transformations = {}
    
    def log_transform_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Apply log transformation to specified features.
        
        Args:
            df: DataFrame with features
            features: List of feature names to transform
            
        Returns:
            DataFrame with log-transformed features
        """
        df_transformed = df.copy()
        
        for feature in features:
            if feature in df_transformed.columns:
                # Add 1 to handle zero values
                df_transformed[f'{feature}_log'] = np.log1p(df_transformed[feature])
                self.transformations[f'{feature}_log'] = 'log_transform'
                self.logger.info(f"Applied log transformation to {feature}")
        
        return df_transformed
    
    def sqrt_transform_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Apply square root transformation to specified features.
        
        Args:
            df: DataFrame with features
            features: List of feature names to transform
            
        Returns:
            DataFrame with sqrt-transformed features
        """
        df_transformed = df.copy()
        
        for feature in features:
            if feature in df_transformed.columns:
                # Ensure non-negative values
                df_transformed[f'{feature}_sqrt'] = np.sqrt(np.maximum(df_transformed[feature], 0))
                self.transformations[f'{feature}_sqrt'] = 'sqrt_transform'
                self.logger.info(f"Applied sqrt transformation to {feature}")
        
        return df_transformed
    
    def create_polynomial_features(self, df: pd.DataFrame, features: List[str], degree: int = 2) -> pd.DataFrame:
        """
        Create polynomial features for specified columns.
        
        Args:
            df: DataFrame with features
            features: List of feature names to create polynomials for
            degree: Polynomial degree
            
        Returns:
            DataFrame with polynomial features
        """
        df_enhanced = df.copy()
        
        for feature in features:
            if feature in df_enhanced.columns:
                for d in range(2, degree + 1):
                    new_feature = f'{feature}_poly_{d}'
                    df_enhanced[new_feature] = df_enhanced[feature] ** d
                    self.transformations[new_feature] = f'polynomial_degree_{d}'
                    
                self.logger.info(f"Created polynomial features for {feature} up to degree {degree}")
        
        return df_enhanced
    
    def create_interaction_features(self, df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between specified feature pairs.
        
        Args:
            df: DataFrame with features
            feature_pairs: List of tuples containing feature pairs
            
        Returns:
            DataFrame with interaction features
        """
        df_enhanced = df.copy()
        
        for feature1, feature2 in feature_pairs:
            if feature1 in df_enhanced.columns and feature2 in df_enhanced.columns:
                # Multiplicative interaction
                interaction_name = f'{feature1}_x_{feature2}'
                df_enhanced[interaction_name] = df_enhanced[feature1] * df_enhanced[feature2]
                self.transformations[interaction_name] = 'multiplicative_interaction'
                
                # Ratio interaction (if both are positive)
                if (df_enhanced[feature1] > 0).all() and (df_enhanced[feature2] > 0).all():
                    ratio_name = f'{feature1}_div_{feature2}'
                    df_enhanced[ratio_name] = df_enhanced[feature1] / (df_enhanced[feature2] + 1e-8)
                    self.transformations[ratio_name] = 'ratio_interaction'
                
                self.logger.info(f"Created interaction features for {feature1} and {feature2}")
        
        return df_enhanced
    
    def create_binned_features(self, df: pd.DataFrame, features: List[str], n_bins: int = 5) -> pd.DataFrame:
        """
        Create binned versions of continuous features.
        
        Args:
            df: DataFrame with features
            features: List of feature names to bin
            n_bins: Number of bins to create
            
        Returns:
            DataFrame with binned features
        """
        df_enhanced = df.copy()
        
        for feature in features:
            if feature in df_enhanced.columns:
                binned_feature = f'{feature}_binned'
                df_enhanced[binned_feature] = pd.cut(
                    df_enhanced[feature], 
                    bins=n_bins, 
                    labels=False,
                    duplicates='drop'
                )
                self.transformations[binned_feature] = f'binned_{n_bins}_bins'
                self.logger.info(f"Created binned feature for {feature} with {n_bins} bins")
        
        return df_enhanced
    
    def create_aggregated_features(self, df: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Create aggregated features from groups of related features.
        
        Args:
            df: DataFrame with features
            feature_groups: Dictionary mapping group names to feature lists
            
        Returns:
            DataFrame with aggregated features
        """
        df_enhanced = df.copy()
        
        for group_name, features in feature_groups.items():
            available_features = [f for f in features if f in df_enhanced.columns]
            
            if len(available_features) > 1:
                # Sum aggregation
                sum_feature = f'{group_name}_sum'
                df_enhanced[sum_feature] = df_enhanced[available_features].sum(axis=1)
                self.transformations[sum_feature] = f'sum_aggregation_{len(available_features)}_features'
                
                # Mean aggregation
                mean_feature = f'{group_name}_mean'
                df_enhanced[mean_feature] = df_enhanced[available_features].mean(axis=1)
                self.transformations[mean_feature] = f'mean_aggregation_{len(available_features)}_features'
                
                # Max aggregation
                max_feature = f'{group_name}_max'
                df_enhanced[max_feature] = df_enhanced[available_features].max(axis=1)
                self.transformations[max_feature] = f'max_aggregation_{len(available_features)}_features'
                
                self.logger.info(f"Created aggregated features for {group_name} group")
        
        return df_enhanced
    
    def create_ratio_features(self, df: pd.DataFrame, numerator_features: List[str], 
                            denominator_features: List[str]) -> pd.DataFrame:
        """
        Create ratio features between numerator and denominator features.
        
        Args:
            df: DataFrame with features
            numerator_features: List of numerator features
            denominator_features: List of denominator features
            
        Returns:
            DataFrame with ratio features
        """
        df_enhanced = df.copy()
        
        for num_feature in numerator_features:
            for den_feature in denominator_features:
                if num_feature in df_enhanced.columns and den_feature in df_enhanced.columns:
                    ratio_feature = f'{num_feature}_per_{den_feature}'
                    df_enhanced[ratio_feature] = (
                        df_enhanced[num_feature] / (df_enhanced[den_feature] + 1e-8)
                    )
                    self.transformations[ratio_feature] = 'ratio_feature'
                    self.logger.info(f"Created ratio feature: {ratio_feature}")
        
        return df_enhanced
    
    def create_temporal_features(self, df: pd.DataFrame, date_columns: List[str]) -> pd.DataFrame:
        """
        Create temporal features from date columns.
        
        Args:
            df: DataFrame with features
            date_columns: List of date column names
            
        Returns:
            DataFrame with temporal features
        """
        df_enhanced = df.copy()
        
        for date_col in date_columns:
            if date_col in df_enhanced.columns:
                # Convert to datetime if not already
                df_enhanced[date_col] = pd.to_datetime(df_enhanced[date_col], errors='coerce')
                
                # Extract temporal components
                df_enhanced[f'{date_col}_year'] = df_enhanced[date_col].dt.year
                df_enhanced[f'{date_col}_month'] = df_enhanced[date_col].dt.month
                df_enhanced[f'{date_col}_day'] = df_enhanced[date_col].dt.day
                df_enhanced[f'{date_col}_dayofweek'] = df_enhanced[date_col].dt.dayofweek
                df_enhanced[f'{date_col}_quarter'] = df_enhanced[date_col].dt.quarter
                
                # Days since reference date
                reference_date = df_enhanced[date_col].min()
                df_enhanced[f'{date_col}_days_since'] = (
                    df_enhanced[date_col] - reference_date
                ).dt.days
                
                # Mark temporal features
                temporal_features = [
                    f'{date_col}_year', f'{date_col}_month', f'{date_col}_day',
                    f'{date_col}_dayofweek', f'{date_col}_quarter', f'{date_col}_days_since'
                ]
                
                for feature in temporal_features:
                    self.transformations[feature] = 'temporal_extraction'
                
                self.logger.info(f"Created temporal features for {date_col}")
        
        return df_enhanced
    
    def apply_automatic_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply automatic transformations based on data characteristics.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with automatically applied transformations
        """
        df_transformed = df.copy()
        
        # Get numerical columns
        numerical_cols = df_transformed.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'customer_id']
        
        # Identify skewed features for log transformation
        skewed_features = []
        for col in numerical_cols:
            if df_transformed[col].skew() > 1:  # Right-skewed
                skewed_features.append(col)
        
        if skewed_features:
            df_transformed = self.log_transform_features(df_transformed, skewed_features)
        
        # Create common interaction features
        if 'monthly_charges' in df_transformed.columns and 'tenure' in df_transformed.columns:
            interaction_pairs = [('monthly_charges', 'tenure')]
            df_transformed = self.create_interaction_features(df_transformed, interaction_pairs)
        
        # Create feature groups for aggregation
        service_features = [col for col in df_transformed.columns if 'service' in col.lower()]
        if len(service_features) > 1:
            feature_groups = {'services': service_features}
            df_transformed = self.create_aggregated_features(df_transformed, feature_groups)
        
        self.logger.info("Applied automatic transformations")
        return df_transformed
    
    def get_transformation_summary(self) -> pd.DataFrame:
        """
        Get summary of all applied transformations.
        
        Returns:
            DataFrame with transformation summary
        """
        summary_data = [
            {'feature': feature, 'transformation': transformation}
            for feature, transformation in self.transformations.items()
        ]
        
        return pd.DataFrame(summary_data)
