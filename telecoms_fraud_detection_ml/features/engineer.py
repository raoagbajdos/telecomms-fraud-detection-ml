"""
Feature engineer for creating fraud detection features from telecoms data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

class FeatureEngineer:
    """Creates and engineers features for telecoms fraud detection."""
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.logger = logging.getLogger(__name__)
        self.created_features = []
        
    def create_fraud_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive fraud detection features.
        
        Args:
            df: Input DataFrame with telecoms data
            
        Returns:
            DataFrame with additional fraud detection features
        """
        df_enhanced = df.copy()
        
        # Usage pattern features
        df_enhanced = self._create_usage_features(df_enhanced)
        
        # Billing anomaly features
        df_enhanced = self._create_billing_features(df_enhanced)
        
        # Customer behavior features
        df_enhanced = self._create_behavior_features(df_enhanced)
        
        # Service pattern features
        df_enhanced = self._create_service_features(df_enhanced)
        
        # Temporal features
        df_enhanced = self._create_temporal_features(df_enhanced)
        
        # Risk scoring features
        df_enhanced = self._create_risk_features(df_enhanced)
        
        self.logger.info(f"Created {len(self.created_features)} fraud detection features")
        return df_enhanced
    
    def _create_usage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create usage pattern features that may indicate fraud."""
        
        # Data usage anomalies
        if 'monthly_data_gb' in df.columns:
            # Extreme usage patterns
            df['is_extreme_data_user'] = (
                (df['monthly_data_gb'] > df['monthly_data_gb'].quantile(0.95)) |
                (df['monthly_data_gb'] < df['monthly_data_gb'].quantile(0.05))
            ).astype(int)
            self.created_features.append('is_extreme_data_user')
            
            # Usage-to-payment ratio
            if 'monthly_charges' in df.columns:
                df['data_per_dollar'] = df['monthly_data_gb'] / (df['monthly_charges'] + 1)
                self.created_features.append('data_per_dollar')
        
        # Call pattern anomalies
        if 'support_calls' in df.columns and 'tenure' in df.columns:
            df['calls_per_month'] = df['support_calls'] / (df['tenure'] + 1)
            df['is_frequent_caller'] = (
                df['calls_per_month'] > df['calls_per_month'].quantile(0.9)
            ).astype(int)
            self.created_features.extend(['calls_per_month', 'is_frequent_caller'])
        
        return df
    
    def _create_billing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create billing-related fraud indicators."""
        
        # Payment behavior
        if 'payment_method' in df.columns:
            # Electronic payments might be lower risk
            df['uses_electronic_payment'] = (
                df['payment_method'].str.contains('Electronic|Credit', case=False, na=False)
            ).astype(int)
            self.created_features.append('uses_electronic_payment')
        
        # Charges consistency
        if 'monthly_charges' in df.columns and 'total_charges' in df.columns and 'tenure' in df.columns:
            # Calculate expected total based on monthly and tenure
            df['expected_total_charges'] = df['monthly_charges'] * df['tenure']
            df['charges_discrepancy'] = abs(
                df['total_charges'] - df['expected_total_charges']
            ) / (df['expected_total_charges'] + 1)
            
            # Flag significant discrepancies
            df['has_billing_anomaly'] = (
                df['charges_discrepancy'] > df['charges_discrepancy'].quantile(0.95)
            ).astype(int)
            
            self.created_features.extend([
                'expected_total_charges', 'charges_discrepancy', 'has_billing_anomaly'
            ])
        
        # Contract risk factors
        if 'contract' in df.columns:
            # Month-to-month contracts might be higher risk
            df['is_month_to_month'] = (
                df['contract'].str.contains('Month', case=False, na=False)
            ).astype(int)
            self.created_features.append('is_month_to_month')
        
        return df
    
    def _create_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create customer behavior features."""
        
        # Service adoption patterns
        service_columns = [col for col in df.columns if 'service' in col.lower()]
        if service_columns:
            # Convert service columns to numeric, treating Yes/No as 1/0
            for col in service_columns:
                if df[col].dtype == 'object':
                    df[col] = (df[col] == 'Yes').astype(int)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            df['total_services'] = df[service_columns].sum(axis=1)
            df['service_adoption_rate'] = df['total_services'] / len(service_columns)
            self.created_features.extend(['total_services', 'service_adoption_rate'])
        
        # Digital engagement
        digital_features = []
        if 'online_security' in df.columns:
            digital_features.append('online_security')
        if 'online_backup' in df.columns:
            digital_features.append('online_backup')
        if 'paperless_billing' in df.columns:
            digital_features.append('paperless_billing')
        
        if digital_features:
            # Convert Yes/No to 1/0 for calculation
            binary_features = []
            for feature in digital_features:
                if df[feature].dtype == 'object':
                    binary_col = f'{feature}_binary'
                    df[binary_col] = (df[feature] == 'Yes').astype(int)
                    binary_features.append(binary_col)
                    
            if binary_features:
                df['digital_engagement_score'] = df[binary_features].mean(axis=1)
                self.created_features.append('digital_engagement_score')
        
        # Customer loyalty indicators
        if 'tenure' in df.columns:
            df['is_new_customer'] = (df['tenure'] <= 6).astype(int)
            df['is_loyal_customer'] = (df['tenure'] >= 24).astype(int)
            self.created_features.extend(['is_new_customer', 'is_loyal_customer'])
        
        return df
    
    def _create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create service-related features."""
        
        # Premium service indicators
        premium_indicators = []
        if 'internet_service' in df.columns:
            # Ensure internet_service is string type
            df['internet_service'] = df['internet_service'].astype(str)
            df['has_fiber_internet'] = (
                df['internet_service'].str.contains('Fiber', case=False, na=False)
            ).astype(int)
            premium_indicators.append('has_fiber_internet')
        
        if 'streaming_tv' in df.columns:
            df['has_streaming_tv'] = (df['streaming_tv'] == 'Yes').astype(int)
            premium_indicators.append('has_streaming_tv')
        
        if 'streaming_movies' in df.columns:
            df['has_streaming_movies'] = (df['streaming_movies'] == 'Yes').astype(int)
            premium_indicators.append('has_streaming_movies')
        
        if premium_indicators:
            df['premium_service_count'] = df[premium_indicators].sum(axis=1)
            self.created_features.extend(premium_indicators + ['premium_service_count'])
        
        # Service complexity score
        if 'multiple_lines' in df.columns and 'phone_service' in df.columns:
            df['service_complexity'] = 0
            
            # Add complexity for each service
            if 'phone_service' in df.columns:
                df['service_complexity'] += (df['phone_service'] == 'Yes').astype(int)
            if 'multiple_lines' in df.columns:
                df['service_complexity'] += (df['multiple_lines'] == 'Yes').astype(int)
            if 'internet_service' in df.columns:
                df['service_complexity'] += (df['internet_service'] != 'No').astype(int)
            
            self.created_features.append('service_complexity')
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        
        # Tenure-based features
        if 'tenure' in df.columns:
            # Categorize tenure
            df['tenure_category'] = pd.cut(
                df['tenure'], 
                bins=[0, 6, 12, 24, 48, float('inf')],
                labels=['New', 'Short', 'Medium', 'Long', 'Very_Long']
            )
            
            # Tenure stability score
            df['tenure_stability'] = np.log1p(df['tenure'])  # Log transform for stability
            self.created_features.extend(['tenure_category', 'tenure_stability'])
        
        # Customer lifecycle stage
        if 'tenure' in df.columns and 'monthly_charges' in df.columns:
            # Early adopter vs late adopter patterns
            df['charges_per_tenure_month'] = df['monthly_charges'] / (df['tenure'] + 1)
            self.created_features.append('charges_per_tenure_month')
        
        return df
    
    def _create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite risk features."""
        
        # Payment risk score
        risk_factors = []
        
        if 'is_month_to_month' in df.columns:
            risk_factors.append('is_month_to_month')
        if 'is_new_customer' in df.columns:
            risk_factors.append('is_new_customer')
        if 'has_billing_anomaly' in df.columns:
            risk_factors.append('has_billing_anomaly')
        if 'is_frequent_caller' in df.columns:
            risk_factors.append('is_frequent_caller')
        
        if risk_factors:
            df['payment_risk_score'] = df[risk_factors].sum(axis=1) / len(risk_factors)
            self.created_features.append('payment_risk_score')
        
        # Usage risk score
        usage_risk_factors = []
        if 'is_extreme_data_user' in df.columns:
            usage_risk_factors.append('is_extreme_data_user')
        if 'charges_discrepancy' in df.columns:
            # Normalize discrepancy to 0-1 scale
            max_discrepancy = df['charges_discrepancy'].max()
            if max_discrepancy > 0:
                df['normalized_charges_discrepancy'] = df['charges_discrepancy'] / max_discrepancy
                usage_risk_factors.append('normalized_charges_discrepancy')
        
        if usage_risk_factors:
            df['usage_risk_score'] = df[usage_risk_factors].mean(axis=1)
            self.created_features.append('usage_risk_score')
        
        # Overall fraud risk score
        risk_components = []
        if 'payment_risk_score' in df.columns:
            risk_components.append('payment_risk_score')
        if 'usage_risk_score' in df.columns:
            risk_components.append('usage_risk_score')
        
        if risk_components:
            df['overall_fraud_risk'] = df[risk_components].mean(axis=1)
            self.created_features.append('overall_fraud_risk')
        
        return df
    
    def get_feature_importance_proxy(self, df: pd.DataFrame, target_col: str = 'is_fraud') -> pd.DataFrame:
        """
        Calculate feature importance proxy using correlation with target.
        
        Args:
            df: DataFrame with features and target
            target_col: Name of the target column
            
        Returns:
            DataFrame with feature importance scores
        """
        if target_col not in df.columns:
            self.logger.warning(f"Target column '{target_col}' not found")
            return pd.DataFrame()
        
        # Calculate correlations for numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != target_col]
        
        correlations = []
        for col in numerical_cols:
            corr = abs(df[col].corr(df[target_col]))
            correlations.append({
                'feature': col,
                'importance_proxy': corr,
                'is_created_feature': col in self.created_features
            })
        
        importance_df = pd.DataFrame(correlations).sort_values(
            'importance_proxy', ascending=False
        )
        
        return importance_df
    
    def get_created_features_list(self) -> List[str]:
        """
        Get list of all created features.
        
        Returns:
            List of created feature names
        """
        return self.created_features.copy()
