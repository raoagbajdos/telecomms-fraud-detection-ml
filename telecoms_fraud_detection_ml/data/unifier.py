"""
Data unifier for combining multiple telecoms data sources.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

class DataUnifier:
    """Combines and unifies multiple telecoms data sources."""
    
    def __init__(self, customer_id_col: str = 'customer_id'):
        """
        Initialize the data unifier.
        
        Args:
            customer_id_col: Name of the customer ID column for joining
        """
        self.customer_id_col = customer_id_col
        self.logger = logging.getLogger(__name__)
        
    def unify_datasets(self, 
                      billing_df: pd.DataFrame,
                      crm_df: pd.DataFrame,
                      social_df: Optional[pd.DataFrame] = None,
                      customer_care_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Unify multiple datasets into a single DataFrame.
        
        Args:
            billing_df: Billing data DataFrame
            crm_df: CRM data DataFrame
            social_df: Social media data DataFrame (optional)
            customer_care_df: Customer care data DataFrame (optional)
            
        Returns:
            Unified DataFrame
        """
        self.logger.info("Starting data unification process...")
        
        # Start with billing data as the base
        unified_df = billing_df.copy()
        
        # Merge with CRM data
        unified_df = self._safe_merge(unified_df, crm_df, "CRM")
        
        # Merge with social data if provided
        if social_df is not None:
            unified_df = self._safe_merge(unified_df, social_df, "Social")
            
        # Merge with customer care data if provided
        if customer_care_df is not None:
            unified_df = self._safe_merge(unified_df, customer_care_df, "Customer Care")
        
        # Create additional unified features
        unified_df = self._create_unified_features(unified_df)
        
        self.logger.info(f"Data unification completed. Final shape: {unified_df.shape}")
        return unified_df
    
    def _safe_merge(self, left_df: pd.DataFrame, right_df: pd.DataFrame, 
                   data_source: str) -> pd.DataFrame:
        """
        Safely merge two DataFrames with error handling.
        
        Args:
            left_df: Left DataFrame
            right_df: Right DataFrame
            data_source: Name of the data source being merged
            
        Returns:
            Merged DataFrame
        """
        try:
            # Check if customer_id column exists in both DataFrames
            if self.customer_id_col not in left_df.columns:
                self.logger.error(f"Customer ID column '{self.customer_id_col}' not found in left DataFrame")
                return left_df
                
            if self.customer_id_col not in right_df.columns:
                self.logger.error(f"Customer ID column '{self.customer_id_col}' not found in {data_source} DataFrame")
                return left_df
            
            # Perform the merge
            merged_df = pd.merge(left_df, right_df, on=self.customer_id_col, how='left')
            
            self.logger.info(f"Successfully merged {data_source} data. "
                           f"Shape before: {left_df.shape}, after: {merged_df.shape}")
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error merging {data_source} data: {str(e)}")
            return left_df
    
    def _create_unified_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features from the unified dataset.
        
        Args:
            df: Unified DataFrame
            
        Returns:
            DataFrame with additional features
        """
        df_enhanced = df.copy()
        
        # Customer lifetime value estimation
        if 'monthly_charges' in df_enhanced.columns and 'tenure' in df_enhanced.columns:
            df_enhanced['customer_lifetime_value'] = (
                df_enhanced['monthly_charges'] * df_enhanced['tenure']
            )
        
        # Service intensity score
        service_cols = [col for col in df_enhanced.columns if 'service' in col.lower()]
        if service_cols:
            df_enhanced['service_intensity'] = df_enhanced[service_cols].sum(axis=1)
        
        # Customer engagement score
        engagement_features = []
        if 'social_media_engagement' in df_enhanced.columns:
            engagement_features.append('social_media_engagement')
        if 'app_usage_minutes' in df_enhanced.columns:
            engagement_features.append('app_usage_minutes')
        if 'website_visits' in df_enhanced.columns:
            engagement_features.append('website_visits')
            
        if engagement_features:
            # Normalize each feature to 0-1 scale and take mean
            for feature in engagement_features:
                max_val = df_enhanced[feature].max()
                if max_val > 0:
                    df_enhanced[f'{feature}_normalized'] = df_enhanced[feature] / max_val
            
            normalized_cols = [f'{feature}_normalized' for feature in engagement_features]
            df_enhanced['customer_engagement_score'] = df_enhanced[normalized_cols].mean(axis=1)
        
        # Support interaction ratio
        if 'support_calls' in df_enhanced.columns and 'tenure' in df_enhanced.columns:
            df_enhanced['support_calls_per_month'] = (
                df_enhanced['support_calls'] / (df_enhanced['tenure'] + 1)  # +1 to avoid division by zero
            )
        
        # High-value customer indicator
        if 'monthly_charges' in df_enhanced.columns:
            high_value_threshold = df_enhanced['monthly_charges'].quantile(0.8)
            df_enhanced['is_high_value_customer'] = (
                df_enhanced['monthly_charges'] >= high_value_threshold
            ).astype(int)
        
        return df_enhanced
    
    def validate_unified_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate the unified dataset and provide quality metrics.
        
        Args:
            df: Unified DataFrame
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_customers': df.duplicated(subset=[self.customer_id_col]).sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Check for missing customer IDs
        missing_customer_ids = df[self.customer_id_col].isnull().sum()
        validation_results['missing_customer_ids'] = missing_customer_ids
        
        # Calculate completeness score
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        validation_results['completeness_score'] = 1 - (missing_cells / total_cells)
        
        self.logger.info(f"Data validation completed. Completeness score: "
                        f"{validation_results['completeness_score']:.3f}")
        
        return validation_results
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get a summary of features in the unified dataset.
        
        Args:
            df: Unified DataFrame
            
        Returns:
            DataFrame containing feature summary
        """
        summary_data = []
        
        for col in df.columns:
            if col == self.customer_id_col:
                continue
                
            col_info = {
                'feature_name': col,
                'data_type': str(df[col].dtype),
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_values': df[col].nunique()
            }
            
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                })
            
            summary_data.append(col_info)
        
        return pd.DataFrame(summary_data)
