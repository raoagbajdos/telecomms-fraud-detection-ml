"""
Data processor for cleaning and preprocessing telecoms data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import logging

class DataProcessor:
    """Handles data cleaning and preprocessing for telecoms fraud detection."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data processor.
        
        Args:
            config: Configuration dictionary containing processing parameters
        """
        self.config = config or {}
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        self.logger = logging.getLogger(__name__)
        
    def clean_billing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean billing data by handling missing values, outliers, and data types.
        
        Args:
            df: Raw billing DataFrame
            
        Returns:
            Cleaned billing DataFrame
        """
        df_clean = df.copy()
        
        # Convert data types
        if 'total_charges' in df_clean.columns:
            df_clean['total_charges'] = pd.to_numeric(df_clean['total_charges'], errors='coerce')
        
        if 'monthly_charges' in df_clean.columns:
            df_clean['monthly_charges'] = pd.to_numeric(df_clean['monthly_charges'], errors='coerce')
            
        # Handle missing values in total_charges
        if 'total_charges' in df_clean.columns and df_clean['total_charges'].isnull().any():
            # Impute missing total_charges based on monthly_charges and tenure
            if 'monthly_charges' in df_clean.columns and 'tenure' in df_clean.columns:
                mask = df_clean['total_charges'].isnull()
                df_clean.loc[mask, 'total_charges'] = (
                    df_clean.loc[mask, 'monthly_charges'] * df_clean.loc[mask, 'tenure']
                )
        
        # Remove outliers in charges
        for col in ['monthly_charges', 'total_charges']:
            if col in df_clean.columns:
                df_clean = self._remove_outliers(df_clean, col)
        
        # Clean categorical variables
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'customer_id':
                df_clean[col] = df_clean[col].str.strip().str.title()
                
        self.logger.info(f"Cleaned billing data: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        return df_clean
    
    def clean_crm_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean CRM data.
        
        Args:
            df: Raw CRM DataFrame
            
        Returns:
            Cleaned CRM DataFrame
        """
        df_clean = df.copy()
        
        # Standardize gender values
        if 'gender' in df_clean.columns:
            df_clean['gender'] = df_clean['gender'].map({
                'M': 'Male', 'F': 'Female', 'Male': 'Male', 'Female': 'Female'
            })
            
        # Clean Yes/No columns
        yes_no_cols = ['partner', 'dependents']
        for col in yes_no_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
        
        # Handle senior citizen
        if 'senior_citizen' in df_clean.columns:
            df_clean['senior_citizen'] = df_clean['senior_citizen'].astype(int)
            
        self.logger.info(f"Cleaned CRM data: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        return df_clean
    
    def clean_social_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean social media and engagement data.
        
        Args:
            df: Raw social DataFrame
            
        Returns:
            Cleaned social DataFrame
        """
        df_clean = df.copy()
        
        # Handle missing values in engagement metrics
        engagement_cols = ['social_media_engagement', 'app_usage_minutes', 'website_visits']
        for col in engagement_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        # Remove outliers in engagement metrics
        for col in engagement_cols:
            if col in df_clean.columns:
                df_clean = self._remove_outliers(df_clean, col)
                
        self.logger.info(f"Cleaned social data: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        return df_clean
    
    def clean_customer_care_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean customer care interaction data.
        
        Args:
            df: Raw customer care DataFrame
            
        Returns:
            Cleaned customer care DataFrame
        """
        df_clean = df.copy()
        
        # Convert interaction dates
        if 'last_interaction_date' in df_clean.columns:
            df_clean['last_interaction_date'] = pd.to_datetime(
                df_clean['last_interaction_date'], errors='coerce'
            )
        
        # Handle missing values in support metrics
        support_cols = ['support_calls', 'complaints', 'satisfaction_score']
        for col in support_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col].fillna(0, inplace=True)
                
        # Clean interaction types
        if 'interaction_type' in df_clean.columns:
            df_clean['interaction_type'] = df_clean['interaction_type'].str.strip().str.title()
            
        self.logger.info(f"Cleaned customer care data: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        return df_clean
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using appropriate encoding methods.
        
        Args:
            df: DataFrame with categorical features
            fit: Whether to fit encoders or use existing ones
            
        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()
        
        # Get categorical columns (excluding customer_id)
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'customer_id']
        
        for col in categorical_cols:
            if col not in self.encoders and fit:
                # Use LabelEncoder for binary categories, OneHotEncoder for multi-class
                unique_values = df_encoded[col].nunique()
                
                if unique_values == 2:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[col] = self.encoders[col].fit_transform(df_encoded[col].astype(str))
                else:
                    self.encoders[col] = OneHotEncoder(sparse_output=False, drop='first')
                    encoded_data = self.encoders[col].fit_transform(df_encoded[[col]])
                    feature_names = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0][1:]]
                    
                    # Add encoded columns and drop original
                    for i, feature_name in enumerate(feature_names):
                        df_encoded[feature_name] = encoded_data[:, i]
                    df_encoded.drop(col, axis=1, inplace=True)
                    
            elif col in self.encoders:
                if isinstance(self.encoders[col], LabelEncoder):
                    df_encoded[col] = self.encoders[col].transform(df_encoded[col].astype(str))
                else:
                    encoded_data = self.encoders[col].transform(df_encoded[[col]])
                    feature_names = [f"{col}_{cat}" for cat in self.encoders[col].categories_[0][1:]]
                    
                    for i, feature_name in enumerate(feature_names):
                        df_encoded[feature_name] = encoded_data[:, i]
                    df_encoded.drop(col, axis=1, inplace=True)
        
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: DataFrame with numerical features
            fit: Whether to fit scaler or use existing one
            
        Returns:
            DataFrame with scaled features
        """
        df_scaled = df.copy()
        
        # Get numerical columns (excluding customer_id)
        numerical_cols = df_scaled.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != 'customer_id']
        
        if fit:
            self.scalers['standard'] = StandardScaler()
            df_scaled[numerical_cols] = self.scalers['standard'].fit_transform(df_scaled[numerical_cols])
        else:
            if 'standard' in self.scalers:
                df_scaled[numerical_cols] = self.scalers['standard'].transform(df_scaled[numerical_cols])
        
        return df_scaled
    
    def _remove_outliers(self, df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove outliers from a specific column using IQR method.
        
        Args:
            df: DataFrame
            column: Column name to remove outliers from
            method: Method to use for outlier detection
            
        Returns:
            DataFrame with outliers removed
        """
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
            return df[mask]
        
        return df
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about processed features.
        
        Returns:
            Dictionary containing feature processing information
        """
        return {
            'scalers': list(self.scalers.keys()),
            'encoders': list(self.encoders.keys()),
            'imputers': list(self.imputers.keys())
        }
