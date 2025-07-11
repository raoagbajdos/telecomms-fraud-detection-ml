"""
Data helpers and utility functions for telecoms fraud detection.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Optional, Any, Union
import json

class DataHelpers:
    """Helper functions for data operations."""
    
    @staticmethod
    def load_data_from_file(filepath: str, **kwargs) -> Optional[pd.DataFrame]:
        """
        Load data from various file formats.
        
        Args:
            filepath: Path to the data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            DataFrame or None if loading fails
        """
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None
        
        try:
            file_ext = os.path.splitext(filepath)[1].lower()
            
            if file_ext == '.csv':
                return pd.read_csv(filepath, **kwargs)
            elif file_ext in ['.xlsx', '.xls']:
                return pd.read_excel(filepath, **kwargs)
            elif file_ext == '.json':
                return pd.read_json(filepath, **kwargs)
            elif file_ext == '.parquet':
                return pd.read_parquet(filepath, **kwargs)
            else:
                print(f"Unsupported file format: {file_ext}")
                return None
                
        except Exception as e:
            print(f"Error loading data from {filepath}: {str(e)}")
            return None
    
    @staticmethod
    def save_data_to_file(df: pd.DataFrame, filepath: str, **kwargs) -> bool:
        """
        Save DataFrame to various file formats.
        
        Args:
            df: DataFrame to save
            filepath: Path to save the file
            **kwargs: Additional arguments for pandas write functions
            
        Returns:
            True if saving was successful
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            file_ext = os.path.splitext(filepath)[1].lower()
            
            if file_ext == '.csv':
                df.to_csv(filepath, index=False, **kwargs)
            elif file_ext in ['.xlsx', '.xls']:
                df.to_excel(filepath, index=False, **kwargs)
            elif file_ext == '.json':
                df.to_json(filepath, **kwargs)
            elif file_ext == '.parquet':
                df.to_parquet(filepath, **kwargs)
            else:
                print(f"Unsupported file format: {file_ext}")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error saving data to {filepath}: {str(e)}")
            return False
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive summary of DataFrame.
        
        Args:
            df: DataFrame to summarize
            
        Returns:
            Dictionary containing data summary
        """
        summary = {
            'shape': df.shape,
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Add statistics for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            summary['numerical_stats'] = df[numerical_cols].describe().to_dict()
        
        # Add statistics for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            summary['categorical_stats'] = {}
            for col in categorical_cols:
                summary['categorical_stats'][col] = {
                    'unique_count': df[col].nunique(),
                    'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                    'unique_values': df[col].unique().tolist()[:10]  # First 10 unique values
                }
        
        return summary
    
    @staticmethod
    def detect_data_quality_issues(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data quality issues in DataFrame.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing detected issues
        """
        issues = {
            'missing_data': DataHelpers._analyze_missing_data(df),
            'duplicates': DataHelpers._analyze_duplicates(df),
            'outliers': DataHelpers._analyze_outliers(df),
            'data_types': DataHelpers._analyze_data_types(df),
            'consistency': {}
        }
        
        return issues
    
    @staticmethod
    def _analyze_missing_data(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_counts = df.isnull().sum()
        missing_percentage = missing_counts / len(df) * 100
        
        return {
            'columns_with_missing': missing_counts[missing_counts > 0].to_dict(),
            'high_missing_columns': missing_counts[missing_percentage > 50].to_dict(),
            'total_missing_cells': int(missing_counts.sum())
        }
    
    @staticmethod
    def _analyze_duplicates(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate data patterns."""
        duplicate_rows = df.duplicated().sum()
        duplicates = {
            'total_duplicate_rows': int(duplicate_rows),
            'duplicate_percentage': float(duplicate_rows / len(df) * 100)
        }
        
        # Check for potential ID columns with duplicates
        for col in df.columns:
            if 'id' in col.lower() and df[col].duplicated().any():
                duplicates[f'{col}_duplicates'] = int(df[col].duplicated().sum())
        
        return duplicates
    
    @staticmethod
    def _analyze_outliers(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in numerical columns."""
        outliers = {}
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                
                if outlier_count > 0:
                    outliers[col] = {
                        'count': int(outlier_count),
                        'percentage': float(outlier_count / len(df) * 100)
                    }
        
        return outliers
    
    @staticmethod
    def _analyze_data_types(df: pd.DataFrame) -> Dict[str, str]:
        """Analyze data type issues."""
        data_type_issues = {}
        
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric data is stored as text
                try:
                    pd.to_numeric(df[col], errors='raise')
                    data_type_issues[col] = 'numeric_stored_as_text'
                except (ValueError, TypeError):
                    pass
        
        return data_type_issues
    
    @staticmethod
    def create_fraud_labels(df: pd.DataFrame, 
                           rules: Dict[str, Any] = None) -> pd.Series:
        """
        Create fraud labels based on business rules.
        
        Args:
            df: DataFrame with customer data
            rules: Dictionary containing fraud detection rules
            
        Returns:
            Series with fraud labels (0 = normal, 1 = fraud)
        """
        if rules is None:
            # Default rules for demonstration
            rules = {
                'high_charges_low_tenure': {
                    'monthly_charges_threshold': 100,
                    'tenure_threshold': 3
                },
                'extreme_usage': {
                    'data_usage_percentile': 95
                },
                'frequent_support_calls': {
                    'calls_per_month_threshold': 5
                }
            }
        
        fraud_flags = pd.Series(0, index=df.index)
        
        # Rule 1: High charges with low tenure
        if 'monthly_charges' in df.columns and 'tenure' in df.columns:
            high_charges_low_tenure = (
                (df['monthly_charges'] > rules['high_charges_low_tenure']['monthly_charges_threshold']) &
                (df['tenure'] < rules['high_charges_low_tenure']['tenure_threshold'])
            )
            fraud_flags |= high_charges_low_tenure.astype(int)
        
        # Rule 2: Extreme data usage
        if 'monthly_data_gb' in df.columns:
            usage_threshold = df['monthly_data_gb'].quantile(
                rules['extreme_usage']['data_usage_percentile'] / 100
            )
            extreme_usage = df['monthly_data_gb'] > usage_threshold
            fraud_flags |= extreme_usage.astype(int)
        
        # Rule 3: Frequent support calls
        if 'support_calls' in df.columns and 'tenure' in df.columns:
            calls_per_month = df['support_calls'] / (df['tenure'] + 1)
            frequent_calls = calls_per_month > rules['frequent_support_calls']['calls_per_month_threshold']
            fraud_flags |= frequent_calls.astype(int)
        
        return fraud_flags
    
    @staticmethod
    def balance_dataset(df: pd.DataFrame, 
                       target_column: str, 
                       method: str = 'undersample') -> pd.DataFrame:
        """
        Balance dataset for fraud detection.
        
        Args:
            df: DataFrame with imbalanced data
            target_column: Name of the target column
            method: Balancing method ('undersample', 'oversample', or 'smote')
            
        Returns:
            Balanced DataFrame
        """
        if target_column not in df.columns:
            print(f"Target column '{target_column}' not found")
            return df
        
        fraud_cases = df[df[target_column] == 1]
        normal_cases = df[df[target_column] == 0]
        
        if method == 'undersample':
            # Undersample majority class
            min_class_size = min(len(fraud_cases), len(normal_cases))
            
            fraud_sample = fraud_cases.sample(n=min_class_size, random_state=42)
            normal_sample = normal_cases.sample(n=min_class_size, random_state=42)
            
            balanced_df = pd.concat([fraud_sample, normal_sample], ignore_index=True)
            
        elif method == 'oversample':
            # Oversample minority class
            max_class_size = max(len(fraud_cases), len(normal_cases))
            
            if len(fraud_cases) < max_class_size:
                fraud_sample = fraud_cases.sample(n=max_class_size, replace=True, random_state=42)
                normal_sample = normal_cases
            else:
                fraud_sample = fraud_cases
                normal_sample = normal_cases.sample(n=max_class_size, replace=True, random_state=42)
            
            balanced_df = pd.concat([fraud_sample, normal_sample], ignore_index=True)
            
        else:
            print(f"Unsupported balancing method: {method}")
            return df
        
        # Shuffle the balanced dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return balanced_df
    
    @staticmethod
    def split_by_date(df: pd.DataFrame, 
                     date_column: str, 
                     split_date: str) -> tuple:
        """
        Split DataFrame by date for time-based validation.
        
        Args:
            df: DataFrame to split
            date_column: Name of the date column
            split_date: Date string to split on
            
        Returns:
            Tuple of (before_split, after_split) DataFrames
        """
        if date_column not in df.columns:
            print(f"Date column '{date_column}' not found")
            return df, pd.DataFrame()
        
        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        split_datetime = pd.to_datetime(split_date)
        
        before_split = df[df[date_column] < split_datetime].copy()
        after_split = df[df[date_column] >= split_datetime].copy()
        
        return before_split, after_split
    
    @staticmethod
    def export_to_json(data: Dict[str, Any], filepath: str) -> bool:
        """
        Export dictionary to JSON file.
        
        Args:
            data: Dictionary to export
            filepath: Path to save JSON file
            
        Returns:
            True if export was successful
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            print(f"Error exporting to JSON: {str(e)}")
            return False
    
    @staticmethod
    def load_from_json(filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load dictionary from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Dictionary or None if loading fails
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            print(f"Error loading from JSON: {str(e)}")
            return None
    
    @staticmethod
    def calculate_correlation_matrix(df: pd.DataFrame, 
                                   method: str = 'pearson') -> pd.DataFrame:
        """
        Calculate correlation matrix for numerical columns.
        
        Args:
            df: DataFrame to analyze
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix DataFrame
        """
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) == 0:
            print("No numerical columns found for correlation analysis")
            return pd.DataFrame()
        
        return df[numerical_cols].corr(method=method)
