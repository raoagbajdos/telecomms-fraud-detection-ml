"""
Data validator for telecoms fraud detection datasets.
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

class DataValidator:
    """Validates data quality and integrity for telecoms fraud detection."""
    
    def __init__(self):
        """Initialize the data validator."""
        self.logger = logging.getLogger(__name__)
        
    def validate_billing_data(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate billing data for common issues.
        
        Args:
            df: Billing DataFrame
            
        Returns:
            Dictionary containing validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Check required columns
        required_cols = ['customer_id', 'monthly_charges', 'total_charges', 'tenure']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results['errors'].append(f"Missing required columns: {missing_cols}")
            results['valid'] = False
        
        # Check for negative charges
        if 'monthly_charges' in df.columns:
            negative_monthly = (df['monthly_charges'] < 0).sum()
            if negative_monthly > 0:
                results['warnings'].append(f"Found {negative_monthly} negative monthly charges")
                
        if 'total_charges' in df.columns:
            negative_total = (df['total_charges'] < 0).sum()
            if negative_total > 0:
                results['warnings'].append(f"Found {negative_total} negative total charges")
        
        # Check for unrealistic tenure values
        if 'tenure' in df.columns:
            unrealistic_tenure = (df['tenure'] < 0).sum() + (df['tenure'] > 100).sum()
            if unrealistic_tenure > 0:
                results['warnings'].append(f"Found {unrealistic_tenure} unrealistic tenure values")
        
        # Calculate metrics
        results['metrics'] = {
            'total_records': len(df),
            'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_customers': df.duplicated(subset=['customer_id']).sum() if 'customer_id' in df.columns else 0
        }
        
        return results
    
    def validate_fraud_labels(self, df: pd.DataFrame, target_col: str = 'is_fraud') -> Dict[str, any]:
        """
        Validate fraud labels in the dataset.
        
        Args:
            df: DataFrame containing fraud labels
            target_col: Name of the target column
            
        Returns:
            Dictionary containing validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }
        
        if target_col not in df.columns:
            results['errors'].append(f"Target column '{target_col}' not found")
            results['valid'] = False
            return results
        
        # Check label distribution
        label_counts = df[target_col].value_counts()
        fraud_rate = label_counts.get(1, 0) / len(df) if len(df) > 0 else 0
        
        results['metrics'] = {
            'fraud_rate': fraud_rate,
            'total_fraud_cases': label_counts.get(1, 0),
            'total_normal_cases': label_counts.get(0, 0),
            'label_distribution': label_counts.to_dict()
        }
        
        # Check for class imbalance
        if fraud_rate < 0.01:
            results['warnings'].append(f"Very low fraud rate: {fraud_rate:.4f}. Consider sampling strategies.")
        elif fraud_rate > 0.5:
            results['warnings'].append(f"High fraud rate: {fraud_rate:.4f}. Verify data quality.")
        
        # Check for missing labels
        missing_labels = df[target_col].isnull().sum()
        if missing_labels > 0:
            results['warnings'].append(f"Found {missing_labels} missing fraud labels")
            
        return results
    
    def validate_feature_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate feature quality across the dataset.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary containing feature quality results
        """
        results = {
            'features': {},
            'overall_quality_score': 0.0,
            'recommendations': []
        }
        
        total_quality_score = 0
        feature_count = 0
        
        for col in df.columns:
            if col == 'customer_id':
                continue
                
            feature_quality = self._assess_feature_quality(df[col], col)
            results['features'][col] = feature_quality
            total_quality_score += feature_quality['quality_score']
            feature_count += 1
        
        # Calculate overall quality score
        if feature_count > 0:
            results['overall_quality_score'] = total_quality_score / feature_count
        
        # Generate recommendations
        results['recommendations'] = self._generate_quality_recommendations(results['features'])
        
        return results
    
    def _assess_feature_quality(self, series: pd.Series, feature_name: str) -> Dict[str, any]:
        """
        Assess the quality of a single feature.
        
        Args:
            series: pandas Series to assess
            feature_name: Name of the feature
            
        Returns:
            Dictionary containing feature quality metrics
        """
        quality_metrics = {
            'feature_name': feature_name,
            'completeness': 1 - (series.isnull().sum() / len(series)),
            'uniqueness': series.nunique() / len(series) if len(series) > 0 else 0,
            'consistency': 1.0,  # Placeholder for consistency check
            'validity': 1.0,     # Placeholder for validity check
            'quality_score': 0.0
        }
        
        # Check for outliers (for numerical features)
        if series.dtype in ['int64', 'float64']:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
            quality_metrics['outlier_percentage'] = outliers / len(series) * 100
        
        # Calculate overall quality score
        quality_metrics['quality_score'] = (
            quality_metrics['completeness'] * 0.4 +
            quality_metrics['uniqueness'] * 0.2 +
            quality_metrics['consistency'] * 0.2 +
            quality_metrics['validity'] * 0.2
        )
        
        return quality_metrics
    
    def _generate_quality_recommendations(self, feature_results: Dict) -> List[str]:
        """
        Generate recommendations based on feature quality assessment.
        
        Args:
            feature_results: Dictionary containing feature quality results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        for feature_name, metrics in feature_results.items():
            if metrics['completeness'] < 0.8:
                recommendations.append(
                    f"Feature '{feature_name}' has low completeness ({metrics['completeness']:.2f}). "
                    "Consider imputation or removal."
                )
            
            if metrics['uniqueness'] < 0.01:
                recommendations.append(
                    f"Feature '{feature_name}' has very low uniqueness ({metrics['uniqueness']:.3f}). "
                    "Consider removing as it may not be informative."
                )
            
            if 'outlier_percentage' in metrics and metrics['outlier_percentage'] > 10:
                recommendations.append(
                    f"Feature '{feature_name}' has high outlier percentage ({metrics['outlier_percentage']:.1f}%). "
                    "Consider outlier treatment."
                )
        
        return recommendations
    
    def generate_data_quality_report(self, df: pd.DataFrame, target_col: str = 'is_fraud') -> str:
        """
        Generate a comprehensive data quality report.
        
        Args:
            df: DataFrame to analyze
            target_col: Name of the target column
            
        Returns:
            String containing the formatted report
        """
        report = ["=" * 50]
        report.append("DATA QUALITY REPORT")
        report.append("=" * 50)
        
        # Basic statistics
        report.append(f"\nDataset Overview:")
        report.append(f"  Total Records: {len(df):,}")
        report.append(f"  Total Features: {len(df.columns)}")
        report.append(f"  Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Validate billing data (if applicable)
        if any(col in df.columns for col in ['monthly_charges', 'total_charges', 'tenure']):
            billing_validation = self.validate_billing_data(df)
            report.append(f"\nBilling Data Validation:")
            report.append(f"  Valid: {billing_validation['valid']}")
            if billing_validation['errors']:
                report.append(f"  Errors: {billing_validation['errors']}")
            if billing_validation['warnings']:
                report.append(f"  Warnings: {billing_validation['warnings']}")
        
        # Validate fraud labels
        if target_col in df.columns:
            fraud_validation = self.validate_fraud_labels(df, target_col)
            report.append(f"\nFraud Label Validation:")
            report.append(f"  Fraud Rate: {fraud_validation['metrics']['fraud_rate']:.4f}")
            report.append(f"  Total Fraud Cases: {fraud_validation['metrics']['total_fraud_cases']:,}")
            if fraud_validation['warnings']:
                report.append(f"  Warnings: {fraud_validation['warnings']}")
        
        # Feature quality assessment
        quality_results = self.validate_feature_quality(df)
        report.append(f"\nFeature Quality Assessment:")
        report.append(f"  Overall Quality Score: {quality_results['overall_quality_score']:.3f}")
        
        if quality_results['recommendations']:
            report.append(f"\nRecommendations:")
            for rec in quality_results['recommendations'][:5]:  # Show top 5
                report.append(f"  - {rec}")
        
        report.append("\n" + "=" * 50)
        
        return "\n".join(report)
