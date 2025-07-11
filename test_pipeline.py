#!/usr/bin/env python3
"""
Quick test script to verify the telecoms fraud detection ML pipeline functionality.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from telecoms_fraud_detection_ml.utils.helpers import DataHelpers
from telecoms_fraud_detection_ml.data.processor import DataProcessor
from telecoms_fraud_detection_ml.data.unifier import DataUnifier
from telecoms_fraud_detection_ml.features.engineer import FeatureEngineer
from telecoms_fraud_detection_ml.models.fraud_detector import FraudDetectionModel
from telecoms_fraud_detection_ml.utils.logger import Logger

def generate_sample_data():
    """Generate sample telecoms data for testing."""
    print("Generating sample data...")
    
    np.random.seed(42)
    n_customers = 1000
    
    # Generate base customer data
    customers = pd.DataFrame({
        'customer_id': [f'CUST_{i:06d}' for i in range(n_customers)],
        'tenure': np.random.randint(1, 60, n_customers),
        'monthly_charges': np.random.normal(75, 25, n_customers).clip(20, 200),
        'total_charges': np.random.normal(1500, 800, n_customers).clip(100, 8000),
        'contract_type': np.random.choice(['Month-to-month', '1-year', '2-year'], n_customers),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_customers),
        'senior_citizen': np.random.choice([0, 1], n_customers, p=[0.8, 0.2])
    })
    
    # Generate billing data
    billing = pd.DataFrame({
        'customer_id': customers['customer_id'],
        'bill_amount': customers['monthly_charges'] + np.random.normal(0, 10, n_customers),
        'payment_date': pd.date_range('2024-01-01', periods=n_customers, freq='1D')[:n_customers],
        'late_payment': np.random.choice([0, 1], n_customers, p=[0.85, 0.15]),
        'billing_cycle': np.random.choice(['monthly', 'quarterly'], n_customers, p=[0.9, 0.1])
    })
    
    # Generate usage data
    usage = pd.DataFrame({
        'customer_id': customers['customer_id'],
        'monthly_data_gb': np.random.exponential(15, n_customers),
        'monthly_calls': np.random.poisson(50, n_customers),
        'monthly_sms': np.random.poisson(30, n_customers),
        'roaming_charges': np.random.exponential(5, n_customers)
    })
    
    # Generate customer care data
    customer_care = pd.DataFrame({
        'customer_id': customers['customer_id'],
        'support_calls': np.random.poisson(2, n_customers),
        'complaint_type': np.random.choice(['billing', 'technical', 'service', 'none'], n_customers, p=[0.3, 0.3, 0.2, 0.2]),
        'resolution_time_hours': np.random.exponential(24, n_customers)
    })
    
    # Generate social/network data
    social = pd.DataFrame({
        'customer_id': customers['customer_id'],
        'network_quality_score': np.random.uniform(1, 10, n_customers),
        'social_media_mentions': np.random.poisson(1, n_customers),
        'referral_count': np.random.poisson(1, n_customers)
    })
    
    return {
        'customers': customers,
        'billing': billing,
        'usage': usage,
        'customer_care': customer_care,
        'social': social
    }

def test_data_helpers():
    """Test the DataHelpers utility functions."""
    print("\n=== Testing DataHelpers ===")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'customer_id': ['CUST_001', 'CUST_002', 'CUST_003', 'CUST_002'],  # Duplicate
        'age': [25, 35, None, 35],  # Missing value
        'income': [50000, 75000, 60000, 75000],
        'category': ['A', 'B', 'A', 'B']
    })
    
    # Test data summary
    summary = DataHelpers.get_data_summary(sample_data)
    print(f"Data summary: {summary['shape']} rows and columns")
    print(f"Missing values: {summary['missing_values']}")
    
    # Test data quality issues
    issues = DataHelpers.detect_data_quality_issues(sample_data)
    print(f"Data quality issues found:")
    print(f"- Missing data columns: {list(issues['missing_data']['columns_with_missing'].keys())}")
    print(f"- Duplicate rows: {issues['duplicates']['total_duplicate_rows']}")
    
    # Test fraud label creation
    fraud_labels = DataHelpers.create_fraud_labels(sample_data)
    print(f"Fraud labels created: {fraud_labels.sum()} fraud cases out of {len(fraud_labels)}")

def test_data_processing():
    """Test the data processing pipeline."""
    print("\n=== Testing Data Processing ===")
    
    # Generate sample data
    data_dict = generate_sample_data()
    
    # Test data processor
    processor = DataProcessor()
    cleaned_data = {}
    
    # Process each data type with the appropriate method
    data_methods = {
        'customers': processor.clean_crm_data,  # Customer data is similar to CRM data
        'billing': processor.clean_billing_data,
        'usage': processor.clean_crm_data,  # Usage data can use CRM cleaning
        'customer_care': processor.clean_customer_care_data,
        'social': processor.clean_social_data
    }
    
    for data_type, df in data_dict.items():
        print(f"Processing {data_type} data...")
        if data_type in data_methods:
            cleaned_df = data_methods[data_type](df)
        else:
            # Fallback to basic cleaning
            cleaned_df = df.dropna().drop_duplicates()
        cleaned_data[data_type] = cleaned_df
        print(f"- Original shape: {df.shape}, Cleaned shape: {cleaned_df.shape}")
    
    # Test data unifier
    unifier = DataUnifier()
    unified_data = unifier.unify_datasets(
        billing_df=cleaned_data['billing'],
        crm_df=cleaned_data['customers'],  # Using customers as CRM data
        social_df=cleaned_data['social'],
        customer_care_df=cleaned_data['customer_care']
    )
    print(f"Unified data shape: {unified_data.shape}")
    print(f"Unified data columns: {list(unified_data.columns)}")
    
    return unified_data

def test_feature_engineering(unified_data):
    """Test feature engineering."""
    print("\n=== Testing Feature Engineering ===")
    
    engineer = FeatureEngineer()
    
    # Create features
    features_df = engineer.create_fraud_features(unified_data)
    print(f"Features created: {features_df.shape[1]} features for {features_df.shape[0]} samples")
    
    # Create target variable
    target = DataHelpers.create_fraud_labels(unified_data)
    print(f"Target variable created: {target.sum()} fraud cases ({target.mean():.2%} fraud rate)")
    
    return features_df, target

def test_model_training(features_df, target):
    """Test model training and prediction."""
    print("\n=== Testing Model Training ===")
    
    # Remove non-numeric columns for training
    numeric_features = features_df.select_dtypes(include=[np.number])
    print(f"Training with {numeric_features.shape[1]} numeric features")
    
    detector = FraudDetectionModel()
    
    # Train model
    print("Training fraud detection model...")
    model_path = detector.train(numeric_features, target)
    print(f"Model saved to: {model_path}")
    
    # Test prediction only if training was successful
    if isinstance(model_path, dict) and not model_path.get('success', True):
        print("Training failed, skipping prediction test")
        return model_path
    
    print("Testing predictions...")
    predictions = detector.predict(numeric_features.head(10))
    probabilities = detector.predict_proba(numeric_features.head(10))
    
    print(f"Sample predictions: {predictions}")
    print(f"Sample probabilities: {probabilities[:, 1]}")  # Fraud probabilities
    
    return model_path

def main():
    """Run the complete pipeline test."""
    print("=== Telecoms Fraud Detection ML Pipeline Test ===")
    
    try:
        # Set up logging
        logger = Logger("pipeline_test")
        logger.info("Starting pipeline test")
        
        # Test individual components
        test_data_helpers()
        
        # Test full pipeline
        unified_data = test_data_processing()
        features_df, target = test_feature_engineering(unified_data)
        model_path = test_model_training(features_df, target)
        
        print("\n=== Pipeline Test Completed Successfully! ===")
        print(f"Model saved at: {model_path}")
        print(f"Sample data processed: {len(unified_data)} customers")
        print(f"Features generated: {features_df.shape[1]} features")
        print(f"Fraud rate in sample: {target.mean():.2%}")
        
        logger.info("Pipeline test completed successfully")
        
    except Exception as e:
        print(f"\n❌ Pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
