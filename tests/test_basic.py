"""Basic tests for the telecoms fraud detection ML package."""

import unittest
import tempfile
import os
import sys
import pandas as pd
import numpy as np

# Add the project root to the path so we can import modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


class TestDataProcessor(unittest.TestCase):
    """Test data processing functionality."""

    def test_import_data_processor(self):
        """Test that DataProcessor can be imported."""
        try:
            from telecoms_fraud_detection_ml.data import DataProcessor
            self.assertIsNotNone(DataProcessor)
        except ImportError as e:
            self.fail(f"Could not import DataProcessor: {e}")

    def test_data_processor_initialization(self):
        """Test DataProcessor can be initialized."""
        try:
            from telecoms_fraud_detection_ml.data import DataProcessor
            processor = DataProcessor()
            self.assertIsNotNone(processor)
        except Exception as e:
            self.fail(f"Could not initialize DataProcessor: {e}")

    def test_data_processor_has_required_methods(self):
        """Test that DataProcessor has the required methods."""
        try:
            from telecoms_fraud_detection_ml.data import DataProcessor
            processor = DataProcessor()
            
            # Check for key methods
            self.assertTrue(hasattr(processor, 'clean_billing_data'))
            self.assertTrue(hasattr(processor, 'clean_crm_data'))
            self.assertTrue(hasattr(processor, 'clean_social_data'))
            self.assertTrue(hasattr(processor, 'clean_customer_care_data'))
            
        except Exception as e:
            self.fail(f"DataProcessor missing required methods: {e}")


class TestDataUnifier(unittest.TestCase):
    """Test data unification functionality."""

    def test_import_data_unifier(self):
        """Test that DataUnifier can be imported."""
        try:
            from telecoms_fraud_detection_ml.data import DataUnifier
            self.assertIsNotNone(DataUnifier)
        except ImportError as e:
            self.fail(f"Could not import DataUnifier: {e}")

    def test_data_unifier_initialization(self):
        """Test DataUnifier can be initialized."""
        try:
            from telecoms_fraud_detection_ml.data import DataUnifier
            unifier = DataUnifier()
            self.assertIsNotNone(unifier)
        except Exception as e:
            self.fail(f"Could not initialize DataUnifier: {e}")


class TestFeatureEngineer(unittest.TestCase):
    """Test feature engineering functionality."""

    def test_import_feature_engineer(self):
        """Test that FeatureEngineer can be imported."""
        try:
            from telecoms_fraud_detection_ml.features import FeatureEngineer
            self.assertIsNotNone(FeatureEngineer)
        except ImportError as e:
            self.fail(f"Could not import FeatureEngineer: {e}")

    def test_feature_engineer_initialization(self):
        """Test FeatureEngineer can be initialized."""
        try:
            from telecoms_fraud_detection_ml.features import FeatureEngineer
            engineer = FeatureEngineer()
            self.assertIsNotNone(engineer)
        except Exception as e:
            self.fail(f"Could not initialize FeatureEngineer: {e}")

    def test_feature_engineer_has_required_methods(self):
        """Test that FeatureEngineer has the required methods."""
        try:
            from telecoms_fraud_detection_ml.features import FeatureEngineer
            engineer = FeatureEngineer()
            
            # Check for key methods
            self.assertTrue(hasattr(engineer, 'create_fraud_features'))
            
        except Exception as e:
            self.fail(f"FeatureEngineer missing required methods: {e}")


class TestFraudDetectionModel(unittest.TestCase):
    """Test fraud detection functionality."""

    def test_import_fraud_detector(self):
        """Test that FraudDetectionModel can be imported."""
        try:
            from telecoms_fraud_detection_ml.models import FraudDetectionModel
            self.assertIsNotNone(FraudDetectionModel)
        except ImportError as e:
            self.fail(f"Could not import FraudDetectionModel: {e}")

    def test_fraud_detector_initialization(self):
        """Test FraudDetectionModel can be initialized."""
        try:
            from telecoms_fraud_detection_ml.models import FraudDetectionModel
            detector = FraudDetectionModel()
            self.assertIsNotNone(detector)
        except Exception as e:
            self.fail(f"Could not initialize FraudDetectionModel: {e}")

    def test_fraud_detector_has_required_methods(self):
        """Test that FraudDetectionModel has the required methods."""
        try:
            from telecoms_fraud_detection_ml.models import FraudDetectionModel
            detector = FraudDetectionModel()
            
            # Check for key methods
            self.assertTrue(hasattr(detector, 'train'))
            self.assertTrue(hasattr(detector, 'predict'))
            self.assertTrue(hasattr(detector, 'predict_proba'))
            self.assertTrue(hasattr(detector, 'save_model'))
            self.assertTrue(hasattr(detector, 'load_model'))
            
        except Exception as e:
            self.fail(f"FraudDetectionModel missing required methods: {e}")


class TestUtilities(unittest.TestCase):
    """Test utility functions."""

    def test_import_helpers(self):
        """Test that DataHelpers can be imported."""
        try:
            from telecoms_fraud_detection_ml.utils import DataHelpers
            self.assertIsNotNone(DataHelpers)
        except ImportError as e:
            self.fail(f"Could not import DataHelpers: {e}")

    def test_data_helpers_fraud_functions(self):
        """Test that DataHelpers has fraud detection functions."""
        try:
            from telecoms_fraud_detection_ml.utils import DataHelpers
            
            # Check for key methods
            self.assertTrue(hasattr(DataHelpers, 'create_fraud_labels'))
            self.assertTrue(hasattr(DataHelpers, 'detect_data_quality_issues'))
            self.assertTrue(hasattr(DataHelpers, 'balance_dataset'))
            
        except Exception as e:
            self.fail(f"DataHelpers missing required methods: {e}")

    def test_fraud_label_creation(self):
        """Test fraud label creation functionality."""
        try:
            from telecoms_fraud_detection_ml.utils import DataHelpers
            
            # Create sample data
            sample_data = pd.DataFrame({
                'customer_id': ['CUST_001', 'CUST_002', 'CUST_003'],
                'monthly_charges': [25.0, 150.0, 75.0],
                'tenure': [12, 2, 24],
                'support_calls': [1, 8, 3]
            })
            
            # Test fraud label creation
            fraud_labels = DataHelpers.create_fraud_labels(sample_data)
            self.assertIsInstance(fraud_labels, pd.Series)
            self.assertEqual(len(fraud_labels), len(sample_data))
            
        except Exception as e:
            self.fail(f"Could not create fraud labels: {e}")


class TestPipelineIntegration(unittest.TestCase):
    """Test end-to-end pipeline integration."""

    def test_basic_pipeline_flow(self):
        """Test that the basic pipeline components work together."""
        try:
            from telecoms_fraud_detection_ml.data import DataProcessor, DataUnifier
            from telecoms_fraud_detection_ml.features import FeatureEngineer
            from telecoms_fraud_detection_ml.models import FraudDetectionModel
            from telecoms_fraud_detection_ml.utils import DataHelpers
            
            # Create minimal sample data
            billing_data = pd.DataFrame({
                'customer_id': ['CUST_001', 'CUST_002'],
                'bill_amount': [75.0, 120.0],
                'payment_date': ['2024-01-01', '2024-01-01'],
                'late_payment': [0, 1],
                'billing_cycle': ['monthly', 'monthly']
            })
            
            crm_data = pd.DataFrame({
                'customer_id': ['CUST_001', 'CUST_002'],
                'tenure': [12, 3],
                'monthly_charges': [75.0, 120.0],
                'total_charges': [900.0, 360.0],
                'contract_type': ['1-year', 'month-to-month'],
                'senior_citizen': [0, 0]
            })
            
            # Test pipeline steps
            processor = DataProcessor()
            cleaned_billing = processor.clean_billing_data(billing_data)
            cleaned_crm = processor.clean_crm_data(crm_data)
            
            unifier = DataUnifier()
            unified_data = unifier.unify_datasets(
                billing_df=cleaned_billing,
                crm_df=cleaned_crm
            )
            
            # Verify unified data
            self.assertIsInstance(unified_data, pd.DataFrame)
            self.assertGreater(len(unified_data.columns), 0)
            
        except Exception as e:
            self.fail(f"Pipeline integration test failed: {e}")


if __name__ == '__main__':
    unittest.main()
