#!/usr/bin/env python3
"""
Comprehensive training script for telecoms fraud detection ML model.
"""

import os
import sys
import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from telecoms_fraud_detection_ml.data import DataProcessor, DataUnifier, DataValidator
from telecoms_fraud_detection_ml.features import FeatureEngineer, FeatureSelector
from telecoms_fraud_detection_ml.models import ModelTrainer
from telecoms_fraud_detection_ml.utils import Logger, ConfigManager, DateUtils, DataHelpers


def main():
    """Main training pipeline for telecoms fraud detection model."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train telecoms fraud detection model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, default='data/generated',
                       help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--models', nargs='+', 
                       default=['random_forest', 'xgboost', 'logistic_regression'],
                       help='Models to train')
    parser.add_argument('--create-fraud-labels', action='store_true',
                       help='Create synthetic fraud labels for training')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = Logger('fraud_detection_training')
    logger.info("Starting telecoms fraud detection model training pipeline")
    
    # Load configuration
    config_manager = ConfigManager(args.config)
    config = config_manager.get_full_config()
    
    try:
        # Step 1: Load and validate data
        logger.info("Step 1: Loading and validating data")
        
        # Load datasets
        billing_df = DataHelpers.load_data_from_file(f"{args.data_dir}/billing.csv")
        crm_df = DataHelpers.load_data_from_file(f"{args.data_dir}/customers.csv")
        social_df = DataHelpers.load_data_from_file(f"{args.data_dir}/social.csv")
        customer_care_df = DataHelpers.load_data_from_file(f"{args.data_dir}/customer_care.csv")
        
        if billing_df is None:
            logger.error("Failed to load billing data")
            return
        
        if crm_df is None:
            logger.error("Failed to load CRM data")
            return
        
        logger.info(f"Loaded datasets - Billing: {billing_df.shape}, CRM: {crm_df.shape}")
        
        # Validate data quality
        validator = DataValidator()
        billing_validation = validator.validate_billing_data(billing_df)
        
        if not billing_validation['valid']:
            logger.error(f"Billing data validation failed: {billing_validation['errors']}")
            return
        
        # Step 2: Data preprocessing and cleaning
        logger.info("Step 2: Data preprocessing and cleaning")
        
        processor = DataProcessor(config.get('data_processing', {}))
        
        # Clean individual datasets
        billing_clean = processor.clean_billing_data(billing_df)
        crm_clean = processor.clean_crm_data(crm_df)
        
        if social_df is not None:
            social_clean = processor.clean_social_data(social_df)
        else:
            social_clean = None
            
        if customer_care_df is not None:
            customer_care_clean = processor.clean_customer_care_data(customer_care_df)
        else:
            customer_care_clean = None
        
        logger.info("Data cleaning completed")
        
        # Step 3: Data unification
        logger.info("Step 3: Unifying datasets")
        
        unifier = DataUnifier()
        unified_df = unifier.unify_datasets(
            billing_df=billing_clean,
            crm_df=crm_clean,
            social_df=social_clean,
            customer_care_df=customer_care_clean
        )
        
        logger.info(f"Unified dataset shape: {unified_df.shape}")
        
        # Step 4: Create fraud labels if requested
        if args.create_fraud_labels:
            logger.info("Step 4: Creating synthetic fraud labels")
            
            fraud_labels = DataHelpers.create_fraud_labels(unified_df)
            unified_df['is_fraud'] = fraud_labels
            
            fraud_rate = fraud_labels.mean()
            logger.info(f"Created fraud labels with {fraud_rate:.4f} fraud rate")
        else:
            # Check if fraud labels already exist
            if 'is_fraud' not in unified_df.columns:
                logger.error("No fraud labels found. Use --create-fraud-labels or provide labels")
                return
        
        # Validate fraud labels
        fraud_validation = validator.validate_fraud_labels(unified_df, 'is_fraud')
        logger.info(f"Fraud label validation: {fraud_validation['metrics']}")
        
        # Step 5: Feature engineering
        logger.info("Step 5: Feature engineering")
        
        engineer = FeatureEngineer()
        feature_enhanced_df = engineer.create_fraud_features(unified_df)
        
        logger.info(f"Created {len(engineer.get_created_features_list())} additional features")
        
        # Step 6: Feature selection
        logger.info("Step 6: Feature selection")
        
        selector = FeatureSelector(max_features=config.get('feature_engineering.max_features'))
        selected_features = selector.automatic_feature_selection(
            feature_enhanced_df, 
            'is_fraud',
            correlation_threshold=0.05,
            variance_threshold=0.01
        )
        
        logger.info(f"Selected {len(selected_features)} features for training")
        
        # Prepare final training dataset
        training_columns = selected_features + ['is_fraud', 'customer_id']
        training_df = feature_enhanced_df[training_columns].copy()
        
        # Step 7: Data preprocessing for ML
        logger.info("Step 7: Final data preprocessing")
        
        # Encode categorical features
        training_df_encoded = processor.encode_categorical_features(training_df, fit=True)
        
        # Scale numerical features
        training_df_final = processor.scale_numerical_features(training_df_encoded, fit=True)
        
        logger.info(f"Final training dataset shape: {training_df_final.shape}")
        
        # Step 8: Model training
        logger.info("Step 8: Training models")
        
        trainer = ModelTrainer(config.get('model_training', {}))
        
        # Train multiple models
        training_results = trainer.train_multiple_models(
            training_data=training_df_final,
            target_column='is_fraud',
            model_types=args.models,
            test_size=config.get('model_training.test_size', 0.2),
            random_state=config.get('model_training.random_state', 42)
        )
        
        # Display training results
        if training_results['best_model']:
            best_model_info = training_results['best_model']
            logger.info(f"Best model: {best_model_info['model_type']} "
                       f"(ROC-AUC: {best_model_info['roc_auc']:.4f})")
        
        # Step 9: Save models and results
        logger.info("Step 9: Saving models and results")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save best model with date
        if training_results['best_model']:
            model_path = trainer.save_best_model(args.output_dir)
            logger.info(f"Best model saved to: {model_path}")
            
            # Also save as standard model.pkl for easy access
            standard_model_path = os.path.join(args.output_dir, "model.pkl")
            best_model = training_results['best_model']['model_object']
            best_model.save_model(standard_model_path, include_date=False)
            logger.info(f"Standard model saved to: {standard_model_path}")
        
        # Save training summary
        training_summary = trainer.get_training_summary()
        summary_path = f"{args.output_dir}/training_summary_{DateUtils.get_datetime_string()}.csv"
        DataHelpers.save_data_to_file(training_summary, summary_path)
        
        # Save feature information
        feature_info = {
            'selected_features': selected_features,
            'created_features': engineer.get_created_features_list(),
            'feature_importance': None
        }
        
        if training_results['best_model']:
            best_model = training_results['best_model']['model_object']
            if hasattr(best_model, 'get_feature_importance'):
                feature_importance = best_model.get_feature_importance()
                feature_info['feature_importance'] = feature_importance.to_dict('records')
        
        feature_info_path = f"{args.output_dir}/feature_info_{DateUtils.get_datetime_string()}.json"
        DataHelpers.export_to_json(feature_info, feature_info_path)
        
        # Save processed data
        processed_data_path = f"data/processed/training_data_{DateUtils.get_datetime_string()}.csv"
        DataHelpers.save_data_to_file(training_df_final, processed_data_path)
        
        # Step 10: Generate comprehensive report
        logger.info("Step 10: Generating training report")
        
        report_lines = [
            "=" * 80,
            "TELECOMS FRAUD DETECTION MODEL TRAINING REPORT",
            "=" * 80,
            f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Configuration: {args.config}",
            "",
            "DATA SUMMARY:",
            "  Original Datasets:",
            f"    - Billing: {billing_df.shape[0]:,} records, {billing_df.shape[1]} columns",
            f"    - CRM: {crm_df.shape[0]:,} records, {crm_df.shape[1]} columns",
            f"    - Social: {social_df.shape[0] if social_df is not None else 0:,} records",
            f"    - Customer Care: {customer_care_df.shape[0] if customer_care_df is not None else 0:,} records",
            f"  Unified Dataset: {unified_df.shape[0]:,} records, {unified_df.shape[1]} columns",
            f"  Final Training Dataset: {training_df_final.shape[0]:,} records, {training_df_final.shape[1]} columns",
            "",
            "FEATURE ENGINEERING:",
            f"  Original Features: {len(unified_df.columns)}",
            f"  Created Features: {len(engineer.get_created_features_list())}",
            f"  Selected Features: {len(selected_features)}",
            "",
            "FRAUD LABEL DISTRIBUTION:",
            f"  Total Cases: {len(unified_df):,}",
            f"  Fraud Cases: {int(unified_df['is_fraud'].sum()):,}",
            f"  Normal Cases: {int((unified_df['is_fraud'] == 0).sum()):,}",
            f"  Fraud Rate: {unified_df['is_fraud'].mean():.4f}",
            "",
            "MODEL TRAINING RESULTS:",
            f"  Models Trained: {training_results['training_summary']['successful_models']}",
            f"  Failed Models: {training_results['training_summary']['failed_models']}",
        ]
        
        if training_results['best_model']:
            best_model_info = training_results['best_model']
            report_lines.extend([
                f"  Best Model: {best_model_info['model_type']}",
                f"  Best ROC-AUC: {best_model_info['roc_auc']:.4f}",
            ])
        
        report_lines.extend([
            "",
            "SAVED FILES:",
            f"  Best Model (dated): {model_path if training_results['best_model'] else 'None'}",
            f"  Standard Model: {os.path.join(args.output_dir, 'model.pkl') if training_results['best_model'] else 'None'}",
            f"  Training Summary: {summary_path}",
            f"  Feature Info: {feature_info_path}",
            f"  Processed Data: {processed_data_path}",
            "",
            "=" * 80
        ])
        
        # Save and display report
        report_content = "\n".join(report_lines)
        report_path = f"{args.output_dir}/training_report_{DateUtils.get_datetime_string()}.txt"
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(report_content)
        logger.info(f"Training report saved to: {report_path}")
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
