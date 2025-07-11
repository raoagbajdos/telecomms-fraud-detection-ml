#!/usr/bin/env python3
"""
Comprehensive demo script for telecoms fraud detection ML pipeline.
"""

import os
import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from telecoms_fraud_detection_ml.data import DataProcessor, DataUnifier, DataValidator
from telecoms_fraud_detection_ml.features import FeatureEngineer, FeatureSelector
from telecoms_fraud_detection_ml.models import ModelTrainer, ModelEvaluator
from telecoms_fraud_detection_ml.utils import Logger, ConfigManager, DateUtils, DataHelpers


def main():
    """Comprehensive demo of the telecoms fraud detection ML pipeline."""
    
    print("=" * 80)
    print("TELECOMS FRAUD DETECTION ML PIPELINE DEMO")
    print("=" * 80)
    print()
    
    # Initialize logger
    logger = Logger('fraud_detection_demo')
    logger.info("Starting telecoms fraud detection demo")
    
    try:
        # Configuration
        config_manager = ConfigManager('config/config.yaml')
        
        print("🔧 Configuration loaded successfully")
        print(f"   Available models: {config_manager.get('model_training.available_models')}")
        print(f"   Test size: {config_manager.get('model_training.test_size')}")
        print()
        
        # Step 1: Load sample data
        print("📊 Step 1: Loading sample telecoms data")
        
        data_dir = "data/generated"
        
        # Load datasets
        billing_df = DataHelpers.load_data_from_file(f"{data_dir}/billing.csv")
        customers_df = DataHelpers.load_data_from_file(f"{data_dir}/customers.csv")
        usage_df = DataHelpers.load_data_from_file(f"{data_dir}/usage.csv")
        network_df = DataHelpers.load_data_from_file(f"{data_dir}/network.csv")
        
        print(f"   ✅ Billing data: {billing_df.shape[0]:,} records, {billing_df.shape[1]} columns")
        print(f"   ✅ Customer data: {customers_df.shape[0]:,} records, {customers_df.shape[1]} columns")
        print(f"   ✅ Usage data: {usage_df.shape[0]:,} records" if usage_df is not None else "   ⚠️  Usage data: Not available")
        print(f"   ✅ Network data: {network_df.shape[0]:,} records" if network_df is not None else "   ⚠️  Network data: Not available")
        print()
        
        # Step 2: Data validation and quality assessment
        print("🔍 Step 2: Data validation and quality assessment")
        
        validator = DataValidator()
        
        # Validate billing data
        billing_validation = validator.validate_billing_data(billing_df)
        print(f"   Billing data validation: {'✅ PASSED' if billing_validation['valid'] else '❌ FAILED'}")
        
        if billing_validation['warnings']:
            for warning in billing_validation['warnings'][:3]:  # Show first 3 warnings
                print(f"      ⚠️  {warning}")
        
        # Generate data quality report
        quality_report = validator.generate_data_quality_report(billing_df)
        print("   📋 Data quality summary generated")
        print()
        
        # Step 3: Data preprocessing and cleaning
        print("🧹 Step 3: Data preprocessing and cleaning")
        
        processor = DataProcessor()
        
        # Clean datasets
        billing_clean = processor.clean_billing_data(billing_df)
        customers_clean = processor.clean_crm_data(customers_df)
        
        print(f"   ✅ Billing data cleaned: {billing_clean.shape}")
        print(f"   ✅ Customer data cleaned: {customers_clean.shape}")
        print()
        
        # Step 4: Data unification
        print("🔗 Step 4: Unifying multiple data sources")
        
        unifier = DataUnifier()
        unified_df = unifier.unify_datasets(
            billing_df=billing_clean,
            crm_df=customers_clean,
            social_df=usage_df,
            customer_care_df=network_df
        )
        
        print(f"   ✅ Unified dataset: {unified_df.shape[0]:,} records, {unified_df.shape[1]} columns")
        
        # Validation of unified data
        unified_validation = unifier.validate_unified_data(unified_df)
        print(f"   📊 Data completeness: {unified_validation['completeness_score']:.3f}")
        print()
        
        # Step 5: Create synthetic fraud labels for demo
        print("🏷️  Step 5: Creating synthetic fraud labels")
        
        fraud_labels = DataHelpers.create_fraud_labels(unified_df)
        unified_df['is_fraud'] = fraud_labels
        
        fraud_count = int(fraud_labels.sum())
        fraud_rate = fraud_labels.mean()
        
        print(f"   ✅ Created fraud labels: {fraud_count:,} fraud cases ({fraud_rate:.4f} fraud rate)")
        
        # Validate fraud labels
        fraud_validation = validator.validate_fraud_labels(unified_df, 'is_fraud')
        print(f"   📊 Label distribution validated")
        print()
        
        # Step 6: Feature engineering
        print("⚙️  Step 6: Advanced feature engineering")
        
        engineer = FeatureEngineer()
        enhanced_df = engineer.create_fraud_features(unified_df)
        
        created_features = engineer.get_created_features_list()
        print(f"   ✅ Created {len(created_features)} fraud detection features")
        print(f"   📊 Total features: {len(enhanced_df.columns)} (original: {len(unified_df.columns)})")
        
        # Show some created features
        if created_features:
            print(f"   🔧 Sample features: {', '.join(created_features[:5])}")
        print()
        
        # Step 7: Feature selection
        print("🎯 Step 7: Intelligent feature selection")
        
        selector = FeatureSelector(max_features=50)
        selected_features = selector.automatic_feature_selection(
            enhanced_df, 'is_fraud',
            correlation_threshold=0.05,
            variance_threshold=0.01
        )
        
        print(f"   ✅ Selected {len(selected_features)} optimal features")
        print(f"   📊 Feature reduction: {len(enhanced_df.columns)} → {len(selected_features)}")
        print()
        
        # Step 8: Prepare training data
        print("📝 Step 8: Preparing training dataset")
        
        # Create final training dataset
        training_columns = selected_features + ['is_fraud', 'customer_id']
        training_df = enhanced_df[training_columns].copy()
        
        # Encode categorical features
        training_encoded = processor.encode_categorical_features(training_df, fit=True)
        
        # Scale numerical features
        training_final = processor.scale_numerical_features(training_encoded, fit=True)
        
        print(f"   ✅ Training dataset prepared: {training_final.shape}")
        print(f"   🔧 Applied encoding and scaling transformations")
        print()
        
        # Step 9: Model training
        print("🤖 Step 9: Training fraud detection models")
        
        trainer = ModelTrainer()
        
        # Train multiple models
        models_to_train = ['random_forest', 'logistic_regression']
        print(f"   🎯 Training models: {', '.join(models_to_train)}")
        
        training_results = trainer.train_multiple_models(
            training_data=training_final,
            target_column='is_fraud',
            model_types=models_to_train,
            test_size=0.2,
            random_state=42
        )
        
        # Display results
        successful_models = training_results['training_summary']['successful_models']
        print(f"   ✅ Successfully trained {successful_models} models")
        
        if training_results['best_model']:
            best_model = training_results['best_model']
            print(f"   🏆 Best model: {best_model['model_type']} (ROC-AUC: {best_model['roc_auc']:.4f})")
        print()
        
        # Step 10: Model evaluation
        print("📊 Step 10: Comprehensive model evaluation")
        
        if training_results['best_model']:
            evaluator = ModelEvaluator()
            best_model_obj = training_results['best_model']['model_object']
            
            # Use the same test split for evaluation (simplified for demo)
            from sklearn.model_selection import train_test_split
            
            features = training_final.drop(columns=['is_fraud', 'customer_id'], errors='ignore')
            target = training_final['is_fraud']
            
            _, X_test, _, y_test = train_test_split(
                features, target, test_size=0.2, random_state=42, stratify=target
            )
            
            test_data_eval = X_test.copy()
            test_data_eval['is_fraud'] = y_test
            test_data_eval['customer_id'] = range(len(test_data_eval))  # Add dummy customer IDs
            
            eval_results = evaluator.evaluate_model(best_model_obj, test_data_eval, 'is_fraud')
            
            if 'error' not in eval_results:
                metrics = eval_results['basic_metrics']
                financial = eval_results['fraud_specific_metrics']['financial_impact']
                
                print(f"   ✅ Model performance:")
                print(f"      🎯 Accuracy: {metrics['accuracy']:.4f}")
                print(f"      🎯 Precision: {metrics['precision']:.4f}")
                print(f"      🎯 Recall: {metrics['recall']:.4f}")
                print(f"      🎯 F1-Score: {metrics['f1_score']:.4f}")
                print(f"      🎯 ROC-AUC: {metrics['roc_auc']:.4f}")
                print()
                print(f"   💰 Business impact:")
                print(f"      🔍 Detected fraud: {financial['detected_fraud_cases']}")
                print(f"      ❌ Missed fraud: {financial['missed_fraud_cases']}")
                print(f"      ⚠️  False positives: {financial['false_positives']}")
            else:
                print(f"   ❌ Evaluation failed: {eval_results['error']}")
        print()
        
        # Step 11: Save models and results
        print("💾 Step 11: Saving models and results")
        
        # Create output directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('reports', exist_ok=True)
        
        # Save best model
        if training_results['best_model']:
            model_path = trainer.save_best_model('models')
            print(f"   ✅ Best model saved: {os.path.basename(model_path)}")
        
        # Save processed data
        processed_data_path = f"data/processed/demo_training_data_{DateUtils.get_datetime_string()}.csv"
        DataHelpers.save_data_to_file(training_final, processed_data_path)
        print(f"   ✅ Training data saved: {os.path.basename(processed_data_path)}")
        
        # Save feature information
        feature_info = {
            'selected_features': selected_features,
            'created_features': created_features,
            'total_features': len(selected_features)
        }
        
        feature_info_path = f"reports/demo_feature_info_{DateUtils.get_datetime_string()}.json"
        DataHelpers.export_to_json(feature_info, feature_info_path)
        print(f"   ✅ Feature info saved: {os.path.basename(feature_info_path)}")
        print()
        
        # Step 12: Demo prediction
        print("🔮 Step 12: Making fraud predictions on sample data")
        
        if training_results['best_model']:
            best_model_obj = training_results['best_model']['model_object']
            
            # Use a sample of the data for prediction demo
            sample_data = training_final.drop(columns=['is_fraud']).head(10)
            
            predictions = best_model_obj.predict(sample_data.drop(columns=['customer_id'], errors='ignore'))
            probabilities = best_model_obj.predict_proba(sample_data.drop(columns=['customer_id'], errors='ignore'))
            
            print(f"   🔍 Analyzed {len(sample_data)} sample records:")
            print(f"      📊 Predicted fraud cases: {predictions.sum()}")
            print(f"      📊 Average fraud probability: {probabilities[:, 1].mean():.4f}")
            print(f"      📊 Max fraud probability: {probabilities[:, 1].max():.4f}")
        print()
        
        # Final summary
        print("✨ Demo Summary")
        print("=" * 40)
        print(f"📊 Data processed: {unified_df.shape[0]:,} records")
        print(f"🔧 Features engineered: {len(created_features)}")
        print(f"🎯 Features selected: {len(selected_features)}")
        print(f"🤖 Models trained: {successful_models}")
        
        if training_results['best_model']:
            best_roc_auc = training_results['best_model']['roc_auc']
            print(f"🏆 Best ROC-AUC: {best_roc_auc:.4f}")
        
        print(f"💾 Models saved to: models/")
        print(f"📈 Reports saved to: reports/")
        print()
        
        logger.info("Telecoms fraud detection demo completed successfully!")
        
        print("🎉 Demo completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"❌ Demo failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
