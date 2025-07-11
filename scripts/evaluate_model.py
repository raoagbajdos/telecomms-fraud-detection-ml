#!/usr/bin/env python3
"""
Model evaluation script for telecoms fraud detection.
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

from telecoms_fraud_detection_ml.models import FraudDetectionModel, ModelEvaluator
from telecoms_fraud_detection_ml.utils import Logger, DateUtils, DataHelpers


def main():
    """Main evaluation pipeline for telecoms fraud detection model."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate fraud detection model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--test-data', type=str, required=True,
                       help='Path to test data file')
    parser.add_argument('--output-dir', type=str, default='evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--target-column', type=str, default='is_fraud',
                       help='Name of the target column')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = Logger('fraud_detection_evaluation')
    logger.info("Starting telecoms fraud detection model evaluation")
    
    try:
        # Step 1: Load trained model
        logger.info(f"Step 1: Loading trained model from {args.model}")
        
        model = FraudDetectionModel()
        if not model.load_model(args.model):
            logger.error("Failed to load model")
            return
        
        model_info = model.get_model_info()
        logger.info(f"Loaded {model_info['model_type']} model with {model_info['n_features']} features")
        
        # Step 2: Load test data
        logger.info(f"Step 2: Loading test data from {args.test_data}")
        
        test_df = DataHelpers.load_data_from_file(args.test_data)
        if test_df is None:
            logger.error("Failed to load test data")
            return
        
        # Validate target column exists
        if args.target_column not in test_df.columns:
            logger.error(f"Target column '{args.target_column}' not found in test data")
            return
        
        logger.info(f"Loaded {test_df.shape[0]:,} test records")
        
        # Step 3: Validate input data
        logger.info("Step 3: Validating test data")
        
        if not model.validate_input(test_df):
            logger.error("Test data validation failed")
            return
        
        # Step 4: Evaluate model
        logger.info("Step 4: Evaluating model performance")
        
        evaluator = ModelEvaluator()
        evaluation_results = evaluator.evaluate_model(model, test_df, args.target_column)
        
        if 'error' in evaluation_results:
            logger.error(f"Model evaluation failed: {evaluation_results['error']}")
            return
        
        # Step 5: Generate evaluation report
        logger.info("Step 5: Generating evaluation report")
        
        evaluation_report = evaluator.generate_evaluation_report(evaluation_results)
        
        # Step 6: Save evaluation results
        logger.info("Step 6: Saving evaluation results")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save detailed evaluation results
        timestamp = DateUtils.get_datetime_string()
        results_file = f"{args.output_dir}/evaluation_results_{timestamp}.json"
        evaluator.export_evaluation_results(evaluation_results, results_file)
        
        # Save evaluation report
        report_file = f"{args.output_dir}/evaluation_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(evaluation_report)
        
        # Save feature importance if available
        if evaluation_results.get('feature_importance'):
            importance_df = pd.DataFrame(evaluation_results['feature_importance'])
            importance_file = f"{args.output_dir}/feature_importance_{timestamp}.csv"
            DataHelpers.save_data_to_file(importance_df, importance_file)
        
        # Step 7: Display results
        print(evaluation_report)
        
        # Log key metrics
        basic_metrics = evaluation_results['basic_metrics']
        logger.info(f"Model Performance - ROC-AUC: {basic_metrics['roc_auc']:.4f}, "
                   f"Precision: {basic_metrics['precision']:.4f}, "
                   f"Recall: {basic_metrics['recall']:.4f}, "
                   f"F1-Score: {basic_metrics['f1_score']:.4f}")
        
        # Business impact summary
        financial_impact = evaluation_results['fraud_specific_metrics']['financial_impact']
        logger.info(f"Business Impact - Detected: {financial_impact['detected_fraud_cases']}, "
                   f"Missed: {financial_impact['missed_fraud_cases']}, "
                   f"False Positives: {financial_impact['false_positives']}")
        
        logger.info(f"Evaluation results saved to: {args.output_dir}")
        logger.info("Model evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
