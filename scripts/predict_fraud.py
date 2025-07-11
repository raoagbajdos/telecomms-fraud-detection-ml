#!/usr/bin/env python3
"""
Prediction script for telecoms fraud detection ML model.
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

from telecoms_fraud_detection_ml.models import FraudDetectionModel
from telecoms_fraud_detection_ml.utils import Logger, DateUtils, DataHelpers


def main():
    """Main prediction pipeline for telecoms fraud detection."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Predict fraud using trained model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data file for prediction')
    parser.add_argument('--output', type=str, 
                       default=f'predictions/fraud_predictions_{DateUtils.get_datetime_string()}.csv',
                       help='Output file for predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Probability threshold for fraud classification')
    parser.add_argument('--include-probabilities', action='store_true',
                       help='Include prediction probabilities in output')
    
    args = parser.parse_args()
    
    # Initialize logger
    logger = Logger('fraud_detection_prediction')
    logger.info("Starting telecoms fraud detection prediction pipeline")
    
    try:
        # Step 1: Load trained model
        logger.info(f"Step 1: Loading trained model from {args.model}")
        
        model = FraudDetectionModel()
        if not model.load_model(args.model):
            logger.error("Failed to load model")
            return
        
        model_info = model.get_model_info()
        logger.info(f"Loaded {model_info['model_type']} model with {model_info['n_features']} features")
        
        # Step 2: Load prediction data
        logger.info(f"Step 2: Loading prediction data from {args.data}")
        
        prediction_df = DataHelpers.load_data_from_file(args.data)
        if prediction_df is None:
            logger.error("Failed to load prediction data")
            return
        
        logger.info(f"Loaded {prediction_df.shape[0]:,} records for prediction")
        
        # Step 3: Validate input data
        logger.info("Step 3: Validating input data")
        
        if not model.validate_input(prediction_df):
            logger.error("Input data validation failed")
            return
        
        # Step 4: Make predictions
        logger.info("Step 4: Making fraud predictions")
        
        # Make predictions
        predictions = model.predict(prediction_df)
        probabilities = model.predict_proba(prediction_df)
        
        # Create results DataFrame
        results_df = prediction_df[['customer_id']].copy()
        results_df['fraud_prediction'] = predictions
        results_df['is_fraud'] = (probabilities[:, 1] >= args.threshold).astype(int)
        
        if args.include_probabilities:
            results_df['fraud_probability'] = probabilities[:, 1]
            results_df['normal_probability'] = probabilities[:, 0]
        
        # Add risk categories
        results_df['risk_category'] = pd.cut(
            probabilities[:, 1],
            bins=[0, 0.3, 0.7, 1.0],
            labels=['Low', 'Medium', 'High']
        )
        
        # Step 5: Generate prediction summary
        logger.info("Step 5: Generating prediction summary")
        
        total_cases = len(results_df)
        predicted_fraud = int(results_df['is_fraud'].sum())
        fraud_rate = predicted_fraud / total_cases
        
        risk_distribution = results_df['risk_category'].value_counts().to_dict()
        
        summary = {
            'prediction_date': datetime.now().isoformat(),
            'model_file': args.model,
            'data_file': args.data,
            'threshold': args.threshold,
            'total_cases': total_cases,
            'predicted_fraud_cases': predicted_fraud,
            'predicted_normal_cases': total_cases - predicted_fraud,
            'fraud_rate': fraud_rate,
            'risk_distribution': risk_distribution,
            'high_risk_cases': risk_distribution.get('High', 0),
            'medium_risk_cases': risk_distribution.get('Medium', 0),
            'low_risk_cases': risk_distribution.get('Low', 0)
        }
        
        # Step 6: Save predictions and summary
        logger.info("Step 6: Saving predictions and summary")
        
        # Create output directory
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        
        # Save predictions
        if not DataHelpers.save_data_to_file(results_df, args.output):
            logger.error("Failed to save predictions")
            return
        
        logger.info(f"Predictions saved to: {args.output}")
        
        # Save summary
        summary_file = args.output.replace('.csv', '_summary.json')
        DataHelpers.export_to_json(summary, summary_file)
        
        # Step 7: Generate and display report
        logger.info("Step 7: Generating prediction report")
        
        report_lines = [
            "=" * 60,
            "FRAUD DETECTION PREDICTION REPORT",
            "=" * 60,
            f"Prediction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model: {args.model}",
            f"Data: {args.data}",
            f"Threshold: {args.threshold}",
            "",
            "PREDICTION RESULTS:",
            f"  Total Cases Analyzed: {total_cases:,}",
            f"  Predicted Fraud Cases: {predicted_fraud:,}",
            f"  Predicted Normal Cases: {total_cases - predicted_fraud:,}",
            f"  Predicted Fraud Rate: {fraud_rate:.4f}",
            "",
            "RISK DISTRIBUTION:",
            f"  High Risk (≥0.7): {risk_distribution.get('High', 0):,}",
            f"  Medium Risk (0.3-0.7): {risk_distribution.get('Medium', 0):,}",
            f"  Low Risk (<0.3): {risk_distribution.get('Low', 0):,}",
            "",
            "MODEL INFORMATION:",
            f"  Model Type: {model_info['model_type']}",
            f"  Features Used: {model_info['n_features']}",
            f"  Training Date: {model_info.get('metadata', {}).get('training_date', 'Unknown')}",
            "",
            "OUTPUT FILES:",
            f"  Predictions: {args.output}",
            f"  Summary: {summary_file}",
            "",
            "=" * 60
        ]
        
        # Save and display report
        report_content = "\n".join(report_lines)
        report_file = args.output.replace('.csv', '_report.txt')
        
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(report_content)
        
        # Log fraud detection summary
        logger.log_fraud_detection(
            n_cases=total_cases,
            fraud_rate=fraud_rate,
            high_risk_cases=risk_distribution.get('High', 0)
        )
        
        logger.info("Prediction pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
