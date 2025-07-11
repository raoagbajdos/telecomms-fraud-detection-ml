"""
Model evaluator for telecoms fraud detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

class ModelEvaluator:
    """Evaluates fraud detection model performance."""
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.logger = logging.getLogger(__name__)
        self.evaluation_history = []
    
    def evaluate_model(self, 
                      model, 
                      test_data: pd.DataFrame, 
                      target_column: str) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a fraud detection model.
        
        Args:
            model: Trained fraud detection model
            test_data: Test DataFrame with features and target
            target_column: Name of the target column
            
        Returns:
            Dictionary containing comprehensive evaluation metrics
        """
        try:
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                roc_auc_score, confusion_matrix, classification_report,
                precision_recall_curve, roc_curve
            )
            
            # Prepare data
            features = test_data.drop(columns=[target_column, 'customer_id'], errors='ignore')
            y_true = test_data[target_column]
            
            # Make predictions
            y_pred = model.predict(features)
            y_proba = model.predict_proba(features)[:, 1]
            
            # Basic classification metrics
            basic_metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'precision': float(precision_score(y_true, y_pred, average='binary')),
                'recall': float(recall_score(y_true, y_pred, average='binary')),
                'f1_score': float(f1_score(y_true, y_pred, average='binary')),
                'roc_auc': float(roc_auc_score(y_true, y_proba))
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            confusion_metrics = {
                'true_negatives': int(cm[0, 0]),
                'false_positives': int(cm[0, 1]),
                'false_negatives': int(cm[1, 0]),
                'true_positives': int(cm[1, 1]),
                'confusion_matrix': cm.tolist()
            }
            
            # Business-specific metrics for fraud detection
            fraud_specific_metrics = self._calculate_fraud_metrics(
                y_true, y_pred, y_proba, test_data
            )
            
            # ROC and PR curves
            curve_data = self._calculate_curves(y_true, y_proba)
            
            # Feature importance if available
            feature_importance = None
            if hasattr(model, 'get_feature_importance'):
                feature_importance = model.get_feature_importance().to_dict('records')
            
            # Compile comprehensive results
            evaluation_results = {
                'basic_metrics': basic_metrics,
                'confusion_metrics': confusion_metrics,
                'fraud_specific_metrics': fraud_specific_metrics,
                'curve_data': curve_data,
                'feature_importance': feature_importance,
                'evaluation_metadata': {
                    'n_samples': len(test_data),
                    'n_features': len(features.columns),
                    'fraud_rate': float(y_true.mean()),
                    'model_type': getattr(model, 'model_type', 'unknown')
                }
            }
            
            # Store evaluation history
            self.evaluation_history.append(evaluation_results)
            
            self.logger.info(f"Model evaluation completed. ROC-AUC: {basic_metrics['roc_auc']:.4f}, "
                           f"Precision: {basic_metrics['precision']:.4f}, "
                           f"Recall: {basic_metrics['recall']:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error during model evaluation: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_fraud_metrics(self, 
                                y_true: pd.Series, 
                                y_pred: np.ndarray, 
                                y_proba: np.ndarray,
                                test_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate fraud-specific business metrics."""
        
        # Detection rate at different thresholds
        detection_rates = {}
        for threshold in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            pred_at_threshold = (y_proba >= threshold).astype(int)
            if y_true.sum() > 0:  # Avoid division by zero
                detection_rate = ((pred_at_threshold == 1) & (y_true == 1)).sum() / y_true.sum()
                false_positive_rate = ((pred_at_threshold == 1) & (y_true == 0)).sum() / (1 - y_true).sum()
            else:
                detection_rate = 0
                false_positive_rate = 0
            
            detection_rates[f'threshold_{threshold}'] = {
                'detection_rate': float(detection_rate),
                'false_positive_rate': float(false_positive_rate)
            }
        
        # Financial impact estimation (assuming cost per fraud case and investigation cost)
        fraud_cost_per_case = 1000  # Example: $1000 per fraud case
        investigation_cost_per_case = 50  # Example: $50 per investigation
        
        total_fraud_cases = int(y_true.sum())
        detected_fraud_cases = int(((y_pred == 1) & (y_true == 1)).sum())
        false_positives = int(((y_pred == 1) & (y_true == 0)).sum())
        missed_fraud_cases = total_fraud_cases - detected_fraud_cases
        
        financial_impact = {
            'total_fraud_cases': total_fraud_cases,
            'detected_fraud_cases': detected_fraud_cases,
            'missed_fraud_cases': missed_fraud_cases,
            'false_positives': false_positives,
            'estimated_fraud_cost_prevented': detected_fraud_cases * fraud_cost_per_case,
            'estimated_fraud_cost_missed': missed_fraud_cases * fraud_cost_per_case,
            'estimated_investigation_cost': (detected_fraud_cases + false_positives) * investigation_cost_per_case
        }
        
        # Risk score distribution
        risk_distribution = {
            'high_risk_count': int((y_proba >= 0.7).sum()),
            'medium_risk_count': int(((y_proba >= 0.3) & (y_proba < 0.7)).sum()),
            'low_risk_count': int((y_proba < 0.3).sum()),
            'mean_fraud_probability': float(y_proba.mean()),
            'std_fraud_probability': float(y_proba.std())
        }
        
        return {
            'detection_rates': detection_rates,
            'financial_impact': financial_impact,
            'risk_distribution': risk_distribution
        }
    
    def _calculate_curves(self, y_true: pd.Series, y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate ROC and Precision-Recall curves."""
        try:
            from sklearn.metrics import roc_curve, precision_recall_curve
            
            # ROC Curve
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
            
            # Precision-Recall Curve
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
            
            return {
                'roc_curve': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': roc_thresholds.tolist()
                },
                'pr_curve': {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'thresholds': pr_thresholds.tolist()
                }
            }
        except Exception as e:
            self.logger.warning(f"Could not calculate curves: {str(e)}")
            return {}
    
    def compare_models(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple model evaluation results.
        
        Args:
            evaluation_results: List of evaluation result dictionaries
            
        Returns:
            Dictionary containing model comparison
        """
        if not evaluation_results:
            return {'error': 'No evaluation results provided'}
        
        comparison_data = []
        
        for i, result in enumerate(evaluation_results):
            if 'error' in result:
                continue
                
            model_info = {
                'model_id': i,
                'model_type': result.get('evaluation_metadata', {}).get('model_type', 'unknown'),
                'accuracy': result['basic_metrics']['accuracy'],
                'precision': result['basic_metrics']['precision'],
                'recall': result['basic_metrics']['recall'],
                'f1_score': result['basic_metrics']['f1_score'],
                'roc_auc': result['basic_metrics']['roc_auc'],
                'detected_fraud_cases': result['fraud_specific_metrics']['financial_impact']['detected_fraud_cases'],
                'false_positives': result['fraud_specific_metrics']['financial_impact']['false_positives']
            }
            comparison_data.append(model_info)
        
        if not comparison_data:
            return {'error': 'No valid evaluation results to compare'}
        
        # Find best model by different metrics
        comparison_df = pd.DataFrame(comparison_data)
        
        best_models = {
            'best_roc_auc': comparison_df.loc[comparison_df['roc_auc'].idxmax()].to_dict(),
            'best_precision': comparison_df.loc[comparison_df['precision'].idxmax()].to_dict(),
            'best_recall': comparison_df.loc[comparison_df['recall'].idxmax()].to_dict(),
            'best_f1': comparison_df.loc[comparison_df['f1_score'].idxmax()].to_dict()
        }
        
        # Overall ranking (weighted score)
        comparison_df['weighted_score'] = (
            0.4 * comparison_df['roc_auc'] +
            0.3 * comparison_df['f1_score'] +
            0.2 * comparison_df['precision'] +
            0.1 * comparison_df['recall']
        )
        
        overall_best = comparison_df.loc[comparison_df['weighted_score'].idxmax()].to_dict()
        
        return {
            'comparison_table': comparison_df.to_dict('records'),
            'best_models': best_models,
            'overall_best': overall_best,
            'summary_statistics': {
                'mean_roc_auc': float(comparison_df['roc_auc'].mean()),
                'std_roc_auc': float(comparison_df['roc_auc'].std()),
                'mean_f1_score': float(comparison_df['f1_score'].mean()),
                'std_f1_score': float(comparison_df['f1_score'].std())
            }
        }
    
    def generate_evaluation_report(self, evaluation_result: Dict[str, Any]) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_result: Dictionary containing evaluation results
            
        Returns:
            Formatted evaluation report string
        """
        if 'error' in evaluation_result:
            return f"Evaluation Error: {evaluation_result['error']}"
        
        report = ["=" * 50]
        report.append("FRAUD DETECTION MODEL EVALUATION REPORT")
        report.append("=" * 50)
        
        # Basic metrics
        basic = evaluation_result['basic_metrics']
        report.append("\nPERFORMANCE METRICS:")
        report.append(f"  Accuracy: {basic['accuracy']:.4f}")
        report.append(f"  Precision: {basic['precision']:.4f}")
        report.append(f"  Recall: {basic['recall']:.4f}")
        report.append(f"  F1-Score: {basic['f1_score']:.4f}")
        report.append(f"  ROC-AUC: {basic['roc_auc']:.4f}")
        
        # Confusion matrix
        cm = evaluation_result['confusion_metrics']
        report.append("\nCONFUSION MATRIX:")
        report.append(f"  True Negatives: {cm['true_negatives']:,}")
        report.append(f"  False Positives: {cm['false_positives']:,}")
        report.append(f"  False Negatives: {cm['false_negatives']:,}")
        report.append(f"  True Positives: {cm['true_positives']:,}")
        
        # Business impact
        financial = evaluation_result['fraud_specific_metrics']['financial_impact']
        report.append("\nBUSINESS IMPACT:")
        report.append(f"  Total Fraud Cases: {financial['total_fraud_cases']:,}")
        report.append(f"  Detected Fraud Cases: {financial['detected_fraud_cases']:,}")
        report.append(f"  Missed Fraud Cases: {financial['missed_fraud_cases']:,}")
        report.append(f"  False Positives: {financial['false_positives']:,}")
        
        # Risk distribution
        risk = evaluation_result['fraud_specific_metrics']['risk_distribution']
        report.append("\nRISK DISTRIBUTION:")
        report.append(f"  High Risk (≥0.7): {risk['high_risk_count']:,}")
        report.append(f"  Medium Risk (0.3-0.7): {risk['medium_risk_count']:,}")
        report.append(f"  Low Risk (<0.3): {risk['low_risk_count']:,}")
        
        # Model metadata
        metadata = evaluation_result['evaluation_metadata']
        report.append("\nMODEL INFORMATION:")
        report.append(f"  Model Type: {metadata['model_type']}")
        report.append(f"  Test Samples: {metadata['n_samples']:,}")
        report.append(f"  Features: {metadata['n_features']}")
        report.append(f"  Dataset Fraud Rate: {metadata['fraud_rate']:.4f}")
        
        report.append("\n" + "=" * 50)
        
        return "\n".join(report)
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all evaluations.
        
        Returns:
            List of evaluation result dictionaries
        """
        return self.evaluation_history.copy()
    
    def export_evaluation_results(self, evaluation_result: Dict[str, Any], filepath: str) -> bool:
        """
        Export evaluation results to a file.
        
        Args:
            evaluation_result: Dictionary containing evaluation results
            filepath: Path to save the results
            
        Returns:
            True if export was successful
        """
        try:
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            serializable_result = convert_numpy(evaluation_result)
            
            with open(filepath, 'w') as f:
                json.dump(serializable_result, f, indent=2)
            
            self.logger.info(f"Evaluation results exported to: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting evaluation results: {str(e)}")
            return False
