# Quick Start Guide - Telecoms Fraud Detection

Welcome to the Telecoms Fraud Detection ML project! This guide will get you up and running in minutes.

## Prerequisites

- Python 3.8+
- Git

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-username/telecomms-fraud-detection-ml.git
cd telecomms-fraud-detection-ml

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Quick Test Run

Test the entire pipeline with generated sample data:

```bash
# Run the complete pipeline test
python test_pipeline.py
```

This will:
- Generate 1000 sample customer records
- Process billing, CRM, social, and customer care data
- Unify the data sources
- Engineer fraud detection features
- Train a fraud detection model
- Test predictions

Expected output:
```
=== Telecoms Fraud Detection ML Pipeline Test ===
✅ Data processing completed: 1000 customers
✅ Feature engineering: 35 features created
✅ Model training successful: Random Forest trained
✅ Predictions working: Model ready for use
```

## 3. Train Your Own Model

### Option A: Use the Complete Training Script

```bash
# Train with generated sample data
python scripts/train_unified_fraud_model.py

# Train with your own data (place CSV files in data/raw/)
python scripts/train_unified_fraud_model.py --data-dir data/raw/
```

### Option B: Step-by-Step Training

```python
from telecoms_fraud_detection_ml.data import DataProcessor, DataUnifier
from telecoms_fraud_detection_ml.features import FeatureEngineer
from telecoms_fraud_detection_ml.models import FraudDetectionModel
from telecoms_fraud_detection_ml.utils.helpers import DataHelpers

# 1. Load and process your data
processor = DataProcessor()
billing_data = processor.clean_billing_data(your_billing_df)
crm_data = processor.clean_crm_data(your_crm_df)

# 2. Unify multiple data sources
unifier = DataUnifier()
unified_data = unifier.unify_datasets(
    billing_df=billing_data,
    crm_df=crm_data,
    social_df=your_social_df,
    customer_care_df=your_care_df
)

# 3. Engineer features
engineer = FeatureEngineer()
features = engineer.create_fraud_features(unified_data)

# 4. Create fraud labels (or use your own)
target = DataHelpers.create_fraud_labels(unified_data)

# 5. Train model
detector = FraudDetectionModel()
model_info = detector.train(features, target)
print(f"Model saved: {model_info}")
```

## 4. Make Fraud Predictions

```bash
# Make predictions on new data
python scripts/predict_fraud.py \
  --input data/new_customers.csv \
  --output fraud_predictions.csv \
  --model models/fraud_model_latest.pkl
```

Or programmatically:

```python
from telecoms_fraud_detection_ml.models import FraudDetectionModel
import pandas as pd

# Load the trained model
detector = FraudDetectionModel()
detector.load_model('models/fraud_model_20250711.pkl')

# Load new customer data
new_data = pd.read_csv('data/new_customers.csv')

# Make predictions
fraud_probabilities = detector.predict_proba(new_data)
fraud_predictions = detector.predict(new_data)

# High-risk customers (>80% fraud probability)
high_risk = new_data[fraud_probabilities[:, 1] > 0.8]
print(f"High-risk customers detected: {len(high_risk)}")
```

## 5. Project Structure

```
telecomms-fraud-detection-ml/
├── data/
│   ├── generated/              # Sample generated data
│   ├── processed/              # Processed datasets
│   └── raw/                    # Your raw data files
├── telecoms_fraud_detection_ml/
│   ├── data/                   # Data processing
│   ├── features/               # Feature engineering
│   ├── models/                 # Model training & evaluation
│   └── utils/                  # Utilities and helpers
├── scripts/
│   ├── train_unified_fraud_model.py  # Complete training pipeline
│   ├── predict_fraud.py              # Make predictions
│   ├── evaluate_model.py             # Model evaluation
│   └── comprehensive_demo.py         # Full demo
├── models/                     # Saved models (.pkl files)
└── test_pipeline.py           # Pipeline validation
```

## 6. Understanding Your Data

The system expects these data sources:

### Required Data Files:
- **Billing Data**: Customer charges, payment history, billing cycles
- **CRM Data**: Customer demographics, contract details, tenure
- **Usage Data**: Call patterns, data usage, roaming behavior
- **Customer Care**: Support interactions, complaints, resolution times

### Optional Data Files:
- **Social Data**: Social media mentions, network quality feedback
- **Network Data**: Service quality metrics, outage history

### Sample Data Format:

```csv
# billing.csv
customer_id,bill_amount,payment_date,late_payment,billing_cycle
CUST_000001,75.50,2024-01-15,0,monthly

# customers.csv (CRM data)
customer_id,tenure,monthly_charges,contract_type,senior_citizen
CUST_000001,24,75.50,2-year,0
```

## 7. Key Features for Fraud Detection

The system automatically creates these fraud indicators:

- **Usage Anomalies**: Extreme data usage, unusual call patterns
- **Billing Patterns**: High charges with low tenure, payment irregularities
- **Service Behavior**: Frequent support calls, complaint patterns
- **Risk Scores**: Composite fraud risk indicators
- **Temporal Features**: Recent activity changes, seasonal patterns

## 8. Model Performance

Typical performance metrics on sample data:
- **Accuracy**: 95%+
- **Precision**: 85%+ (low false positives)
- **Recall**: 80%+ (catches most fraud)
- **F1-Score**: 82%+

## 9. Common Commands

```bash
# Generate sample data for testing
python scripts/comprehensive_demo.py

# Evaluate a trained model
python scripts/evaluate_model.py --model models/fraud_model_latest.pkl

# Quick pipeline validation
python test_pipeline.py

# View model information
python -c "
from telecoms_fraud_detection_ml.models import FraudDetectionModel
detector = FraudDetectionModel()
detector.load_model('models/fraud_model_latest.pkl')
print(detector.get_model_info())
"
```

## 10. Next Steps

1. **Customize Fraud Rules**: Edit `telecoms_fraud_detection_ml/utils/helpers.py` to adjust fraud detection rules
2. **Add More Features**: Extend `telecoms_fraud_detection_ml/features/engineer.py` with domain-specific features
3. **Model Tuning**: Modify hyperparameters in `telecoms_fraud_detection_ml/models/fraud_detector.py`
4. **Deploy**: Use the saved `.pkl` model files in your production environment

## Troubleshooting

**Q: "Module not found" errors?**
A: Make sure you've activated your virtual environment and installed requirements.txt

**Q: "No fraud cases detected"?**
A: The fraud detection rules may need adjustment for your data. Edit the rules in `DataHelpers.create_fraud_labels()`

**Q: Low model performance?**
A: Try collecting more diverse data sources or adjusting the feature engineering parameters

Your telecoms fraud detection system is ready to help identify suspicious customer behavior and protect your business! 🛡️📱
