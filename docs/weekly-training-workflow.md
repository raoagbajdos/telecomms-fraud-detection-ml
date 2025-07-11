# GitHub Actions Weekly Model Training

This document explains the automated weekly model training workflow for the telecoms fraud detection ML project.

## Overview

The GitHub Actions workflow automatically trains the fraud detection model every Sunday at 2 AM UTC using the sample telecoms data. This ensures the model stays up-to-date and provides a consistent `model.pkl` file for deployment.

## Workflow Details

### Schedule
- **Frequency**: Weekly (every Sunday)
- **Time**: 2:00 AM UTC
- **Trigger**: Cron schedule `0 2 * * 0`

### Manual Triggering
The workflow can also be triggered manually from the GitHub Actions tab with optional parameters:
- **Model Type**: Choose specific model (random_forest, xgboost, logistic_regression) or auto-select
- **Create Fraud Labels**: Option to generate synthetic fraud labels for training

### Workflow Steps

1. **Environment Setup**
   - Sets up Ubuntu latest with Python 3.11
   - Installs project dependencies from `requirements.txt`
   - Creates necessary directories (models, logs, data/processed)

2. **Model Training**
   - Runs the training pipeline using sample telecoms data
   - Creates both dated model files and standard `model.pkl`
   - Generates training reports and logs

3. **Validation**
   - Validates model file creation
   - Tests model loading functionality
   - Checks model file sizes and formats
   - Runs pipeline validation tests

4. **Artifact Management**
   - Archives model files, reports, and logs
   - Commits new model files to the repository
   - Creates GitHub releases with model details
   - Retains artifacts for 90 days

5. **Notification**
   - Reports training completion status
   - Provides summary of generated files

## Generated Files

Each weekly training run generates:

### Model Files
- `models/model.pkl` - Standard model file for deployment
- `models/fraud_model_YYYYMMDD_HHMMSS.pkl` - Dated model file for versioning

### Reports
- `models/training_report.json` - JSON report with training metadata
- `logs/*.log` - Training logs for debugging

### Metadata
- Training timestamp
- Model performance metrics
- Feature information
- Data processing statistics

## Usage

### Accessing the Model
```python
from telecoms_fraud_detection_ml.models import FraudDetectionModel

# Load the latest model
detector = FraudDetectionModel()
detector.load_model('models/model.pkl')

# Make predictions
predictions = detector.predict(customer_data)
```

### Checking Model Status
```bash
# Check model file
ls -la models/model.pkl

# View training report
cat models/training_report.json

# Check recent logs
ls -la logs/
```

## Monitoring

### GitHub Actions Tab
- View workflow run history
- Check training success/failure status
- Download artifacts from recent runs

### Repository Releases
- Each successful training creates a release
- Tagged as `model-weekly-{run_number}`
- Includes model files and documentation

### Notifications
- Workflow sends status notifications
- Success/failure reports in GitHub Actions

## Configuration

The workflow uses configuration from:
- `config/config.yaml` - Model training parameters
- `requirements.txt` - Python dependencies
- Sample data in `data/generated/` directory

## Benefits

1. **Automated Maintenance**: Model stays current without manual intervention
2. **Consistent Deployment**: Always provides `model.pkl` for production use
3. **Version Control**: Maintains history of model changes
4. **Quality Assurance**: Automated testing and validation
5. **Artifact Preservation**: Keeps training history and metrics

## Troubleshooting

### Common Issues
- **Training Failures**: Check logs in workflow run details
- **Missing Dependencies**: Verify requirements.txt completeness
- **Data Issues**: Ensure sample data is available and properly formatted

### Manual Recovery
If automatic training fails:
1. Go to GitHub Actions tab
2. Select "Weekly Model Training" workflow
3. Click "Run workflow" to trigger manually
4. Review logs for error details

## Security

- Uses GitHub's secure runners
- No sensitive data in repositories
- Model files are version controlled
- Access controlled through GitHub permissions

## Future Enhancements

Potential improvements:
- Model performance monitoring
- A/B testing between model versions
- Automatic model deployment to staging
- Integration with monitoring systems
- Email notifications for training status
