# mlzoomcamp2024-midterm

# Diabetes Risk Prediction System

## Problem Description
This project implements a machine learning system to predict diabetes risk based on various health metrics. The system analyzes patient health data including BMI, blood glucose levels, HbA1c, and other medical indicators to assess the likelihood of diabetes. This tool can assist healthcare providers in early risk assessment and preventive care decisions.

### Target Audience
- Healthcare providers
- Medical screening facilities
- Preventive care clinics

### Key Features
- Risk categorization (Very Low to Very High)
- Medical threshold-based predictions
- Real-time API predictions
- Containerized deployment

## Dataset

The dataset includes the following features:
- gender: Patient's gender
- age: Patient's age
- hypertension: Hypertension history (0 or 1)
- heart_disease: Heart disease history (0 or 1)
- smoking_history: Smoking status
- bmi: Body Mass Index
- HbA1c_level: Hemoglobin A1c level
- blood_glucose_level: Blood glucose level
- diabetes: Target variable (0 or 1)


## Exploratory Data Analysis

Key insights from data analysis (`ingestion.ipynb`):
1. Feature Distributions
   - BMI distribution shows 47.1% overweight, 21.8% normal
   - HbA1c levels: 42.7% prediabetic, 37.9% normal, 19.5% diabetic
   - Blood glucose: 71.9% in diabetes range, 21.1% normal

2. Feature Engineering
   - Created medical category features (BMI, HbA1c, glucose)
   - Combined heart_disease and hypertension into cardiovascular_risk
   - Standardized numerical features

3. Data Quality
   - No missing values
   - Outlier handling using Z-score method
   - Feature correlations analyzed

## Model Development

1. Feature Selection
   - Key correlations with diabetes:
     * HbA1c_level: 0.296
     * blood_glucose_level: 0.280
     * cardiovascular_risk: 0.247
     * bmi: 0.200

2. Model Training
   - Algorithm: XGBoost Classifier
   - Optimized threshold: 0.2
   - Risk categorization system implemented

### API Service
- Flask-based REST API
- JSON request/response format
- Error handling and input validation
- Risk categorization in responses

### Containerization
- Docker implementation
- Python 3.10 slim base image
- Security considerations (non-root user)
- Environment variable configuration

## Project Structure

```
project/
├── data/
│   └── diabetes_dataset_processed.csv
├── models/
│   ├── final_model.pkl
│   ├── threshold_optimization.pkl
│   └── evaluation_results.pkl
├── notebooks/
│   ├── ingestion.ipynb         # Data Ingestion Pipeline + EDA
│   ├── evaluate-1.ipynb        # Initial model evaluation
│   ├── evaluate-2.ipynb        # Model tuning & optimization
│   └── evaluate_final.ipynb    # Final model implementation
├── scripts/
│   └── predict.py             # Flask API implementation
│   └── train.py               # Model training script
├── Dockerfile
└── requirements.txt
```

## Model Details

### Features
- Age
- BMI
- Blood Glucose Level
- HbA1c Level
- Hypertension
- Heart Disease
- Smoking History
- Gender

### Performance Metrics
- AUC-ROC: 0.972
- Precision: 0.912
- Recall: 0.593
- Cross-validation mean: 0.971 (±0.003)

### Risk Categories
- Very Low Risk: < 20%
- Low Risk: 20-40%
- Moderate Risk: 40-60%
- High Risk: 60-80%
- Very High Risk: > 80%

## API Usage

### Endpoint
`POST /predict`

### Request Format
```json
{
    "age": 45,
    "bmi": 28.5,
    "HbA1c_level": 6.5,
    "blood_glucose_level": 140,
    "cardiovascular_risk": 1,
    "gender": "female",
    "smoking_history": "never",
    "bmi_category": "Overweight",
    "HbA1c_category": "Prediabetes",
    "glucose_category": "Normal"
}
```

### Response Format
```json
{
    "probability": 0.291,
    "prediction": "Diabetes Risk",
    "risk_category": "Low Risk",
    "risk_score": "29.1%"
}
```
## Setup and Deployment

### Local Development
```bash
# Clone repository
git clone https://github.com/hypertoast/mlzoomcamp2024-midterm

# Install dependencies
pip install -r requirements.txt

# Run Flask API
python scripts/predict.py
```

### Docker Deployment
```bash
# Build Docker image
docker build -t diabetes-prediction-api .

# Run container
docker run -p 9696:9696 diabetes-prediction-api
```
## Model Training Process

1. Initial Evaluation (evaluate-1.ipynb)
   - Base model implementation with XGBoost
   - Performance assessment
   - Validation metrics:
     * AUC-ROC
     * Precision & Recall
     * Confusion Matrix
     * Cross-validation

2. Threshold Optimization (evaluate-2.ipynb)
   - Analysis of different prediction thresholds
   - Optimization for medical context (prioritizing recall)
   - Testing different thresholds:
     * Conservative (0.5)
     * Balanced (0.4)
     * Sensitive (0.2)
   - Risk category implementation

3. Final Implementation (evaluate_final.ipynb)
   - Implementation with optimized threshold (0.2)
   - Risk categorization system
   - Final model export
   - Comprehensive test cases

## Security Considerations

- Non-root user in Docker container
- Input validation
- Error handling
- Secure dependencies
