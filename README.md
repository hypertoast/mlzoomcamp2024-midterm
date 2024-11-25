# mlzoomcamp2024-midterm

# Diabetes Risk Prediction System

## Project Overview
This project implements a machine learning system for predicting diabetes risk based on various health metrics. It includes data preprocessing, model development, API implementation, and containerization.

### Key Features
- Machine learning model to predict diabetes risk
- REST API for real-time predictions
- Docker containerization for easy deployment
- Detailed validation and model tuning
- Risk categorization system

## Technical Architecture

### Data Pipeline
1. Data Ingestion & Cleaning
   - Feature engineering
   - Data validation
   - Outlier handling

2. Model Development
   - XGBoost classifier
   - Cross-validation
   - Hyperparameter tuning
   - Threshold optimization for medical context

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
project/
├── data/
│   └── diabetes_dataset_processed.csv
├── models/
│   ├── final_model.pkl
│   ├── threshold_optimization.pkl
│   └── evaluation_results.pkl
├── notebooks/
│   ├── evaluate-1.ipynb        # Initial model evaluation
│   ├── evaluate-2.ipynb        # Model tuning & optimization
│   └── evaluate_final.ipynb    # Final model implementation
├── scripts/
│   └── predict.py             # Flask API implementation
├── Dockerfile
└── requirements.txt

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


### Response Format
```json
{
    "probability": 0.291,
    "prediction": "Diabetes Risk",
    "risk_category": "Low Risk",
    "risk_score": "29.1%"
}

## Setup and Deployment

### Local Development
```bash
# Clone repository
git clone https://github.com/hypertoast/mlzoomcamp2024-midterm

# Install dependencies
pip install -r requirements.txt

# Run Flask API
python scripts/predict.py


### Docker Deployment
```bash
# Build Docker image
docker build -t diabetes-prediction-api .

# Run container
docker run -p 9696:9696 diabetes-prediction-api

## Model Training Process

### Initial Evaluation (evaluate-1.ipynb)

- Base model implementation
- Performance assessment
- Validation strategy


### Model Tuning (evaluate-2.ipynb)

- Threshold optimization
- Hyperparameter tuning
- Cross-validation


### Final Implementation (evaluate_final.ipynb)

- Best model selection
- Final evaluation
- Model export

## Security Considerations

- Non-root user in Docker container
- Input validation
- Error handling
- Secure dependencies
