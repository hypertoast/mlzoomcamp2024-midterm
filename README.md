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
│   ├── model_selection.ipynb    # Model comparison & selection
│   ├── evaluate-1.ipynb        # Initial model evaluation
│   ├── evaluate-2.ipynb        # Model tuning & optimization
│   └── evaluate_final.ipynb    # Final model implementation
├── scripts/
│   └── predict.py             # Flask API implementation
│   └── train.py               # Model training script
├── Dockerfile
└── requirements.txt
```



## Exploratory Data Analysis

### About Dataset

It is procured from Kaggle [Diabetes prediction dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)

The Diabetes prediction dataset is a collection of medical and demographic data from patients, along with their diabetes status (positive or negative). The data includes features such as age, gender, body mass index (BMI), hypertension, heart disease, smoking history, HbA1c level, and blood glucose level. This dataset can be used to build machine learning models to predict diabetes in patients based on their medical history and demographic information. This can be useful for healthcare professionals in identifying patients who may be at risk of developing diabetes and in developing personalized treatment plans. Additionally, the dataset can be used by researchers to explore the relationships between various medical and demographic factors and the likelihood of developing diabetes.

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

## Analysis & Model Development Process

### 1. Data Analysis & Feature Engineering (ingestion_slim.ipynb)
- Data cleaning and preprocessing
- Feature distribution analysis
- Outlier detection and handling
- Feature engineering:
  * BMI categories
  * HbA1c categories
  * Glucose categories
  * Cardiovascular risk score
- Correlation analysis

### 2. Model Selection (model_selection.ipynb)
Compared multiple models with comprehensive evaluation:

| Model                | Accuracy | Precision | Recall   | F1       | ROC_AUC  |
|---------------------|----------|-----------|----------|----------|----------|
| Logistic Regression | 0.93880  | 0.760215  | 0.413934 | 0.536012 | 0.943884 |
| Random Forest       | 0.95320  | 0.823826  | 0.574941 | 0.677241 | 0.957245 |
| XGBoost            | 0.95935  | 0.888118  | 0.599532 | 0.715834 | 0.970273 |

XGBoost selected as final model due to:
- Highest ROC-AUC score (0.970)
- Best F1 score (0.716)
- Superior precision-recall balance

### 3. Model Optimization Process
1. Initial Evaluation (evaluate-1.ipynb)
   - Base XGBoost implementation
   - Performance metrics analysis
   - Cross-validation

2. Threshold Optimization (evaluate-2.ipynb)
   - Medical context consideration
   - Threshold analysis
   - Risk category implementation

3. Final Implementation (evaluate_final.ipynb)
   - Implementation with optimized threshold
   - Final validation
   - Model export preparation

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
- Flask-based REST API
- JSON request/response format
- Error handling and input validation
- Risk categorization in responses

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
- Docker implementation
- Python 3.10 slim base image
- Security considerations (non-root user)
- Environment variable configuration

```bash
# Build Docker image
docker build -t diabetes-prediction-api .

# Run container
docker run -p 9696:9696 diabetes-prediction-api
```

## Security Considerations

- Non-root user in Docker container
- Input validation
- Error handling
- Secure dependencies
