import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle
from typing import Dict, Any
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask('diabetes_prediction')

# Load the model and components
try:
    with open('../models/final_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        encoders = model_data['encoders']
        scaler = model_data['scaler']
        feature_columns = model_data['feature_columns']
        threshold = model_data['threshold']
        risk_categories = model_data['risk_categories']
    
    logger.info(f"Loaded risk categories: {risk_categories}")
    logger.info(f"Risk categories type: {type(risk_categories)}")
    if isinstance(risk_categories, dict):
        logger.info(f"Risk category keys type: {type(list(risk_categories.keys())[0])}")
    
    categorical_values = {
        column: list(encoder.classes_) 
        for column, encoder in encoders.items()
    }
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

def get_risk_category(probability: float) -> str:
    """Determine risk category based on probability."""
    prob_percentage = probability * 100
    
    if prob_percentage < 20:
        return "Very Low Risk"
    elif prob_percentage < 40:
        return "Low Risk"
    elif prob_percentage < 60:
        return "Moderate Risk"
    elif prob_percentage < 80:
        return "High Risk"
    else:
        return "Very High Risk"

def prepare_features(data: Dict[str, Any]) -> pd.DataFrame:
    """Prepare features for prediction using encoders and scaler."""
    try:
        # Create initial DataFrame
        df = pd.DataFrame([data])
        logger.info(f"Initial DataFrame columns: {df.columns.tolist()}")
        
        # Validate categorical values before processing
        categorical_features = ['gender', 'smoking_history', 'bmi_category', 'HbA1c_category', 'glucose_category']
        for col in categorical_features:
            if col in df.columns:
                is_valid = df[col].iloc[0] in categorical_values[col]
                if not is_valid:
                    raise ValueError(f"Invalid value for {col}. Must be one of: {categorical_values[col]}")
        
        # Scale numeric features
        numeric_features = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'cardiovascular_risk']
        logger.info(f"Scaling numeric features: {numeric_features}")
        df[numeric_features] = scaler.transform(df[numeric_features])
        
        # Encode categorical features
        for col in categorical_features:
            if col in df.columns:
                logger.info(f"Encoding {col} with value: {df[col].iloc[0]}")
                df[col] = encoders[col].transform(df[col])
                logger.info(f"Encoded {col} result: {df[col].values}")
        
        # Ensure correct column order
        df = df[feature_columns]
        
        return df
    
    except Exception as e:
        logger.error(f"Error in prepare_features: {str(e)}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for diabetes prediction."""
    try:
        patient_data = request.get_json()
        logger.info(f"Received request data: {patient_data}")
        
        if not patient_data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Request body is empty'
            }), 400
        
        features_df = prepare_features(patient_data)
        
        probability = float(model.predict_proba(features_df)[0][1])
        logger.info(f"Predicted probability: {probability}")
        
        prediction = "Diabetes Risk" if probability >= threshold else "No Diabetes Risk"
        risk_category = get_risk_category(probability)
        
        logger.info(f"Final prediction: {prediction}, category: {risk_category}")
        
        return jsonify({
            'probability': probability,
            'prediction': prediction,
            'risk_category': risk_category,
            'risk_score': f"{probability*100:.1f}%"
        })
    
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({
            'error': 'Invalid input',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500

@app.route('/info', methods=['GET'])
def info():
    """Endpoint providing information about the model and valid input values."""
    return jsonify({
        'model_info': {
            'features': feature_columns,
            'categorical_values': categorical_values,
            'threshold': threshold,
            'risk_categories': {
                'very_low': '< 20%',
                'low': '20-40%',
                'moderate': '40-60%',
                'high': '60-80%',
                'very_high': '> 80%'
            }
        },
        'input_ranges': {
            'age': {'min': 0, 'max': 120},
            'bmi': {'min': 10, 'max': 100},
            'blood_glucose_level': {'min': 0, 'max': 500},
            'HbA1c_level': {'min': 0, 'max': 15},
            'cardiovascular_risk': {'min': 0, 'max': 1}
        },
        'example_request': {
            'age': 45,
            'bmi': 28.5,
            'HbA1c_level': 6.5,
            'blood_glucose_level': 140,
            'cardiovascular_risk': 1,
            'gender': 'female',
            'smoking_history': 'never',
            'bmi_category': 'Overweight',
            'HbA1c_category': 'Prediabetes',
            'glucose_category': 'Normal'
        }
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)