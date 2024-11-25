# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import pickle

def prepare_features(df):
    """Prepare features for modeling"""
    numerical_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'cardiovascular_risk']
    categorical_cols = ['gender', 'smoking_history', 'bmi_category', 'HbA1c_category', 'glucose_category']
    
    # Create feature matrix
    X_num = df[numerical_cols].copy()
    X_cat = df[categorical_cols].copy()
    
    # Scale numerical features
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)
    X_num_scaled = pd.DataFrame(X_num_scaled, columns=numerical_cols)
    
    # Encode categorical features
    encoders = {}
    X_cat_encoded = pd.DataFrame()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X_cat_encoded[col] = le.fit_transform(X_cat[col])
        encoders[col] = le
    
    # Combine features
    X = pd.concat([X_num_scaled, X_cat_encoded], axis=1)
    y = df['diabetes']
    
    return X, y, encoders, scaler

def train_model():
    # Load data
    print("Loading data...")
    df = pd.read_csv('../data/diabetes_dataset_processed.csv')
    
    # Prepare features
    print("Preparing features...")
    X, y, encoders, scaler = prepare_features(df)
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    model = xgb.XGBClassifier(
        max_depth=4,
        learning_rate=0.1,
        n_estimators=200,
        min_child_weight=1,
        subsample=0.9,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Save model and components
    print("Saving model and components...")
    model_data = {
        'model': model,
        'encoders': encoders,
        'scaler': scaler,
        'feature_columns': X.columns.tolist(),
        'threshold': 0.2,  # Optimized threshold
        'risk_categories': {
            'very_low': {'min': 0.0, 'max': 0.2},
            'low': {'min': 0.2, 'max': 0.4},
            'moderate': {'min': 0.4, 'max': 0.6},
            'high': {'min': 0.6, 'max': 0.8},
            'very_high': {'min': 0.8, 'max': 1.0}
        }
    }
    
    with open('../models/final_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Training completed successfully!")
    
if __name__ == "__main__":
    train_model()