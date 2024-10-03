#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from flask import Flask, request, jsonify
import joblib

# Create Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('loan_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the request data
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame(data, index=[0])

        # Feature engineering and preprocessing
        # Map categorical variables
        education_mapping = {'Graduate': 1, 'Not Graduate': 0}
        self_employed_mapping = {'Yes': 1, 'No': 0}
        
        df['education'] = df['education'].map(education_mapping)
        df['self_employed'] = df['self_employed'].map(self_employed_mapping)

        # Scale numerical features
        numerical_cols = ['no_of_dependents', 'income_annum', 'loan_amount', 
                          'loan_term', 'cibil_score', 'residential_assets_value', 
                          'commercial_assets_value', 'luxury_assets_value', 
                          'bank_asset_value']
        
        # Create the income_loan_ratio feature
        df['income_loan_ratio'] = df['income_annum'] / df['loan_amount']

        # Scale the numerical features
        df[numerical_cols] = scaler.transform(df[numerical_cols])

        # Ensure only relevant features are passed
        features = ['no_of_dependents', 'education', 'self_employed', 'income_annum',
                    'loan_amount', 'loan_term', 'cibil_score', 
                    'residential_assets_value', 'commercial_assets_value', 
                    'luxury_assets_value', 'bank_asset_value', 
                    'income_loan_ratio']  # Include the new feature

        # Reindex the DataFrame to match the expected feature set
        df = df.reindex(columns=features)

        # Ensure all necessary features are present
        for feature in features:
            if feature not in df.columns:
                df[feature] = 0  # If any feature is missing, fill it with 0

        # Make prediction
        prediction = model.predict(df)

        # Convert prediction to readable format
        result = 'Approved' if prediction[0] == 1 else 'Rejected'

        return jsonify({'loan_status': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
