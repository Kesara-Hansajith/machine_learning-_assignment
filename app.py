import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
model = joblib.load('random_forest_income_prediction_model.pkl')

st.title('Income Prediction App')

# Create input fields for all features
age = st.number_input('Age', min_value=17, max_value=90)
workclass = st.selectbox('Work Class', ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov'])
education = st.selectbox('Education', ['Bachelors', 'HS-grad', 'Some-college', 'Masters', 'Doctorate'])
# Add other input fields

if st.button('Predict'):
    # Create input data dictionary
    input_data = {
        'age': age,
        'workclass': workclass,
        'education': education,
        # Add other fields
    }
    
    # Convert to DataFrame and preprocess
    input_df = pd.DataFrame([input_data])
    # Apply the same preprocessing as during training
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    st.success(f'Predicted Income: {"More than $50K" if prediction == 1 else "$50K or less"}')
