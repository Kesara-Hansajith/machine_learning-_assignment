import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('random_forest_income_prediction_model.pkl')

# Define manual encoding based on training data categories (alphabetical order)
category_mappings = {
    'workclass': ['Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'],
    'education': ['Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Some-college'],
    'marital-status': ['Divorced', 'Married-civ-spouse', 'Never-married', 'Separated', 'Widowed'],
    'occupation': ['Lecturer', 'Docter' 'student', 'Craft-repair', 'Exec-managerial', 'Prof-specialty', 'Sales'],
    'relationship': ['Husband', 'Own-child', 'Wife'],
    'race': ['Black', 'White'],
    'sex': ['Female', 'Male'],
    'native-country': ['Sri Lanka', 'India', 'United-States', 'England', 'Brazil', 'France', 'China', 'Finland', 'Poland']
}

# Create numerical mapping dictionaries
encoding_dict = {}
for feature, categories in category_mappings.items():
    encoding_dict[feature] = {category: idx for idx, category in enumerate(categories)}

st.title("Income Prediction App")
st.write("Predict whether income exceeds $50K/year")

# Create input fields with manual encoding
input_data = {}
cols = st.columns(2)

with cols[0]:
    input_data['age'] = st.number_input('Age', 17, 90)
    input_data['workclass'] = encoding_dict['workclass'][st.selectbox('Work Class', category_mappings['workclass'])]
    input_data['education'] = encoding_dict['education'][st.selectbox('Education', category_mappings['education'])]
    input_data['education-num'] = st.number_input('Education Years', 1, 16)
    input_data['sex'] = encoding_dict['sex'][st.selectbox('Sex', category_mappings['sex'])]
    input_data['capital-gain'] = st.number_input('Capital Gain ($)', 0)
    input_data['hours-per-week'] = st.number_input('Hours/Week', 0, 100)

with cols[1]:
    input_data['marital-status'] = encoding_dict['marital-status'][st.selectbox('Marital Status', category_mappings['marital-status'])]
    input_data['occupation'] = encoding_dict['occupation'][st.selectbox('Occupation', category_mappings['occupation'])]
    input_data['relationship'] = encoding_dict['relationship'][st.selectbox('Relationship', category_mappings['relationship'])]
    input_data['fnlwgt'] = st.number_input('Final Weight (fnlwgt)', min_value=0, value=189778)
    input_data['race'] = encoding_dict['race'][st.selectbox('Race', category_mappings['race'])]
    input_data['capital-loss'] = st.number_input('Capital Loss ($)', 0)
    input_data['native-country'] = encoding_dict['native-country'][st.selectbox('Native Country', category_mappings['native-country'])]


# Add missing numerical features with default values


if st.button("Predict Income"):
    try:
        # Create feature order array matching model expectations
        feature_order = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
        ]
        
        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([input_data])[feature_order]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        
        # Display results
        st.success(f"Predicted Income:  {'>50K' if prediction == 1 else '<=50K'}")
        st.info(f"Probability of earning >$50K:  {proba*100:.2f}%")
        
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
