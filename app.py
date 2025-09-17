import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the trained Linear Regression model
# Make sure 'linear_regression_model.pkl' is in the same directory as your app.py file
try:
    model = joblib.load('linear_regression_model.pkl')
except FileNotFoundError:
    st.error("Model file 'linear_regression_model.pkl' not found. Please make sure it's in the correct directory.")
    st.stop()


# Define the columns and their order as they were in the training data
# This is crucial for correct one-hot encoding and prediction
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
numerical_cols_for_prediction = ['reading score', 'writing score'] # Features used for prediction

# To handle categorical features consistently, we need to replicate the one-hot encoding
# used during training. We'll create a dummy preprocessor here.
# In a real application, you would save and load the preprocessor fitted on the training data
# along with the model. For simplicity here, we'll define the categories explicitly
# based on the unique values observed in your original dataframe.
# IMPORTANT: Ensure these categories match the unique values in your training data exactly.
gender_categories = ['female', 'male']
race_ethnicity_categories = ['group A', 'group B', 'group C', 'group D', 'group E']
parental_education_categories = ["associate's degree", "bachelor's degree", "high school", "master's degree", "some college", "some high school"]
lunch_categories = ['free/reduced', 'standard']
test_prep_categories = ['completed', 'none']


# Create a ColumnTransformer for one-hot encoding.
# This needs to match how the training data was transformed.
preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', categories=[
            gender_categories,
            race_ethnicity_categories,
            parental_education_categories,
            lunch_categories,
            test_prep_categories
        ]), categorical_cols)
    ],
    remainder='passthrough' # Keep numerical columns as they are
)


st.title('Student Math Score Predictor')

st.sidebar.header('Input Student Information')

# Create input fields in the sidebar
gender = st.sidebar.selectbox('Gender', gender_categories)
race_ethnicity = st.sidebar.selectbox('Race/Ethnicity', race_ethnicity_categories)
parental_level_of_education = st.sidebar.selectbox('Parental Level of Education', parental_education_categories)
lunch = st.sidebar.selectbox('Lunch', lunch_categories)
test_preparation_course = st.sidebar.selectbox('Test Preparation Course', test_prep_categories)
reading_score = st.sidebar.number_input('Reading Score', min_value=0, max_value=100, value=70)
writing_score = st.sidebar.number_input('Writing Score', min_value=0, max_value=100, value=70)

# Create a dictionary with the input data
input_data = {
    'gender': gender,
    'race/ethnicity': race_ethnicity,
    'parental level of education': parental_level_of_education,
    'lunch': lunch,
    'test preparation course': test_preparation_course,
    'reading score': reading_score,
    'writing score': writing_score
}

# Convert the input data to a pandas DataFrame
input_df = pd.DataFrame([input_data])


# Add a button to trigger the prediction
if st.sidebar.button('Predict Math Score'):
    # Apply the same preprocessing as the training data
    # Important: The order of columns in input_df must match the order expected by the preprocessor
    # and the model.
    try:
        # Manually order columns to match training data for preprocessing
        input_df_ordered = input_df[categorical_cols + numerical_cols_for_prediction]
        processed_input = preprocessor.fit_transform(input_df_ordered) # Use fit_transform for single sample

        # Make prediction
        predicted_math_score = model.predict(processed_input)

        st.header('Predicted Math Score')
        st.success(f'{predicted_math_score[0]:.2f}')

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure the input values are valid.")

st.write("""
This app predicts the math score of a student based on their demographic information,
lunch type, test preparation course completion, reading score, and writing score.
Input the student's details in the sidebar and click 'Predict Math Score'.
""")

# Optional: Display the input data (for debugging)
# st.subheader("Input Data")
# st.dataframe(input_df)