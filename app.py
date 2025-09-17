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

# Define the columns and their order as they were in the original dataframe
categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
numerical_cols_for_prediction = ['reading score', 'writing score']

# Define the categories explicitly based on the unique values observed in your original dataframe.
# This is needed for consistent one-hot encoding.
gender_categories = ['female', 'male']
race_ethnicity_categories = ['group A', 'group B', 'group C', 'group D', 'group E']
parental_education_categories = ["associate's degree", "bachelor's degree", "high school", "master's degree", "some college", "some high school"]
lunch_categories = ['free/reduced', 'standard']
test_prep_categories = ['completed', 'none']

# Create a OneHotEncoder for categorical features
# handle_unknown='ignore' is important for deployment
onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[
    gender_categories,
    race_ethnicity_categories,
    parental_education_categories,
    lunch_categories,
    test_prep_categories
])

# Define the exact column names and their order as expected by the trained model
# This list comes from printing X_train.columns.tolist() after one-hot encoding during training
expected_training_columns = [
    'reading score',
    'writing score',
    'gender_male',
    'race/ethnicity_group B',
    'race/ethnicity_group C',
    'race/ethnicity_group D',
    'race/ethnicity_group E',
    "parental level of education_bachelor's degree",
    "parental level of education_high school",
    "parental level of education_master's degree",
    "parental level of education_some college",
    "parental level of education_some high school",
    'lunch_standard',
    'test preparation course_none'
]


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
    try:
        # Separate categorical and numerical columns in the input DataFrame
        input_categorical_df = input_df[categorical_cols]
        input_numerical_df = input_df[numerical_cols_for_prediction]

        # Apply one-hot encoding to the categorical features
        input_categorical_encoded = onehot_encoder.fit_transform(input_categorical_df)

        # Convert the encoded categorical features to a DataFrame
        processed_categorical_df = pd.DataFrame(input_categorical_encoded)
        processed_categorical_df.columns = onehot_encoder.get_feature_names_out(categorical_cols) # Get names for encoded columns


        # Combine processed categorical and numerical features
        input_numerical_df.reset_index(drop=True, inplace=True)
        processed_categorical_df.reset_index(drop=True, inplace=True)
        combined_input_df = pd.concat([input_numerical_df, processed_categorical_df], axis=1)


        # Reindex the combined DataFrame to match the exact columns and order of X_train
        # Fill any missing columns (due to categories not present in input) with 0
        final_input_df = combined_input_df.reindex(columns=expected_training_columns, fill_value=0)


        # Make prediction
        # The model expects a numpy array or similar, so convert the DataFrame
        predicted_math_score = model.predict(final_input_df)


        st.header('Predicted Math Score')
        st.success(f'{predicted_math_score[0]:.2f}')

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure the input values are valid and the expected training columns are correct.")

st.write("""
This app predicts the math score of a student based on their demographic information,
lunch type, test preparation course completion, reading score, and writing score.
Input the student's details in the sidebar and click 'Predict Math Score'.
""")
