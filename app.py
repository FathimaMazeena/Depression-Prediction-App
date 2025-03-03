import streamlit as st
import joblib
import pandas as pd
from MentalHealthAnalysis import TestDataCleaner

# Load models
model = joblib.load('depression_prediction_stack.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Streamlit app title
st.title("Mental Health Prediction App")

# Input fields for user data
st.write("Please fill out the form below:")

# Create a form for user input
with st.form("mental_health_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=10, max_value=100)
    city = st.text_input("City")
    work_or_student = st.selectbox("Are you a student or a working professional?", ["Student", "Working Professional"])
    profession = st.text_input("Profession")
    degree = st.text_input("Degree")
    pressure = st.slider("Work/Academic Pressure (1-5)", min_value=1, max_value=5, value=1)
    satisfaction = st.slider("Study/Work Satisfaction (1-5)", min_value=1, max_value=5, value=1)
    cgpa = st.number_input("CGPA: If you are a student (0-4 scale)", min_value=0.0, max_value=4.0, step=0.01)
    financial_stress = st.slider("Financial Stress (1-5)", min_value=1, max_value=5, value=1)
    hours = st.number_input("Work/Study Hours per Day", min_value=1, max_value=18)
    diet = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])
    sleep = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
    family_history = st.radio("Family History of Mental Illness", ["Yes", "No"])
    suicidal_thoughts = st.radio("Have you ever had suicidal thoughts?", ["Yes", "No"])

    # Submit button
    submitted = st.form_submit_button("Submit")

# Process the form submission
if submitted:
    try:
        # Create a dictionary from the form data
        data = {
            'Gender': gender,
            'Age': age,
            'City': city,
            'Working Professional or Student': work_or_student,
            'Profession': profession,
            'Academic Pressure': 0,
            'Work Pressure': 0,
            'CGPA': cgpa,
            'Study Satisfaction': 0,
            'Job Satisfaction': 0,
            'Sleep Duration': sleep,
            'Dietary Habits': diet,
            'Degree': degree,
            'Have you ever had suicidal thoughts ?': suicidal_thoughts,
            'Work/Study Hours': hours,
            'Financial Stress': financial_stress,
            'Family History of Mental Illness': family_history
        }

        # Adjust fields based on whether the user is a student or working professional
        if work_or_student == 'Student':
            data['Study Satisfaction'] = satisfaction
            data['Academic Pressure'] = pressure
        else:
            data['Job Satisfaction'] = satisfaction
            data['Work Pressure'] = pressure

        # Convert the dictionary to a DataFrame
        data_df = pd.DataFrame([data])

        # Clean the input data
        cleaner = TestDataCleaner(data_df)
        cleaned_input = cleaner.clean_data()

        # Ensure the cleaned input is still a DataFrame
        if not isinstance(cleaned_input, pd.DataFrame):
            cleaned_input = pd.DataFrame([cleaned_input])

        # Define categorical columns
        categorical_cols = [
            'Gender', 'City', 'Profession', 'Degree', 'Dietary Habits', 'Sleep Duration',
            'Family History of Mental Illness', 'Working Professional or Student',
            'Have you ever had suicidal thoughts ?'
        ]

        # Get the one-hot encoder and feature names from the encoder
        one_hot_encoder = encoder["one_hot"]
        feature_names = encoder["feature_names"]

        # Apply the encoder to the user input data
        encoded_input = one_hot_encoder.transform(cleaned_input[categorical_cols])
        encoded_input = pd.DataFrame(encoded_input, columns=one_hot_encoder.get_feature_names_out(categorical_cols))

        # Add numerical columns to the encoded input
        numerical_cols = [
            'Age', 'Academic Pressure', 'Work Pressure', 'CGPA', 'Study Satisfaction',
            'Job Satisfaction', 'Work/Study Hours', 'Financial Stress'
        ]

        for col in numerical_cols:
            encoded_input[col] = cleaned_input[col].values

        # Ensure the encoded input includes all feature names
        encoded_input = encoded_input.reindex(columns=feature_names, fill_value=0)

        # Reorder columns to match the feature names used during training
        encoded_input = encoded_input[model.feature_names_in_]
       
        # Scale numerical features using the pre-fitted scaler
        encoded_input[numerical_cols] = scaler.transform(encoded_input[numerical_cols])

        # Make prediction
        prediction = model.predict(encoded_input)

        # Display the prediction result
        st.write(f"Prediction: {'Depressed' if prediction[0] == 1 else 'Not Depressed'}")

    except Exception as e:
        st.error(f"An error occurred: {e}")