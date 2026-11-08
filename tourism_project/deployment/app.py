import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# Same decision threshold used when evaluating the model in train.py
CLASSIFICATION_THRESHOLD = 0.45

# Streamlit UI for Wellness Tourism Package purchase prediction
st.title("Wellness Tourism Package Prediction")
st.write("A tool for the Visit with Us sales team that predicts whether a customer is likely to purchase the Wellness Tourism Package.")
st.write("Enter the customer and sales-interaction details below to get a prediction.")

# Collect customer details
st.subheader("Customer details")
Age = st.number_input("Age (years)", min_value=18, max_value=100, value=36)
Gender = st.selectbox("Gender", ["Male", "Female"])
MaritalStatus = st.selectbox("Marital status", ["Married", "Divorced", "Single", "Unmarried"])
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
MonthlyIncome = st.number_input("Gross monthly income", min_value=0, max_value=200000, value=22000, step=500)
CityTier = st.selectbox("City tier (1 = most developed)", [1, 2, 3])
Passport = st.selectbox("Holds a valid passport?", ["No", "Yes"])
OwnCar = st.selectbox("Owns a car?", ["Yes", "No"])
NumberOfTrips = st.number_input("Average number of trips per year", min_value=0, max_value=30, value=3)
PreferredPropertyStar = st.selectbox("Preferred hotel rating (stars)", [3, 4, 5])
NumberOfPersonVisiting = st.number_input("Number of people on the trip", min_value=1, max_value=10, value=3)
NumberOfChildrenVisiting = st.number_input("Number of children under 5 on the trip", min_value=0, max_value=5, value=1)

# Collect sales-interaction details
st.subheader("Sales interaction")
TypeofContact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
ProductPitched = st.selectbox("Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
DurationOfPitch = st.number_input("Duration of pitch (minutes)", min_value=1, max_value=120, value=14)
NumberOfFollowups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=4)
PitchSatisfactionScore = st.selectbox("Pitch satisfaction score (1-5)", [1, 2, 3, 4, 5], index=2)

# Convert user input into a DataFrame with the same column names used in training
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": 1 if Passport == "Yes" else 0,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": 1 if OwnCar == "Yes" else 0,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "MonthlyIncome": MonthlyIncome,
}])

# Predict and display the result
if st.button("Predict"):
    probability = model.predict_proba(input_data)[0, 1]
    prediction = int(probability >= CLASSIFICATION_THRESHOLD)
    st.subheader("Prediction result")
    if prediction == 1:
        st.success(f"Likely to purchase the Wellness Tourism Package (probability {probability:.0%}).")
    else:
        st.info(f"Unlikely to purchase the Wellness Tourism Package (probability {probability:.0%}).")
