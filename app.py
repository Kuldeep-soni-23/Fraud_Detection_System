import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
from geopy.distance import geodesic

# Load model and encoder safely
@st.cache_resource
def load_artifacts():
    model = joblib.load("fraud_detection_model.jb")
    encoder = joblib.load("label_encoders.jb")
    return model, encoder

model, encoder = load_artifacts()

# Helper function: Distance calculator
def haversine(lat1, lon1, lat2, lon2):
    try:
        return geodesic((lat1, lon1), (lat2, lon2)).km
    except Exception:
        return 0.0

# Streamlit UI
st.set_page_config(page_title="Fraud Detection System", page_icon="💳", layout="centered")
st.title("💳 Fraud Detection System")
st.write("Fill in the transaction details below to check if it is **fraudulent or legitimate**.")

# Input fields
st.header("Transaction Details")

col1, col2 = st.columns(2)
with col1:
    merchant = st.text_input("Merchant Name")
    category = st.text_input("Category")
    amt = st.number_input("Transaction Amount", min_value=0.0, format="%.2f")
    gender = st.selectbox("Gender", ["Male", "Female"])
    cc_num = st.text_input("Credit Card Number")

with col2:
    lat = st.number_input("User Latitude", format="%.6f")
    long = st.number_input("User Longitude", format="%.6f")
    merch_lat = st.number_input("Merchant Latitude", format="%.6f")
    merch_long = st.number_input("Merchant Longitude", format="%.6f")
    hour = st.slider("Transaction Hour", 0, 23, 12)
    day = st.slider("Transaction Day", 1, 31, 15)
    month = st.slider("Transaction Month", 1, 12, 6)

# Distance calculation
distance = haversine(lat, long, merch_lat, merch_long)

# Prediction
if st.button("🚨 Check For Fraud"):
    if merchant and category and cc_num:
        input_data = pd.DataFrame([[
            merchant, category, amt, distance, hour, day, month, gender, cc_num
        ]], columns=['merchant', 'category', 'amt', 'distance', 'hour', 'day', 'month', 'gender', 'cc_num'])

        # Encode categorical variables
        categorical_col = ['merchant', 'category', 'gender']
        for col in categorical_col:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col] = -1  # Unknown category

        # Hash credit card number (to avoid direct leakage)
        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))

        # Model prediction
        prediction = model.predict(input_data)[0]
        result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"

        # Show result with color
        if prediction == 1:
            st.error(result)
        else:
            st.success(result)

        # Debug: Show processed data
        with st.expander("🔍 Processed Input Data"):
            st.dataframe(input_data)

    else:
        st.warning("⚠️ Please fill all required fields before checking for fraud.")
