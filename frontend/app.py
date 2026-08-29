

import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend, resolved by Docker's internal DNS via the container name
BACKEND_URL = "http://backend:7860"

st.set_page_config(page_title="SuperKart Sales Forecast", page_icon="🛒")

# Page title
st.title("SuperKart Sales Forecasting System")
st.write(
    "Enter the product and store details below to predict the total sales "
    "revenue for that product in that store."
)

# Input fields for product and store data.
# The options below match exactly the category levels present in the training data.
st.subheader("Product details")
Product_Weight = st.number_input("Product Weight", min_value=0.0, max_value=50.0, value=12.66)
Product_MRP = st.number_input("Product MRP", min_value=0.0, max_value=500.0, value=117.08)
Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
Product_Allocated_Area = st.number_input(
    "Product Allocated Area (share of total display area)",
    min_value=0.0, max_value=1.0, value=0.027, format="%.3f"
)
Product_Id_char = st.selectbox(
    "Product ID Prefix", ["FD", "DR", "NC"],
    help="FD = Food, DR = Drinks, NC = Non-Consumable"
)
Product_Type_Category = st.selectbox("Product Type Category", ["Perishables", "Non Perishables"])

st.subheader("Store details")
Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
Store_Type = st.selectbox(
    "Store Type",
    ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"]
)
Store_Age_Years = st.number_input("Store Age (Years)", min_value=0, max_value=100, value=16)

# Create JSON payload
product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Id_char": Product_Id_char,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}

# Single Prediction
if st.button("Predict", type='primary'):
    try:
        response = requests.post(f"{BACKEND_URL}/v1/predict", json=product_data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            predicted_sales = result["Sales"]
            st.success(f"Predicted Product Store Sales Total: {predicted_sales:,.2f}")
        else:
            st.error(f"Prediction failed ({response.status_code}): {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"Unable to reach the prediction API: {e}")

# Batch Prediction
st.subheader("Batch Prediction")
st.caption(
    "Upload a CSV containing the ten feature columns: Product_Weight, Product_Sugar_Content, "
    "Product_Allocated_Area, Product_MRP, Store_Size, Store_Location_City_Type, Store_Type, "
    "Product_Id_char, Store_Age_Years, Product_Type_Category."
)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    if st.button("Predict for Batch", type='primary'):
        try:
            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files={"file": uploaded_file},
                timeout=60,
            )
            if response.status_code == 200:
                results = response.json()
                st.success("Predictions completed successfully.")
                df = pd.DataFrame(
                    {"Row": list(results.keys()), "Predicted Sales": list(results.values())}
                )
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Download predictions as CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    "superkart_predictions.csv",
                    "text/csv",
                )
            else:
                st.error(f"Prediction failed ({response.status_code}): {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Unable to reach the prediction API: {e}")
