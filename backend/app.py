

# Import necessary libraries
import numpy as np
import joblib                                     # For loading the serialized model
import pandas as pd                               # For data manipulation
from flask import Flask, request, jsonify         # For creating the Flask API

# Initialize Flask app with a name
superkart_api = Flask("SuperKart")

# Load the trained model pipeline (preprocessing + regressor in a single artifact)
model = joblib.load("superkart_model.joblib")

# The exact feature contract the model was trained on
FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]


# Define a route for the home page
@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Prediction API"


# Health check - useful for confirming the container is up before sending real traffic
@superkart_api.get('/v1/health')
def health():
    return jsonify({"status": "ok", "features_expected": FEATURES})


# Define an endpoint to predict sales for a single product
@superkart_api.post('/v1/predict')
def predict_sales():
    # Get JSON data from the request
    data = request.get_json()

    # Validate that every required feature is present, and say which are missing if not
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify({"error": "Missing required features", "missing": missing}), 400

    # Extract relevant features from the input data
    sample = {feature: data[feature] for feature in FEATURES}

    # Convert the extracted data into a DataFrame
    input_data = pd.DataFrame([sample])

    # Make a prediction using the trained model
    prediction = model.predict(input_data).tolist()[0]

    # Return the prediction as a JSON response
    return jsonify({'Sales': round(prediction, 2)})


# Define an endpoint to predict sales for a batch of products
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the file into a DataFrame
    input_data = pd.read_csv(file)

    # Validate the uploaded file carries every required column
    missing = [f for f in FEATURES if f not in input_data.columns]
    if missing:
        return jsonify({"error": "Missing required columns", "missing": missing}), 400

    # Make predictions for the batch data
    predictions = model.predict(input_data[FEATURES]).tolist()

    # Create an output dictionary mapping row index to predicted sales
    output_dict = {str(i): round(pred, 2) for i, pred in enumerate(predictions)}

    return output_dict


# Run the Flask app in debug mode when executed directly (production uses gunicorn)
if __name__ == '__main__':
    superkart_api.run(debug=True)
