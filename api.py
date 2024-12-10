# Import necessary libraries
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from typing import Optional

# Set Pandas options to suppress future warnings about silent downcasting
#pd.set_option('future.no_silent_downcasting', True)

# Initialize FastAPI application
app = FastAPI()

# Load the trained model, scaler, and training columns
rf_model = load("results/rf_model.joblib")  # Load Random Forest model
scaler = load("results/scaler.joblib")      # Load scaler used during training
with open("results/features.txt", "r") as f:
    TRAINING_COLUMNS = f.read().splitlines()  # Load training column names

# Define the input data schema for the API
class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: Optional[int]
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

def preprocess_input(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input data to match the training format.
    """
    # Handle missing values for 'tenure'
    input_data['tenure'] = input_data['tenure'].fillna(0).astype(int)

    # Ensure 'TotalCharges' is numeric
    input_data['TotalCharges'] = pd.to_numeric(input_data['TotalCharges'], errors='coerce').fillna(0)

    # One-hot encode categorical variables
    input_encoded = pd.get_dummies(input_data)

    # Align columns with the training dataset
    input_encoded = input_encoded.reindex(columns=TRAINING_COLUMNS, fill_value=0)

    return input_encoded

# Define the prediction endpoint
@app.post("/predict")
def predict_churn(input_data: ChurnInput):
    """
    Predict customer churn based on input data.

    Args:
        input_data (ChurnInput): Input data received from the API request.

    Returns:
        dict: A dictionary containing the churn prediction or an error message.
    """
    try:
        # Convert input data to a pandas DataFrame
        input_df = pd.DataFrame([input_data.dict()])

        # Validate the input data for negative values
        if input_df["MonthlyCharges"].iloc[0] < 0 or input_df["TotalCharges"].iloc[0] < 0:
            return {"error": "MonthlyCharges and TotalCharges cannot be negative"}

        # Preprocess the input data
        processed_data = preprocess_input(input_df)

        # Scale the input data
        scaled_data = scaler.transform(processed_data)

        # Make a prediction
        prediction = rf_model.predict(scaled_data)
        churn = "Yes" if prediction[0] == 1 else "No"

        return {"churn": churn}
    except Exception as e:
        # Return error details for debugging
        return {"error": str(e)}


# Endpoint for health check
@app.get("/")
def read_root():
    """
    Health check endpoint to confirm the API is running.

    Returns:
        dict: A simple confirmation message.
    """
    return {"message": "API is running successfully"}

