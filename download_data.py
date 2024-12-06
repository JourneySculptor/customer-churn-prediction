import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Retrieve Kaggle API credentials from environment variables
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

# Dataset and destination details
dataset = "blastchar/telco-customer-churn"  # Kaggle dataset identifier
destination = "data/"  # Directory to save the dataset

# Download and extract the dataset
print("Downloading dataset from Kaggle...")
api.dataset_download_files(dataset, path=destination, unzip=True)
print("Dataset downloaded and extracted successfully!")
