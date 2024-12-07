import os
from kaggle.api.kaggle_api_extended import KaggleApi
from google.cloud import storage

# Retrieve Kaggle API credentials from environment variables
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY")

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

def download_kaggle_data():
    """
    Downloads the dataset from Kaggle.
    """
    dataset = "blastchar/telco-customer-churn"  # Kaggle dataset identifier
    destination = "data/"  # Directory to save the dataset

    print("Downloading dataset from Kaggle...")
    api.dataset_download_files(dataset, path=destination, unzip=True)
    print("Dataset downloaded and extracted successfully!")

def load_dummy_data():
    """
    Downloads the dummy_customer_churn.csv from Google Cloud Storage 
    to a local directory (/tmp).
    """
    client = storage.Client()
    bucket_name = "your-bucket-name"  # Replace with your actual bucket name
    blob_name = "dummy_customer_churn.csv"
    local_path = "/tmp/dummy_customer_churn.csv"

    try:
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        print(f"Downloaded dummy data to {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading dummy data: {e}")
        raise

if __name__ == "__main__":
    # Kaggle Data Download
    print("Starting Kaggle data download...")
    download_kaggle_data()

    # Google Cloud Storage Data Download
    print("Starting Google Cloud Storage data download...")
    load_dummy_data()
