
# Customer Churn Analysis

## Project Overview
This project predicts **customer churn** in the telecommunications industry using machine learning models. Churn prediction enables businesses to retain valuable customers by identifying who is likely to leave.
In the competitive telecommunications industry, understanding customer churn is critical for retaining customers and reducing costs.

### Use Cases
- **CRM Integration**: Integrate this API with customer relationship management (CRM) systems to enable real-time churn predictions for targeted marketing campaigns.
- **Proactive Retention**: Use the predictions to proactively offer discounts or rewards to customers identified as likely to churn.
- **Business Insights**: Identify factors contributing to churn and adjust marketing or customer service strategies accordingly.

## Methodology Summary:

  - Built and evaluated **Logistic Regression**, **Random Forest**, and **XGBoost** models.
  - Selected **Random Forest** for deployment due to its balance of accuracy and interpretability.
  - Deployed as a real-time API using **FastAPI**.

## Key Highlights:
- **Multiple Machine Learning Models**: Logistic Regression, Random Forest, and XGBoost were compared.
- **Highest Accuracy**: Logistic Regression achieved **81.33%** accuracy.
- **Deployment**: The Random Forest model was deployed as a real-time prediction API using **FastAPI**.
- **Comprehensive Analysis**: Includes data preprocessing, exploratory analysis, and model evaluation.

## Dataset Overview
- **Source**: Telecom customer churn dataset
- **Rows**: 7043
- **Columns**: 21
- **Target Variable**: `Churn` (Yes/No)
- **Access**: 
  - This dataset is publicly available on Kaggle.
  - You can download it from the following link and place it in the `data/` folder as `customer_churn.csv`:
    - [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

### Important Note on Dataset Usage
This project **does not include the dataset** due to copyright restrictions. 

#### Steps to Use the Dataset:
1. Download the dataset from the [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) provided in the [Dataset Overview](#dataset-overview) section.
2. Place the file in the `data/` directory within the project.
3. Ensure the file is named `customer_churn.csv`.

**Without the dataset**, the project **will not run successfully**. 
If you encounter issues, refer to the [Troubleshooting](#troubleshooting) section for detailed guidance.

### Skills Demonstrated
- **Data Preprocessing**: Handling missing values, encoding categorical features.
- **Exploratory Data Analysis (EDA)**: Visualizing churn correlations with various features.
- **Machine Learning**: Model training, hyperparameter tuning, and evaluation.
- **API Development**: Deploying a trained model using **FastAPI**.
- **Version Control**: Managing the project with **Git**.
- **Deployment Skills**: Using Replit to host APIs and managing API testing workflows.

---

## Methodology
1. **Data Cleaning**
    - Converted `TotalCharges` to numeric and handled missing values.
    - One-hot encoding for categorical variables (e.g., `Contract`, `InternetService`).
    - Binned `tenure` into categories for better interpretability.

2. **Exploratory Data Analysis (EDA)**
    - Visualized churn distribution by features like `Contract` and `MonthlyCharges`.
    - Examined correlations and feature importance.

3. **Model Building**
    - Trained Logistic Regression, Random Forest, and XGBoost models.
    - Hyperparameter tuning for Random Forest using grid search.

4. **Model Evaluation**
    - Compared models using accuracy, precision, recall, and F1-score.
    - Selected Random Forest for deployment due to its high recall and balanced F1-score.

## Results
### Model Comparison
| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression | 81.33%   | 0.85      | 0.90   | 0.88     |
| Random Forest       | 79.27%   | 0.82      | 0.92   | 0.87     |
| XGBoost             | 80.12%   | 0.80      | 0.91   | 0.85     |

### Model Insights
- **Logistic Regression** achieved the highest accuracy but had slightly lower recall than Random Forest.
- **Random Forest** recall score of 92% ensures the model is effective at identifying customers likely to churn, minimizing false negatives.
- **XGBoost** showed solid performance but was outperformed by Random Forest and Logistic Regression in terms of precision and F1-score.

### Visualizations
- **Confusion Matrix Heatmap**:
  ![Confusion Matrix Heatmap](results/confusion_matrix.png)
- **Churn by Gender**:
  ![Churn by Gender](results/churn_by_gender.png)


## How to Test the API
### 1. Cloning the Repository
Clone this repository to your local machine using Git:
```bash
git clone https://github.com/yourusername/churn_analysis_project.git
cd churn_analysis_project
```
### 2. Installing Dependencies
Make sure you have Python 3.12+ installed. Then, install the required libraries using `pip`:
```bash
pip install -r requirements.txt
```
### 3. Running Locally
To run the FastAPI app locally, navigate to the project directory and use Uvicorn to launch the app:
```bash
uvicorn main:app --reload
```
he application will be accessible at `http://127.0.0.1:8000` in your browser.
### 4. Replit Deployment (Current Setup)
1. Visit the hosted API at this URL:
   - **Base URL**: [https://b5fdc89d-a172-453b-b01a-bd1f31a71217-00-p5trej93zd6q.pike.replit.dev](https://b5fdc89d-a172-453b-b01a-bd1f31a71217-00-p5trej93zd6q.pike.replit.dev)
2. Open Swagger UI to test the API:   
   - **Swagger UI**: [https://b5fdc89d-a172-453b-b01a-bd1f31a71217-00-p5trej93zd6q.pike.replit.dev/docs](https://b5fdc89d-a172-453b-b01a-bd1f31a71217-00-p5trej93zd6q.pike.replit.dev/docs)

- Input data in the provided JSON format.
   - Example POST request:
```json
   {
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 12,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "No",
      "OnlineBackup": "Yes",
      "DeviceProtection": "Yes",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "Yes",
      "Contract": "One year",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 90.65,
      "TotalCharges": 1083.3
   }
```
This should return a prediction response, similar to:
```json
{
  "prediction": "Yes"
}
```

### 5. Testing the API
You can test the API using `curl` or Postman. Here's an example `POST` request:
```bash
curl -X 'POST' \
  'https://b5fdc89d-a172-453b-b01a-bd1f31a71217-00-p5trej93zd6q.pike.replit.dev/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "One year",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 49.99,
  "TotalCharges": "600.5"
}'
```

### Swagger UI: Testing and Results
You can use Swagger UI to explore the API's functionality and test its endpoints interactively.

1. **Access URL**:  
   - For local testing: `http://127.0.0.1:8000/docs`  
   - For hosted deployment (Replit): Use the provided Replit URL.

2. **Testing Steps**:  
   - Click the **"Try it out"** button for the `/predict` endpoint.  
   - Enter the required JSON data in the provided fields.  
   - Press **"Execute"** to get the prediction result.

3. **Sample Response**:
```json
{
  "prediction": "Yes"
}
```
4. **Example Screenshot**: Below is a successful API response tested using the REST Client extension in Visual Studio Code: 
![Swagger UI Success](results/swagger_success.png)

![REST Client in VSCode Success](results/churn_rest_client_response.png)
## Troubleshooting

Here are some common errors you might encounter while running the API and their solutions:

1. **500 Internal Server Error**:
   - **Cause**: Missing dependencies or the required data file is not found.
   - **Solution**:
     - Ensure all dependencies are installed:
       ```bash
       pip install -r requirements.txt
       ```
     - Verify that the `customer_churn.csv` file is correctly placed in the `data/` folder.

2. **404 Not Found**:
   - **Cause**: Incorrect URL or improperly configured API endpoint.
   - **Solution**:
     - Check that the endpoint URL (e.g., `/predict`) is correct.
     - Ensure the server is running:
       ```bash
       uvicorn main:app --reload
       ```

3. **Connection Refused**:
   - **Cause**: The server is not running.
   - **Solution**:
     - Start the server using:
       ```bash
       uvicorn main:app --reload
       ```
     - For Replit deployments, ensure the provided URL is active and correctly linked.


## How to Contribute
1. Fork the repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Create a new Pull Request




---

## Files and Directory Structure
```bash
churn_analysis_project/
├── analysis.py             # Contains data analysis and model training code.
├── main.py                 # Starts the FastAPI application for predictions.
├── api.py                  # Handles API endpoints and prediction logic.
├── requirements.txt        # Lists all dependencies required to run the project.
├── results/                # Contains model and visualization outputs.
│   ├── rf_model.joblib     # Saved Random Forest model for predictions.
│   ├── scaler.joblib       # StandardScaler for preprocessing features.
│   ├── features.txt        # List of features used for model training.
│   ├── confusion_matrix.png # Confusion matrix heatmap.
│   ├── churn_by_gender.png # Visualization of churn rates by gender.
├── README.md               # Project documentation (this file).
├── .gitignore              # Specifies files and directories to ignore in version control.
```
## Requirements
- Python 3.12+
- FastAPI
- Uvicorn
- scikit-learn
- joblib
- pandas
- pydantic


---

## Technologies Used
- **Python**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost
- **Framework**: FastAPI, Uvicorn
- **Deployment**: Replit (for easy hosting and deployment)

## Future Work
- Experiment with additional models such as **Gradient Boosting** and **Support Vector Machines**.
- Integrate the API with **AWS Lambda** and **API Gateway** for a fully serverless deployment.
- Implement autoscaling for handling high-frequency requests.
- Incorporate advanced feature engineering techniques to improve model accuracy.
- Create automated testing and implement unit tests using **Pytest** to ensure code reliability.

---

## Limitations
While this project provides a robust foundation for churn prediction, there are some limitations:

1. **Data Bias**: The model relies on a specific dataset, which may not represent all customer demographics or regions.
2. **Feature Constraints**: Some important features that could improve prediction accuracy may be missing from the dataset.
3. **Model Scalability**: The current deployment setup might not scale well for large, high-frequency requests without additional optimizations.
4. **Interpretability**: Although Random Forest provides feature importance, it lacks the full interpretability of simpler models like Logistic Regression.

Addressing these limitations through advanced techniques and broader datasets is a potential area for future development.

---

## Conclusion
This project demonstrates how machine learning can be used to predict customer churn in the telecommunications industry. By comparing various models, a **Random Forest** model was chosen for deployment due to its accuracy and interpretability. This project provides a foundation for building real-time prediction systems that can be integrated into business operations to help retain valuable customers.

---

## Contribution
Feel free to fork this repository and submit pull requests with improvements or new features.

## License
MIT License
