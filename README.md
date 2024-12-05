
# Customer Churn Analysis

## Project Overview
This project predicts customer churn using machine learning models, including Logistic Regression, Random Forest, and XGBoost. 
The Random Forest model, after hyperparameter tuning, was chosen for deployment as a FastAPI service for real-time predictions.

### Highlights:
- Multiple machine learning models trained and compared.
- Logistic Regression achieved the highest accuracy (**81.33%**) in initial evaluation.
- Random Forest was selected for deployment due to its performance and feature importance interpretability.

### Skills Demonstrated
- Data preprocessing and feature engineering.
- Training, evaluating, and tuning machine learning models.
- API development and deployment with FastAPI.

---

## Dataset Overview
- **Source**: Telecom customer churn dataset
- **Rows**: 7043
- **Columns**: 21
- **Target Variable**: `Churn` (Yes/No)
- **Access**: 
  - This dataset is publicly available on Kaggle.
  - You can download it from the following link and place it in the `data/` folder as `customer_churn.csv`:
    - [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Methodology

### **1. Data Cleaning**
- Converted `TotalCharges` to numeric and handled missing values.
- Encoded categorical variables using one-hot encoding.
- Grouped `tenure` into categorized bins for better analysis.

### **2. Exploratory Data Analysis (EDA)**
- Analyzed correlations between churn and features such as `Contract` and `MonthlyCharges`.
- Visualized churn distribution by gender.

### **3. Model Building**
- Trained and evaluated the following models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Performed hyperparameter tuning for Random Forest.

### **4. Model Evaluation**
- **Logistic Regression**: Accuracy 81.33%, Precision 85%, Recall 90%.
- **Random Forest**: Accuracy 79.27%, Precision 82%, Recall 92%.
- **XGBoost**: Accuracy 79.13%, Precision 84%, Recall 89%.

### **5. Model Deployment**
- Random Forest model was deployed as a FastAPI-based API for real-time churn predictions.

---

## Results
### Model Comparison
| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression | 81.33%   | 0.85      | 0.90   | 0.88     |
| Random Forest       | 79.27%   | 0.82      | 0.92   | 0.87     |
| XGBoost             | 79.13%   | 0.84      | 0.89   | 0.86     |

### Visualizations
- **Confusion Matrix Heatmap**:
  ![Confusion Matrix Heatmap](results/confusion_matrix.png)
- **Churn by Gender**:
  ![Churn by Gender](results/churn_by_gender.png)

---

## Technologies Used
- **Python**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, XGBoost
- **Framework**: FastAPI

---

## File Structure
```bash
churn_analysis_project/
├── analysis.py             # Data analysis and model training script
├── api.py                  # FastAPI code for predictions
├── requirements.txt        # Required Python libraries
├── results/                # Saved models and visualizations
│   ├── rf_model.joblib     # Trained Random Forest model
│   ├── scaler.joblib       # Scaler for preprocessing
│   ├── confusion_matrix.png
│   ├── churn_by_gender.png
├── README.md               # Project documentation
├── .gitignore              # To ignore unnecessary files like data/
```

---

## How to Run
### 1. Setup
1. Clone the repository:
```bash
git clone <your-repo-url>
```
2. Navigate to the project directory:
```bash
cd churn_analysis_project
```
3. Install required libraries:
```bash
pip install -r requirements.txt
```

### 2. Running the Analysis
1. Train and evaluate the models:
```bash
python analysis.py
```

### 3. Running the API
1. Start the API server:
```bash
uvicorn api:app --reload
```
2. Open Swagger UI to test the API:
   - Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.

3. Example POST Request:
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
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 1407.00
}
```

---

## Next Steps
- **Cloud Deployment**:
  - Deploy the API to AWS, GCP, or Heroku for public access.
- **Model Optimization**:
  - Further tune the Logistic Regression and XGBoost models.
- **Interactive Visualizations**:
  - Create dashboards with tools like Plotly or Dash.

---

## Contribution
Feel free to fork this repository and submit pull requests with improvements or new features.

## License
MIT License
