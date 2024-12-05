# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump
import os
from xgboost import XGBClassifier  # Ensure xgboost is installed: pip install xgboost

# Create the results directory if it does not exist
os.makedirs('results', exist_ok=True)

# Step 1: Load the dataset
df = pd.read_csv('data/customer_churn.csv')

# Step 2: Ensure 'TotalCharges' is numeric and handle missing values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# Step 3: Group 'tenure' into categories for better analysis
df['tenure_group'] = pd.cut(
    df['tenure'],
    bins=[0, 12, 24, 36, 48, 60, 72],
    labels=['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
)

# Step 4: Drop unnecessary columns (e.g., 'customerID')
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])

# Step 5: Display dataset information
print("Dataset Info:")
df.info()

# Step 6: Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Step 7: Define features (X) and target variable (y)
X = df_encoded.drop('Churn_Yes', axis=1)
y = df_encoded['Churn_Yes']

# Step 8: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Initialize and apply StandardScaler to scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 10: Train Logistic Regression model
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)
print("\nLogistic Regression Results:")
print("Accuracy:", accuracy_score(y_test, log_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_pred))
print("Classification Report:\n", classification_report(y_test, log_pred))

# Step 11: Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
print("\nRandom Forest Results:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))

# Step 12: Train XGBoost model
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
print("\nXGBoost Results:")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred))
print("Classification Report:\n", classification_report(y_test, xgb_pred))

# Step 13: Perform hyperparameter tuning on Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
print("\nBest Parameters from GridSearchCV for Random Forest:", grid_search.best_params_)
rf_model = grid_search.best_estimator_

# Step 14: Save the best model and scaler
dump(rf_model, 'results/rf_model.joblib')
dump(scaler, 'results/scaler.joblib')

# Step 15: Plot feature importances for Random Forest
feature_importances = rf_model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, feature_importances)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importances")
plt.savefig('results/feature_importance.png')  # Save as an image
plt.close()

# Step 16: Save the list of features used during training
with open('results/features.txt', 'w') as f:
    f.write('\n'.join(X_train.columns))

# Step 17: Save confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix Heatmap for Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('results/confusion_matrix.png')
plt.close()

# Step 18: Additional visualization - Churn by Gender
if 'gender' in df.columns and 'Churn' in df.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x='gender', hue='Churn', data=df)
    plt.title('Churn by Gender')
    plt.savefig('results/churn_by_gender.png')
    plt.close()

# Final output
print("\nTraining Columns:", X_train.columns.tolist())
print("\nAnalysis and training complete. Results are saved.")
