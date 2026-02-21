# Rain-Prediction-using-Logistic-Regression-Weather-Australia-Dataset-
Logistic Regression model to predict rainfall using Weather Australia dataset with full preprocessing pipeline (imputation, scaling, encoding).

The complete ML workflow is implemented manually using NumPy, Pandas, and Scikit-learn, including:
Data cleaning
Time-based train/validation/test split
Missing value imputation
Feature scaling
One-hot encoding
Model training
Accuracy evaluation
Confusion matrix analysis

Dataset
Dataset: Weather Australia
Target Variable: RainTomorrow (Yes/No)
Approach: Time-based split using year from Date column
Split strategy:
Train: Before 2015
Validation: 2015
Test: After 2015
This avoids data leakage and simulates real-world forecasting.

Preprocessing Pipeline
1) Handling Missing Values
Numeric columns → Imputed using SimpleImputer(strategy="mean")

2) Feature Scaling
Applied MinMaxScaler()
Scaled numeric features to range (0,1)

3) Categorical Encoding
Applied OneHotEncoder(handle_unknown="ignore")
Converted categorical features into numerical format
Concatenated encoded columns back into dataset

Model
Algorithm: Logistic Regression

model = LogisticRegression(
    solver="liblinear",
    max_iter=500,
    tol=0.001
)
Why Logistic Regression?
Interpretable
Works well for binary classification
Efficient on medium-sized datasets

Evaluation Metrics
✔ Accuracy
Training Accuracy
Validation Accuracy
Test Accuracy

Confusion Matrix
Normalized confusion matrix used to analyze prediction quality.
from sklearn.metrics import confusion_matrix

What I Learned
Importance of proper preprocessing
Avoiding data leakage
Correctly handling categorical variables
Proper feature concatenation
Understanding model .fit() and .predict() workflow
Handling shape mismatches and indexing issues

Project Structure
├── model.py
├── train_encoded.parquet
├── val_encoded.parquet
├── test_encoded.parquet
├── train_target.parquet
├── val_target.parquet
├── test_target.parquet
 Future Improvements

Implement full sklearn Pipeline
Hyperparameter tuning with GridSearchCV
ROC-AUC evaluation
Feature importance analysis
Try advanced models (Random Forest, XGBoost)

Tech Stack

Python
NumPy
Pandas
Scikit-learn

Key Takeaway
This project demonstrates a complete end-to-end machine learning workflow from raw dataset to evaluated model using structured preprocessing and clean implementation practices.
