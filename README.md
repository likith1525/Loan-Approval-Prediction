# Loan-Approval-Prediction
Predicting bank loan approvals using Random Forest and Logistic Regression.
# Loan Approval Prediction using Machine Learning

## Overview
Loans are a major requirement of the modern world, and evaluating an applicant's profile accurately is critical for banks to minimize risk and maximize profit. This project uses machine learning algorithms to predict whether a candidate's loan application should be approved or rejected based on key features like Marital Status, Education, Applicant Income, and Credit History.

## Dataset
The dataset contains 13 features capturing applicant details and their loan status. 
* **Categorical Features:** Gender, Married, Dependents, Education, Self_Employed, Property_Area, Loan_Status
* **Numerical Features:** ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History

## Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn

## Workflow
1. **Data Preprocessing:** Handled missing values by replacing them with the column mean and converted categorical variables into integers using Label Encoding.
2. **Exploratory Data Analysis (EDA):** Visualized feature distributions using bar plots and generated a correlation heatmap to identify highly impactful features (e.g., Credit History).
3. **Model Training:** Split the data into 60% training and 40% testing sets. 
4. **Evaluation:** Tested multiple classifiers to find the best-performing model.

## Model Performance
I evaluated four different classification models. Below are the accuracy scores achieved on the testing dataset:

| Model | Test Accuracy |
| :--- | :--- |
| Random Forest Classifier | 82.50% |
| Logistic Regression | 80.83% |
| Support Vector Classifier (SVC) | 69.16% |
| K-Nearest Neighbors (KNN) | 63.75% |

## Conclusion
The **Random Forest Classifier** outperformed the other models, achieving the highest testing accuracy of **82.5%**. Future improvements could include implementing ensemble learning techniques like Bagging and Boosting to further enhance prediction accuracy.

## Deployment
I deployed the code using streamlit by running it in the command prompt.
Imported neccesary libraries:
step by step in command prompt:-
* pip install streamlit
* pip install joblib scikit-learn nltk
* python -m streamlit run loan_approval_app

This runs and executes my code using the guven files in a streamlit platform.
