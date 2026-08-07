# Customer Churn Prediction

A Machine Learning project that predicts whether a customer is likely to churn based on demographic, behavioral, and subscription-related information. The project follows a complete machine learning workflow, including data preprocessing, exploratory data analysis, feature engineering, model comparison, cross-validation, hyperparameter tuning, and model evaluation.

---

## Project Overview

Customer churn prediction helps businesses identify customers who are likely to leave their service. By predicting churn in advance, companies can take preventive actions to improve customer retention.

In this project, multiple machine learning classification algorithms were trained and compared to identify the best-performing model for churn prediction.

---

## Dataset

The dataset contains customer demographic information, subscription details, spending behavior, and customer interaction history.

### Features

- Age
- Gender
- Tenure
- Usage Frequency
- Support Calls
- Payment Delay
- Subscription Type
- Contract Length
- Total Spend
- Last Interaction

### Target Variable

- Churn
  - 0 → Customer Retained
  - 1 → Customer Churned

---

## Project Workflow

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
  - One-Hot Encoding
  - Standard Scaling
- Model Building
- Model Comparison
- Cross Validation
- Hyperparameter Tuning
- Model Evaluation
- Final Model Selection

---

## Machine Learning Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- CatBoost
- HistGradientBoosting
- AdaBoost
- Naive Bayes

---

## Model Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- 5-Fold Cross Validation

---

## Best Model

After comparing multiple machine learning models, HistGradientBoosting achieved the best overall performance on the test dataset.

Hyperparameter tuning was also performed on XGBoost using RandomizedSearchCV. Although the tuned XGBoost model achieved excellent performance, HistGradientBoosting slightly outperformed it and was selected as the final model.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- CatBoost
- Joblib

---

## Project Structure

```
Customer_Churn_Prediction/
│
├── Customer_Churn_Prediction.ipynb
├── customer_churn_pipeline.joblib
├── app.py
├── requirements.txt
└── README.md
```

---

## Key Learnings

Through this project, I gained practical experience in:

- Data Cleaning and Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- Building Multiple Classification Models
- Model Comparison
- Cross Validation
- Hyperparameter Tuning
- Model Evaluation
- Machine Learning Pipeline Creation
- Model Serialization using Joblib

---

## Future Improvements

- Deploy the model using Streamlit
- Improve model interpretability using SHAP
- Test the model on additional customer datasets
- Add real-time prediction functionality

---

## Author

**Sujal Chauhan**

Aspiring Machine Learning Engineer

GitHub: https://github.com/sujal-chauhan2006
