
# 🚢 Titanic Survival Prediction

## 📌 Overview

This project focuses on predicting passenger survival on the Titanic using Machine Learning. The notebook covers the complete workflow of a classification problem, including data cleaning, exploratory data analysis (EDA), feature engineering, data preprocessing, model building, and evaluation.

The goal of this project is to understand how Logistic Regression can be applied to a real-world dataset and how different passenger attributes influence survival chances.

---

## 📂 Dataset

The dataset contains information about Titanic passengers, including:

- Passenger Class (Pclass)
- Gender (Sex)
- Age
- Fare
- Embarked Port
- Number of Siblings/Spouses (SibSp)
- Number of Parents/Children (Parch)
- Survival Status (Target Variable)

### Target Variable

- 0 = Did Not Survive
- 1 = Survived

---

## 🔍 Exploratory Data Analysis (EDA)

The following analyses were performed:

- Dataset overview and information
- Missing value analysis
- Statistical summary
- Age distribution analysis
- Survival distribution analysis
- Correlation heatmap
- Feature relationship exploration

---

## 🛠 Data Preprocessing

The following preprocessing steps were applied:

- Handling missing values
  - Age column filled using Median
  - Embarked column filled using Mode
- Categorical feature encoding
- Feature scaling using StandardScaler
- Removal of unnecessary columns
- Feature engineering by creating a Family feature

---

## 🤖 Model Used

### Logistic Regression

The Logistic Regression algorithm was used to classify passengers into:

- Survived
- Not Survived

---

## 📊 Model Evaluation

The model performance was evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report

---

## 📚 Libraries Used

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## 🎯 Key Learning Outcomes

- Understanding binary classification problems
- Data cleaning and preprocessing
- Handling missing values
- Feature engineering
- Feature scaling
- Logistic Regression implementation
- Model evaluation techniques
- Data visualization and EDA

---

## 🏆 Conclusion

This project demonstrates the complete machine learning workflow for a classification problem using the Titanic dataset. Through data preprocessing, feature engineering, visualization, and Logistic Regression, the model predicts passenger survival and provides insights into the factors that influenced survival during the Titanic disaster.

⭐ Feel free to explore the notebook and provide feedback.
