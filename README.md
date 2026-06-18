# 🚗 Car Price Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting used car selling prices using various Machine Learning Regression algorithms.

The workflow includes data preprocessing, feature engineering, exploratory data analysis (EDA), model training, evaluation, and comparison of multiple regression models.

The goal is to build an accurate model capable of estimating the selling price of a car based on its features.

---

## 📊 Dataset Information

The dataset contains information about used cars, including:

- Brand
- Model
- Year
- Engine Size (CC)
- Maximum Power (BHP)
- Mileage
- Fuel Type
- Transmission
- Number of Seats
- Kilometers Driven
- Selling Price

---

## 🛠️ Data Preprocessing

The following preprocessing techniques were applied:

- Handling missing values
- Removing unnecessary columns
- Label Encoding for categorical features
- Feature Scaling
- Train-Test Split

---

## 📈 Exploratory Data Analysis (EDA)

Performed data analysis to understand:

- Distribution of car prices
- Relationship between features and selling price
- Correlation analysis
- Feature importance insights

---

## 🤖 Machine Learning Models Used

The following regression algorithms were implemented and compared:

### 1. Linear Regression

A baseline regression model used for comparison.

### 2. SGD Regressor

Linear Regression trained using Gradient Descent optimization.

### 3. Random Forest Regressor

An ensemble learning algorithm that combines multiple decision trees.

### 4. XGBoost Regressor

A gradient boosting algorithm used for improving predictive performance.

---

## 📊 Model Performance

| Model | R² Score |
|---------|---------|
| Linear Regression | 0.62 |
| SGD Regressor | 0.61 |
| Random Forest Regressor | 0.81 |
| XGBoost Regressor | 0.84 |

### 🏆 Best Model

**XGBoost Regressor** achieved the highest performance with:

- R² Score: **0.84**
- MAE: **63,857**
- RMSE: **93,772**

---

## 📚 Libraries Used

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
```

---

## 🎯 Key Learnings

Through this project, I learned:

- Data Cleaning and Preprocessing
- Feature Engineering
- Label Encoding
- Exploratory Data Analysis (EDA)
- Regression Algorithms
- Model Evaluation Metrics
- Hyperparameter Tuning
- Model Comparison and Selection

---

## 🚀 Conclusion

The project successfully predicts car selling prices using Machine Learning techniques.

Among all tested models, **XGBoost Regressor** delivered the best performance with an R² score of **0.84**, making it the final selected model for car price prediction.

---

## 👨‍💻 Author

**Sujal Chauhan**

BCA Student | Data Science & Machine Learning Enthusiast
