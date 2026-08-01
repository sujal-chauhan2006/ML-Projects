# 🏠 Ahmedabad House Price Prediction

🔗 **Repo:** [ML-Projects/House_Price_Prediction](https://github.com/sujal-chauhan2006/ML-Projects/tree/main/House_Price_Prediction)

A machine learning project that predicts residential property prices (in ₹ Crore) in Ahmedabad, India, based on location, area, property type, and other listing features. The pipeline covers data cleaning, exploratory data analysis, outlier handling, feature engineering, and comparison of 12+ regression models, ending with a tuned ensemble model saved for deployment.

## 📊 Dataset

- **File:** `ahmendabad_house_price.csv`
- **Size:** 20,060 rows × 10 columns
- **Original columns:** `name`, `location`, `description`, `rate_per_sqft`, `area_in_sqft`, `area_type`, `property_title`, `property_type`, `bhk_type`, `price_in_cr`
- **Target variable:** `price_in_cr` (property price in ₹ Crore)

> Note: The dataset file is not included in this repo. Place `ahmendabad_house_price.csv` in the project root (or update the path in the notebook) before running.

## 🔍 Project Workflow

1. **Data Understanding** – inspect shape, types, summary statistics, and value counts
2. **Missing Value Handling**
   - Dropped high-cardinality/unstructured text columns: `description`, `name`, `property_title`
   - Filled numeric columns with median, categorical columns with mode
   - Grouped rare locations (fewer than 20 listings) into an `"Other"` category
3. **Exploratory Data Analysis (EDA)**
   - Univariate analysis (boxplots) for `area_in_sqft`, `bhk_type`, `price_in_cr`
   - Distribution plots for categorical and numeric features
   - Scatter plots and a correlation heatmap against `price_in_cr`
4. **Outlier Handling** – IQR-based removal/clipping on `price_in_cr` and `area_in_sqft`
5. **Train/Test Split** – 80/20 split (`random_state=42`)
6. **Feature Encoding & Scaling**
   - `LabelEncoder` for `area_type`, `property_type`, `location`
   - `StandardScaler` for `area_in_sqft` and `bhk_type`
7. **Model Training & Comparison** – 12 regression algorithms trained and evaluated on R², MAE, MSE, RMSE
8. **Hyperparameter Tuning** – `RandomizedSearchCV` on `GradientBoostingRegressor`
9. **Ensembling** – `StackingRegressor` and `VotingRegressor` built from Random Forest, Extra Trees, Gradient Boosting, and XGBoost
10. **Model Export** – final model and preprocessing objects saved with `pickle`/`joblib`

## 🏆 Model Results

| Model | R² Score | MAE | MSE | RMSE |
|---|---|---|---|---|
| Random Forest | 0.9225 | 0.0491 | 0.0230 | 0.1515 |
| Extra Trees | 0.9204 | 0.0473 | 0.0236 | 0.1536 |
| **XGBoost (tuned)** | 0.9154 | 0.0731 | 0.0250 | 0.1582 |
| Gradient Boosting (tuned) | 0.9101 | 0.0762 | 0.0266 | 0.1632 |
| **Stacking Regressor** | **0.9279** | 0.0491 | 0.0214 | 0.1461 |
| Voting Regressor | 0.9267 | 0.0577 | 0.0217 | 0.1473 |
| Linear Regression / Ridge | 0.5051 | 0.2685 | 0.1465 | 0.3828 |
| Decision Tree | 0.8788 | 0.0542 | 0.0359 | 0.1894 |
| AdaBoost | 0.7235 | 0.2071 | 0.0819 | 0.2861 |
| KNN | 0.4624 | 0.2642 | 0.1592 | 0.3989 |
| SVR | 0.0214 | 0.3835 | 0.2897 | 0.5383 |
| Lasso / ElasticNet | ~0.00 | ~0.41 | ~0.29 | ~0.54 |

5-fold cross-validation confirmed the **Stacking Regressor** as the most stable and accurate model (mean R² = 0.957, std = 0.008), closely followed by the Voting Regressor (mean R² = 0.957). The **Voting Regressor** was chosen as the final deployed model.

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Modeling:** `scikit-learn`, `xgboost`
- **Serialization:** `pickle`, `joblib`

## 📦 Installation

```bash
git clone https://github.com/sujal-chauhan2006/ML-Projects.git
cd ML-Projects/House_Price_Prediction
pip install -r requirements.txt
```

**requirements.txt**
```
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
joblib
```

## 🚀 Usage

1. Place the dataset (`ahmendabad_house_price.csv`) in the project directory.
2. Open and run `House_price_prediction.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run all cells sequentially — the notebook will train and evaluate all models, tune the best-performing ones, and save the final model and preprocessing objects.

## 📁 Output Files

Running the notebook produces the following artifacts:

| File | Description |
|---|---|
| `model.pkl` | Trained Voting Regressor model |
| `voting_model.joblib` | Same model saved via joblib |
| `le_area.pkl` | Label encoder for `area_type` |
| `le_property.pkl` | Label encoder for `property_type` |
| `le_location.pkl` | Label encoder for `location` |
| `scaler.pkl` | StandardScaler for numeric features |

To make predictions on new data, load these files and apply the same encoding/scaling steps used during training before calling `model.predict()`.

## 📌 Key Features Used for Prediction

`location`, `rate_per_sqft`, `area_in_sqft`, `area_type`, `property_type`, `bhk_type`
