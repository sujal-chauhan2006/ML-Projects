# 🚗 Car Price Prediction

A production-ready machine learning project that predicts used-car selling prices from listing attributes (brand, age, mileage, engine specs, etc.). Unlike a typical notebook, this version is explicitly built to avoid **data leakage**: every aggregate/statistic-based feature and outlier bound is fit on the training set only, and the full pipeline (preprocessing + model) is wrapped in a single reusable `sklearn.Pipeline`. It ends with a tuned Gradient Boosting model and a ready-to-use `predict_price()` inference function.

## 📊 Dataset

- **File:** `processes2.csv`
- **Size:** 2,095 rows × 14 columns
- **Original columns:** `name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`, `transmission`, `owner`, `seats`, `Mileage Unit`, `Mileage`, `Engine (CC)`, `max_power (in bph)`
- **Target variable:** `selling_price` (log-transformed with `log1p` during training to reduce skew; predictions are converted back with `expm1`)

> Note: The dataset file is not included in this repo. Place `processes2.csv` in the project root (or update the path in the *Load Data* cell) before running.

## ⚠️ What This Version Fixes (vs. a naive first pass)

- **Target leakage removed** — `brand_freq` and `brand_price_mean` are computed from the training set only, then mapped onto the test set (with fallbacks for brands unseen in training)
- **Outlier clipping fitted on train only**, then applied to test using the same bounds
- **`ColumnTransformer` + `Pipeline`** replaces manual `pd.get_dummies` + column alignment, bundling scaling, encoding, and the model into one reusable object
- **`random_state` set everywhere** for reproducibility
- **`SGDRegressor` divergence fixed** (previously exploded to R² ≈ -1e28) via proper in-pipeline scaling and `early_stopping`

## 🔍 Project Workflow

1. **Imports & Setup** — installs `catboost`; sets a global `RANDOM_STATE`
2. **Load Data** — reads `processes2.csv`, renames the messy `max_power (in bph)` column to `Power`
3. **EDA** — histograms, count plots, and boxplots across numeric and categorical columns
4. **Feature Engineering** (row-level only, safe to do before the split)
   - `car_age` = current year − `year`
   - `km_per_year` = `km_driven` / (`car_age` + 1)
   - `premium_brand` flag for BMW, Audi, Mercedes, Jaguar, Volvo
   - `brand` extracted from `name`
   - `old_car`, `is_first_owner`, `is_automatic`, `dealer_sale` boolean flags
   - Log-transform of `selling_price`
5. **Train/Test Split** — 80/20 (`random_state=42`), performed **before** any aggregate feature is computed
6. **Leak-Free Aggregate Features** — `brand_freq` and `brand_price_mean` computed on train, mapped to test with fallbacks
7. **Outlier Handling** — IQR bounds computed on train only, applied identically to both sets
8. **Preprocessing Pipeline** — `ColumnTransformer` with `StandardScaler` (numeric) + `OneHotEncoder(handle_unknown='ignore')` (categorical)
9. **Model Comparison** — 13 regressors, each wrapped in `Pipeline([preprocessor, model])`, evaluated on R², MSE, MAE, RMSE
10. **Hyperparameter Tuning** — `RandomizedSearchCV` (5-fold CV) on CatBoost and LightGBM; manually tuned Random Forest and Gradient Boosting
11. **Ensembling** — `VotingRegressor` and `StackingRegressor` combining Gradient Boosting, CatBoost, and Random Forest pipelines
12. **Final Model Selection & Persistence** — best model and lookup tables (`brand_freq_map`, `brand_price_map`, `clip_bounds`) saved with `joblib`
13. **Inference Helper** — `predict_price()` function that takes raw listing fields and returns a price in original currency units

## 🏆 Model Results (initial comparison, sorted by R²)

| Model | R² Score | MAE | MSE | RMSE |
|---|---|---|---|---|
| CatBoost | 0.9054 | 0.1283 | 0.0309 | 0.1759 |
| SVR | 0.9025 | 0.1300 | 0.0319 | 0.1786 |
| LightGBM | 0.9005 | 0.1316 | 0.0325 | 0.1804 |
| Gradient Boosting | 0.8972 | 0.1340 | 0.0336 | 0.1833 |
| Random Forest | 0.8952 | 0.1340 | 0.0343 | 0.1852 |
| Bagging | 0.8905 | 0.1362 | 0.0358 | 0.1892 |
| XGBoost | 0.8875 | 0.1426 | 0.0368 | 0.1918 |
| Extra Trees | 0.8831 | 0.1452 | 0.0382 | 0.1955 |
| Linear Regression | 0.8754 | 0.1504 | 0.0407 | 0.2019 |
| KNN | 0.8721 | 0.1494 | 0.0418 | 0.2045 |
| Decision Tree | 0.8537 | 0.1643 | 0.0479 | 0.2187 |
| AdaBoost | 0.8477 | 0.1729 | 0.0498 | 0.2232 |
| SGD | 0.7955 | 0.1714 | 0.0669 | 0.2586 |

### After tuning & ensembling

| Model | R² Score | MAE |
|---|---|---|
| CatBoost (tuned, `RandomizedSearchCV`) | 0.9046 | 0.1273 |
| LightGBM (tuned, `RandomizedSearchCV`) | 0.9025 | 0.1285 |
| Random Forest (tuned) | 0.9027 | 0.1309 |
| **Gradient Boosting (tuned)** | **0.9039** | 0.1300 |
| Voting Regressor (GBR + CatBoost + RF) | 0.9069 | 0.1270 |
| Stacking Regressor (GBR + CatBoost + RF) | 0.9066 | 0.1266 |

The Voting and Stacking ensembles edge out the individual tuned models, but the tuned **Gradient Boosting Regressor** was selected as the final deployed model for its strong balance of performance and simplicity — swap `final_model` in the notebook to `vote` or `stack` if you want the ensemble instead.

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Modeling:** `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
- **Serialization:** `joblib`

## 📦 Installation

```bash
git clone https://github.com/sujal-chauhan2006/ML-Projects.git
cd ML-Projects/Car_Price_Prediction
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
lightgbm
catboost
joblib
```

## 🚀 Usage

1. Place the dataset (`processes2.csv`) in the project directory.
2. Open and run `Car_Price_Prediction.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run all cells sequentially — the notebook will engineer features, train and compare all models, tune the top performers, ensemble them, and save the final model plus lookup tables.
4. Use the built-in `predict_price()` function to price a new listing:

```python
predict_price({
    'name': 'BMW 3 Series', 'year': 2016, 'km_driven': 45000,
    'fuel': 'Diesel', 'transmission': 'Automatic', 'owner': 'First Owner',
    'seller_type': 'Dealer', 'Mileage': 18.5, 'Engine (CC)': 1995, 'Power': 187
})
# -> 607098.53
```

## 📁 Output Files

Running the notebook produces the following artifacts:

| File | Description |
|---|---|
| `car_price_model.pkl` | Final trained pipeline (preprocessing + Gradient Boosting model) |
| `brand_freq_map.pkl` | Train-set brand frequency lookup table |
| `brand_price_map.pkl` | Train-set brand average (log) price lookup table |
| `clip_bounds.pkl` | IQR outlier clipping bounds fitted on train |

To predict on new data outside the notebook, load these four files and reuse the same `predict_price()` logic (re-derive `car_age`, `km_per_year`, `premium_brand`, `brand`, `is_automatic`, `dealer_sale`, map `brand_freq`/`brand_price_mean`, clip with `clip_bounds`, then call `model.predict()` and invert with `np.expm1`).

## 📌 Key Features Used for Prediction

`Mileage`, `Engine (CC)`, `Power`, `car_age`, `km_per_year`, `brand_freq`, `brand_price_mean`, `fuel`, `transmission`, `brand`, `premium_brand`, `is_automatic`, `dealer_sale`
