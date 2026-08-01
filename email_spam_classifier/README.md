# 📧 Email Spam Classifier

A machine learning project that classifies SMS/email messages as **Spam** or **Ham (not spam)** using NLP-based text preprocessing and classical ML classifiers. The pipeline covers data cleaning, EDA, text preprocessing (tokenization, stemming, stopword removal), TF-IDF vectorization, comparison of 11 classification models, and ensembling for the final model.

## 📊 Dataset

- **File:** `email.csv`
- **Original size:** 5,573 rows × 2 columns (`Category`, `Message`)
- **Target variable:** `Category` — encoded as `0 = Ham`, `1 = Spam`
- One invalid/stray label row was dropped, leaving **5,572 messages** for training (imbalanced: ~4,825 ham vs ~747 spam)

> Note: The dataset file is not included in this repo. Place `email.csv` in the project root (or update the path in the notebook) before running.

## 🔍 Project Workflow

1. **Data Cleaning**
   - Encoded `Category` labels with `LabelEncoder`
   - Removed an invalid label row
   - Checked for missing and duplicate values
2. **Exploratory Data Analysis (EDA)**
   - Class distribution (count plot + pie chart) — confirmed class imbalance
   - Derived features: `Num_charecter`, `Num_word`, `Num_sentance` per message (via NLTK tokenization)
   - Distribution comparisons between spam and ham messages
   - Correlation heatmap and pairplot across derived features
3. **Text Preprocessing** (custom `transformed_data()` function)
   - Lowercasing
   - Tokenization (NLTK)
   - Removing special characters (keeping only alphanumeric tokens)
   - Removing stopwords and punctuation
   - Stemming (Porter Stemmer)
4. **Word Cloud & Frequency Analysis**
   - WordClouds for spam vs. ham messages
   - Most frequent words in each class (bar plots)
5. **Feature Extraction** — `TfidfVectorizer` (`max_features=3000`) on the cleaned text
6. **Train/Test Split** — 80/20 split (`random_state=2`)
7. **Model Training & Comparison** — 11 classifiers evaluated on Accuracy and Precision
8. **Ensembling**
   - `VotingClassifier` (soft voting: MultinomialNB + Random Forest + Extra Trees)
   - `StackingClassifier` (same base learners, Random Forest as final estimator)
9. **Model Export** — final Random Forest model and TF-IDF vectorizer saved with `pickle`

## 🏆 Model Results

| Model | Accuracy | Precision |
|---|---|---|
| Logistic Regression | 0.9552 | 0.9909 |
| SVC (sigmoid kernel) | 0.9695 | 0.9559 |
| Multinomial Naive Bayes | 0.9704 | 1.0000 |
| Decision Tree | 0.9462 | 0.8603 |
| K-Nearest Neighbors | 0.8879 | 1.0000 |
| Random Forest | 0.9749 | 0.9924 |
| AdaBoost | 0.9265 | 0.8958 |
| Bagging Classifier | 0.9623 | 0.8816 |
| Extra Trees | 0.9749 | 0.9851 |
| Gradient Boosting | 0.9596 | 0.9669 |
| XGBoost | 0.9677 | 0.9485 |
| **Voting Classifier** (MNB + RF + ExtraTrees) | 0.9731 | **1.0000** |
| **Stacking Classifier** (MNB + RF + ExtraTrees → RF) | **0.9803** | 0.9857 |

The **Stacking Classifier** achieved the best overall accuracy (98.03%), while the **Voting Classifier** achieved perfect precision (no false positives on spam detection) — important for avoiding legitimate emails being flagged as spam. The final exported model uses a **Random Forest Classifier** on top of the TF-IDF features.

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`, `wordcloud`
- **NLP:** `nltk` (tokenization, stopwords, Porter stemming)
- **Modeling:** `scikit-learn`, `xgboost`
- **Serialization:** `pickle`

## 📦 Installation

```bash
git clone https://github.com/sujal-chauhan2006/ML-Projects.git
cd ML-Projects/Email_Spam_Classifier
pip install -r requirements.txt
```

**requirements.txt**
```
numpy
pandas
matplotlib
seaborn
nltk
wordcloud
scikit-learn
xgboost
```

After installing, download the required NLTK data (also done inside the notebook):

```python
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 🚀 Usage

1. Place the dataset (`email.csv`) in the project directory.
2. Open and run `Email_Spam_Classifier.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. Run all cells sequentially — the notebook will clean the data, preprocess text, train and compare all models, and save the final model and vectorizer.

## 📁 Output Files

Running the notebook produces the following artifacts:

| File | Description |
|---|---|
| `model.pkl` | Trained Random Forest classifier |
| `vectorizer.pkl` | Fitted `TfidfVectorizer` (3000 features) |

To classify a new message, load these two files, transform the raw text through the same `transformed_data()` preprocessing function, vectorize it with `vectorizer.pkl`, and call `model.predict()`.

## 📌 Key Features Used for Prediction

TF-IDF representation (3000 features) of the cleaned, stemmed message text.

