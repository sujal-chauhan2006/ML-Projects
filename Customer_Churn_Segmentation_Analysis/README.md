# Customer Churn Segmentation Analysis

## 📌 Project Overview

This project focuses on **segmenting customers based on their purchasing behavior** and identifying which customer groups are more likely to churn.

Instead of looking at every transaction separately, I converted the transaction-level data into **customer-level data** using features such as recency, frequency, total spending, average order value, quantity purchased, return rate, and age.

Then I used **K-Means Clustering** to divide customers into different groups and compared the churn rate and behavior of each group.

---

## 🎯 Problem Statement

The main goal of this project is to answer the following questions:

* What types of customers exist?
* Which customer segment is more valuable?
* Which segment has higher support/risk-related behavior?
* Which segment contains more churned customers?
* What actions can the company take for different customer segments?

---

## 📂 Dataset

The dataset contains e-commerce customer transaction information.

### Dataset Size

* **12,249 transactions**
* **2,040 unique customers**
* **13 columns initially**

Some of the important columns include:

* Purchase Date
* Product Category
* Product Price
* Quantity
* Total Purchase Amount
* Payment Method
* Returns
* Customer Name
* Age
* Gender
* Churn

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook / Google Colab

---

## 🔍 Project Workflow

### 1. Data Loading

The dataset was loaded using Pandas and basic information such as shape, columns, data types, and statistical summary was checked.

### 2. Data Cleaning

The following preprocessing steps were performed:

* Removed unnecessary columns such as `Customer ID` and `Customer Age`
* Checked missing values
* Checked duplicate records
* Converted `Purchase Date` into datetime format
* Filled missing `Returns` values with `0`
* Checked the unique values of different columns

### 3. Exploratory Data Analysis

EDA was performed to understand customer and transaction behavior.

The analysis included:

* Numerical feature distributions
* Categorical feature distributions
* Correlation heatmap
* Churn distribution
* Box plots for numerical features
* Missing-value analysis

The dataset contained **2,405 churned transactions/customers records and 9,843 non-churn records** at the transaction-level churn count before customer aggregation.

---

## 👤 Customer-Level Feature Engineering

Since the goal was customer segmentation, transaction-level data was converted into customer-level data.

For every customer, I created the following features:

| Feature           | Meaning                                           |
| ----------------- | ------------------------------------------------- |
| Recency           | Number of days since the customer's last purchase |
| Frequency         | Number of transactions made by the customer       |
| Monetary          | Total amount spent by the customer                |
| Avg_Order_Value   | Average purchase amount per transaction           |
| Total_Quantity    | Total number of products purchased                |
| Avg_Product_Price | Average price of products purchased               |
| Return_Rate       | Percentage of transactions that were returned     |
| Age               | Customer age                                      |
| Churn             | Customer churn status                             |

The final customer-level dataset contained **2,040 customers**.

---

## 📊 Recency Calculation

Recency was calculated using the latest purchase date available in the dataset.

The latest purchase date was:

`2023-09-15`

For each customer:

**Recency = Latest Purchase Date − Customer's Last Purchase Date**

A lower recency means the customer purchased more recently, while a higher recency means the customer has not purchased for a longer period.

---

## ⚙️ Data Preprocessing for Clustering

The features used for customer segmentation were:

```text
Recency
Frequency
Monetary
Avg_Order_Value
Total_Quantity
Avg_Product_Price
Return_Rate
Age
```

Before applying K-Means:

1. Missing values were handled using **median imputation**.
2. Features were standardized using **StandardScaler**.

This was important because the features have very different scales. For example, `Monetary` can contain values in thousands while `Return_Rate` is between 0 and 1.

---

## 🤖 Customer Segmentation Using K-Means

I used **K-Means Clustering** to divide customers into groups.

To find a suitable number of clusters, I tested values of **K from 2 to 10** and calculated the **Silhouette Score** for each value.

The best result was:

* **Best K = 2**
* **Silhouette Score = 0.1924**

Therefore, the final model was created with **2 customer segments**.

---

## 📈 Cluster Results

The final clusters showed the following behavior:

| Metric             | Cluster 0 | Cluster 1 |
| ------------------ | --------: | --------: |
| Customers          |     1,257 |       783 |
| Churn Rate         |    19.65% |    20.95% |
| Avg. Recency       |    255.57 |    138.41 |
| Avg. Frequency     |      4.59 |      8.27 |
| Avg. Monetary      | 11,856.31 | 23,692.09 |
| Avg. Order Value   |  2,609.43 |  2,896.37 |
| Avg. Quantity      |     13.43 |     25.27 |
| Avg. Product Price |    256.98 |    252.33 |
| Avg. Return Rate   |    41.83% |    39.66% |
| Avg. Age           |     42.77 |     44.79 |

---

## 💡 Key Insights

### Cluster 0 — Lower Engagement Customers

Cluster 0 contains the larger number of customers with **1,257 customers**.

These customers have:

* Higher average recency
* Lower purchase frequency
* Lower total spending
* Lower total quantity purchased

Their churn rate is around **19.65%**.

This group can be considered a lower-engagement customer segment and may need strategies to increase repeat purchases.

### Cluster 1 — High-Value and More Active Customers

Cluster 1 contains **783 customers**.

These customers:

* Purchase more frequently
* Spend significantly more
* Purchase more products
* Have lower average recency
* Have a slightly higher churn rate of **20.95%**

This makes Cluster 1 an important segment because these customers generate much higher revenue, but losing them could also have a larger financial impact.

---

## 📌 Business Recommendations

### For Cluster 0

The company can focus on improving engagement through:

* Personalized offers
* Re-engagement campaigns
* Discounts for returning customers
* Product recommendations
* Reminder campaigns for inactive customers

### For Cluster 1

Since this is a higher-value customer group, the company should focus on retention:

* Loyalty programs
* Personalized recommendations
* Exclusive offers
* Priority customer support
* Early access to new products
* Monitoring customers with increasing recency

The slightly higher churn rate in this segment should be monitored because these customers contribute significantly more revenue.

---

## 🚀 Conclusion

This project shows how **customer segmentation can be combined with churn analysis** to understand different types of customers.

Using customer-level behavioral features and K-Means clustering, I identified **2 major customer segments**.

The main finding is that Cluster 1 customers are more active and valuable, with an average monetary value of around **23,692**, compared with around **11,856** for Cluster 0. However, Cluster 1 also has a slightly higher churn rate, making customer retention especially important for this group.

This type of segmentation can help businesses move from treating all customers the same to using **different strategies for different customer groups**.

---

## 📁 Project Structure

```text
Customer-Churn-Segmentation/
│
├── Customer_Churn_Segmentation_Analysis.ipynb
├── README.md
└── ecommerce_customer_data_custom_ratios.csv
```

---



---

