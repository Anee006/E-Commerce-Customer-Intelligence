# E-Commerce Customer Intelligence Platform

An end-to-end Machine Learning and Business Intelligence system built on the **Olist Brazilian E-Commerce Dataset**. This platform transforms raw transactional CSVs into a **MySQL Star Schema Data Warehouse**, applying **Data Mining & Machine Learning algorithms** (RFM Clustering, Churn Classification, Market Basket Analysis, and Revenue Forecasting), and serving real-time analytics via a **Streamlit Web Application** and a **Power BI Executive Dashboard**.

---

## Application Overview
![alt text]({1CF4FD96-FD8E-4C28-8F2A-BD1A2EA0C48C}.png)
![alt text]({77BE9256-2B67-4D43-9FCA-4FBEFCB9E0C6}.png)
![alt text]({36ECC7BA-15AD-4E18-A8EC-383A5AAA00D0}.png)
![alt text]({46AC77FA-7BF6-42FB-A9F0-FD186968BDA2}.png)
![alt text]({03D1D473-84C2-4E32-B9B6-A1782964D33E}.png)
![alt text]({CCB3D4FA-A153-401B-87E6-0797319FBC7B}.png)
![alt text]({0553F924-BDFA-41D0-B33B-75AD82B7247A}.png)
![alt text]({306AE6D9-C944-400D-964C-10BA1A54476A}.png)

---

## System Architecture

```text
Olist CSVs ➔ Python ETL ➔ MySQL Star Schema ➔ ML / Forecasting ➔ Power BI / Streamlit App
```

### Tech Stack
| Layer | Technology |
|---|---|
| **Data Storage** | MySQL 8.x |
| **Data Engineering & ETL** | Python 3.9+, Pandas, SQLAlchemy, PyMySQL |
| **Customer Segmentation** | Scikit-learn (K-Means, PCA) |
| **Predictive Classification** | Scikit-learn (Random Forest) |
| **Cross-Sell Mining** | Mlxtend (Apriori Algorithm) |
| **Time Series Forecasting** | Statsmodels (Holt-Winters ES) |
| **Deployment & UI** | Streamlit, Plotly, Power BI Desktop |

---

## ETL & MySQL Star Schema Design
The Data Engineering pipeline extracts data from 9 CSV files, translates Portuguese category names into formatted English, handles missing value imputations using medians, and loads structured dimensional tables into MySQL:

- **fact_sales:** Central fact table containing order item metrics, monetary spend, customer keys, product keys, seller IDs, and date keys.
- **dim_customer:** Customer geographic and unique identity metadata.
- **dim_product:** Product categories (translated to English) and physical dimensions.
- **dim_date:** Calendar metadata (year, quarter, month, day, day of week, weekend flags).

---

## Machine Learning & Data Mining Modules
- **RFM Customer Segmentation:** Unsupervised K-Means Clustering operating on log-transformed ($\ln(1 + x)$) Recency, Frequency, and Monetary metrics, paired with 2D Principal Component Analysis (PCA) for visualization.

- **Churn / Repeat Purchase Classifier:** Binary classification using Random Forest to predict the likelihood of repeat purchases based on historical order behaviors.

- **Market Basket Cross-Sell Engine:** Apriori Association Rule Mining evaluated on lifetime customer category baskets to generate cross-selling recommendations based on Support, Confidence, and Lift metrics.

- **Time Series Revenue Forecasting:** Holt-Winters Exponential Smoothing models aggregated on monthly sales data ('MS') to project 6-month future revenue trends.

---

## Dataset Insights & RFM Segmentation Behavior
### 1. Dataset Reality & Distribution Characteristics
The **Olist Brazilian E-Commerce dataset** exhibits a heavy long-tail distribution typical of consumer marketplace environments, characterized by an extreme **97% single-purchase rate**:
* **Single-Order Customers ($\sim 97\%$):** The vast majority of unique customers in the dataset recorded a single historical purchase ($\text{Frequency} = 1$).
* **Repeat Purchasers ($\sim 3\%$):** Only a small fraction of unique buyers returned for multiple orders ($\text{Frequency} \ge 2$).

### 2. Mathematical Impact on K-Means Clustering
Because K-Means clustering operates on log-transformed features ($\ln(1 + x)$) to mitigate extreme skewness:
1. **Feature Invariance on Frequency:** For single-purchase customers, the transformed frequency feature evaluates to:
   $$\text{Log\_Frequency} = \ln(1 + 1) = 0.693$$
   Since $97\%$ of observations share this identical frequency coordinate, the K-Means algorithm relies primarily on **Monetary Spend** and **Recency** to segment the primary customer base.

2. **Dynamic Bucket Allocation:** Consequently, single-purchase customers naturally fall into two primary clusters based on their single-transaction monetary value:
   * **Regular Customer:** Moderate lifetime spenders (e.g., $\$10 - \$200$).
   * **VIP / High Spender:** High lifetime spenders (e.g., $\$300+$).

### 3. Cluster Verification & Slider Test Bench
The remaining segments (**Low-Value / At-Risk** and **VIP Loyal Customer**) are activated when evaluating customers with elevated **Recency** (extended inactivity) or **Frequency** (repeat purchases). 

To verify that all four cluster boundaries function correctly in the Streamlit application, test the following feature parameter combinations:

| Target Segment | Recency (Days) | Frequency (Orders) | Monetary ($ Spend) | Expected Model Outcome |
| :--- | :---: | :---: | :---: | :--- |
| **Low-Value / At-Risk** | **300+** | **1** | **$20.00** | `Low-Value / At-Risk` |
| **Regular Customer** | **30** | **1** | **$80.00** | `Regular Customer` |
| **High Spender** | **60** | **2** | **$450.00** | `High Spender` |
| **VIP Loyal Customer** | **15** | **4+** | **$1,200.00** | `VIP Loyal Customer` |

## Project Setup & Execution Guide

### 1. Prerequisites
- Python 3.9+
- MySQL Server 8.x
- Power BI Desktop 
- MySQL Connector/NET 8.0.x 
- Git

### 2. Environment Setup
```bash
# Clone the repository
git clone [https://github.com/YOUR_USERNAME/E-Commerce-Customer-Intelligence.git](https://github.com/YOUR_USERNAME/E-Commerce-Customer-Intelligence.git)
cd E-Commerce-Customer-Intelligence

# Create & activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Configuration
1. Run `sql/01_star_schema.sql` inside MySQL Workbench to create the `olist_dw` database schema.
2. Copy `src/config.example.py` to `src/config.py` and update your MySQL password.

### 4. Pipeline Execution
```bash
# Run ETL Pipeline 
python scripts/01_run_etl.py

# Train & Save ML Models
python scripts/02_train_rfm_kmeans.py
python scripts/03_train_churn_classifier.py
python scripts/04_train_apriori.py
python scripts/05_train_forecasting.py

# Power BI Dashboard
# 1. Ensure MySQL Connector/NET is installed on your machine.
# 2. Open power_bi/olist_customer_intelligence.pbix in Power BI Desktop.
# 3. Connect using Server: localhost:3306, Database: olist_dw.
# 4. Click Refresh to load live data directly from your MySQL Data Warehouse.

# Launch Streamlit Dashboard
streamlit run app/main.py
```