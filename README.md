# E-Commerce Customer Intelligence Platform

An end-to-end Machine Learning and Business Intelligence system built on the **Olist Brazilian E-Commerce Dataset**. This platform transforms raw transactional CSVs into a **MySQL Star Schema Data Warehouse**, extracts machine learning features, trains predictive models, generates time series revenue forecasts, builds Power BI reports, and serves an interactive **Streamlit Web App**.

---

## System Architecture

```text
Olist CSVs ➔ Python ETL ➔ MySQL Star Schema ➔ ML / Forecasting ➔ Power BI / Streamlit App
```

### Core Modules
1. **Star Schema Data Warehouse:** Fact and Dimension modeling (`fact_sales`, `dim_customer`, `dim_product`, `dim_date`).
2. **RFM & K-Means Clustering:** Unsupervised customer segmentation using Recency, Frequency, and Monetary metrics.
3. **Churn / Repeat Purchase Classifier:** Machine learning classification using Random Forest.
4. **Market Basket Analysis:** Association Rule Mining via the Apriori algorithm for product recommendations.
5. **Time Series Sales Forecasting:** Exponential Smoothing models to aggregate sales and project future monthly revenue.
6. **Power BI Executive Dashboard:** MySQL-connected interactive dashboard featuring executive sales KPIs, customer demographic maps, logistics analytics, and order distribution.
7. **Streamlit Web App:** Interactive control center for business KPIs, live model inferences, product cross-selling engine, time series revenue projections, and embedded Power BI reports.

---

## Cloning Guidelines

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
1. Ensure MySQL Connector/NET is installed on your machine.
2. Open power_bi/olist_customer_intelligence.pbix in Power BI Desktop.
3. If prompted for credentials:
    - Server: localhost:3306
    - Database: olist_dw
    - User: root
    - Password: Your local MySQL password
4. Click Refresh to load live data directly from your MySQL Data Warehouse.

# Launch Streamlit Dashboard
streamlit run app/main.py
```