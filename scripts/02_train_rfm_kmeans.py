# Computes Recency, Frequency, and Monetary metrics via SQL queries, standardizes features and fits a K-Means model

import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATABASE_URL

def train_rfm_kmeans():
    engine = create_engine(DATABASE_URL)

    query = """
    SELECT 
        c.customer_unique_id,
        MAX(d.full_date) AS max_date,
        COUNT(DISTINCT f.order_id) AS frequency,
        SUM(f.total_value) AS monetary
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_key = c.customer_key
    JOIN dim_date d ON f.date_key = d.date_key
    GROUP BY c.customer_unique_id
    """
    rfm_df = pd.read_sql(query, engine)
    
    # calculate Recency in days
    max_dataset_date = pd.to_datetime(rfm_df["max_date"]).max()
    rfm_df["recency"] = (max_dataset_date - pd.to_datetime(rfm_df["max_date"])).dt.days
    
    features = rfm_df[["recency", "frequency", "monetary"]]

    print("Scaling Features & Training K-Means Model...")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(scaled_features)

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(scaler, "saved_models/rfm_scaler.joblib")
    joblib.dump(kmeans, "saved_models/kmeans_rfm.joblib")
    print("Saved `rfm_scaler.joblib` and `kmeans_rfm.joblib` to saved_models/")

if __name__ == "__main__":
    train_rfm_kmeans()