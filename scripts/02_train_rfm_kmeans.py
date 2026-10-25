# Computes Recency, Frequency, and Monetary metrics via SQL queries, standardizes features and fits a K-Means model

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATABASE_URL

def train_rfm_kmeans():
    print("Querying Sales Data for RFM Analysis...")
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
    
    # Calculate Recency in days
    max_dataset_date = pd.to_datetime(rfm_df["max_date"]).max()
    rfm_df["recency"] = (max_dataset_date - pd.to_datetime(rfm_df["max_date"])).dt.days
    
    # Log Transformation to handle severe skewness in monetary and frequency values
    rfm_log = pd.DataFrame()
    rfm_log["recency"] = np.log1p(rfm_df["recency"])
    rfm_log["frequency"] = np.log1p(rfm_df["frequency"])
    rfm_log["monetary"] = np.log1p(rfm_df["monetary"])

    print("Scaling Features & Training K-Means Model...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(rfm_log)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm_df["cluster"] = kmeans.fit_predict(scaled_features)

    # Dynamic Label Mapping based on Cluster Centers
    cluster_summary = rfm_df.groupby("cluster")[["recency", "frequency", "monetary"]].mean()
    sorted_clusters = cluster_summary.sort_values(by="monetary").index.tolist()

    # Create mapping: lowest spend -> At-Risk/Low Value, highest spend -> VIP
    label_names = ["Low-Value / At-Risk", "Regular Customer", "High Spender", "VIP Loyal Customer"]
    cluster_mapping = {cluster_id: label_names[i] for i, cluster_id in enumerate(sorted_clusters)}

    print("\nDiscovered Cluster Centers & Mapped Names:")
    for cid, name in cluster_mapping.items():
        avg_rec = cluster_summary.loc[cid, "recency"]
        avg_freq = cluster_summary.loc[cid, "frequency"]
        avg_mon = cluster_summary.loc[cid, "monetary"]
        print(f"Cluster {cid} ({name}): Recency={avg_rec:.1f}d, Frequency={avg_freq:.1f}, Monetary=${avg_mon:.2f}")

    # Save artifacts including the dynamic mapping dictionary
    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(scaler, "saved_models/rfm_scaler.joblib")
    joblib.dump(kmeans, "saved_models/kmeans_rfm.joblib")
    joblib.dump(cluster_mapping, "saved_models/rfm_cluster_mapping.joblib")
    
    print("\nModels & Cluster Mappings saved to saved_models/")

if __name__ == "__main__":
    train_rfm_kmeans()