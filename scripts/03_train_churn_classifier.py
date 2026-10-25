# Trains a RandomForestClassifier to determine if a customer is a repeat buyer (1) or a single-purchase customer (0)

import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATABASE_URL

def train_churn_model():
    engine = create_engine(DATABASE_URL)

    query = """
    SELECT 
        c.customer_unique_id,
        COUNT(DISTINCT f.order_id) AS total_orders,
        AVG(f.total_value) AS avg_order_value,
        SUM(f.total_value) AS total_spent
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_key = c.customer_key
    GROUP BY c.customer_unique_id
    """
    df = pd.read_sql(query, engine)

    # Label: 1 if customer ordered more than once, else 0
    df["repeat_customer"] = (df["total_orders"] > 1).astype(int)

    X = df[["avg_order_value", "total_spent"]]
    y = df["repeat_customer"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Fitting Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    preds = rf_model.predict(X_test)
    print("Model Performance:")
    print(classification_report(y_test, preds, zero_division=0))

    joblib.dump(rf_model, "saved_models/churn_rf.joblib")
    print("Saved `churn_rf.joblib` to saved_models/")

if __name__ == "__main__":
    train_churn_model()