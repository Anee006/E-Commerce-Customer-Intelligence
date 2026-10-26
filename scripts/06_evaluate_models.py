# Evaluate ML models

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, silhouette_score
from statsmodels.tools.eval_measures import rmse, mae

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATABASE_URL

def evaluate_all_models():
    engine = create_engine(DATABASE_URL)
    evaluation_results = {}

    # 1. RFM K-Means Clustering Evaluation
    print("[1/4] Evaluating K-Means RFM Segmentation...")
    try:
        scaler = joblib.load("saved_models/rfm_scaler.joblib")
        kmeans = joblib.load("saved_models/kmeans_rfm.joblib")

        query_rfm = """
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
        rfm_df = pd.read_sql(query_rfm, engine)
        max_date = pd.to_datetime(rfm_df["max_date"]).max()
        rfm_df["recency"] = (max_date - pd.to_datetime(rfm_df["max_date"])).dt.days

        log_rfm = np.log1p(rfm_df[["recency", "frequency", "monetary"]])
        scaled_rfm = scaler.transform(log_rfm)
        clusters = kmeans.predict(scaled_rfm)

        # Calculate Inertia & sampled Silhouette Score
        inertia = float(kmeans.inertia_)

        sample_idx = np.random.choice(scaled_rfm.shape[0], size=min(10000, scaled_rfm.shape[0]), replace=False)
        sil_score = float(silhouette_score(scaled_rfm[sample_idx], clusters[sample_idx]))

        evaluation_results["rfm_kmeans"] = {
            "inertia": round(inertia, 2),
            "silhouette_score": round(sil_score, 4),
            "num_clusters": int(kmeans.n_clusters)
        }
        print(f"   Silhouette Score: {sil_score:.4f} | Inertia: {inertia:,.2f}")

    except Exception as e:
        print(f"   ⚠️ RFM Evaluation Error: {e}")

    # 2. Random Forest Repeat Purchase (Churn Evaluation)
    print("\n[2/4] Evaluating Random Forest Repeat Purchase Classifier...")
    try:
        model_data = joblib.load("saved_models/churn_model.joblib")
        rf_model = model_data["model"]
        scaler = model_data["scaler"]

        query_churn = """
        SELECT 
            c.customer_unique_id,
            COUNT(DISTINCT f.order_id) AS frequency,
            SUM(f.total_value) AS monetary,
            AVG(f.total_value) AS avg_basket_value,
            MAX(d.full_date) AS max_date
        FROM fact_sales f
        JOIN dim_customer c ON f.customer_key = c.customer_key
        JOIN dim_date d ON f.date_key = d.date_key
        GROUP BY c.customer_unique_id
        """
        df_churn = pd.read_sql(query_churn, engine)
        max_date = pd.to_datetime(df_churn["max_date"]).max()
        df_churn["recency"] = (max_date - pd.to_datetime(df_churn["max_date"])).dt.days
        df_churn["is_repeat"] = (df_churn["frequency"] > 1).astype(int)

        X = df_churn[["recency", "frequency", "monetary", "avg_basket_value"]]
        y = df_churn["is_repeat"]
        X_scaled = scaler.transform(X)

        y_pred = rf_model.predict(X_scaled)
        y_proba = rf_model.predict_proba(X_scaled)[:, 1]

        acc = float(accuracy_score(y, y_pred))
        prec = float(precision_score(y, y_pred, zero_division=0))
        rec = float(recall_score(y, y_pred, zero_division=0))
        f1 = float(f1_score(y, y_pred, zero_division=0))
        auc = float(roc_auc_score(y, y_proba))

        evaluation_results["churn_random_forest"] = {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4)
        }
        print(f"   Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f}")

    except Exception as e:
        print(f"   ⚠️ Churn Classifier Evaluation Error: {e}")

    # 3. Apriori Market Basket Association Rules Evaluation
    print("\n[3/4] Evaluating Apriori Association Rules...")
    try:
        rules = joblib.load("saved_models/apriori_rules.joblib")

        num_rules = int(len(rules))
        avg_lift = float(rules["lift"].mean()) if num_rules > 0 else 0.0
        max_lift = float(rules["lift"].max()) if num_rules > 0 else 0.0
        avg_confidence = float(rules["confidence"].mean()) if num_rules > 0 else 0.0

        evaluation_results["apriori_cross_sell"] = {
            "total_rules_discovered": num_rules,
            "average_lift": round(avg_lift, 2),
            "max_lift": round(max_lift, 2),
            "average_confidence_pct": round(avg_confidence * 100, 2)
        }
        print(f"   Rules Discovered: {num_rules} | Avg Lift: {avg_lift:.2f} | Max Lift: {max_lift:.2f} | Avg Confidence: {avg_confidence*100:.2f}%")
        
    except Exception as e:
        print(f"   ⚠️ Apriori Evaluation Error: {e}")

    # 4. Holt-Winters Sales Forecasting Evaluation
    print("\n[4/4] Evaluating Holt-Winters Exponential Smoothing Sales Forecast...")
    try:
        forecast_data = joblib.load("saved_models/sales_forecast.joblib")
        hist_sales = forecast_data["historical_sales"]

        fitted_values = forecast_data.get("fitted_values", hist_sales)
        
        calc_rmse = float(rmse(hist_sales, fitted_values))
        calc_mae = float(mae(hist_sales, fitted_values))
        calc_mape = float(np.mean(np.abs((hist_sales - fitted_values) / hist_sales)) * 100)

        evaluation_results["holt_winters_forecasting"] = {
            "rmse": round(calc_rmse, 2),
            "mae": round(calc_mae, 2),
            "mape_pct": round(calc_mape, 2)
        }
        print(f"   RMSE: ${calc_rmse:,.2f} | MAE: ${calc_mae:,.2f} | MAPE: {calc_mape:.2f}%")
    except Exception as e:
        print(f"   ⚠️ Forecasting Evaluation Error: {e}")

    # Save evaluation to JSON
    os.makedirs("saved_models", exist_ok=True)
    report_path = "saved_models/model_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"\nAll Evaluation Metrics Exported Successfully to `{report_path}`!")

if __name__ == "__main__":
    evaluate_all_models()