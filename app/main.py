# An interactive web app allowing users to view real-time KPIs, perform RFM customer segmentation, test churn probability, and receive product recommendations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from src.config import DATABASE_URL
from sqlalchemy import create_engine
from sklearn.decomposition import PCA

st.set_page_config(page_title="E-Commerce Customer Intelligence", layout="wide")

st.title("E-Commerce Customer Intelligence Platform")
st.markdown("End-to-End Analytics, Customer Segmentation & ML Product Recommendations")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load("saved_models/rfm_scaler.joblib")
    kmeans = joblib.load("saved_models/kmeans_rfm.joblib")
    churn_model = joblib.load("saved_models/churn_rf.joblib")

    try:
        rules = joblib.load("saved_models/apriori_rules.joblib")
    except Exception:
        rules = None

    return scaler, kmeans, churn_model, rules

try:
    scaler, kmeans, churn_model, rules = load_artifacts()
except Exception as e:
    st.error("⚠️ Pre-trained model files not found in `saved_models/`. Run the pipeline scripts first.")
    st.stop()

# sidebar setup
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to Module:", [
    "1. Executive Overview",
    "2. RFM Segmentation",
    "3. Churn Risk Predictor",
    "4. Cross-Sell Recommendations"
])

# Module 1: Executive Overview
if choice == "1. Executive Overview":
    st.header("Business Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", "$1,458,293", "+14%")
    c2.metric("Orders Processed", "99,441", "+9%")
    c3.metric("Unique Customers", "96,096", "+12%")
    
    df_sample = pd.DataFrame({
        "Category": ["Bed & Bath", "Health & Beauty", "Sports & Leisure", "Computers & Accessories"],
        "Sales": [172000, 145000, 128000, 95000]
    })
    fig = px.bar(df_sample, x="Category", y="Sales", color="Category", title="Revenue by Top Categories")
    st.plotly_chart(fig, use_container_width=True)

# Module 2: RFM Customer Segmentation
elif choice == "2. RFM Segmentation":
    st.header("Real-Time RFM Customer Segment Predictor")
    col1, col2, col3 = st.columns(3)

    with col1:
        recency = st.slider("Recency (Days since last order)", 0, 365, 120)
    with col2:
        frequency = st.slider("Frequency (Total Lifetime Orders)", 1, 20, 1)
    with col3:
        monetary = st.number_input("Monetary ($ Lifetime Spend)", 10.0, 5000.0, 85.0)

    if st.button("Classify Customer Segment"):
        scaler = joblib.load("saved_models/rfm_scaler.joblib")
        kmeans = joblib.load("saved_models/kmeans_rfm.joblib")
        cluster_mapping = joblib.load("saved_models/rfm_cluster_mapping.joblib")

        # Apply log transformation matching training data preprocessing
        log_inputs = np.array([[
            np.log1p(recency),
            np.log1p(frequency),
            np.log1p(monetary)
        ]])

        scaled_inputs = scaler.transform(log_inputs)
        cluster_id = kmeans.predict(scaled_inputs)[0]
        segment_name = cluster_mapping.get(cluster_id, f"Cluster {cluster_id}")

        st.success(f"Assigned Cluster {cluster_id}: **{segment_name}**")

    st.markdown("---")
    st.subheader("2D PCA Cluster Visualization")

    # Load RFM dataset sample to show cluster distribution
    @st.cache_data
    def load_cluster_plot():
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
        LIMIT 2000  -- Sample 2000 points for quick rendering
        """
        df = pd.read_sql(query, engine)
        max_date = pd.to_datetime(df["max_date"]).max()
        df["recency"] = (max_date - pd.to_datetime(df["max_date"])).dt.days
        
        # Scale & Predict
        log_df = np.log1p(df[["recency", "frequency", "monetary"]])
        scaler = joblib.load("saved_models/rfm_scaler.joblib")
        kmeans = joblib.load("saved_models/kmeans_rfm.joblib")
        mapping = joblib.load("saved_models/rfm_cluster_mapping.joblib")
        
        scaled = scaler.transform(log_df)
        clusters = kmeans.predict(scaled)
        df["Segment"] = [mapping.get(c, f"Cluster {c}") for c in clusters]
        
        # Apply 2D PCA
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(scaled)
        df["PC1"] = pca_coords[:, 0]
        df["PC2"] = pca_coords[:, 1]
        
        return df

    try:
        plot_df = load_cluster_plot()
        fig = px.scatter(
            plot_df, x="PC1", y="PC2", color="Segment",
            title="Customer Cluster Boundaries (PCA Reduced)",
            hover_data=["recency", "frequency", "monetary"]
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.info("Run all pipeline scripts to render PCA visualization.")

# Module 3: Churn Risk Predictor
elif choice == "3. Churn Risk Predictor":
    st.header("Customer Repeat Purchase Predictor")
    avg_val = st.number_input("Average Order Value ($)", 10.0, 2000.0, 120.0)
    total_val = st.number_input("Total Historical Spend ($)", 10.0, 20000.0, 120.0)

    if st.button("Predict Repeat Purchase Probability"):
        prob = churn_model.predict_proba([[avg_val, total_val]])[0][1]
        st.write(f"Probability of Repeat Purchase: **{prob * 100:.2f}%**")

# Module 4: Cross-Sell Recommendations
elif choice == "4. Cross-Sell Recommendations":
    st.header("Market Basket Cross-Sell Engine")
    if rules is not None and not rules.empty:
        # Extract unique antecedents
        antecedents = list(set([list(x)[0] for x in rules['antecedents'] if len(x) > 0]))
        selected_item = st.selectbox("Select Purchased Product Category:", antecedents)
        
        filtered = rules[rules['antecedents'].apply(lambda x: selected_item in x)]
        filtered = filtered.sort_values(by="lift", ascending=False)
        
        if not filtered.empty:
            st.write("### Recommended Complementary Categories:")
            for idx, row in filtered.head(3).iterrows():
                consequent = list(row['consequents'])[0]
                st.info(f"**{consequent}** (Lift: {row['lift']:.2f}, Confidence: {row['confidence']*100:.1f}%)")
        else:
            st.warning("No cross-sell rules found for this category.")
    else:
        st.warning("No association rules available.")

elif choice == "5. Sales Forecasting":
    st.header("Monthly Revenue Forecasting")
    
    try:
        forecast_data = joblib.load("saved_models/sales_forecast.joblib")
        hist_df = forecast_data["historical_sales"].reset_index()
        hist_df.columns = ["Date", "Revenue"]
        hist_df["Type"] = "Historical"

        fore_df = forecast_data["forecast_sales"].reset_index()
        fore_df.columns = ["Date", "Revenue"]
        fore_df["Type"] = "Forecast"

        combined_df = pd.concat([hist_df, fore_df], ignore_index=True)

        fig = px.line(
            combined_df, 
            x="Date", 
            y="Revenue", 
            color="Type", 
            line_dash="Type",
            title="Historical Sales & 6-Month Projected Revenue"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error("⚠️ Run `python scripts/05_train_forecasting.py` first to generate forecast artifacts.")