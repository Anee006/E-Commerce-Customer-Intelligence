# An interactive web app allowing users to view real-time KPIs, perform RFM customer segmentation, test churn probability, and receive product recommendations

import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

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
    recency = st.slider("Recency (Days since last order)", 0, 365, 30)
    frequency = st.slider("Frequency (Total Lifetime Orders)", 1, 50, 2)
    monetary = st.number_input("Monetary ($ Lifetime Spend)", 10.0, 10000.0, 250.0)

    if st.button("Classify Customer Segment"):
        scaled = scaler.transform([[recency, frequency, monetary]])
        cluster_id = kmeans.predict(scaled)[0]
        segment_map = {
            0: "Low-Value / At-Risk",
            1: "Regular Customer",
            2: "High-Spending Customer",
            3: "VIP Customer"
        }
        st.success(f"Assigned Cluster {cluster_id}: **{segment_map.get(cluster_id, 'Segment ' + str(cluster_id))}**")

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
                st.info(f"👉 **{consequent}** (Lift: {row['lift']:.2f}, Confidence: {row['confidence']*100:.1f}%)")
        else:
            st.warning("No cross-sell rules found for this category.")
    else:
        st.warning("No association rules available.")

