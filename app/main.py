# An interactive web app allowing users to view real-time KPIs, perform RFM customer segmentation, test churn probability, and receive product recommendations

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine
from sklearn.decomposition import PCA
from src.config import DATABASE_URL

st.set_page_config(
    page_title="E-Commerce Customer Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLUE = "#1677D2"
NAVY = "#18324F"
MUTED = "#64748B"
PALE_BLUE = "#EAF3FF"

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

      /* Streamlit theme variables adapt to the selected Light/Dark theme. */
      .stApp {
        background: var(--background-color);
        color: var(--text-color);
      }
      [data-testid="stHeader"] {
        background: color-mix(in srgb, var(--background-color) 92%, transparent);
      }
      [data-testid="stSidebar"] {
        background: var(--secondary-background-color);
        border-right: 1px solid var(--border-color, rgba(128,128,128,.22));
      }
      [data-testid="stSidebar"] h1 { color: var(--text-color); font-size: 1.25rem; }
      .block-container { padding-top: 1.8rem; padding-bottom: 2.5rem; max-width: 1500px; }

      .hero, div[data-testid="stMetric"], .chart-heading {
        background: var(--secondary-background-color);
        color: var(--text-color);
        border: 1px solid var(--border-color, rgba(128,128,128,.22));
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,.04);
      }
      .hero { padding: 1.5rem 1.25rem; text-align: center; margin-bottom: 1.5rem; }
      .hero h1 { color: var(--text-color); font-size: 2.1rem; font-weight: 800; margin: 0 0 .35rem 0; }
      .hero p { color: var(--text-color); opacity: .72; margin: 0; font-size: .98rem; }
      .section-heading { color: var(--text-color); font-size: 1.55rem; font-weight: 700; margin: .4rem 0 1rem 0; }
      div[data-testid="stMetric"] { padding: 1rem 1.1rem; }
      div[data-testid="stMetricLabel"] { color: var(--text-color); opacity: .72; font-size: .88rem; }
      div[data-testid="stMetricValue"] { color: var(--text-color); font-weight: 750; font-size: 1.65rem; }
      .chart-heading {
        border-radius: 10px 10px 0 0;
        padding: .8rem 1rem .55rem 1rem;
        margin-top: .2rem;
        box-shadow: none;
      }
      .chart-heading h4 { margin: 0; color: var(--text-color); font-size: 1rem; font-weight: 700; }
      .chart-heading p { margin: .25rem 0 0 0; color: var(--text-color); opacity: .68; font-size: .8rem; }
      div[data-testid="stPlotlyChart"] {
        background: var(--secondary-background-color);
        border: 1px solid var(--border-color, rgba(128,128,128,.22));
        border-top: 0; border-radius: 0 0 10px 10px;
        padding: .3rem .35rem .1rem .35rem; margin-bottom: 1rem;
      }
      .small-note { color: var(--text-color); opacity: .68; font-size: .85rem; }
      div.stButton > button { border-radius: 8px; font-weight: 600; }
      hr { border-color: var(--border-color, rgba(128,128,128,.22)); }
      /* Keep form labels and other common widgets readable in either theme. */
      [data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] {
        color: var(--text-color);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

engine = create_engine(DATABASE_URL)

def get_theme_colors():
    """Return colors suitable for the active Streamlit theme when detectable."""
    base = "light"
    try:
        theme_context = getattr(st.context, "theme", None)
        if isinstance(theme_context, dict):
            base = str(theme_context.get("base", base)).lower()
        elif isinstance(theme_context, str):
            base = theme_context.lower()
    except Exception:
        pass
    try:
        configured_base = st.get_option("theme.base")
        if configured_base in ("light", "dark") and base == "light":
            base = configured_base
    except Exception:
        pass

    if "dark" in base:
        return {
            "base": "dark",
            "template": "plotly_dark",
            "text": "#E5E7EB",
            "muted": "#AAB4C3",
            "grid": "#374151",
            "line": "#4B5563",
            "hover_bg": "#111827",
            "paper": "rgba(0,0,0,0)",
            "plot": "rgba(0,0,0,0)",
            "card": "#1F2937",
        }
    
    return {
        "base": "light",
        "template": "plotly_white",
        "text": "#18324F",
        "muted": "#64748B",
        "grid": "#EDF1F6",
        "line": "#E7EDF5",
        "hover_bg": "#FFFFFF",
        "paper": "rgba(255,255,255,0)",
        "plot": "rgba(255,255,255,0)",
        "card": "#FFFFFF",
    }


def style_chart(fig, height=360):
    colors = get_theme_colors()
    fig.update_layout(
        template=colors["template"],
        height=height,
        margin=dict(l=18, r=18, t=22, b=18),
        paper_bgcolor=colors["paper"],
        plot_bgcolor=colors["plot"],
        font=dict(family="Inter, sans-serif", color=colors["text"], size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor=colors["hover_bg"], font_color=colors["text"]),
        uniformtext_minsize=10,
        uniformtext_mode="show",
    )
    fig.update_traces(
        selector=dict(type="bar"),
        marker_line_width=0,
        textfont=dict(
            color=get_theme_colors()["text"],
            size=12,
        ),
    )
    fig.update_xaxes(showgrid=False, linecolor=colors["line"], zeroline=False, color=colors["muted"])
    fig.update_yaxes(showgrid=True, gridcolor=colors["grid"], zeroline=False, linecolor=colors["line"], color=colors["muted"])
    return fig

def chart_header(title, subtitle):
    st.markdown(
        f'<div class="chart-heading"><h4>{title}</h4><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )

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
except Exception:
    st.error("Pre-trained model files were not found in `saved_models/`. Run the pipeline scripts first.")
    st.stop()

with st.sidebar:
    choice = st.radio(
        "NAVIGATION",
        [
            "Executive Overview",
            "RFM Segmentation",
            "Churn Risk Predictor",
            "Cross-Sell Recommendations",
            "Sales Forecasting",
            "Model Evaluation Metrics",
        ],
        label_visibility="visible",
    )

st.markdown(
    """
    <div class="hero">
      <h1>E-Commerce Customer Intelligence</h1>
      <p>Analytics, customer segmentation, repeat-purchase insights and product recommendations</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Module 1: Executive Overview
if choice == "Executive Overview":
    st.markdown('<div class="section-heading">Executive Overview</div>', unsafe_allow_html=True)
    total_revenue = 1_458_293
    orders_processed = 99_441
    unique_customers = 96_096
    average_order_value = total_revenue / orders_processed

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue", f"${total_revenue:,.0f}", "+14%")
    k2.metric("Orders Processed", f"{orders_processed:,}", "+9%")
    k3.metric("Unique Customers", f"{unique_customers:,}", "+12%")
    k4.metric("Average Order Value", f"${average_order_value:,.2f}")

    category_df = pd.DataFrame(
        {
            "Category": ["Bed & Bath", "Health & Beauty", "Sports & Leisure", "Computers & Accessories"],
            "Sales": [172000, 145000, 128000, 95000],
        }
    )

    left, right = st.columns(2, gap="large")

    with left:
        chart_header("Revenue by Category", "Sales contribution across selected product categories")
        fig_category = px.bar(
            category_df,
            x="Category",
            y="Sales",
            color_discrete_sequence=[BLUE],
            text_auto=".2s",
        )

        fig_category.update_traces(marker_line_width=0, textposition="outside", cliponaxis=False)
        fig_category.update_yaxes(title="Revenue ($)", tickprefix="$", separatethousands=True)
        fig_category.update_xaxes(title="", tickangle=-15)
        style_chart(fig_category, 370)
        st.plotly_chart(fig_category, use_container_width=True, config={"displayModeBar": False})

    with right:
        chart_header("Customer Geographic Distribution", "Unique customers by state")

        try:
            @st.cache_data
            def load_geo_data():
                query = """
                    SELECT customer_state AS State,
                           COUNT(DISTINCT customer_unique_id) AS Total_Customers
                    FROM dim_customer
                    GROUP BY customer_state
                    ORDER BY Total_Customers DESC
                """

                return pd.read_sql(query, engine)
            
            geo_df = load_geo_data()

            if geo_df.empty:
                st.info("No customer geography data was returned from the database.")
            else:
                geo_df = geo_df.sort_values("Total_Customers", ascending=True).tail(15)
                fig_geo = px.bar(
                    geo_df,
                    x="Total_Customers",
                    y="State",
                    orientation="h",
                    color_discrete_sequence=[BLUE],
                )

                fig_geo.update_traces(marker_line_width=0)
                fig_geo.update_xaxes(title="Unique customers", separatethousands=True)
                fig_geo.update_yaxes(title="")
                style_chart(fig_geo, 370)
                st.plotly_chart(fig_geo, use_container_width=True, config={"displayModeBar": False})

        except Exception:
            st.info("Could not load customer geography. Check the database connection and `dim_customer` table.")

# Module 2: RFM Customer Segmentation
elif choice == "RFM Segmentation":
    st.markdown('<div class="section-heading">RFM Customer Segmentation</div>', unsafe_allow_html=True)
    st.caption("Estimate a customer segment using recency, purchase frequency and lifetime monetary value")

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        recency = st.slider("Recency · Days since last order", 0, 365, 120)

    with col2:
        frequency = st.slider("Frequency · Lifetime orders", 1, 20, 1)

    with col3:
        monetary = st.number_input("Monetary · Lifetime spend ($)", min_value=10.0, max_value=5000.0, value=85.0, step=10.0)

    if st.button("Classify Customer Segment", type="primary"):
        try:
            cluster_mapping = joblib.load("saved_models/rfm_cluster_mapping.joblib")
            log_inputs = np.array([[np.log1p(recency), np.log1p(frequency), np.log1p(monetary)]])
            scaled_inputs = scaler.transform(log_inputs)
            cluster_id = int(kmeans.predict(scaled_inputs)[0])
            segment_name = cluster_mapping.get(cluster_id, f"Cluster {cluster_id}")
            st.success(f"Assigned segment: **{segment_name}** (Cluster {cluster_id})")

        except Exception as exc:
            st.error(f"Could not classify this customer. Check the RFM model files. Details: {exc}")

    st.markdown("---")
    chart_header("Customer Segments (PCA View)", "A two-dimensional view of customers grouped by their predicted RFM segment")

    @st.cache_data
    def load_cluster_plot():
        query = """
            SELECT c.customer_unique_id,
                   MAX(d.full_date) AS max_date,
                   COUNT(DISTINCT f.order_id) AS frequency,
                   SUM(f.total_value) AS monetary
            FROM fact_sales f
            JOIN dim_customer c ON f.customer_key = c.customer_key
            JOIN dim_date d ON f.date_key = d.date_key
            GROUP BY c.customer_unique_id
            LIMIT 2000
        """

        df = pd.read_sql(query, engine)

        if df.empty:
            return df

        df["max_date"] = pd.to_datetime(df["max_date"])
        max_date = df["max_date"].max()
        df["recency"] = (max_date - df["max_date"]).dt.days

        log_df = np.log1p(df[["recency", "frequency", "monetary"]].clip(lower=0))
        scaled = scaler.transform(log_df)
        clusters = kmeans.predict(scaled)
        mapping = joblib.load("saved_models/rfm_cluster_mapping.joblib")

        df["Segment"] = [mapping.get(int(c), f"Cluster {c}") for c in clusters]

        pca_coords = PCA(n_components=2).fit_transform(scaled)
        df["PC1"], df["PC2"] = pca_coords[:, 0], pca_coords[:, 1]

        return df

    try:
        plot_df = load_cluster_plot()

        if plot_df.empty:
            st.warning("No customer data was returned for the PCA chart.")
        else:
            fig = px.scatter(
                plot_df,
                x="PC1",
                y="PC2",
                color="Segment",
                hover_data=["recency", "frequency", "monetary"],
                color_discrete_sequence=px.colors.qualitative.Set2,
            )

            fig.update_traces(
                marker=dict(size=7, opacity=0.8)
            )
            fig.update_xaxes(title="Principal Component 1")
            fig.update_yaxes(title="Principal Component 2")

            style_chart(fig, 480)

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    except Exception as exc:
        st.error(f"PCA chart error: {type(exc).__name__}: {exc}")


# Module 3: Repeat Purchase / Churn Risk Predictor
elif choice == "Churn Risk Predictor":
    st.markdown('<div class="section-heading">Repeat Purchase Predictor</div>', unsafe_allow_html=True)
    st.caption("Estimate the probability of a repeat purchase")
    left, right = st.columns(2, gap="large")

    with left:
        avg_val = st.number_input("Average Order Value ($)", min_value=10.0, max_value=2000.0, value=120.0, step=10.0)

    with right:
        total_val = st.number_input("Total Historical Spend ($)", min_value=10.0, max_value=20000.0, value=120.0, step=50.0)

    if st.button("Predict Repeat Purchase Probability", type="primary"):
        try:
            prob = float(churn_model.predict_proba([[avg_val, total_val]])[0][1])
            m1, m2 = st.columns([1, 2], gap="large")

            with m1:
                st.metric("Repeat Purchase Probability", f"{prob * 100:.1f}%")

            with m2:
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"color": get_theme_colors()["text"]}},
                    gauge={
                        "axis": {"range": [0, 100], "ticksuffix": "%", "tickcolor": MUTED},
                        "bar": {"color": BLUE},
                        "bgcolor": get_theme_colors()["card"],
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 35], "color": "#FDECEC"},
                            {"range": [35, 70], "color": "#FFF4D6"},
                            {"range": [70, 100], "color": "#E5F7ED"},
                        ],
                    },
                ))

                style_chart(gauge, 280)
                st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
            st.caption("Interpret this score alongside business context; it is a model estimate, not a guarantee.")

        except Exception as exc:
            st.error(f"Prediction failed. Check the saved model and its expected input features. Details: {exc}")

# Module 4: Cross-Sell Recommendations
elif choice == "Cross-Sell Recommendations":
    st.markdown('<div class="section-heading">Cross-Sell Recommendations</div>', unsafe_allow_html=True)

    if rules is not None and not rules.empty:
        antecedents = sorted({list(items)[0] for items in rules["antecedents"] if len(items) > 0}, key=str)

        if antecedents:
            selected_item = st.selectbox("Purchased product category", antecedents)
            filtered = rules[rules["antecedents"].apply(lambda items: selected_item in items)].copy()
            filtered = filtered.sort_values("lift", ascending=False)

            if not filtered.empty:
                st.markdown("### Recommended complementary categories")
                rec_cols = st.columns(min(3, len(filtered.head(3))), gap="large")

                for i, (_, row) in enumerate(filtered.head(3).iterrows()):
                    consequent_items = list(row["consequents"])
                    consequent = ", ".join(map(str, consequent_items))

                    with rec_cols[i]:
                        st.markdown(
                            f"<div style='background:var(--secondary-background-color);color:var(--text-color);border:1px solid var(--border-color, rgba(128,128,128,.22));border-radius:10px;padding:1.1rem;min-height:150px'>"
                            f"<div style='color:var(--text-color);opacity:.7;font-size:.8rem'>RECOMMENDATION {i+1}</div>"
                            f"<div style='font-size:1.1rem;font-weight:700;color:var(--text-color);margin:.5rem 0'>{consequent}</div>"
                            f"<div style='color:var(--primary-color);font-size:.9rem'>Lift: {row['lift']:.2f}</div>"
                            f"<div style='color:var(--text-color);opacity:.75;font-size:.85rem'>Confidence: {row['confidence']*100:.1f}%</div>"
                            "</div>",

                            unsafe_allow_html=True,
                        )

                with st.expander("View matching association rules"):
                    display_rules = filtered.copy()
                    display_rules["antecedents"] = display_rules["antecedents"].apply(lambda x: ", ".join(map(str, x)))
                    display_rules["consequents"] = display_rules["consequents"].apply(lambda x: ", ".join(map(str, x)))

                    cols = [c for c in ["antecedents", "consequents", "support", "confidence", "lift"] if c in display_rules.columns]

                    st.dataframe(display_rules[cols], use_container_width=True, hide_index=True)
            else:
                st.warning("No cross-sell rules were found for this category.")
        else:
            st.warning("The saved rules do not contain any usable product categories.")
    else:
        st.warning("No association rules are available. Run the market-basket pipeline to generate them.")

# Module 5: Sales Forecasting
elif choice == "Sales Forecasting":
    st.markdown('<div class="section-heading">Sales Forecasting</div>', unsafe_allow_html=True)

    try:
        forecast_data = joblib.load("saved_models/sales_forecast.joblib")
        hist_sales = forecast_data["historical_sales"]
        forecast_sales = forecast_data["forecast_sales"]

        if len(hist_sales) == 0 or len(forecast_sales) == 0:
            st.warning("The forecast artifact contains no historical or forecast values.")
            st.stop()

        avg_hist_revenue = float(hist_sales.mean())
        next_month_revenue = float(forecast_sales.iloc[0])
        mom_growth = ((next_month_revenue - avg_hist_revenue) / avg_hist_revenue * 100) if avg_hist_revenue else 0.0
        forecast_horizon = min(6, len(forecast_sales))
        selected_total = float(forecast_sales.iloc[:forecast_horizon].sum())
        projected_run_rate = float(forecast_sales.iloc[:forecast_horizon].mean())

        f1, f2, f3 = st.columns(3, gap="large")
        f1.metric("Historical Avg. Monthly Revenue", f"${avg_hist_revenue:,.0f}")
        f2.metric("Next Month Projected Revenue", f"${next_month_revenue:,.0f}", f"{mom_growth:+.1f}% vs. historical avg.")
        f3.metric(f"Projected Revenue · Next {forecast_horizon} Months", f"${selected_total:,.0f}", f"Avg. ${projected_run_rate:,.0f}/month")

        st.markdown("---")

        title_col, control_col = st.columns([2, 1], gap="large")

        with title_col:
            st.subheader("Sales Trajectory & Projection")

        with control_col:
            allowed_horizons = [n for n in [3, 6] if n <= len(forecast_sales)]

            if not allowed_horizons:
                allowed_horizons = [len(forecast_sales)]
            months_to_show = st.selectbox("Forecast window", options=allowed_horizons, index=len(allowed_horizons)-1, format_func=lambda n: f"Next {n} Months")

        selected_forecast = forecast_sales.iloc[:months_to_show]
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=hist_sales.index, y=hist_sales.values, mode="lines+markers", name="Historical Revenue",
            line=dict(color=BLUE, width=3), marker=dict(size=6, color="#69B6F5"),
            hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Historical:</b> $%{y:,.2f}<extra></extra>",
        ))

        fig_forecast.add_trace(go.Scatter(
            x=selected_forecast.index, y=selected_forecast.values, mode="lines+markers", name=f"Forecast · {months_to_show} months",
            line=dict(color="#10A778", width=3, dash="dash"), marker=dict(size=8, symbol="diamond", color="#34C99A"),
            hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Projected:</b> $%{y:,.2f}<extra></extra>",
        ))

        fig_forecast.update_xaxes(title="Timeline")
        fig_forecast.update_yaxes(title="Monthly revenue", tickprefix="$", separatethousands=True)
        style_chart(fig_forecast, 460)
        st.plotly_chart(fig_forecast, use_container_width=True, config={"displayModeBar": False})

        with st.expander("View month-by-month projected revenue"):
            breakdown_df = pd.DataFrame({
                "Forecast Period": pd.to_datetime(selected_forecast.index).strftime("%B %Y") if not isinstance(selected_forecast.index, pd.PeriodIndex) else selected_forecast.index.strftime("%B %Y"),
                "Projected Monthly Revenue": selected_forecast.values,
                "Difference from Historical Average ($)": selected_forecast.values - avg_hist_revenue,
                "Growth vs. Historical Average (%)": ((selected_forecast.values - avg_hist_revenue) / avg_hist_revenue * 100) if avg_hist_revenue else np.zeros(len(selected_forecast)),
            })

            st.dataframe(
                breakdown_df.style.format({
                    "Projected Monthly Revenue": "${:,.2f}",
                    "Difference from Historical Average ($)": "${:+,.2f}",
                    "Growth vs. Historical Average (%)": "{:+.1f}%",
                }),

                use_container_width=True,
                hide_index=True,
            )

    except Exception:
        st.error("The forecasting artifact `saved_models/sales_forecast.joblib` could not be loaded.")
        st.info("Run `python scripts/05_train_forecasting.py` in your terminal to regenerate it.")

# Module 6: Model Evaluation Metrics
elif choice == "Model Evaluation Metrics":
    st.markdown(
        '<div class="section-heading">Model Performance & Evaluation Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Quantitative evaluation for Data Mining and Machine Learning algorithms"
    )

    try:
        with open("saved_models/model_evaluation_report.json", "r") as f:
            metrics = json.load(f)

        # 1. Random Forest Repeat Purchase Classifier
        st.subheader("1. Random Forest Repeat Purchase Classifier")
        rf_m = metrics.get("churn_random_forest", {})

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("Accuracy", f"{rf_m.get('accuracy', 0) * 100:.2f}%")
        col2.metric("Precision", f"{rf_m.get('precision', 0) * 100:.2f}%")
        col3.metric("Recall", f"{rf_m.get('recall', 0) * 100:.2f}%")
        col4.metric("F1-Score", f"{rf_m.get('f1_score', 0) * 100:.2f}%")
        col5.metric("ROC-AUC", f"{rf_m.get('roc_auc', 0):.4f}")

        with st.expander("View Classification Performance Breakdown"):
            st.markdown("""
            - **Evaluation Target:** Repeat buyers (Orders > 1) versus
              single-purchase customers (Orders = 1).
            - **Input Features:** `avg_order_value` and `total_spent`.
            - **Performance Summary:** Accuracy, precision, recall, F1-score,
              and ROC-AUC summarize the classifier's performance.
            """)

        st.markdown("---")

        # 2. RFM K-Means Customer Segmentation
        st.subheader("2. RFM K-Means Clustering")
        rfm_m = metrics.get("rfm_kmeans", {})

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Silhouette Score",
            f"{rfm_m.get('silhouette_score', 0):.4f}",
        )
        c2.metric(
            "Inertia",
            f"{rfm_m.get('inertia', 0):,.2f}",
        )
        c3.metric(
            "Optimal Clusters (k)",
            f"{rfm_m.get('num_clusters', 4)}",
        )

        with st.expander("View Clustering Details"):
            st.markdown("""
            - **Transformation:** Log-transformed features using `ln(1 + x)`
              to reduce the impact of skewness.
            - **Cluster Quality:** The Silhouette Score measures how well
              observations fit within their assigned clusters compared
              with other clusters.
            """)

        st.markdown("---")

        # 3. Apriori Cross-Sell Engine
        st.subheader("3. Apriori Market Basket Engine")
        apr_m = metrics.get("apriori_cross_sell", {})

        a1, a2, a3, a4 = st.columns(4)

        a1.metric(
            "Rules Discovered",
            f"{apr_m.get('total_rules_discovered', 0):,}",
        )
        a2.metric(
            "Average Lift",
            f"{apr_m.get('average_lift', 0):.2f}",
        )
        a3.metric(
            "Maximum Lift",
            f"{apr_m.get('max_lift', 0):.2f}",
        )
        a4.metric(
            "Average Confidence",
            f"{apr_m.get('average_confidence_pct', 0):.2f}%",
        )

        with st.expander("View Association Rule Summaries"):
            st.markdown("""
            - **Support Threshold:** Reflects the frequency of itemset
              co-occurrences used to identify association rules.
            - **Lift:** Values greater than 1 indicate a positive
              association between the antecedent and consequent,
              relative to their independent occurrence.
            """)

        st.markdown("---")

        # 4. Holt-Winters Sales Forecasting
        st.subheader("4. Holt-Winters Revenue Forecasting")
        hw_m = metrics.get("holt_winters_forecasting", {})

        h1, h2, h3 = st.columns(3)

        h1.metric(
            "MAPE (Percentage Error)",
            f"{hw_m.get('mape_pct', 0):.2f}%",
        )
        h2.metric(
            "RMSE (USD)",
            f"${hw_m.get('rmse', 0):,.2f}",
        )
        h3.metric(
            "MAE (USD)",
            f"${hw_m.get('mae', 0):,.2f}",
        )

        with st.expander("View Time Series Evaluation Details"):
            st.markdown("""
            - **MAPE:** Expresses forecast error as a percentage.
            - **RMSE:** Measures the square root of the mean squared
              forecast errors and penalizes larger errors more heavily.
            - **MAE:** Measures the average absolute difference between
              actual and forecast values.
            - RMSE and MAE are displayed in USD, because the forecast
              target is revenue measured in USD.
            """)

    except Exception as e:
        st.error("Could not load the model evaluation report.")
        st.info(
            "Run `python scripts/06_evaluate_models.py` from your project "
            "root to generate `saved_models/model_evaluation_report.json`."
        )
        st.caption(f"Details: {type(e).__name__}: {e}")


st.markdown("---")
st.markdown('<p class="small-note">E-Commerce Customer Intelligence · Analytics dashboard</p>', unsafe_allow_html=True)
