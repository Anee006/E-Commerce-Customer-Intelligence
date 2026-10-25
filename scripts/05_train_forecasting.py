# Time Series Sales Forecasting using statsmodels (ARIMA / Holt-Winters Exponential Smoothing)

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATABASE_URL

def train_sales_forecasting():
    print("Querying Sales Data for Time Series Forecasting...")
    engine = create_engine(DATABASE_URL)

    # Query daily aggregated sales
    query = """
    SELECT 
        d.full_date,
        SUM(f.total_value) AS daily_revenue
    FROM fact_sales f
    JOIN dim_date d ON f.date_key = d.date_key
    GROUP BY d.full_date
    ORDER BY d.full_date ASC
    """
    df = pd.read_sql(query, engine)
    df['full_date'] = pd.to_datetime(df['full_date'])
    
    df = df.set_index('full_date')

    # Resample to Monthly Sales ('MS' = Month Start)
    print("📅 Aggregating Daily Sales to Monthly Totals...")
    monthly_sales = df['daily_revenue'].resample('MS').sum()

    # Filter for complete months with revenue over $10,000
    monthly_sales = monthly_sales[monthly_sales > 10000]

    monthly_sales.index.freq = 'MS'

    print(f"Total Valid Months Analyzed: {len(monthly_sales)}")
    print(monthly_sales.tail(6))

    # Fit Holt-Winters Exponential Smoothing Model
    print("Fitting Exponential Smoothing Time Series Model...")
    model = ExponentialSmoothing(
        monthly_sales, 
        trend='add', 
        seasonal='add', 
        seasonal_periods=12
    )
    fitted_model = model.fit()

    # Generate 6-Month Future Forecast
    forecast_steps = 6
    forecast = fitted_model.forecast(forecast_steps)
    
    print("\nNext 6 Months Sales Forecast:")
    print(forecast)

    # Save Artifacts for Streamlit Deployment
    os.makedirs("saved_models", exist_ok=True)
    
    forecast_data = {
        "historical_sales": monthly_sales,
        "forecast_sales": forecast,
        "fitted_model": fitted_model
    }
    
    joblib.dump(forecast_data, "saved_models/sales_forecast.joblib")
    print("\nSaved `sales_forecast.joblib` to saved_models/")

if __name__ == "__main__":
    train_sales_forecasting()