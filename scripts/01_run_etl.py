# Reads raw CSV files, handles missing values, builds dimension and fact df, and streams them into MySQL

import pandas as pd
from sqlalchemy import create_engine, text
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATABASE_URL

def run_etl():
    engine = create_engine(DATABASE_URL)
    data_path = "data/raw/"

    # Load CSV Files
    customers = pd.read_csv(os.path.join(data_path, "olist_customers_dataset.csv"))
    products = pd.read_csv(os.path.join(data_path, "olist_products_dataset.csv"))
    orders = pd.read_csv(os.path.join(data_path, "olist_orders_dataset.csv"))
    order_items = pd.read_csv(os.path.join(data_path, "olist_order_items_dataset.csv"))
    translations = pd.read_csv(os.path.join(data_path, "product_category_name_translation.csv"))

    # Transform Customer Dimension
    print("Cleaning & Extracting Customer Dimension...")
    customers["customer_key"] = customers["customer_id"]

    customers_dim = customers[[
        "customer_key", "customer_id", "customer_unique_id",
        "customer_zip_code_prefix", "customer_city", "customer_state"
    ]].drop_duplicates(subset=["customer_key"])

    # Transform Product Dimension with English Translation
    print("Translating Product Categories to English...")
    products = products.merge(translations, on="product_category_name", how="left")

    # Use English category if available, fallback to original or 'Unknown'
    products["product_category_name"] = products["product_category_name_english"].fillna(
        products["product_category_name"]
    ).fillna("Unknown")

    # Format Category Display Names 
    products["product_category_name"] = (
        products["product_category_name"].astype(str).str.replace("_", " ").str.title()
    )

    # Impute Missing Values for Physical Dimensions
    for col in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        products[col] = products[col].fillna(products[col].median())

    products["product_key"] = products["product_id"]

    products_dim = products[[
        "product_key", "product_id", "product_category_name",
        "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"
    ]].drop_duplicates(subset=["product_key"])

    # Transform Date Dimension
    print("Generating Date Dimension Table...")

    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    unique_dates = orders["order_purchase_timestamp"].dt.date.unique()

    date_df = pd.DataFrame({"full_date": pd.to_datetime(unique_dates)})
    date_df["date_key"] = date_df["full_date"].dt.strftime("%Y%m%d").astype(int)
    date_df["year"] = date_df["full_date"].dt.year
    date_df["quarter"] = date_df["full_date"].dt.quarter
    date_df["month"] = date_df["full_date"].dt.month
    date_df["day"] = date_df["full_date"].dt.day
    date_df["day_of_week"] = date_df["full_date"].dt.day_of_week
    date_df["is_weekend"] = date_df["day_of_week"].isin([5, 6])
    date_dim = date_df.drop_duplicates(subset=["date_key"])

    # Transform Fact Sales Table
    print("Constructing Fact Sales Table...")

    sales_fact = order_items.merge(orders, on="order_id", how="inner")
    sales_fact["customer_key"] = sales_fact["customer_id"]
    sales_fact["product_key"] = sales_fact["product_id"]
    sales_fact["date_key"] = pd.to_datetime(sales_fact["order_purchase_timestamp"]).dt.strftime("%Y%m%d").astype(int)
    sales_fact["total_value"] = sales_fact["price"] + sales_fact["freight_value"]

    sales_fact_dim = sales_fact[[
        "order_id", "order_item_id", "customer_key", "product_key",
        "seller_id", "date_key", "price", "freight_value", "total_value"
    ]].drop_duplicates(subset=["order_id", "order_item_id"])

    # Stream Clean Data to MySQL Warehouse
    print("Clearing and Re-loading MySQL Warehouse...")

    with engine.begin() as conn:
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
        conn.execute(text("TRUNCATE TABLE fact_sales;"))
        conn.execute(text("TRUNCATE TABLE dim_customer;"))
        conn.execute(text("TRUNCATE TABLE dim_product;"))
        conn.execute(text("TRUNCATE TABLE dim_date;"))
        conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))

    # Load Data into Dimensions & Fact
    customers_dim.to_sql("dim_customer", engine, if_exists="append", index=False, chunksize=5000)
    products_dim.to_sql("dim_product", engine, if_exists="append", index=False, chunksize=5000)
    date_dim.to_sql("dim_date", engine, if_exists="append", index=False, chunksize=5000)
    sales_fact_dim.to_sql("fact_sales", engine, if_exists="append", index=False, chunksize=5000)

    print("ETL Execution Finished Successfully!")

if __name__ == "__main__":
    run_etl()