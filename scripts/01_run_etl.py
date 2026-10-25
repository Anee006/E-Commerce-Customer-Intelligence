# Reads raw CSV files, handles missing values, builds dimension and fact df, and streams them into MySQL

import pandas as pd
from sqlalchemy import create_engine
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATABASE_URL

def run_etl():
    engine = create_engine(DATABASE_URL)
    data_path = "data/raw/"

    # load CSV files
    customers = pd.read_csv(os.path.join(data_path, "olist_customers_dataset.csv"))
    products = pd.read_csv(os.path.join(data_path, "olist_products_dataset.csv"))
    translations = pd.read_csv(os.path.join(data_path, "product_category_name_translation.csv"))
    orders = pd.read_csv(os.path.join(data_path, "olist_orders_dataset.csv"))
    order_items = pd.read_csv(os.path.join(data_path, "olist_order_items_dataset.csv"))

    # transform customer dimension
    print("Cleaning & Extracting Customer Dimension...")
    customers["customer_key"] = customers["customer_id"]

    customers_dim = customers[[
        "customer_key", "customer_id", "customer_unique_id",
        "customer_zip_code_prefix", "customer_city", "customer_state"
    ]].drop_duplicates(subset=["customer_key"])

    # Map Portuguese category names to English
    print("Translating Product Categories to English...")
    products = products.merge(translations, on="product_category_name", how="left")

    # Fallback: Use English name if available, else original, else 'Unknown'
    products["product_category_name"] = products["product_category_name_english"].fillna(
        products["product_category_name"]
    ).fillna("Unknown")

    # Format category display names 
    products["product_category_name"] = (
        products["product_category_name"].str.replace("_", " ").str.title()
    )

    # transform product dimension 
    print("Cleaning & Imputing Missing Values in Product Dimension...")

    products["product_key"] = products["product_id"]

    for col in ["product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]:
        products[col] = products[col].fillna(products[col].median())
    
    products_dim = products[[
        "product_key", "product_id", "product_category_name",
        "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"
    ]].drop_duplicates(subset=["product_key"])

    # transform date dimension
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

    # transform fact sales table
    print("Constructing Fact Sales Table...")
    sales_fact = order_items.merge(orders, on="order_id", how="inner")

    sales_fact["customer_key"] = sales_fact["customer_id"]
    sales_fact["product_key"] = sales_fact["product_id"]
    sales_fact["date_key"] = pd.to_datetime(sales_fact["order_purchase_timestamp"]).dt.strftime("%Y%m%d").astype(int)
    sales_fact["total_value"] = sales_fact["price"] + sales_fact["freight_value"]

    sales_fact_dim = sales_fact[[
        "order_id", "order_item_id", "customer_key", "product_key",
        "seller_id", "date_key", "price", "freight_value", "total_value"
    ]]

    # database load operation
    print("Streaming Clean Data to MySQL Warehouse...")

    customers_dim.to_sql("dim_customer", engine, if_exists="append", index=False, chunksize=5000)
    products_dim.to_sql("dim_product", engine, if_exists="append", index=False, chunksize=5000)
    date_dim.to_sql("dim_date", engine, if_exists="append", index=False, chunksize=5000)
    sales_fact_dim.to_sql("fact_sales", engine, if_exists="append", index=False, chunksize=5000)

    print("ETL Execution Finished Successfully!")

if __name__ == "__main__":
    run_etl()