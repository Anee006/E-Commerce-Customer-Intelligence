CREATE DATABASE IF NOT EXISTS olist_dw;
USE olist_dw;

-- Dimension 1: Customer
CREATE TABLE IF NOT EXISTS dim_customer (
    customer_key VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50),
    customer_unique_id VARCHAR(50),
    customer_zip_code_prefix INT,
    customer_city VARCHAR(100),
    customer_state VARCHAR(5)
);

-- Dimension 2: Product
CREATE TABLE IF NOT EXISTS dim_product (
    product_key VARCHAR(50) PRIMARY KEY,
    product_id VARCHAR(50),
    product_category_name VARCHAR(100),
    product_weight_g FLOAT,
    product_length_cm FLOAT,
    product_height_cm FLOAT,
    product_width_cm FLOAT
);

-- Dimension 3: Date
CREATE TABLE IF NOT EXISTS dim_date (
    date_key INT PRIMARY KEY,
    full_date DATE,
    year INT,
    quarter INT,
    month INT,
    day INT,
    day_of_week INT,
    is_weekend BOOLEAN
);

-- Fact Table: Sales
CREATE TABLE IF NOT EXISTS fact_sales (
    order_id VARCHAR(50),
    order_item_id INT,
    customer_key VARCHAR(50),
    product_key VARCHAR(50),
    seller_id VARCHAR(50),
    date_key INT,
    price DECIMAL(10, 2),
    freight_value DECIMAL(10, 2),
    total_value DECIMAL(10, 2),
    PRIMARY KEY (order_id, order_item_id),
    FOREIGN KEY (customer_key) REFERENCES dim_customer(customer_key),
    FOREIGN KEY (product_key) REFERENCES dim_product(product_key),
    FOREIGN KEY (date_key) REFERENCES dim_date(date_key)
);