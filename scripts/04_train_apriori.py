# Applies mlxtend implementation of the Apriori on product purchases to compute support, confidence, and lift for cross-selling recommendations

import pandas as pd
from sqlalchemy import create_engine
from mlxtend.frequent_patterns import apriori, association_rules
import joblib
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import DATABASE_URL

def train_apriori():
    print("Querying Customer Lifetime Baskets for Apriori Rule Mining...")
    engine = create_engine(DATABASE_URL)

    query = """
    SELECT 
        c.customer_unique_id AS basket_id, 
        p.product_category_name
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_key = c.customer_key
    JOIN dim_product p ON f.product_key = p.product_key
    WHERE p.product_category_name NOT IN ('Unknown', 'Portateis Cozinha E Bandejas')
    """
    df = pd.read_sql(query, engine)

    print("Building One-Hot Encoded Customer Basket Matrix...")
    basket = (df.groupby(['basket_id', 'product_category_name'])['product_category_name']
              .count().unstack().reset_index().fillna(0)
              .set_index('basket_id'))
    
    basket_sets = basket.astype(bool)
    multi_category_baskets = basket_sets[basket_sets.sum(axis=1) > 1]

    print(f"Total Multi-Category Customer Baskets: {len(multi_category_baskets)}")

    if len(multi_category_baskets) > 0:
        print("Mining Frequent Itemsets...")
        frequent_itemsets = apriori(multi_category_baskets, min_support=0.0005, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)

        print(f"Discovered {len(rules)} English association rules!")
        
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump(rules, "saved_models/apriori_rules.joblib")
        print("Saved `apriori_rules.joblib` to saved_models/")

if __name__ == "__main__":
    train_apriori()