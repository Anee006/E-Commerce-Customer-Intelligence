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
    engine = create_engine(DATABASE_URL)

    query = """
    SELECT c.customer_unique_id AS basket_id, p.product_category_name
    FROM fact_sales f
    JOIN dim_customer c ON f.customer_key = c.customer_key
    JOIN dim_product p ON f.product_key = p.product_key
    WHERE p.product_category_name != 'Unknown'
    """
    df = pd.read_sql(query, engine)

    print("Building Basket Matrix...")
    basket = (df.groupby(['basket_id', 'product_category_name'])['product_category_name']
          .count().unstack().reset_index().fillna(0)
          .set_index('basket_id'))
    
    basket_sets = basket.astype(bool)

    print("Mining Frequent Itemsets & Generating Rules...")
    frequent_itemsets = apriori(basket_sets, min_support=0.003, use_colnames=True)
    
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        print(f"Discovered {len(rules)} association rules!")
        joblib.dump(rules, "saved_models/apriori_rules.joblib")
    else:
        print("No frequent itemsets found with current support threshold.")

if __name__ == "__main__":
    train_apriori()