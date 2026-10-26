# Replace with your OWN credentials

import os
from sqlalchemy.engine import URL

DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "YOUR_MYSQL_PASSWORD_HERE")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "olist_dw")

DATABASE_URL = URL.create(
    drivername="mysql+pymysql",
    username=DB_USER,
    password=DB_PASS,
    host=DB_HOST,
    port=int(DB_PORT),
    database=DB_NAME,
)
