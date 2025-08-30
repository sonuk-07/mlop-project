import pandas as pd
from sqlalchemy import create_engine
import os
import sys

def load_data_from_db():
    """
    Load data from 'shoppers' table in MySQL/MariaDB into a Pandas DataFrame.

    Steps:
    1. Connect to the database using SQLAlchemy.
    2. Load data from 'shoppers' table into a DataFrame.
    3. Return the DataFrame for further processing.
    """

    try:
        # -------------------------
        # 1. Database Connection
        # -------------------------
        db_password = os.getenv("MYSQL_PASSWORD", "Yunachan10")
        connection_string = f"mysql+pymysql://sonu:{db_password}@mariadb-columnstore:3306/shoppers_db"
        eng_conn = create_engine(connection_string)

        # -------------------------
        # 2. Load data into DataFrame
        # -------------------------
        print("[INFO] Connecting to database and reading 'shoppers' table...")
        df = pd.read_sql_table('shoppers', con=eng_conn)
        print(f"[INFO] Loaded {len(df)} rows from database")

        return df

    except Exception as e:
        print(f"[ERROR] Failed to load data from database: {e}")
        print("Make sure:")
        print("1. MariaDB container is running (`docker ps`)")
        print("2. Database 'shoppers_db' and table 'shoppers' exist")
        print("3. User 'sonu' has proper privileges")
        return None

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    df = load_data_from_db()
    if df is None:
        sys.exit(1)
    else:
        # Example: show first 5 rows
        print(df.head())
