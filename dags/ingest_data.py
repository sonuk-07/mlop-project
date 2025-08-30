import pandas as pd
from sqlalchemy import create_engine
import os
import sys

def ingest_data():
    """
    Ingest CSV data into MySQL/MariaDB database.

    Steps:
    1. Check if CSV file exists.
    2. Load CSV data into a Pandas DataFrame.
    3. Connect to the MySQL/MariaDB database using SQLAlchemy.
    4. Test the database connection.
    5. Ingest data into the 'shoppers' table (replace if exists), using chunking for large datasets.
    6. Handle errors gracefully and provide meaningful messages for debugging.

    Notes:
    - Password can be set via environment variable 'MYSQL_PASSWORD' for security.
    - This function is designed for MLOps pipelines where data validation is handled externally (e.g., Great Expectations).
    """

    try:
        # -------------------------
        # 1. CSV File Path
        # -------------------------
        data_path = '/opt/airflow/data/online_shoppers_intention.csv'
        if not os.path.exists(data_path):
            print(f"[ERROR] Data file not found at {data_path}")
            return False  # Stop execution if CSV is missing

        # -------------------------
        # 2. Load CSV into DataFrame
        # -------------------------
        print("[INFO] Loading CSV data...")
        df = pd.read_csv(data_path)
        print(f"[INFO] Loaded {len(df)} rows from CSV")

        # -------------------------
        # 3. Database Connection
        # -------------------------
        # Use environment variable for password if set; fallback to default
        db_password = os.getenv("MYSQL_PASSWORD", "Yunachan10")
        connection_string = f"mysql+pymysql://sonu:{db_password}@mariadb-columnstore:3306/shoppers_db"
        eng_conn = create_engine(connection_string)

        # -------------------------
        # 4. Test Connection
        # -------------------------
        try:
            with eng_conn.connect() as conn:
                print("[INFO] Database connection successful")
        except Exception as conn_err:
            print(f"[ERROR] Could not connect to database: {conn_err}")
            return False

        # -------------------------
        # 5. Ingest Data
        # -------------------------
        # - if_exists='replace' ensures table is replaced if it exists
        # - chunksize=5000 helps for large datasets to avoid memory issues
        print("[INFO] Ingesting data to MySQL...")
        df.to_sql('shoppers', con=eng_conn, if_exists='replace', index=False, chunksize=5000)
        print(f"[INFO] Data ingested successfully. Table 'shoppers' now has {len(df)} rows.")

        return True  # Success

    # -------------------------
    # 6. Exception Handling
    # -------------------------
    except FileNotFoundError as e:
        print(f"[ERROR] File error: {e}")
        return False

    except Exception as e:
        print(f"[ERROR] Database ingestion failed: {e}")
        print("Make sure:")
        print("1. MariaDB container is running (`docker ps`)")
        print("2. Database 'shoppers_db' exists")
        print("3. User 'sonu' has proper privileges")
        return False

# -------------------------
# Main Execution
# -------------------------
if __name__ == "__main__":
    success = ingest_data()
    if not success:
        sys.exit(1)  # Exit with error code if ingestion fails
