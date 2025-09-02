import pandas as pd
from sqlalchemy import create_engine
import os
import logging

log = logging.getLogger(__name__)

def ingest_data(
    csv_path: str = '/opt/airflow/data/online_shoppers_intention.csv',
    db_url: str = None,
    table_name: str = 'shoppers',
    chunksize: int = 5000
) -> pd.DataFrame:
    """
    Ingest CSV data into MySQL/MariaDB database and return DataFrame.
    """
    if not os.path.exists(csv_path):
        log.error(f"Data file not found at {csv_path}")
        raise FileNotFoundError(f"{csv_path} does not exist")

    log.info(f"Loading CSV data from {csv_path}...")
    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df)} rows from CSV.")

    if not db_url:
        db_password = os.getenv("MYSQL_PASSWORD", "Yunachan10")
        db_url = f"mysql+pymysql://sonu:{db_password}@mariadb-columnstore:3306/shoppers_db"

    engine = create_engine(db_url)

    try:
        with engine.connect() as conn:
            log.info("Database connection successful.")
        df.to_sql(table_name, con=engine, if_exists='replace', index=False, chunksize=chunksize)
        log.info(f"Data ingested successfully. Table '{table_name}' now has {len(df)} rows.")
    except Exception as e:
        log.error(f"Database ingestion failed: {e}")
        raise

    return df  # ✅ return DataFrame for DAG
