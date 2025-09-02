import pandas as pd
from sqlalchemy import create_engine
import os
import logging
import redis
from typing import Optional

log = logging.getLogger(__name__)

def load_data_from_db(
    redis_host: str = 'redis',
    redis_port: int = 6379,
    db_url: Optional[str] = None,
    table_name: str = 'shoppers'
) -> Optional[pd.DataFrame]:
    """
    Load data from the 'shoppers' table in MariaDB into a Pandas DataFrame,
    using Redis as a caching layer. Returns DataFrame.
    """
    redis_key = f"{table_name}_data_loaded"
    r = None

    # 1️⃣ Try connecting to Redis
    try:
        r = redis.Redis(host=redis_host, port=redis_port, db=0)
        log.info("Connected to Redis successfully.")
        if r.exists(redis_key):
            log.info("Found cached data in Redis. Loading from cache...")
            cached_data = r.get(redis_key)
            df = pd.read_json(cached_data.decode('utf-8', errors='ignore'), orient='split')
            log.info(f"Loaded {len(df)} rows from Redis cache.")
            return df
    except redis.exceptions.ConnectionError as e:
        log.warning(f"Could not connect to Redis: {e}. Proceeding without caching.")
        r = None

    # 2️⃣ If Redis cache is empty or unavailable, load from database
    try:
        if not db_url:
            db_password = os.getenv("MYSQL_PASSWORD", "Yunachan10")
            db_url = f"mysql+pymysql://sonu:{db_password}@mariadb-columnstore:3306/shoppers_db"

        engine = create_engine(db_url)
        log.info(f"Connecting to database and reading '{table_name}' table...")
        df = pd.read_sql_table(table_name, con=engine)
        log.info(f"Loaded {len(df)} rows from database.")

        # 3️⃣ Cache data in Redis if available
        if r:
            r.set(redis_key, df.to_json(orient='split'))
            log.info("Data cached successfully in Redis.")

        return df

    except Exception as e:
        log.error(f"Failed to load data from database: {e}")
        log.error("Make sure:\n"
                  "1. MariaDB container is running (`docker ps`)\n"
                  f"2. Database and table '{table_name}' exist\n"
                  "3. User has proper privileges")
        return None
