import pandas as pd
import logging
from typing import Dict

log = logging.getLogger(__name__)

def feature_engineer_data(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Performs feature engineering and separates features (X) and target (y).

    Args:
        df (pd.DataFrame): The cleaned DataFrame from the previous task.

    Returns:
        dict: {'X': X_dict, 'y': y_dict} where X and y are serialized for Airflow XCom.
    """
    if df is None or df.empty:
        log.error("Input DataFrame is empty or None. Cannot perform feature engineering.")
        raise ValueError("Input DataFrame is empty or None.")

    log.info("Starting feature engineering.")
    
    df_fe = df.copy()

    # ----- Create New Features -----
    if set(['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration']).issubset(df_fe.columns):
        df_fe['total_duration'] = (
            df_fe['Administrative_Duration'] +
            df_fe['Informational_Duration'] +
            df_fe['ProductRelated_Duration']
        )
        log.info("Created 'total_duration' feature.")

    if set(['Administrative', 'Informational', 'ProductRelated']).issubset(df_fe.columns):
        df_fe['total_pages_visited'] = (
            df_fe['Administrative'] +
            df_fe['Informational'] +
            df_fe['ProductRelated']
        )
        log.info("Created 'total_pages_visited' feature.")

    # ----- Check target column -----
    if "Revenue" not in df_fe.columns:
        log.error("Target column 'Revenue' not found in dataframe.")
        raise ValueError("Target column 'Revenue' not found in dataframe.")

    # ----- Split Features & Target -----
    X = df_fe.drop('Revenue', axis=1)
    y = df_fe['Revenue']

    # ----- One-hot encode remaining categoricals -----
    X = pd.get_dummies(X, drop_first=True)
    log.info("One-hot encoded remaining categorical columns.")

    log.info(f"Features (X) shape: {X.shape}, Target (y) shape: {y.shape}")
    log.info("Feature engineering complete.")

    # ----- XCom-safe serialization -----
    # For X (DataFrame): use orient='split'
    X_dict = X.to_dict(orient='split')

    # For y (Series): convert to list
    y_dict = {'values': y.tolist()}

    return {'X': X_dict, 'y': y_dict}
