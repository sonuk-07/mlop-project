import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
import logging

log = logging.getLogger(__name__)

def prepare_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float = 0.3, 
    random_state: int = 42, 
    resample: bool = True
) -> dict:
    """
    Split dataset into train/test sets and optionally apply SMOTE resampling.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        test_size (float): Proportion of test data.
        random_state (int): Random seed for reproducibility.
        resample (bool): Whether to apply SMOTE on training data.

    Returns:
        dict: {'X_train': dict, 'X_test': dict, 'y_train': dict, 'y_test': dict} XCom-safe

    Raises:
        ValueError: If input features or target are empty.
    """
    if X is None or X.empty:
        log.error("Input features DataFrame is empty or None.")
        raise ValueError("Input features DataFrame is empty or None.")
    if y is None or y.empty:
        log.error("Target series is empty or None.")
        raise ValueError("Target series is empty or None.")

    log.info("Starting data preparation (train/test split and resampling).")

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    log.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    log.info(f"Original training class distribution: {Counter(y_train)}")

    # Apply SMOTE resampling if required
    if resample:
        log.info("Applying SMOTE for resampling.")
        try:
            sm = SMOTE(random_state=random_state)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            log.info(f"Resampled training set shape: {X_train.shape}")
            log.info(f"Resampled training class distribution: {Counter(y_train)}")
        except Exception as e:
            log.error(f"SMOTE resampling failed: {e}")
            raise

    log.info("Data preparation complete.")

    # Convert to XCom-safe dictionaries
    return {
        'X_train': X_train.to_dict(orient='split'),
        'X_test': X_test.to_dict(orient='split'),
        'y_train': y_train.to_dict(),
        'y_test': y_test.to_dict()
    }
