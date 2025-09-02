import pandas as pd
import logging
from evidently.dashboard.dashboard import Dashboard
from evidently.dashboard.tabs import ClassificationPerformanceTab, DataDriftTab
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, ClassificationPerformanceProfileSection
import os

log = logging.getLogger(__name__)

def generate_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    y_true: pd.Series = None,
    y_pred: pd.Series = None,
    report_dir: str = "/mlflow/reports"  # <-- MLflow shared folder
) -> dict:
    """
    Generates Evidently reports for data drift and model performance.
    Returns paths to HTML reports relative to /mlflow (XCom-friendly).
    """
    os.makedirs(report_dir, exist_ok=True)

    # -------------------------
    # 1️⃣ Data Drift Dashboard
    # -------------------------
    log.info("Generating data drift dashboard...")
    drift_dashboard = Dashboard(tabs=[DataDriftTab()])
    drift_dashboard.calculate(reference_df=reference_df, current_df=current_df)
    drift_dashboard_path = os.path.join(report_dir, "data_drift_report.html")
    drift_dashboard.save(drift_dashboard_path)
    log.info(f"Data drift report saved at {drift_dashboard_path}")

    # -------------------------
    # 2️⃣ Model Performance Profile (optional)
    # -------------------------
    profile_path = None
    if y_true is not None and y_pred is not None:
        log.info("Generating model performance profile...")
        profile = Profile(sections=[ClassificationPerformanceProfileSection(), DataDriftProfileSection()])
        current_df_copy = current_df.copy()
        current_df_copy['target'] = y_true
        current_df_copy['prediction'] = y_pred
        profile.calculate(reference_df=reference_df, current_df=current_df_copy)
        profile_path = os.path.join(report_dir, "model_performance_report.html")
        profile.save_html(profile_path)
        log.info(f"Model performance report saved at {profile_path}")

    # Return paths **relative to /mlflow**, suitable for MLflow logging
    relative_paths = {
        "data_drift_report": drift_dashboard_path.replace("/mlflow/", ""),
        "model_performance_report": profile_path.replace("/mlflow/", "") if profile_path else None
    }

    return relative_paths
