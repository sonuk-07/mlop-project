# -------------------------
# monitoring.py
# -------------------------
import os
import logging
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.metric_preset.classification_performance import ClassificationPreset

log = logging.getLogger(__name__)

def generate_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_col: str,
    y_pred: pd.Series = None,
    task_type: str = "classification",
    report_dir: str = None
) -> dict:
    """
    Generates Evidently monitoring reports for feature drift and model performance.

    Returns dict with paths:
        - feature_drift_report
        - performance_report (if valid)
    """

    # -------------------------
    # 1️⃣ Prepare report directory
    # -------------------------
    if report_dir is None:
        report_dir = os.path.join(os.environ.get("AIRFLOW_HOME", "/opt/airflow"), "reports")
    os.makedirs(report_dir, exist_ok=True)
    log.info(f"Reports will be saved in: {report_dir}")

    # -------------------------
    # 2️⃣ Feature Drift Report
    # -------------------------
    log.info("Generating feature drift report (DataDriftPreset)...")
    reference_features = reference_df.drop(columns=[target_col], errors="ignore")
    current_features = current_df.drop(columns=[target_col], errors="ignore")

    feature_report = Report(metrics=[DataDriftPreset()])
    feature_report.run(reference_data=reference_features, current_data=current_features)

    feature_report_path = os.path.join(report_dir, "feature_drift_report.html")
    feature_report.save_html(feature_report_path)
    log.info(f"Feature drift report saved at {feature_report_path}")

    # -------------------------
    # 3️⃣ Model Performance Report
    # -------------------------
    perf_report_path = None
    if y_pred is not None and len(y_pred) > 0:
        log.info(f"Generating {task_type} performance report...")

        current_df_copy = current_df.copy()
        reference_df_copy = reference_df.copy()

        current_df_copy["target"] = current_df_copy[target_col]
        current_df_copy["prediction"] = y_pred

        # Classification safety check
        if task_type == "classification":
            labels = sorted(reference_df[target_col].dropna().unique())
            if len(labels) == 0 or current_df_copy["prediction"].nunique() == 0:
                log.warning("No valid labels in target or prediction. Skipping performance report.")
                perf_report_path = None
            else:
                # Set categories to avoid confusion matrix errors
                current_df_copy["target"] = current_df_copy["target"].astype("category").cat.set_categories(labels)
                current_df_copy["prediction"] = current_df_copy["prediction"].astype("category").cat.set_categories(labels)
                reference_df_copy["target"] = reference_df_copy[target_col]
                reference_df_copy["prediction"] = reference_df_copy[target_col]  # perfect reference

                perf_report = Report(metrics=[ClassificationPreset()])
                perf_report.run(reference_data=reference_df_copy, current_data=current_df_copy)

                perf_report_path = os.path.join(report_dir, "performance_report.html")
                perf_report.save_html(perf_report_path)
                log.info(f"Performance report saved at {perf_report_path}")

        elif task_type == "regression":
            reference_df_copy["target"] = reference_df_copy[target_col]
            reference_df_copy["prediction"] = reference_df_copy[target_col]
            perf_report = Report(metrics=[RegressionPreset()])
            perf_report.run(reference_data=reference_df_copy, current_data=current_df_copy)

            perf_report_path = os.path.join(report_dir, "performance_report.html")
            perf_report.save_html(perf_report_path)
            log.info(f"Regression performance report saved at {perf_report_path}")

    else:
        log.warning("y_pred is None or empty. Skipping performance report.")

    return {
        "feature_drift_report": os.path.relpath(feature_report_path, report_dir),
        "performance_report": os.path.relpath(perf_report_path, report_dir) if perf_report_path else None
    }