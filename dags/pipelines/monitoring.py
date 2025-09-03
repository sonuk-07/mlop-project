import pandas as pd
import logging
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, ClassificationPerformanceProfileSection
from evidently import ColumnMapping
import os

log = logging.getLogger(__name__)

def generate_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_col: str,
    y_pred: pd.Series = None,
    report_dir: str = "/mlflow/reports"
) -> dict:
    """
    Generates:
    1. Feature drift report
    2. Concept drift / model performance report
    """

    os.makedirs(report_dir, exist_ok=True)

    # -------------------------
    # 1️⃣ Feature Drift Profile
    # -------------------------
    log.info("Generating feature drift profile...")
    profile_feature = Profile(sections=[DataDriftProfileSection()])
    column_mapping_feature = ColumnMapping()
    column_mapping_feature.target = target_col

    # Use the new Evidently API
    profile_feature.run(reference_data=reference_df, current_data=current_df, column_mapping=column_mapping_feature)

    feature_report_path = os.path.join(report_dir, "feature_drift_report.html")
    profile_feature.save_html(feature_report_path)
    log.info(f"Feature drift report saved at {feature_report_path}")

    # -------------------------
    # 2️⃣ Concept Drift / Model Performance Profile
    # -------------------------
    concept_report_path = None
    if y_pred is not None:
        log.info("Generating concept drift / model performance profile...")
        profile_concept = Profile(sections=[ClassificationPerformanceProfileSection(), DataDriftProfileSection()])

        current_df_copy = current_df.copy()
        current_df_copy['prediction'] = y_pred

        # Use the new Evidently API
        profile_concept.run(reference_data=reference_df, current_data=current_df_copy, column_mapping=column_mapping_feature)

        concept_report_path = os.path.join(report_dir, "concept_drift_report.html")
        profile_concept.save_html(concept_report_path)
        log.info(f"Concept drift report saved at {concept_report_path}")

    # -------------------------
    # Return relative paths
    # -------------------------
    relative_paths = {
        "feature_drift_report": feature_report_path.replace("/mlflow/", ""),
        "concept_drift_report": concept_report_path.replace("/mlflow/", "") if concept_report_path else None
    }

    return relative_paths
