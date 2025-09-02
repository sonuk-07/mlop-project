import great_expectations as gx
import pandas as pd
import logging
from typing import Dict

log = logging.getLogger(__name__)

def validate_shoppers_data(df: pd.DataFrame) -> Dict:
    """
    Validate the online shoppers dataset using Great Expectations.

    Args:
        df (pd.DataFrame): DataFrame passed from the DAG.

    Returns:
        dict: Validated DataFrame serialized as dict (orient='split') for XCom.

    Raises:
        ValueError: If input DataFrame is empty.
        Exception: If validation fails.
    """
    if df is None or df.empty:
        log.error("[ERROR] Received empty DataFrame for validation")
        raise ValueError("[ERROR] Received empty DataFrame for validation")

    log.info("Starting data validation with Great Expectations.")

    # -------------------------------
    # Initialize GE context (in-memory)
    # -------------------------------
    context = gx.get_context(context_root_dir=None, runtime_environment={"config_variables": {}})

    context.add_datasource(
        name="default_pandas_datasource",
        class_name="Datasource",
        execution_engine={"class_name": "PandasExecutionEngine"},
        data_connectors={
            "default_runtime_data_connector_name": {
                "class_name": "RuntimeDataConnector",
                "batch_identifiers": ["default_identifier_name"]
            }
        }
    )

    # -------------------------------
    # Expectation Suite
    # -------------------------------
    suite_name = "online_shoppers_dynamic_suite"
    context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

    # -------------------------------
    # Runtime Batch Request
    # -------------------------------
    batch_request = gx.core.batch.RuntimeBatchRequest(
        datasource_name="default_pandas_datasource",
        data_connector_name="default_runtime_data_connector_name",
        data_asset_name="online_shoppers_data",
        runtime_parameters={"batch_data": df},
        batch_identifiers={"default_identifier_name": "default_identifier"}
    )

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name
    )

    # -------------------------------
    # Numeric columns expectations
    # -------------------------------
    numeric_columns = [
        "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"
    ]
    for col in numeric_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        validator.expect_column_values_to_be_between(column=col, min_value=min_val, max_value=max_val)

    # -------------------------------
    # Categorical columns expectations
    # -------------------------------
    categorical_columns = ["Month","OperatingSystems","Browser","Region","TrafficType","VisitorType","Weekend","Revenue"]
    for col in categorical_columns:
        unique_vals = df[col].dropna().unique().tolist()
        validator.expect_column_values_to_be_in_set(column=col, value_set=unique_vals)

    # -------------------------------
    # Save suite & validate
    # -------------------------------
    validator.save_expectation_suite(discard_failed_expectations=False)
    validation_result = validator.validate()

    if validation_result.success:
        log.info("✅ Data validation passed successfully!")
        return df.to_dict(orient='split')  # XCom-friendly
    else:
        log.error("❌ Data validation failed. Check Airflow logs for detailed results.")
        for idx, result in enumerate(validation_result.results, 1):
            if not result.success:
                expectation_type = result.expectation_config.expectation_type
                kwargs = result.expectation_config.kwargs
                log.error(f"Expectation {idx}: {expectation_type} failed. ➤ Kwargs: {kwargs}")
        raise Exception("Data validation failed.")
