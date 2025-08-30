# validate_data.py
import great_expectations as gx

def validate_shoppers_data(df):
    """
    Validate the online shoppers dataset using Great Expectations.
    df: Pandas DataFrame passed from the DAG.
    """
    if df is None or df.empty:
        raise ValueError("[ERROR] Received empty DataFrame for validation")

    # -------------------------------
    # Initialize GE
    # -------------------------------
    context = gx.get_context()

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
    # Create expectation suite
    # -------------------------------
    suite_name = "online_shoppers_dynamic_suite"
    context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

    # -------------------------------
    # RuntimeBatchRequest
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
    # Add expectations
    # -------------------------------
    numeric_columns = [
        "Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
        "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"
    ]
    for col in numeric_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        validator.expect_column_values_to_be_between(column=col, min_value=min_val, max_value=max_val)

    categorical_columns = ["Month","OperatingSystems","Browser","Region","TrafficType","VisitorType","Weekend","Revenue"]
    for col in categorical_columns:
        unique_vals = df[col].dropna().unique().tolist()
        validator.expect_column_values_to_be_in_set(column=col, value_set=unique_vals)

    # -------------------------------
    # Save suite & validate
    # -------------------------------
    validator.save_expectation_suite(discard_failed_expectations=False)
    validation_result = validator.validate()

    # -------------------------------
    # Print detailed results
    # -------------------------------
    print(f"Validation success: {validation_result.success}")
    print(f"Statistics: {validation_result.statistics}\n")

    for idx, result in enumerate(validation_result.results, 1):
        expectation_type = result.expectation_config.expectation_type
        kwargs = result.expectation_config.kwargs
        success = result.success

        print(f"🔹 Expectation {idx}: {expectation_type}")
        print(f"   ➤ Kwargs: {kwargs}")
        print(f"   ✅ Success: {success}")

        if not success:
            if "unexpected_list" in result.result:
                print(f"   ⚠️ Unexpected values: {result.result['unexpected_list']}")
            elif "unexpected_percent" in result.result:
                print(f"   ⚠️ Unexpected %: {result.result['unexpected_percent']:.2f}%")
            elif "unexpected_index_list" in result.result:
                print(f"   ⚠️ Failed rows index: {result.result['unexpected_index_list']}")
        print("-" * 80)
