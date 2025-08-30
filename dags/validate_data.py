# Import and env setup
import great_expectations as gx
import pandas as pd

print(gx.__version__)  # Should print 1.4.4

# Load Titanic dataset
titanic_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(titanic_url)

# Initialize Data Context
context = gx.get_context()

# Add a Pandas datasource to the context
context.add_datasource(
    name="default_pandas_datasource",
    class_name="Datasource",
    execution_engine={
        "class_name": "PandasExecutionEngine"
    },
    data_connectors={
        "default_runtime_data_connector_name": {
            "class_name": "RuntimeDataConnector",
            "batch_identifiers": ["default_identifier_name"]
        }
    }
)

# Print context type for debugging
print(f"Context type: {type(context).__name__}")

# Create Expectation Suite
suite_name = "titanic_expectation_suite"
context.add_or_update_expectation_suite(expectation_suite_name=suite_name)

# Create a RuntimeBatchRequest
batch_request = gx.core.batch.RuntimeBatchRequest(
    datasource_name="default_pandas_datasource",
    data_connector_name="default_runtime_data_connector_name",
    data_asset_name="titanic_data",  # This can be any string
    runtime_parameters={"batch_data": df},
    batch_identifiers={"default_identifier_name": "default_identifier"}
)

# Get validator using the RuntimeBatchRequest
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name=suite_name
)

# Add expectations
validator.expect_column_values_to_not_be_null(column="sex")
validator.expect_column_distinct_values_to_be_in_set(
    column="sex",
    value_set=["male", "female"]
)
validator.expect_column_values_to_be_in_set(
    column="pclass",
    value_set=[1, 2, 3]
)
validator.expect_column_values_to_be_between(
    column="age",
    min_value=0,
    max_value=100
)
validator.expect_column_values_to_be_between(
    column="fare",
    min_value=0,
)

validator.expect_column_distinct_values_to_be_in_set(
    column="embarked",
    value_set=['C', 'Q', 'S'],
)

validator.expect_column_values_to_be_between(
    column="sibsp",
    min_value=0,
    max_value=10,
)

validator.expect_column_values_to_be_between(
    column="age",
    min_value=20,
    max_value=40,
)

validator.expect_column_values_to_be_between(
    column="parch",
    min_value=0,
    max_value=9,
)

validator.expect_column_values_to_be_in_set(
    column="fare",
    value_set=[0, 100],
)


validator.expect_column_values_to_be_in_set(
    column="survived",
    value_set=[0, 1]
)
validator.save_expectation_suite(discard_failed_expectations=False)

# Run validation directly on the validator
validation_result = validator.validate()

# Print validation results
print("Validation success:", validation_result.success)
print("Validation statistics:", validation_result.statistics)

print("\n📋 Detailed Expectation Results:\n")
for idx, result in enumerate(validation_result.results, 1):
    expectation_type = result.expectation_config.expectation_type
    kwargs = result.expectation_config.kwargs
    success = result.success

    print(f"🔹 Expectation {idx}: {expectation_type}")
    print(f"   ➤ Kwargs: {kwargs}")
    print(f"   ✅ Success: {success}")

    # Optionally show result details like unexpected values
    if not success:
        unexpected = result.result.get("unexpected_list", None)
        if unexpected:
            print(f"   ⚠️ Unexpected values: {unexpected}")
        elif "unexpected_percent" in result.result:
            print(f"   ⚠️ Unexpected %: {result.result['unexpected_percent']:.2f}%")
        elif "element_count" in result.result:
            print(f"   ⚠️ Details: {result.result}")
    print("-" * 80)