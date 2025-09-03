from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import os
import joblib
import logging

log = logging.getLogger(__name__)

# -------------------------
# Default DAG args
# -------------------------
default_args = {
    'owner': 'sonu',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# -------------------------
# Helper functions
# -------------------------
def reconstruct_df(obj):
    if isinstance(obj, dict):
        return pd.DataFrame(**obj)
    return obj

def reconstruct_series(obj):
    if isinstance(obj, pd.Series):
        return obj
    elif isinstance(obj, list):
        return pd.Series(obj)
    elif isinstance(obj, dict):
        if 'index' in obj and 'data' in obj:
            return pd.Series(obj['data'], index=obj['index'], name=obj.get('name'))
        else:
            return pd.Series(obj)
    else:
        raise ValueError(f"Cannot reconstruct Series from type {type(obj)}")

# -------------------------
# DAG definition
# -------------------------
with DAG(
    dag_id='ml_pipeline_dag',
    default_args=default_args,
    description='End-to-end ML pipeline for online shoppers dataset',
    schedule_interval=None,
    start_date=datetime(2025, 8, 30),
    catchup=False,
    tags=['mlflow', 'catboost', 'pipeline'],
    doc_md="This DAG performs ingestion, validation, preprocessing, feature engineering, "
           "model training, hyperparameter tuning, and Evidently monitoring.",
) as dag:

    # -------------------------
    # 1️⃣ Ingest Data
    # -------------------------
    def ingest_task_callable(**kwargs):
        from pipelines.ingest_data import ingest_data
        ti = kwargs['ti']
        df = ingest_data()
        if df.empty:
            raise ValueError("Ingested DataFrame is empty")
        ti.xcom_push(key='ingested_df', value=df.to_dict('split'))
        return "Ingest successful"

    ingest_task = PythonOperator(
        task_id='ingest_data_task',
        python_callable=ingest_task_callable
    )

    # -------------------------
    # 2️⃣ Load Data
    # -------------------------
    def load_task_callable(**kwargs):
        from pipelines.load_data_from_db import load_data_from_db
        ti = kwargs['ti']
        df = load_data_from_db()
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")
        ti.xcom_push(key='loaded_df', value=df.to_dict('split'))
        return "Load successful"

    load_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_task_callable
    )

    # -------------------------
    # 3️⃣ Validate Data
    # -------------------------
    def validate_task_callable(**kwargs):
        from pipelines.validate_data import validate_shoppers_data
        ti = kwargs['ti']
        df_dict = ti.xcom_pull(key='loaded_df', task_ids='load_data_task')
        df = reconstruct_df(df_dict)
        validated_df = validate_shoppers_data(df)
        ti.xcom_push(
            key='validated_df',
            value=validated_df.to_dict('split') if isinstance(validated_df, pd.DataFrame) else validated_df
        )
        return "Validation done"

    validate_task = PythonOperator(
        task_id='validate_data_task',
        python_callable=validate_task_callable
    )

    # -------------------------
    # 4️⃣ Preprocess / Clean Data
    # -------------------------
    def preprocess_task_callable(**kwargs):
        from pipelines.preprocess_data import clean_data
        ti = kwargs['ti']
        df_dict = ti.xcom_pull(key='validated_df', task_ids='validate_data_task')
        df = reconstruct_df(df_dict)
        cleaned_df = clean_data(df)
        ti.xcom_push(
            key='cleaned_df',
            value=cleaned_df.to_dict('split') if isinstance(cleaned_df, pd.DataFrame) else cleaned_df
        )
        return "Preprocessing done"

    preprocess_task = PythonOperator(
        task_id='preprocess_task',
        python_callable=preprocess_task_callable
    )

    # -------------------------
    # 5️⃣ Feature Engineering
    # -------------------------
    def feature_engineering_task_callable(**kwargs):
        from pipelines.feature_engineering import feature_engineer_data
        ti = kwargs['ti']
        df_dict = ti.xcom_pull(key='cleaned_df', task_ids='preprocess_task')
        df = reconstruct_df(df_dict)
        data_dict = feature_engineer_data(df)
        X, y = data_dict['X'], data_dict['y']
        ti.xcom_push(key='X', value=X.to_dict('split') if isinstance(X, pd.DataFrame) else X)
        ti.xcom_push(
            key='y',
            value=y['values'] if isinstance(y, dict) else y.tolist() if isinstance(y, pd.Series) else y
        )
        return "Feature engineering done"

    feature_engineering_task = PythonOperator(
        task_id='feature_engineering_task',
        python_callable=feature_engineering_task_callable
    )

    # -------------------------
    # 6️⃣ Prepare Data (Train/Test Split)
    # -------------------------
    def prepare_data_task_callable(**kwargs):
        from pipelines.data_preparation import prepare_data
        ti = kwargs['ti']
        X_dict = ti.xcom_pull(key='X', task_ids='feature_engineering_task')
        y_dict = ti.xcom_pull(key='y', task_ids='feature_engineering_task')
        X = reconstruct_df(X_dict)
        y = reconstruct_series(y_dict)
        data_splits = prepare_data(X, y)
        for key, df_split in data_splits.items():
            ti.xcom_push(
                key=key,
                value=df_split.to_dict('split') if isinstance(df_split, pd.DataFrame) else df_split
            )
        return "Data prepared"

    prepare_data_task = PythonOperator(
        task_id='prepare_data_task',
        python_callable=prepare_data_task_callable
    )

    # -------------------------
    # 7️⃣ Hyperparameter Tuning
    # -------------------------
    def tune_model_task_callable(**kwargs):
        from pipelines.modeling import tune_catboost
        ti = kwargs['ti']
        X_train_dict = ti.xcom_pull(key='X_train', task_ids='prepare_data_task')
        y_train_dict = ti.xcom_pull(key='y_train', task_ids='prepare_data_task')
        X_train = reconstruct_df(X_train_dict)
        y_train = reconstruct_series(y_train_dict)
        best_params = tune_catboost(X_train, y_train)
        ti.xcom_push(key='best_params', value=best_params)
        return "Tuning done"

    hyperparameter_tuning_task = PythonOperator(
        task_id='hyperparameter_tuning_task',
        python_callable=tune_model_task_callable
    )

    # -------------------------
    # 8️⃣ Train Final Model
    # -------------------------
    def train_model_task_callable(**kwargs):
        from pipelines.train_models import train_catboost_model
        ti = kwargs['ti']

        hyperparams = ti.xcom_pull(key='best_params', task_ids='hyperparameter_tuning_task')
        X_train = reconstruct_df(ti.xcom_pull(key='X_train', task_ids='prepare_data_task'))
        y_train = reconstruct_series(ti.xcom_pull(key='y_train', task_ids='prepare_data_task'))
        X_test = reconstruct_df(ti.xcom_pull(key='X_test', task_ids='prepare_data_task'))
        y_test = reconstruct_series(ti.xcom_pull(key='y_test', task_ids='prepare_data_task'))

        model_dict = train_catboost_model(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hyperparams=hyperparams
        )

        ti.xcom_push(key='trained_model', value=model_dict)
        return "Model trained"

    train_model_task = PythonOperator(
        task_id='train_model_task',
        python_callable=train_model_task_callable
    )

    # -------------------------
    # 9️⃣ Evidently Monitoring
    # -------------------------
    def evidently_monitoring_task_callable(**kwargs):
        from pipelines.monitoring import generate_evidently_report
        ti = kwargs['ti']
        import joblib

        # Load trained model
        trained_model_dict = ti.xcom_pull(key='trained_model', task_ids='train_model_task')
        model_path = trained_model_dict['model_path']
        trained_model = joblib.load(model_path)

        # Reconstruct train/test data
        X_train = reconstruct_df(ti.xcom_pull(key='X_train', task_ids='prepare_data_task'))
        y_train = reconstruct_series(ti.xcom_pull(key='y_train', task_ids='prepare_data_task'))
        X_test = reconstruct_df(ti.xcom_pull(key='X_test', task_ids='prepare_data_task'))
        y_test = reconstruct_series(ti.xcom_pull(key='y_test', task_ids='prepare_data_task'))

        # Predict on test set
        y_pred = trained_model.predict(X_test)

        # Make sure `target` exists in current_df for Evidently
        current_df_with_target = X_test.copy()
        current_df_with_target['target'] = y_test

        # Generate reports (use the updated generate_evidently_report)
        report_paths = generate_evidently_report(
            reference_df=X_train.assign(target=y_train),  # train set with target
            current_df=current_df_with_target,            # test set with target
            y_pred=y_pred,                                # predictions
            target_col='target'                           # explicitly pass target column
        )

        ti.xcom_push(key='evidently_reports', value=report_paths)
        return "Monitoring done"



    evidently_monitoring_task = PythonOperator(
        task_id='evidently_monitoring_task',
        python_callable=evidently_monitoring_task_callable
    )

    # -------------------------
    # ✅ DAG Task Dependencies
    # -------------------------
    ingest_task >> load_task >> validate_task >> preprocess_task
    preprocess_task >> feature_engineering_task >> prepare_data_task
    prepare_data_task >> hyperparameter_tuning_task >> train_model_task
    train_model_task >> evidently_monitoring_task
