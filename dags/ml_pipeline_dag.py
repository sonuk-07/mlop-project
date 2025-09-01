# dags/ml_pipeline_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pipelines.ingest_data import ingest_data
from pipelines.load_data_from_db import load_data_from_db
from pipelines.validate_data import validate_shoppers_data

default_args = {
    'owner': 'sonu',
    'depends_on_past': False,
    'email': ['skjais04@gmail.com'],
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ml_pipeline_dag',
    default_args=default_args,
    description='Ingest and validate online shoppers data',
    schedule_interval=None,
    start_date=datetime(2025, 8, 30),
    catchup=False,
) as dag:

    ingest_task = PythonOperator(
        task_id='ingest_data_task',
        python_callable=ingest_data
    )

    load_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data_from_db
    )

    validate_task = PythonOperator(
        task_id='validate_data_task',
        python_callable=lambda **kwargs: validate_shoppers_data(kwargs['ti'].xcom_pull(task_ids='load_data_task')),
        provide_context=True
    )

    ingest_task >> load_task >> validate_task
