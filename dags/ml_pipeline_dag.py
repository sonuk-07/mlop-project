# dags/ml_pipeline_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from ingest_data import ingest_data  # import your function

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
    description='Ingest online shoppers data',
    schedule_interval=None,
    start_date=datetime(2025, 8, 30),
    catchup=False,
) as dag:

    ingest_task = PythonOperator(
        task_id='ingest_data_task',
        python_callable=ingest_data # call the function directly
    )
