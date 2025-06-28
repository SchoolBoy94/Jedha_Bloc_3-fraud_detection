import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../scripts"))

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from realtime_predict import score_and_store

def extract_predictions():
    score_and_store(return_rows=False)

with DAG(
    dag_id="fraud_detection_dag",
    schedule_interval="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={
        "owner": "airflow",
        "retries": 1,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["fraud"]
) as dag:
    run_predictions = PythonOperator(
        task_id="score_and_store_predictions",
        python_callable=extract_predictions
    )
