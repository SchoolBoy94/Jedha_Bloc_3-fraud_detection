from datetime import timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import sys, pathlib

# ➜ ajoute /opt/airflow au PYTHONPATH pour que « scripts » soit trouvable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import extract     # scripts/extract.py
import consume     # scripts/consume.py
import transform   # scripts/transform.py

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="dag_2",
    default_args=default_args,
    schedule_interval=None,      # change en @hourly si besoin
    catchup=False,
    tags=["etl", "kafka", "fraud"],
) as dag:

    extract_task = PythonOperator(
        task_id="extract_data_to_kafka",
        python_callable=extract.run_extract,     # ← nom exact
        op_kwargs={"run_id": "{{ run_id }}"},
    )

    consume_task = PythonOperator(
        task_id="consume_raw_to_t1",
        python_callable=consume.run_consume,     # ← nom exact
        op_kwargs={"run_id": "{{ run_id }}"},
    )

    transform_task = PythonOperator(
        task_id="transform_to_t2",
        python_callable=transform.run_transform, # ← nom exact
        op_kwargs={"run_id": "{{ run_id }}"},
    )

    extract_task >> consume_task >> transform_task
