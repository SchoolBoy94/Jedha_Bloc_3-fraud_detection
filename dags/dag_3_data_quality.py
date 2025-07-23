from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import sys, pathlib

# ➜ ajoute /opt/airflow afin que "scripts" soit importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.data_quality import run_quality_checks

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": days_ago(1),
    "email_on_failure": False,
    "retries": 0,
}

with DAG(
    dag_id="dag_3",
    default_args=default_args,
    description="Contrôles Data‑Quality quotidiens sur la table t2",
    schedule_interval="@daily",      # change en None pour déclenchement manuel
    catchup=False,
    tags=["data-quality", "fraud"],
) as dag:

    dq_task = PythonOperator(
        task_id="run_data_quality_checks",
        python_callable=run_quality_checks,
    )
