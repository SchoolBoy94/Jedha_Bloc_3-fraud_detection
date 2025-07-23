
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from scripts.store_csv_to_t1 import store_csv_to_t1
from scripts.create_tables import create_table
from scripts.migrate_t1_to_t2 import migrate  

default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
}

with DAG(
    dag_id="dag_1_store_csv",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=["fraud"],
) as dag:

    create_t1 = PythonOperator(
        task_id="create_t1_table",
        python_callable=create_table,
    )

    load_csv = PythonOperator(
        task_id="load_csv_into_t1",
        python_callable=store_csv_to_t1,
    )


     # 3) migration / transformation t1 ➜ t2
    migrate_t1_t2 = PythonOperator(
        task_id="migrate_t1_t2",
        python_callable=migrate,
    )


    create_t1 >> load_csv >> migrate_t1_t2

