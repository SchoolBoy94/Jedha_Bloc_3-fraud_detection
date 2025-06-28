import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from pathlib import Path
from datetime import datetime, timedelta, timezone
import pandas as pd
from sqlalchemy import create_engine
#import psycopg2

DB_URI = os.getenv("DB_URI")
REPORT_DIR = Path("/opt/airflow/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

query = """
        SELECT *
        FROM fraud_predictions
        WHERE prediction = 1
          AND trans_date_trans_time::date = CURRENT_DATE - INTERVAL '1 day';
    """

# def export_daily_report():
#     conn = psycopg2.connect(dbname="fraud", user="postgres", password="postgres", host="postgres")
#     query = """
#         SELECT *
#         FROM fraud_predictions
#         WHERE prediction = 1
#           AND trans_date_trans_time::date = CURRENT_DATE - INTERVAL '1 day';
#     """
#
#     df = pd.read_sql(query, conn)
#     filename = f"/opt/airflow/reports/fraud_report_{datetime.now().strftime('%Y-%m-%d')}.csv"
#     df.to_csv(filename, index=False)
#     conn.close()


def export_daily_report(**_) -> None:
    """     Exporte les fraudes (prediction = 1) observés hier (UTC) vers un CSV    """
    # 1) Connexion
    engine = create_engine(DB_URI)
    df = pd.read_sql(query, engine)
    if df.empty:
        engine.dispose()
        return

    # 2) Ecriture du rapport
    report_date = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    outfile = REPORT_DIR / f"fraud_report_{report_date}.csv"
    df.to_csv(outfile, index = False)
    engine.dispose()









with DAG(
    dag_id="daily_reporting_dag",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["report"]
) as dag:
    export_task = PythonOperator(
        task_id="export_fraud_report",
        python_callable=export_daily_report
    )
