# dags/daily_online_metrics.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import mlflow, mlflow.pyfunc, os, io

DB_URI = os.getenv("DB_URI")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

default_args = {"owner": "airflow", "retries": 1, "retry_delay": timedelta(minutes=5)}
dag = DAG(
    dag_id="daily_online_metrics",
    default_args=default_args,
    schedule_interval="0 7 * * *",     # chaque jour à 07h UTC
    start_date=datetime(2025, 6, 20),
    catchup=False,
)

def compute_daily_metrics(**_):
    engine = create_engine(DB_URI)
    df = pd.read_sql(
        """
        SELECT * FROM online_eval
        WHERE created_at::date = CURRENT_DATE - INTERVAL '1 day'
          AND y_true IN (0,1)
        """,
        engine,
    )
    if df.empty:
        return "No labelled data for yesterday"

    auc = roc_auc_score(df["y_true"], df["y_proba"])
    fpr, tpr, _ = roc_curve(df["y_true"], df["y_proba"])

    # # tracer ROC
    # plt.figure()
    # plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    # plt.plot([0, 1], [0, 1], "--", color="grey")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title("ROC – online transactions J-1")
    # plt.legend(loc="lower right")
    # buf = io.BytesIO()
    # plt.savefig(buf, format="png")
    # buf.seek(0)

    # # log dans MLflow avec run daté
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # mlflow.set_experiment("online_daily_metrics")
    # with mlflow.start_run(run_name="online_daily_metrics") as run:
    #     mlflow.set_tag("eval_type", "online_daily")
    #     mlflow.log_metric("roc_auc", auc)
    #     mlflow.log_metric("accuracy",  (df["y_true"] == (df["y_proba"] > 0.5)).mean())
    #     mlflow.log_figure(buf, "roc_curve.png")

    # buf.close()
    # plt.close()


    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC – online transactions J-1")
    ax.legend(loc="lower right")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("online_daily_metrics")
    with mlflow.start_run(run_name="online_daily_metrics") as run:
        mlflow.set_tag("eval_type", "online_daily")
        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("accuracy",  (df["y_true"] == (df["y_proba"] > 0.5)).mean())
        mlflow.log_figure(fig, "roc_curve.png")

    plt.close(fig)


PythonOperator(
    task_id="compute_daily_metrics",
    python_callable=compute_daily_metrics,
    dag=dag,
)
