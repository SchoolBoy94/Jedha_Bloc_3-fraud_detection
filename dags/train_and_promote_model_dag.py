# dags/train_and_promote_model_dag.py
"""
DAG hebdomadaire : entraînement d’un GradientBoostingClassifier puis promotion
s’il est meilleur que la version Production courante.
"""
from __future__ import annotations

import math                               # ⚠️  nouveau
from datetime import datetime, timedelta
import os
import pandas as pd
import numpy as np                        # ⚠️  nouveau
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from airflow import DAG
from airflow.operators.python import PythonOperator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# ───────────────────────────── config Airflow
default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="train_and_promote_model",
    description="Train GB model weekly and promote if better",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

# ───────────────────────────── variables globales
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "fraud_model"
DATA_PATH = "/opt/airflow/data/fraudTest.csv"  # volume ./data monté

# ───────────────────────────── helper
def promote_version(client, model_name, run_id, new_roc, prod_auc):
    """Passe la version `run_id` en Production et archive l’ancienne."""
    mv = next(
        v for v in client.search_model_versions(f"name='{model_name}'")
        if v.run_id == run_id
    )
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"✅ Modèle v{mv.version} promu (ROC-AUC {new_roc:.4f} > {prod_auc:.4f})")


# ───────────────────────────── tâche training
def train_gb_model(**context):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("fraud-detection")

    # ---------- 1. sampling équilibré sur les deux classes ----------
    df_full = pd.read_csv(DATA_PATH)

    fraud = df_full[df_full.is_fraud == 1]
    legit = df_full[df_full.is_fraud == 0]
    n = min(5_000, len(fraud), len(legit))  # 5 000 / classe max

    df = pd.concat([
        fraud.sample(n, random_state=42),
        legit.sample(n, random_state=42),
    ])

    # ---------- 2. feature engineering ----------
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["weekday"] = df["trans_date_trans_time"].dt.weekday
    df["month"] = df["trans_date_trans_time"].dt.month
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
    df["age"] = df["trans_date_trans_time"].dt.year - df["dob"].dt.year
    df["lat_diff"] = (df["lat"] - df["merch_lat"]).abs()
    df["long_diff"] = (df["long"] - df["merch_long"]).abs()

    num = [
        "amt", "city_pop", "unix_time", "lat", "long", "merch_lat", "merch_long",
        "hour", "day", "weekday", "month", "age", "lat_diff", "long_diff",
    ]
    cat = ["category", "state", "gender"]

    X = df[num + cat]
    y = df["is_fraud"]

    # ---------- 3. pipeline sklearn ----------
    num_pipe = Pipeline([("scaler", StandardScaler())])
    cat_pipe = Pipeline([("enc", OneHotEncoder(handle_unknown="ignore", drop="first"))])
    preproc = ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])

    clf = Pipeline([
        ("preproc", preproc),
        ("gb", GradientBoostingClassifier()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    clf.fit(X_train, y_train)

    # ---------- 4. métrique ----------
    try:
        roc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
    except ValueError:
        roc = float("nan")

    # ---------- 5. log MLflow ----------
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(
            clf,
            "model",
            registered_model_name=MODEL_NAME,
        )
        mlflow.log_metric("roc_auc", roc)

        # push XComs pour la tâche promote
        context["ti"].xcom_push(key="run_id", value=run.info.run_id)
        context["ti"].xcom_push(key="roc_auc", value=roc)


# ───────────────────────────── tâche promotion
def promote_if_better(**context):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    new_roc = float(context["ti"].xcom_pull(key="roc_auc", task_ids="train"))
    run_id  = context["ti"].xcom_pull(key="run_id",  task_ids="train")

    # ROC-AUC du modèle en Production
    try:
        current_prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
        prod_run = client.get_run(current_prod.run_id)
        prod_auc = prod_run.data.metrics.get("roc_auc", -1.0)
    except IndexError:
        prod_auc = -1.0  # aucun modèle en prod

    # -- promotion automatique s’il n’y a pas encore de prod --
    if prod_auc == -1.0 and not math.isnan(new_roc):
        promote_version(client, MODEL_NAME, run_id, new_roc, prod_auc)
        return

    # -- promotion si meilleur --
    if math.isnan(new_roc) or new_roc <= prod_auc:
        print(f"ℹ️  Nouveau ROC-AUC {new_roc:.4f} ≤ Production {prod_auc:.4f} → pas de promotion")
        return

    promote_version(client, MODEL_NAME, run_id, new_roc, prod_auc)


# ───────────────────────────── définition des tâches
train_task = PythonOperator(
    task_id="train",
    python_callable=train_gb_model,
    dag=dag,
)

promote_task = PythonOperator(
    task_id="promote",
    python_callable=promote_if_better,
    dag=dag,
)

train_task >> promote_task
