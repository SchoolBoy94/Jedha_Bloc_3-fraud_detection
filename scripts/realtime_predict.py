
"""realtime_predict.py – scoring temps réel + archivage + suivi online
----------------------------------------------------------------------
1. Appel de l’API Jedha  /current-transactions  (1 transaction)
2. Conversion horodatage + archivage dans fraudArchive.csv
3. Préparation des features ➜ modèle Production (MLflow)
4. Insertion dans fraud_predictions  +  online_eval
5. Mise à jour des compteurs cumulés dans le run « online-eval »
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException, RestException
from sqlalchemy import create_engine
from dotenv import load_dotenv

# ─────────────────────────────── Config & constantes
load_dotenv()

DB_URI                = os.getenv("DB_URI")
MLFLOW_TRACKING_URI   = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MLFLOW_MODEL_NAME     = os.getenv("MLFLOW_MODEL_NAME", "fraud_model")

DATA_DIR              = Path("/opt/airflow/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_ARCHIVE_PATH     = DATA_DIR / "fraudArchive.csv"

API_URL              = "https://charlestng-real-time-fraud-detection.hf.space/current-transactions"

# quota 5 req / min  → intervalle mini 12 s
_MIN_INTERVAL         = 12.1
_LAST_CALL_TS         = 0.0

# ─────────────────────────────── Session HTTP robuste
SESSION = requests.Session()
SESSION.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=[429, 502, 503, 504],
            allowed_methods=["GET"],
        )
    ),
)

# ─────────────────────────────── API
def fetch_transaction(timeout: int = 30) -> pd.DataFrame:
    """Retourne un DataFrame (une seule ligne)."""
    global _LAST_CALL_TS
    delay = _MIN_INTERVAL - (time.time() - _LAST_CALL_TS)
    if delay > 0:
        time.sleep(delay)

    resp = SESSION.get(API_URL, timeout=timeout, headers={"accept": "application/json"})
    _LAST_CALL_TS = time.time()
    resp.raise_for_status()

    outer: str = resp.json()  # chaîne JSON orient="split"
    d: Dict[str, Any] = json.loads(outer)
    return pd.DataFrame(d["data"], columns=d["columns"], index=d["index"])


# ─────────────────────────────── Feature engineering
ALL_FEATURES: List[str] = [
    # numériques
    "amt", "city_pop", "unix_time", "lat", "long", "merch_lat", "merch_long",
    "hour", "day", "weekday", "month", "age", "lat_diff", "long_diff",
    # catégorielles
    "category", "state", "gender",
]

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # datetime dérivé d’unix_time déjà prêt
    df["trans_date_trans_time"] = pd.to_datetime(df["unix_time"], unit="s", utc=True)
    df["hour"]      = df["trans_date_trans_time"].dt.hour
    df["day"]       = df["trans_date_trans_time"].dt.day
    df["weekday"]   = df["trans_date_trans_time"].dt.weekday
    df["month"]     = df["trans_date_trans_time"].dt.month
    df["dob"]       = pd.to_datetime(df["dob"], errors="coerce")
    df["age"]       = df["trans_date_trans_time"].dt.year - df["dob"].dt.year
    df["lat_diff"]  = (df["lat"]  - df["merch_lat"]).abs()
    df["long_diff"] = (df["long"] - df["merch_long"]).abs()

    return df


# ─────────────────────────────── Modèle Production
def load_production_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(MLFLOW_MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(f"Aucune version Production pour {MLFLOW_MODEL_NAME}")
    return mlflow.sklearn.load_model(f"models:/{MLFLOW_MODEL_NAME}/Production")


# ─────────────────────────────── Pipeline principal
def score_and_store(*, return_rows: bool = False):
    # 1) extraction
    raw = fetch_transaction()

    # 2) conversion timestamp AVANT archivage
    ts = pd.to_numeric(raw["current_time"], errors="coerce").astype(float)
    ts[(ts >= 1e10)] = ts[(ts >= 1e10)] / 1_000
    ts[(ts < 1) & (ts >= 0)] *= 86_400
    ts = ts.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)

    raw["unix_time"]    = ts.round().astype("Int64")
    raw["current_time"] = pd.to_datetime(ts, unit="s", utc=True)\
                               .dt.strftime("%Y-%m-%d %H:%M:%S")

    # 3) archivage
    if DATA_ARCHIVE_PATH.exists() and list(raw.columns) == \
       list(pd.read_csv(DATA_ARCHIVE_PATH, nrows=1).columns):
        raw.to_csv(DATA_ARCHIVE_PATH, mode="a", header=False, index=False)
    else:
        raw.to_csv(DATA_ARCHIVE_PATH, mode="w", header=True, index=False)

    # 4) features + scoring
    feat = feature_engineering(raw)

    # ─── NOUVEAU : récupère la liste des colonnes attendues par le modèle
    model = load_production_model()
    preproc = model.named_steps["preproc"]  # ColumnTransformer
    model_features = list(preproc.feature_names_in_)


    # → Toutes les colonnes numériques sont déjà créées
    # → `category`, `state`, `gender` proviennent directement de l’API
    missing = [c for c in model_features if c not in feat.columns]
    if missing:
        raise KeyError(f"Colonnes manquantes dans l’API : {missing}")

    X    = feat[model_features]

    proba  = model.predict_proba(X)[0, 1]
    pred   = int(proba >= 0.5)

    # Notification Transaction frauduleuse
    if pred == 1:
        send_discord_alert(raw.iloc[0], proba)

    feat["prediction"] = pred
    feat["created_at"] = pd.Timestamp.utcnow()

    # 5) insertion DB
    engine = create_engine(DB_URI)
    y_true_val = (
        int(raw["is_fraud"].iloc[0]) if "is_fraud" in raw.columns else -1
    )
    df_eval = pd.DataFrame(
        {
            "created_at": [feat["created_at"].iloc[0]],
            "y_true":     [y_true_val],
            "y_proba":    [float(proba)],
        }
    )
    with engine.begin() as conn:
        feat.to_sql("fraud_predictions", con=conn, if_exists="append", index=False)
        df_eval.to_sql("online_eval",     con=conn, if_exists="append", index=False)

    # 6) métriques cumulées MLflow (run « online-eval »)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    runs = client.search_runs(
        experiment_ids=["0"],
        filter_string="tags.run_name = 'online-eval'",
        max_results=1,
    )
    if runs:
        mlflow.start_run(run_id=runs[0].info.run_id)
    else:
        mlflow.start_run(run_name="online-eval")
        mlflow.set_tag("eval_type", "online_cumulative")

    # récup compteurs
    # ─── récup compteurs existants pour le run online-eval ───────────────
    run_id = mlflow.active_run().info.run_id  # id du run ouvert

    def last(metric: str) -> int:
        hist = client.get_metric_history(run_id, metric)  # API ≥ 2.9
        return hist[-1].value if hist else 0

    tp, fp, fn, tn = map(last, ("tp", "fp", "fn", "tn"))
    # ────────────────────────────────────────────────────────────────────

    y_true, y_pred = y_true_val, pred

    # ⚠️  ignorer les transactions dont la vérité terrain est inconnue
    if y_true == -1:
        mlflow.end_run()
        if return_rows:
            return feat.to_dict(orient="records")
        return          # On s'arrête là, rien à loguer dans les métriques

    if y_true == 1 and y_pred == 1: tp += 1
    elif y_true == 1 and y_pred == 0: fn += 1
    elif y_true == 0 and y_pred == 1: fp += 1
    elif y_true == 0 and y_pred == 0: tn += 1

    n     = tp + fp + fn + tn
    acc   = (tp + tn) / n
    prec  = tp / (tp + fp) if tp + fp else 0
    rec   = tp / (tp + fn) if tp + fn else 0
    f1    = 2 * prec * rec / (prec + rec) if prec + rec else 0

    # log
    mlflow.log_metric("tp", tp, step=n)
    mlflow.log_metric("fp", fp, step=n)
    mlflow.log_metric("fn", fn, step=n)
    mlflow.log_metric("tn", tn, step=n)
    mlflow.log_metric("accuracy",  acc, step=n)
    mlflow.log_metric("precision", prec, step=n)
    mlflow.log_metric("recall",    rec, step=n)
    mlflow.log_metric("f1",        f1,  step=n)
    mlflow.end_run()

    if return_rows:
        return feat.to_dict(orient="records")


# ─────────────────────────────── Script CLI
if __name__ == "__main__":
    from pprint import pprint
    pprint(score_and_store(return_rows=True))
