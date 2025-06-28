# scripts/utils/mlflow_utils.py
import os
import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient()

EXPERIMENT_NAME = "fraud-detection"
MODEL_NAME = "fraud_model"

def get_latest_production_model():
    """Charge le modèle currently en stage Production, ou lève RuntimeError."""
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(f"Aucun modèle en Production pour {MODEL_NAME}")
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")

def compare_and_promote_best_model(metric: str = "roc_auc"):
    """
    Cherche le dernier run de l'expérience, compare la métrique `metric`
    avec la version Production, promeut si meilleur.
    """
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Expérience {EXPERIMENT_NAME} introuvable")

    latest_run = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )[0]

    new_metric = latest_run.data.metrics.get(metric)
    if new_metric is None:
        raise ValueError(f"Métrique {metric} absente du run {latest_run.info.run_id}")

    # récupère la version Production (s'il y en a une)
    prod_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if prod_versions:
        prod_version = prod_versions[0]
        prod_metric = prod_version.metrics.get(metric, -1)
        if new_metric <= prod_metric:
            print(f"Pas de promotion : {metric} {new_metric:.4f} ≤ {prod_metric:.4f}")
            return

    # trouver la version du modèle liée à ce run
    model_versions = [
        mv for mv in client.search_model_versions(f"name='{MODEL_NAME}'")
        if mv.run_id == latest_run.info.run_id
    ]
    if not model_versions:
        # Si le modèle n'a pas encore été enregistré, on le crée
        mv = client.create_model_version(
            name=MODEL_NAME,
            source=latest_run.info.artifact_uri + "/model",
            run_id=latest_run.info.run_id,
        )
    else:
        mv = model_versions[0]

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"✅ Modèle v{mv.version} promu ({metric}: {new_metric:.4f})")
