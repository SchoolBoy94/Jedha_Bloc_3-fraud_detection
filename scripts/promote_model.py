from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "fraud_model"
latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version

# Archive old Production versions
for m in client.get_latest_versions(model_name, stages=["Production"]):
    client.transition_model_version_stage(model_name, m.version, "Archived")

# Promote the latest model to Production
client.transition_model_version_stage(model_name, latest_version, "Production")
