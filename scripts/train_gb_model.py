# scripts/train_gb_model.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
# (facultatif) import mlflow

# ---------------------------------------------------------------------
# 1. Charger et échantillonner
# ---------------------------------------------------------------------
DATA_PATH = Path("data/fraudTest.csv")       # adapte si besoin
df = pd.read_csv(DATA_PATH)

# Échantillon aléatoire (20 000 lignes)
df = df.sample(n=1_000, random_state=42).reset_index(drop=True)

# ---------------------------------------------------------------------
# 2. Ingénierie de variables
# ---------------------------------------------------------------------
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
df["hour"]     = df["trans_date_trans_time"].dt.hour
df["day"]      = df["trans_date_trans_time"].dt.day
df["weekday"]  = df["trans_date_trans_time"].dt.weekday
df["month"]    = df["trans_date_trans_time"].dt.month

df["dob"] = pd.to_datetime(df["dob"], errors="coerce")
df["age"] = df["trans_date_trans_time"].dt.year - df["dob"].dt.year

df["lat_diff"]  = (df["lat"]  - df["merch_lat"]).abs()
df["long_diff"] = (df["long"] - df["merch_long"]).abs()

# ---------------------------------------------------------------------
# 3. Features / target
# ---------------------------------------------------------------------
num_features = [
    "amt", "city_pop", "unix_time", "lat", "long",
    "merch_lat", "merch_long",
    "hour", "day", "weekday", "month",
    "age", "lat_diff", "long_diff",
]
cat_features = ["category", "state", "gender"]
target = "is_fraud"

X = df[num_features + cat_features]
y = df[target]

# ---------------------------------------------------------------------
# 4. Pipeline pré-processing + modèle
# ---------------------------------------------------------------------
numeric_trf = Pipeline(steps=[("scaler", StandardScaler())])
categorical_trf = Pipeline(
    steps=[("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_trf, num_features),
        ("cat", categorical_trf, cat_features),
    ]
)

model = GradientBoostingClassifier()

clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier",   model),
])

# ---------------------------------------------------------------------
# 5. Split, entraînement, évaluation
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

clf.fit(X_train, y_train)

y_pred  = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, digits=4))
print("ROC-AUC :", roc_auc_score(y_test, y_proba).round(4))

# ---------------------------------------------------------------------
# 6. (Facultatif) log dans MLflow
# ---------------------------------------------------------------------
# import mlflow.sklearn
# mlflow.set_tracking_uri("http://mlflow:5000")
# mlflow.set_experiment("fraud-detection")
# with mlflow.start_run():
#     mlflow.sklearn.log_model(clf, "model")
#     mlflow.log_param("estimator", "GradientBoostingClassifier")
#     mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))
#     mlflow.log_metric("accuracy", clf.score(X_test, y_test))
