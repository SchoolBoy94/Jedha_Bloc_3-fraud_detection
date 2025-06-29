# Pipeline de Prévision et de Surveillance des Transactions frauduleuses

Prédire l’**Transactions frauduleuses**

| Couche             | Technologie                | Fonctionnalité principale                                                 |
| ------------------ | -------------------------- | ------------------------------------------------------------------------- |
| **Données & Jobs** | **Airflow 2.9**            | Orchestration quotidienne des pipelines                                   |
| **Modélisation**   | **LightGBM + MLflow 2.11** | Entraîne et enregistre le modèle.                                         |
| **Enregistrement** | **PostgreSQL**             | Base de donneés pour "Airflow", "MLFlow" et "Fraud", stockés les fraudes. |

---

## 1 · Sources de données

- **FraudTEST.csv** : [https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv](https://lead-program-assets.s3.eu-west-3.amazonaws.com/M05-Projects/fraudTest.csv)
- **Fraud Detection API** : [https://huggingface.co/spaces/charlestng/real-time-fraud-detection](https://huggingface.co/spaces/charlestng/real-time-fraud-detection)

---

## 2 · Démarrage rapide

```bash
# Construire et lancer l’ensemble des services
docker compose up -d --build

# Accéder aux interfaces
# Airflow (admin / admin) : http://localhost:8080
# MLflow : http://localhost:5000

```

---

## 3 · Organisation du dépôt

```
project/
├── dags/
│   ├── train_and_promote_model_dag.py   # Entraînement hebdomadaire + promotion MLflow
│   ├── fraud_detection_dag.py          # Scoring temps réel toutes les 5 min
│   ├── daily_reporting_dag.py          # Export CSV des fraudes de J-1
│   └── daily_online_metrics.py         # Calcul quotidien des métriques en ligne
├── scripts/
│   ├── train_gb_model.py               # Entraînement GradientBoosting standalone
│   ├── realtime_predict.py             # Scoring temps réel + archivage + alertes
│   └── utils/
│       └── mlflow_utils.py             # Helpers MLflow (chargement, promotion)
├── data/                               # CSV montés (fraudTest.csv, archives)
├── mlruns/                             # Artifacts & registre MLflow
├── reports/                            # Sortie des rapports CSV Airflow
├── docker-compose.yml                  # Orchestration Docker (Postgres, Airflow, MLflow)
├── Dockerfile.mlflow                   # Image MLflow avec psycopg2
├── requirements.txt                    # Dépendances Python
└── .env                                # Variables d’environnement
```

---

## 4 · Variables d’environnement clés

Voir la liste complète dans `.env`.

---

## 5 · DAGs Airflow par ordre chronologique

| ID du DAG                     | Programmation              | Description                                                                          |
| ----------------------------- | -------------------------- | ------------------------------------------------------------------------------------ |
| `train_and_promote_model_dag` | `@weekly` ou personnalisée | `train_gb_model → promote_if_better`                                                 |
| `fraud_detection_dag`         | `@hourly` ou personnalisée | Détection des fraudes à partir d'API, enregistrement dans fraudArchive et postgreSQL |
| `daily_reporting_dag`         | `@daily` ou personnalisée  | Génération des rapports                                                              |
| `daily_online_metrics`        | `@daily` ou personnalisée  | Génération des métriques                                                             |

### Enregistrement

1. Charger le dernier `fraudArchive.csv`
2. Dans la base de données "Fraud" dans PostGreSQL

## 9 · Feuille de route

- À venir : ajout d’une interface graphique pour la visualisation temps réel.

---

### Licence

MIT — utilisation, modification et partage libres. Contributions bienvenues !

![Python](https://img.shields.io/badge/lang-Python-blue)
![YAML](https://img.shields.io/badge/lang–YAML-blueviolet)
