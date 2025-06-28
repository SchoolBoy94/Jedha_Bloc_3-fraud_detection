# Fraud Detection Pipeline

## Technologies utilisées
- Apache Airflow
- MLflow
- PostgreSQL
- scikit-learn
- Docker

## Déploiement local
```bash
docker compose up --build -d
```

## Sécurité
- Variables d’environnement via `.env`
- Utilisateur admin pour Airflow et MLflow (à sécuriser en prod)
- PostgreSQL isolé en réseau local

## Structure
- `dags/` : DAGs Airflow
- `scripts/` : Scripts ML
- `reports/` : CSV exportés
- `mlruns/` : artefacts MLflow
