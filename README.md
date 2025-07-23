# Pipeline de Prévision et de Surveillance des Transactions frauduleuses

Prédire l’**Transactions frauduleuses**

| Couche             | Technologie                | Fonctionnalité principale                                                 |
| ------------------ | -------------------------- | ------------------------------------------------------------------------- |
| **Données & Jobs** | **Airflow 2.9**            | Orchestration quotidienne des pipelines                                   |
| **Stockage & Traîtement**   | **Kafka** | Stockage & Traîtement des données                                         |
| **Enregistrement** | **PostgreSQL**             | Base de donneés pour "Airflow" et "Fraud", stockés les fraudes. |

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
# PostgreSQL docker exec -it fraud_postgres psql -U postgres -d fraud

```

---

## 3 · Organisation du dépôt

```
│   .env
│   docker-compose.yml
│   requirements.txt
│
├───dags
│       dag_1.py
│       dag_2.py
│       dag_3_data_quality.py
│
├───data
├───init
│       init-multiple-db.sh
│
├───monitoring
│   └───prometheus.yml
├───reports
└───scripts
        config.py
        consume.py
        create_tables.py
        data_quality.py
        etl_utils.py
        extract.py
        migrate_t1_to_t2.py
        store_csv_to_t1.py
        transform.py
        __init__.py
```

---

## 4 · Variables d’environnement clés

Voir la liste complète dans `.env`.

---

## 5 · DAGs Airflow par ordre chronologique

| ID du DAG                     | Programmation              | Description                                                                          |
| ----------------------------- | -------------------------- | ------------------------------------------------------------------------------------ |
| `dag_1` | `@once` | extraction + stockage et transformation pour les données historiques                                                 |
| `dag_2`         | `@hourly` ou personnalisée | extraction + stockage et transformation pour les nouvelles données |
| `dag_3_data_quality`         | `@daily` ou personnalisée  | data_quality                                                              |


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
