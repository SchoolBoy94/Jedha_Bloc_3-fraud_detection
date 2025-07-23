from datetime import timedelta
from airflow.utils.dates import days_ago

KAFKA_BROKER = "kafka:9092"
TOPIC_RAW = "api_raw_data"
TOPIC_TRANSFORMED = "transformed_data"

PG_CONN_INFO = {
    'host': 'postgres',
    'database': 'fraud',
    'user': 'postgres',
    'password': 'postgres',
    'port': 5432
}

API_URL = "https://charlestng-real-time-fraud-detection.hf.space/current-transactions"
_MIN_INTERVAL = 12.1

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 0,
    'retry_delay': timedelta(minutes=1),
}
