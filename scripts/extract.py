import time, json
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from kafka import KafkaProducer



API_URL = "https://charlestng-real-time-fraud-detection.hf.space/current-transactions"
_MIN_INTERVAL = 12.1

KAFKA_BROKER = "kafka:9092"
TOPIC_RAW = "api_raw_data"

SESSION = requests.Session()
SESSION.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 502, 503, 504],
        allowed_methods=["GET"],
    ))
)

_LAST_CALL_TS = 0.0

def fetch_transaction(timeout: int = 30) -> pd.DataFrame:
    global _LAST_CALL_TS
    delay = _MIN_INTERVAL - (time.time() - _LAST_CALL_TS)
    if delay > 0:
        time.sleep(delay)

    resp = SESSION.get(API_URL, timeout=timeout, headers={"accept": "application/json"})
    _LAST_CALL_TS = time.time()
    d = json.loads(resp.json())
    return pd.DataFrame(d["data"], columns=d["columns"], index=d["index"])

def run_extract(run_id):
    df = fetch_transaction()
    if "current_time" in df.columns:
        df.rename(columns={"current_time": "date_trans"}, inplace=True)

    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    for _, row in df.iterrows():
        record = row.to_dict()
        record['run_id'] = run_id
        producer.send(TOPIC_RAW, value=record)

    producer.flush()
    producer.close()

    return f"Envoyé {len(df)} messages à Kafka"
