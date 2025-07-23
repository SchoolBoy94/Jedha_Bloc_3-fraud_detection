import json
import pandas as pd
from kafka import KafkaConsumer

from scripts.etl_utils import df_to_postgres_batch

KAFKA_BROKER = "kafka:9092"
TOPIC_RAW = "api_raw_data"


def run_consume(run_id):
    consumer = KafkaConsumer(
        TOPIC_RAW,
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset='earliest',
        enable_auto_commit=False,
        group_id="table1_consumer",
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        consumer_timeout_ms=10000,
    )

    records = []
    for message in consumer:
        record = message.value
        if record.get('run_id') == run_id:
            records.append(record)
            if len(records) >= 100:
                break

    consumer.commit()
    consumer.close()

    if not records:
        return "Aucun message à insérer dans table1"

    df = pd.DataFrame(records)
    if 'run_id' in df.columns:
        df = df.drop(columns=['run_id'])
    df_to_postgres_batch(df, "t1")

    return f"{len(records)} messages insérés dans table1"
