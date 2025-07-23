import json
from kafka import KafkaConsumer
import pandas as pd
from scripts.etl_utils import df_to_postgres_batch

TOPIC_RAW = "api_raw_data"
KAFKA_BROKER = "kafka:9092"
TARGET_TABLE = "t2"

def run_transform(run_id: str = None):

    print(f"⏳ Consommation depuis le topic : {TOPIC_RAW}")
    consumer = KafkaConsumer(
        TOPIC_RAW,
        bootstrap_servers=KAFKA_BROKER,
        auto_offset_reset='earliest',
        enable_auto_commit=False,
        group_id="transform_t2_consumer",
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        consumer_timeout_ms=10000,
    )

    records = []
    for message in consumer:
        record = message.value

        if run_id and record.get("run_id") != run_id:
            continue  # ignore les anciens messages

        if "current_time" in record:
            record["date_trans"] = record.pop("current_time")

        records.append(record)
        print(f"✅ Message préparé pour insertion : {record}")

        if len(records) >= 100:
            break

    consumer.commit()
    consumer.close()

    if records:
        df = pd.DataFrame(records)

        # Nettoyage des colonnes inutiles
        cols_to_drop = ["run_id", "first", "last", "street", "city", "state", "job", "trans_num", "cc_num"]
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True, errors='ignore')

        # Conversion en datetime
        df["date_trans"] = pd.to_datetime(df["date_trans"], unit='ms', errors='coerce')

        df_to_postgres_batch(df, TARGET_TABLE)
        print(f"✅ {len(df)} lignes insérées dans la table {TARGET_TABLE}")
    else:
        print("⚠️ Aucun message traité.")

if __name__ == "__main__":
    import sys
    rid = sys.argv[1] if len(sys.argv) > 1 else None
    
