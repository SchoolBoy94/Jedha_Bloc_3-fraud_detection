import psycopg2
import pandas as pd
from psycopg2.extras import execute_batch
from scripts.config import PG_CONN_INFO

def get_postgres_conn():
    return psycopg2.connect(**PG_CONN_INFO)

def df_to_postgres_batch(df: pd.DataFrame, table_name: str):
    conn = get_postgres_conn()
    cur = conn.cursor()
    cols = list(df.columns)
    values = df.values.tolist()
    placeholders = ','.join(['%s'] * len(cols))
    insert_query = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({placeholders}) ON CONFLICT DO NOTHING"
    execute_batch(cur, insert_query, values)
    conn.commit()
    cur.close()
    conn.close()

def transform_record(record: dict) -> dict:
    # Conversion de date_trans en ISO 8601
    ts = record.get('date_trans')
    if ts:
        try:
            record['date_trans'] = datetime.utcfromtimestamp(int(ts) / 1000).isoformat()
        except Exception:
            pass  # ignore si erreur de conversion

    return record
