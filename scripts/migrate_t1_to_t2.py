
"""
Copie / transforme les données de t1 → t2.

Usage manuel :
    docker exec -it airflow_webserver python /opt/airflow/scripts/migrate_t1_to_t2.py
– ou directement via un DAG Airflow.
"""
import psycopg2
from psycopg2.extras import execute_batch
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

PG_CONN_INFO = {
    "host": "postgres",
    "database": "fraud",
    "user": "postgres",
    "password": "postgres",
    "port": 5432,
}

SRC = "t1"
DST = "t2"

# Colonnes à copier et leur ordre cible
COLS_DST = [
    "merchant", "category", "amt", "gender", "zip",
    "lat", "long", "city_pop", "dob",
    "merch_lat", "merch_long", "date_trans", "is_fraud",
]

def fetch_t1_rows():
    """Récupère toutes les lignes de t1 ; retourne une liste de tuples dans l’ordre COLS_DST."""
    sel_cols = ", ".join([
        "merchant", "category", "amt", "gender", "zip",
        "lat", "long", "city_pop", "dob",
        "merch_lat", "merch_long", "date_trans", "is_fraud"
    ])
    sql = f"SELECT {sel_cols} FROM {SRC};"

    with psycopg2.connect(**PG_CONN_INFO) as conn, conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    transformed = []
    for row in rows:
        *first_parts, date_ms, is_fraud = row
        # Conversion BIGINT (ms) → datetime ISO
        if isinstance(date_ms, (int, float)):
            date_ts = datetime.utcfromtimestamp(date_ms / 1000.0)
        else:
            date_ts = None
        transformed.append(tuple(first_parts + [date_ts, is_fraud]))
    return transformed

def insert_into_t2(rows):
    if not rows:
        log.info("Aucune donnée à migrer.")
        return 0

    placeholders = ", ".join(["%s"] * len(COLS_DST))
    insert_sql = f"""
        INSERT INTO {DST} ({', '.join(COLS_DST)})
        VALUES ({placeholders})
        ON CONFLICT DO NOTHING;
    """

    with psycopg2.connect(**PG_CONN_INFO) as conn, conn.cursor() as cur:
        execute_batch(cur, insert_sql, rows, page_size=1000)
        conn.commit()
        log.info("✅ %d lignes insérées dans %s", len(rows), DST)
    return len(rows)

def migrate():
    log.info("⚙️  Migration %s → %s", SRC, DST)
    rows = fetch_t1_rows()

    # dé‑doublonnage simple en mémoire (sur toutes les colonnes)
    unique_rows = list({tuple(r) for r in rows})
    inserted = insert_into_t2(unique_rows)
    log.info("🎉 Migration terminée : %d lignes migrées.", inserted)

if __name__ == "__main__":
    migrate()
