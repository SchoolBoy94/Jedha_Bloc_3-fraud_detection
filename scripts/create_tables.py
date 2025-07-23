import os
import psycopg2

# Charger les variables d'environnement si besoin
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Connexion PostgreSQL
PG_CONN_INFO = {
    'host': os.getenv("POSTGRES_HOST"),
    'database': os.getenv("POSTGRES_DB"),
    'user': os.getenv("POSTGRES_USER"),
    'password': os.getenv("POSTGRES_PASSWORD"),
    'port': int(os.getenv("POSTGRES_PORT"))
}






CREATE_TABLE_SQL = """


CREATE TABLE IF NOT EXISTS t1 (
    cc_num BIGINT,
    merchant TEXT,
    category TEXT,
    amt FLOAT,
    first TEXT,
    last TEXT,
    gender TEXT,
    street TEXT,
    city TEXT,
    state TEXT,
    zip INT,
    lat FLOAT,
    long FLOAT,
    city_pop INT,
    job TEXT,
    dob DATE,
    trans_num TEXT PRIMARY KEY,
    merch_lat FLOAT,
    merch_long FLOAT,
    is_fraud INT,
    date_trans BIGINT
);


CREATE TABLE IF NOT EXISTS t2 (
    merchant       TEXT,
    category       TEXT,
    amt            FLOAT,
    gender         VARCHAR(1),
    zip            INTEGER,
    lat            FLOAT,
    long           FLOAT,
    city_pop       INTEGER,
    dob            DATE,
    merch_lat      FLOAT,
    merch_long     FLOAT,
    date_trans     TIMESTAMP,
    is_fraud       INTEGER
);




"""

def create_table():
    try:
        conn = psycopg2.connect(**PG_CONN_INFO)
        cur = conn.cursor()
        cur.execute(CREATE_TABLE_SQL)
        conn.commit()
        cur.close()
        conn.close()
        print("✅ Table `table1` créée avec succès dans la base `fraud`.")
    except Exception as e:
        print(f"❌ Erreur lors de la création de la table : {e}")

if __name__ == "__main__":
    create_table()
