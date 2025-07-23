import pandas as pd
from scripts.etl_utils import df_to_postgres_batch

def store_csv_to_t1(csv_path: str = "/opt/airflow/data/fraudTest.csv"):
    # 1. Lire le fichier CSV (en ignorant la colonne d’index vide éventuelle)
    df = pd.read_csv(csv_path)
    df = df.head(2)
     # ✅ Supprimer la colonne d’index inutile
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    if 'trans_date_trans_time' in df.columns:
        df.drop(columns=['trans_date_trans_time'], inplace=True)    

    # 2. Renommer la colonne temporelle
    if "unix_time" in df.columns:
        df.rename(columns={"unix_time": "date_trans"}, inplace=True)

    # 3. Supprimer la première colonne vide s’il y en a une (index du CSV)
    if df.columns[0] == "":
        df = df.drop(columns=df.columns[0])

    # 4. Insertion en base de données
    df_to_postgres_batch(df, "t1")
    print(f"✅ {len(df)} lignes de {csv_path} insérées dans la table t1.")

if __name__ == "__main__":
    store_csv_to_t1()