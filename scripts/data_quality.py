import os, psycopg2, logging, requests
from datetime import datetime, timedelta

logger  = logging.getLogger(__name__)
WEBHOOK = os.getenv("DISCORD_WEBHOOK")  # ← variable d’environnement
TABLE   = "t2"

# ──────────────────────────────────────────────────────────────────
PG_CONN_INFO = {
    "host": "postgres",
    "database": "fraud",
    "user": "postgres",
    "password": "postgres",
    "port": 5432,
}

def _fetchone(sql: str):
    with psycopg2.connect(**PG_CONN_INFO) as conn, conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchone()[0]

# ──────────────────────────────────────────────────────────────────
def check_nulls():
    cols = ["merchant", "category", "amt", "gender", "date_trans"]
    issues = []
    for col in cols:
        cnt = _fetchone(f"SELECT COUNT(*) FROM {TABLE} WHERE {col} IS NULL;")
        if cnt:
            issues.append(f"• `{col}` : **{cnt} NULL**")
    return issues

def check_duplicates():
    dup = _fetchone(
        f"""
        SELECT COUNT(*) FROM (
            SELECT merchant, category, amt, date_trans, COUNT(*)
            FROM {TABLE}
            GROUP BY merchant, category, amt, date_trans
            HAVING COUNT(*) > 1
        ) AS t;
        """
    )
    return [f"• **{dup} doublons** détectés"] if dup else []

def check_amount_range():
    bad = _fetchone(
        f"SELECT COUNT(*) FROM {TABLE} WHERE amt <= 0 OR amt > 100000;"
    )
    return [f"• **{bad} montants hors plage** (<=0 ou >100 000)"] if bad else []

def check_freshness(hours=24):
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    recent = _fetchone(
        f"SELECT COUNT(*) FROM {TABLE} WHERE date_trans >= '{cutoff.isoformat()}';"
    )
    return [] if recent else [f"• **Aucune ligne récente (<{hours} h)**"]

# ──────────────────────────────────────────────────────────────────
def _notify_discord(content: str, level="ERROR"):
    if not WEBHOOK:
        logger.warning("Webhook Discord non configuré.")
        return
    color = {"ERROR": 0xE74C3C, "INFO": 0x2ECC71}.get(level, 0x95A5A6)
    payload = {"embeds": [{
        "title": f"Data‑Quality {level}",
        "description": content,
        "color": color
    }]}
    try:
        requests.post(WEBHOOK, json=payload, timeout=10)
        logger.info("Notification Discord envoyée.")
    except Exception as exc:
        logger.error("Échec d’envoi Discord : %s", exc)

# ──────────────────────────────────────────────────────────────────
def run_quality_checks():
    logger.info("⚙️  Début des contrôles DQ sur %s", TABLE)

    issues  = []
    issues += check_nulls()
    issues += check_duplicates()
    issues += check_amount_range()
    issues += check_freshness()

    if issues:
        msg = "**Anomalies détectées :**\n" + "\n".join(issues)
        logger.error(msg.replace("**", ""))           # log Airflow
        _notify_discord(msg, level="ERROR")            # notif rouge
        raise ValueError("Data‑Quality FAILED")        # échec de la task
    else:
        ok_msg = "Tous les contrôles *PASS* ✔️"
        logger.info(ok_msg)
        _notify_discord(ok_msg, level="INFO")          # notif verte

# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_quality_checks()
