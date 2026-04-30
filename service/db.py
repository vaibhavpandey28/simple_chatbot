import psycopg2
from psycopg2.extras import RealDictCursor
from core.logger import get_logger

logger  = get_logger(__name__)


def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="csv_db",
        user="csv_user",
        password="csv_pass",
        port=5432
    )


def run_query(query: str):
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    try:
        cur.execute(query)
        result = cur.fetchall()
        logger.info("Query executed successfully")
        return result
    except Exception as e:
        logger.error(f"Error occurred while executing query: {e}")
        return {"error": str(e)}
    finally:
        cur.close()
        conn.close()