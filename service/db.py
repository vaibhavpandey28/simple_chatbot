import os

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

from core.logger import get_logger

load_dotenv()
logger = get_logger(__name__)


DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "csv_db")
DB_USER = os.getenv("DB_USER", "csv_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "csv_pass")
DB_PORT = int(os.getenv("DB_PORT", "5432"))


def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT,
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
        logger.error("Error occurred while executing query: %s", e)
        return {"error": str(e)}
    finally:
        cur.close()
        conn.close()
