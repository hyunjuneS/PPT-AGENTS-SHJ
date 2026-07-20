"""PostgreSQL의 sources 테이블에서 id별 raw_text를 조회."""

import logging
import os

import psycopg2

logger = logging.getLogger(__name__)


def _get_connection():
    return psycopg2.connect(
        host=os.environ["DB_HOST"],
        port=os.environ.get("DB_PORT", "5432"),
        dbname=os.environ["DB_NAME"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
    )


def fetch_raw_texts(ids: list[str]) -> dict[str, str]:
    """sources 테이블에서 id 목록에 해당하는 raw_text를 조회해 {id: raw_text} 로 반환.

    ids 중 sources 테이블에 없는 값이 하나라도 있으면 ValueError.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, raw_text FROM sources WHERE id = ANY(%s)", (ids,))
            rows = dict(cur.fetchall())
    finally:
        conn.close()

    missing = [i for i in ids if i not in rows]
    if missing:
        raise ValueError(f"sources not found for id(s): {', '.join(missing)}")

    logger.info("[DB] fetched %d raw_text row(s) from sources", len(rows))
    return rows
