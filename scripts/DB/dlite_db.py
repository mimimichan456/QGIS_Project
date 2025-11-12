import json
import os
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional

load_dotenv()


def _get_connection():
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL が設定されていません。")
    return psycopg2.connect(database_url, sslmode="require")


def save_session_state(
    session_id: str,
    *,
    g: Dict,
    rhs: Dict,
    queue: List[Dict[str, Any]],
    start_id: int,
    goal_id: int,
    blocked_edges: List[Dict[str, int]],
    start_point: Dict[str, float],
    goal_point: Dict[str, float],
):
    blocked_payload = {
        "edges": blocked_edges,
        "start_point": start_point,
        "goal_point": goal_point,
    }
    query = """
        INSERT INTO dlite_session (session_id, g, rhs, U, start_id, goal_id, blocked_edges)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (session_id)
        DO UPDATE SET g = EXCLUDED.g, rhs = EXCLUDED.rhs, U = EXCLUDED.U,
                      start_id = EXCLUDED.start_id, goal_id = EXCLUDED.goal_id,
                      blocked_edges = EXCLUDED.blocked_edges,
                      updated_at = CURRENT_TIMESTAMP;
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                (
                    session_id,
                    Json(g),    # D* Lite 側で有限値のみに間引いてから渡している
                    Json(rhs),  # → DB に書き込むペイロードを最小限に抑える
                    Json(queue),
                    int(start_id),
                    int(goal_id),
                    Json(blocked_payload),
                ),
            )


def load_session_state(session_id: str) -> Optional[Dict[str, Any]]:
    query = """
        SELECT g, rhs, U, start_id, goal_id, blocked_edges
        FROM dlite_session
        WHERE session_id = %s;
    """
    with _get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (session_id,))
            row = cur.fetchone()

    if not row:
        return None

    g, rhs, queue, start_id, goal_id, blocked_json = row
    if isinstance(blocked_json, str):
        blocked_payload = json.loads(blocked_json)
    else:
        blocked_payload = blocked_json or {}

    return {
        "g": g or {},
        "rhs": rhs or {},
        "queue": queue or [],
        "km": 0,
        "start_id": int(start_id),
        "goal_id": int(goal_id),
        "blocked_edges": blocked_payload.get("edges", []),
        "start_point": blocked_payload.get("start_point"),
        "goal_point": blocked_payload.get("goal_point"),
    }
