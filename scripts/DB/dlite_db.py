import json
import os
import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
from psycopg2 import pool

_connection_pool = None

load_dotenv()


def _get_connection():
    global _connection_pool
    if _connection_pool is None:
        _connection_pool = pool.SimpleConnectionPool(
            1, 10, os.getenv("DATABASE_URL"), sslmode="require"
        )
    return _connection_pool.getconn()


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
    _execute_query(
        query,
        (
            session_id,
            Json(g),
            Json(rhs),
            Json(queue),
            int(start_id),
            int(goal_id),
            Json(blocked_payload),
        ),
    )
            
def _execute_query(query, params=(), fetch=False):
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if fetch:
                return cur.fetchone()
            conn.commit()
    finally:
        _connection_pool.putconn(conn)


def load_session_state(session_id: str) -> Optional[Dict[str, Any]]:
    query = """
        SELECT g, rhs, U, start_id, goal_id, blocked_edges
        FROM dlite_session
        WHERE session_id = %s;
    """
    row = _execute_query(query, (session_id,), fetch=True)

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