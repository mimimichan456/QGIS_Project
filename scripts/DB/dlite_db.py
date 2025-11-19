import json
import os
from psycopg2.extras import Json
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
from psycopg2 import pool

load_dotenv()

def _get_connection():
    if not hasattr(_get_connection, "pool"):
        _get_connection.pool = pool.SimpleConnectionPool(
            1, 10, os.getenv("DATABASE_URL"), sslmode="require"
        )
    return _get_connection.pool.getconn()


def _put_connection(conn):
    if hasattr(_get_connection, "pool"):
        _get_connection.pool.putconn(conn)


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
    goal_points: List[Dict[str, float]],
    goal_node_ids: Optional[List[int]] = None,
):
    blocked_payload = {
        "edges": blocked_edges,
        "start_point": start_point,
        "goal_points": goal_points,
        "goal_node_ids": goal_node_ids,
    }
    query = """
        INSERT INTO dlite_db (session_id, g, rhs, U, start_id, goal_id, blocked_edges)
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
        _put_connection(conn)


def load_session_state(session_id: str) -> Optional[Dict[str, Any]]:
    query = """
        SELECT g, rhs, U, start_id, goal_id, blocked_edges
        FROM public.dlite_db
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

    goal_points = blocked_payload.get("goal_points") or []
    if not goal_points and blocked_payload.get("goal_point"):
        goal_points = [blocked_payload.get("goal_point")]

    return {
        "g": g or {},
        "rhs": rhs or {},
        "queue": queue or [],
        "km": 0,
        "start_id": int(start_id),
        "goal_id": int(goal_id),
        "blocked_edges": blocked_payload.get("edges", []),
        "start_point": blocked_payload.get("start_point"),
        "goal_points": goal_points,
        "goal_point": blocked_payload.get("goal_point"),
        "goal_node_ids": blocked_payload.get("goal_node_ids") or [],
    }


def reset_blocked_point(session_id: str) -> bool:
    query = """
        SELECT blocked_edges
        FROM public.dlite_db
        WHERE session_id = %s;
    """
    row = _execute_query(query, (session_id,), fetch=True)
    if not row:
        return False

    blocked_json = row[0]
    if isinstance(blocked_json, str):
        payload = json.loads(blocked_json) if blocked_json else {}
    else:
        payload = blocked_json or {}

    payload["blocked_point"] = {}
    payload["edges"] = []

    update_query = """
        UPDATE dlite_db
        SET blocked_edges = %s,
            updated_at = CURRENT_TIMESTAMP
        WHERE session_id = %s;
    """
    _execute_query(update_query, (Json(payload), session_id))
    return True
