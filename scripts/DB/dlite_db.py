import json
import os
import requests
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

headers = {
    "apikey": SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates",
}

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

    url = f"{SUPABASE_URL}/rest/v1/dlite_db?on_conflict=session_id"
    payload = {
        "session_id": session_id,
        "g": g,
        "rhs": rhs,
        "u": queue,
        "start_id": int(start_id),
        "goal_id": int(goal_id),
        "blocked_edges": blocked_payload
    }

    res = requests.post(url, headers=headers, json=payload, timeout=10)

    # 成功時（200, 201）はログも JSON パースも行わない
    if res.status_code in (200, 201):
        return None

    # 失敗時のみエラーを表示
    print("DEBUG_SAVE_FAIL:", res.status_code, res.text)

    try:
        return res.json()
    except ValueError:
        print("DEBUG_JSON_PARSE_FAIL(save):", res.status_code, res.text)
        return {"error": res.text, "status_code": res.status_code}

def load_session_state(session_id: str) -> Optional[Dict[str, Any]]:
    url = f"{SUPABASE_URL}/rest/v1/dlite_db?session_id=eq.{session_id}&select=*"
    res = requests.get(url, headers=headers, timeout=10)
    if res.status_code != 200:
        print("DEBUG_LOAD_FAIL:", res.status_code, res.text)
    try:
        rows = res.json()
    except (ValueError, json.JSONDecodeError):
        print("DEBUG_JSON_PARSE_FAIL(load):", res.status_code, res.text)
        return None

    if not rows:
        return None

    row = rows[0]

    blocked_payload = row.get("blocked_edges") or {}
    if isinstance(blocked_payload, str):
        blocked_payload = json.loads(blocked_payload)

    goal_points = blocked_payload.get("goal_points") or []
    if not goal_points and blocked_payload.get("goal_point"):
        goal_points = [blocked_payload.get("goal_point")]

    return {
        "g": row.get("g") or {},
        "rhs": row.get("rhs") or {},
        "queue": row.get("u") or [],
        "km": 0,
        "start_id": int(row.get("start_id")),
        "goal_id": int(row.get("goal_id")),
        "blocked_edges": blocked_payload.get("edges", []),
        "start_point": blocked_payload.get("start_point"),
        "goal_points": goal_points,
        "goal_point": blocked_payload.get("goal_point"),
        "goal_node_ids": blocked_payload.get("goal_node_ids") or [],
    }

def reset_blocked_point(session_id: str) -> bool:
    url = f"{SUPABASE_URL}/rest/v1/dlite_db?session_id=eq.{session_id}&select=blocked_edges"
    res = requests.get(url, headers=headers, timeout=10)
    try:
        rows = res.json()
    except (ValueError, json.JSONDecodeError):
        print("DEBUG_JSON_PARSE_FAIL(reset):", res.status_code, res.text)
        return False

    if not rows:
        return False

    blocked_json = rows[0].get("blocked_edges")

    if isinstance(blocked_json, str):
        payload = json.loads(blocked_json) if blocked_json else {}
    else:
        payload = blocked_json or {}

    payload["blocked_point"] = {}
    payload["edges"] = []

    update_url = f"{SUPABASE_URL}/rest/v1/dlite_db?session_id=eq.{session_id}"
    res2 = requests.patch(update_url, headers=headers, json={"blocked_edges": payload}, timeout=10)
    if res2.status_code not in (200, 204):
        print("DEBUG_UPDATE_FAIL:", res2.status_code, res2.text)

    return res2.status_code in (200, 204)
