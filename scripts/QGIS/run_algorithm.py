import os
import math
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from scripts.QGIS.find_shelter import find_nearest_shelter
from scripts.QGIS.dlite_algorithm import DStarLite
# from scripts.QGIS.save_route import save_route_to_shapefile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def _ensure_point(point):
    if isinstance(point, Point):
        return point
    if isinstance(point, (tuple, list)) and len(point) == 2:
        return Point(float(point[0]), float(point[1]))
    if isinstance(point, dict) and {"lon", "lat"} <= set(point):
        return Point(float(point["lon"]), float(point["lat"]))

#  ポイントの座標列正規化
def _normalize_point_sequence(points):
    if points is None:
        return []
    if isinstance(points, np.ndarray):
        points = points.tolist()
    if isinstance(points, (list, tuple, set)):
        seq = list(points)
        if not seq:
            return []
        first = seq[0]
        if isinstance(first, (Point, dict, list, tuple)):
            return [_ensure_point(p) for p in seq]
        if len(seq) == 2 and all(isinstance(v, (int, float)) for v in seq):
            return [_ensure_point(seq)]
        return [_ensure_point(seq)]
    return [_ensure_point(points)]

# 通行止め道路のエッジ正規化
def _normalize_edges(edges):
    pairs = []
    for edge in edges or []:
        if isinstance(edge, dict):
            u, v = edge.get("u"), edge.get("v")
        else:
            u, v = edge
        if u is None or v is None:
            continue
        pairs.append((int(u), int(v)))
    return list(dict.fromkeys(pairs))

# 通行止め更新を統一
def _normalize_edge_updates(edges):
    updates = []
    for edge in edges or []:
        blocked = True
        weight = None
        if isinstance(edge, dict):
            u, v = edge.get("u"), edge.get("v")
            if "blocked" in edge:
                blocked = bool(edge["blocked"])
            if "weight" in edge and edge["weight"] is not None:
                weight = float(edge["weight"])
        else:
            if len(edge) == 3:
                u, v, blocked = edge
            elif len(edge) >= 2:
                u, v = edge[:2]
            else:
                continue
        if u is None or v is None:
            continue
        updates.append(
            {
                "u": int(u),
                "v": int(v),
                "blocked": bool(blocked),
                "weight": weight,
            }
        )
    dedup = {}
    for upd in updates:
        dedup[(upd["u"], upd["v"])] = upd
    return list(dedup.values())

#  ポイントの座標を lon/lat タプルに変換
def _point_to_lonlat(point: Point):
    return {"lon": float(point.x), "lat": float(point.y)}

#   numpy配列化して高速化
def _node_lookup_arrays(node_positions):
    node_ids = np.array(list(node_positions.keys()), dtype=np.int64)
    node_coords = np.array(list(node_positions.values()), dtype=float)
    return node_ids, node_coords


# ポイントからの最近某ノードを高速に探索
def _nearest_node(point, node_ids, node_coords):
    target = np.array([point.x, point.y], dtype=float)
    deltas = node_coords - target
    dist_sq = np.einsum("ij,ij->i", deltas, deltas)
    min_idx = int(np.argmin(dist_sq))
    return int(node_ids[min_idx])

#  ノード列からルート座標列を構築
def _coords_close(c1, c2, tol=1e-9):
    return abs(c1[0] - c2[0]) <= tol and abs(c1[1] - c2[1]) <= tol


def build_route_coords(path, graph, node_positions):
    route_coords = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        geom_line = graph[u][v].get("geometry")
        if not geom_line:
            continue
        coords = list(geom_line)
        if len(coords) < 2:
            continue
        node_u = tuple(node_positions.get(u, coords[0]))
        node_v = tuple(node_positions.get(v, coords[-1]))
        start_matches_u = _coords_close(coords[0], node_u)
        end_matches_u = _coords_close(coords[-1], node_u)
        if not start_matches_u and end_matches_u:
            coords = list(reversed(coords))
        elif not start_matches_u and not _coords_close(coords[0], node_v):
            coords = [node_u, node_v]
        if route_coords and _coords_close(route_coords[-1], coords[0]):
            coords = coords[1:]
        route_coords.extend(coords)
    return route_coords


def _simplify_route_nodes(nodes):
    if not nodes:
        return []
    simplified = []
    index_map = {}
    for node in nodes:
        if node in index_map:
            loop_start = index_map[node]
            for removed in simplified[loop_start + 1 :]:
                index_map.pop(removed, None)
            simplified = simplified[: loop_start + 1]
        else:
            simplified.append(node)
            index_map[node] = len(simplified) - 1
    return simplified

def run_dlite_algorithm(
    loads_path=os.path.join(DATA_DIR, "processed/roads/ube_roads.shp"),
    start_point=None,
    goal_point=None,
    start_node_id=None,
    goal_node_id=None,
    initial_state=None,
    blocked_edges=None,
    new_blocked_edges=None,
):
    candidate_payload = []#避難所候補
    shelter_result = None

    # --- 出発点とゴール候補を避難所検索から取得 ---
    if start_point is None:
        shelter_result = find_nearest_shelter()
        start_point = shelter_result["start_point"]

    if goal_point is None:
        if shelter_result is None:
            shelter_result = find_nearest_shelter(start_point=start_point)
        candidate_payload = shelter_result.get("candidate_shelters", [])
        goal_point = [item["goal_point"] for item in candidate_payload]

    elif not candidate_payload:
        if isinstance(goal_point, (list, tuple)):
            goal_list = list(goal_point)
            if goal_list and isinstance(goal_list[0], dict) and "goal_point" in goal_list[0]:
                candidate_payload = goal_list
                goal_point = [item["goal_point"] for item in goal_list]
        elif isinstance(goal_point, dict) and "goal_point" in goal_point:
            candidate_payload = [goal_point]
            goal_point = [goal_point["goal_point"]]

    start_point = _ensure_point(start_point)
    # ゴールは単一/複数いずれの入力形式でも扱えるよう配列化
    goal_points = _normalize_point_sequence(goal_point)

    # 避難所候補ごとに座標と属性をまとめて保持
    goal_candidates = []
    for idx, point in enumerate(goal_points):
        payload = candidate_payload[idx] if idx < len(candidate_payload) else {}
        goal_candidates.append(
            {
                "point": point,
                "goal_point": payload.get("goal_point") or _point_to_lonlat(point),
                "shelter_attr": payload.get("shelter_attr", {}),
                "distance_m": payload.get("distance_m"),
            }
        )

    blocked_edges = _normalize_edges(blocked_edges)
    edge_updates = _normalize_edge_updates(new_blocked_edges)

    # --- 道路レイヤ読込＆グラフ生成 ---
    roads = gpd.read_file(loads_path, usecols=["geometry", "u", "v", "length"])

    # --- グラフ構築 ---
    G = nx.Graph()
    node_positions = {}

    for f in roads.itertuples(index=False):
        if f.geometry is None:
            continue
        u, v = getattr(f, "u", None), getattr(f, "v", None)
        if u is None or v is None:
            continue

        geom = f.geometry
        if geom.geom_type == "MultiLineString":
            if not geom.geoms:
                continue
            coords = list(geom.geoms[0].coords)
        else:
            coords = list(geom.coords)

        if len(coords) < 2:
            continue

        node_positions[u] = node_positions.get(u, coords[0])
        node_positions[v] = node_positions.get(v, coords[-1])

        length_attr = getattr(f, "length", None)
        edge_length = float(length_attr) if length_attr is not None else float(geom.length)
        G.add_edge(
            u,
            v,
            weight=edge_length,
            base_weight=edge_length,
            geometry=coords,
        )

    # --- 出発点・到着点を最寄りノードへスナップ ---
    node_ids_arr, node_coords_arr = _node_lookup_arrays(node_positions)

    if start_node_id is not None:
        start_id = int(start_node_id)

        if start_point is None:
            sx, sy = node_positions[start_id]
            start_point = Point(sx, sy)
    else:
        start_id = _nearest_node(start_point, node_ids_arr, node_coords_arr)

    if goal_node_id is not None:
        if isinstance(goal_node_id, (list, tuple, set)):
            goal_ids = [int(g) for g in goal_node_id]
        else:
            goal_ids = [int(goal_node_id)]
    else:
        goal_ids = []

    goal_nodes = []
    goal_metadata = {}
    goal_candidates_payload = []

    for idx, cand in enumerate(goal_candidates):
        if idx < len(goal_ids):
            node_id = goal_ids[idx]
            point = cand["point"]
        else:
            point = cand["point"]
            node_id = _nearest_node(point, node_ids_arr, node_coords_arr)

        cand["node_id"] = node_id

        if node_id not in goal_nodes:
            goal_nodes.append(node_id)
            goal_metadata[node_id] = cand

        goal_candidates_payload.append(
            {
                "node_id": int(node_id),
                "goal_point": cand["goal_point"],
                "shelter_attr": cand.get("shelter_attr", {}),
                "distance_m": cand.get("distance_m"),
            }
        )


    # --- 通行止め適用 ---
    if blocked_edges:
        for u, v in blocked_edges:
            if G.has_edge(u, v):
                G[u][v]["weight"] = float("inf")

    dlite = DStarLite(G, start_id, goal_nodes, node_positions, initial_state=initial_state)

    if edge_updates:
        for upd in edge_updates:
            u, v = upd["u"], upd["v"]
            if upd["weight"] is not None:
                dlite.update_edge_cost(u, v, upd["weight"])
            elif upd["blocked"]:
                dlite.update_edge_cost(u, v, float("inf"))
            else:
                dlite.update_edge_cost(u, v, None)

    dlite.compute_shortest_path()
    route = _simplify_route_nodes(dlite.extract_path())

    reached_goal_id = dlite.get_reached_goal() if route else None
    if reached_goal_id is None and goal_nodes:
        reached_goal_id = goal_nodes[0]

    total_dist = sum(
        float(G[route[i]][route[i + 1]]["weight"])
        for i in range(len(route) - 1)
        if math.isfinite(G[route[i]][route[i + 1]]["weight"])
    )

    route_coords = build_route_coords(route, G, node_positions)

    print(f"📏 距離: {total_dist:.2f} m")
    print(f"🛣️ ノード数: {len(route)}")

    selected_goal_meta = goal_metadata.get(reached_goal_id)
    if selected_goal_meta:
        selected_goal_point = selected_goal_meta["goal_point"]
        selected_goal_attr = selected_goal_meta.get("shelter_attr", {})
    else:
        if reached_goal_id in node_positions:
            gx, gy = node_positions[reached_goal_id]
            selected_goal_point = _point_to_lonlat(Point(gx, gy))
        else:
            selected_goal_point = goal_candidates_payload[0]["goal_point"]
        selected_goal_attr = {}

    return {
        "start": _point_to_lonlat(start_point),
        "goal": selected_goal_point,
        "distance_m": float(total_dist),
        "route_nodes": route,
        "route_coords": route_coords,
        "start_id": int(start_id),
        "goal_id": int(reached_goal_id) if reached_goal_id is not None else None,
        "goal_node_ids": [int(g) for g in goal_nodes],
        "goal_candidates": goal_candidates_payload,
        "selected_shelter_attr": selected_goal_attr,
        "blocked_edges": [{"u": int(u), "v": int(v)} for u, v in (blocked_edges or [])],
        "dlite_state": dlite.export_state(),
    }


if __name__ == "__main__":
    result = run_dlite_algorithm()

    # save_route_to_shapefile(result)
