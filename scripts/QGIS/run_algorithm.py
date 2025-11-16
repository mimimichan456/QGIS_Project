import os
import math
import numpy as np
import networkx as nx
import geopandas as gpd
import time
from shapely.geometry import Point, LineString
from shapely.ops import split as shapely_split
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


def _normalize_node_id(value):
    if value is None:
        return None
    if isinstance(value, str):
        if value.startswith("__"):
            return value
        try:
            return int(float(value))
        except ValueError:
            return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _serialize_node_id(value):
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        if value.startswith("__"):
            return value
        try:
            return int(value)
        except ValueError:
            return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _add_edge_with_geometry(graph, a, b, coords):
    line_geom = LineString(coords)
    length = float(line_geom.length)
    graph.add_edge(
        a,
        b,
        weight=length,
        base_weight=length,
        geometry=list(coords),
    )

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


def _find_nearest_edge(point_geom, graph):
    best = None
    best_dist = float("inf")
    for u, v, data in graph.edges(data=True):
        if not math.isfinite(data.get("weight", float("inf"))):
            continue
        coords = data.get("geometry")
        if not coords:
            continue
        line = LineString(coords) if not isinstance(coords, LineString) else coords
        dist = line.distance(point_geom)
        if dist < best_dist:
            best_dist = dist
            best = (u, v, line)
    return best


def _generate_pseudo_node_id(node_positions):
    max_id = 0
    for key in node_positions:
        try:
            val = int(key)
            if val > max_id:
                max_id = val
        except (TypeError, ValueError):
            continue
    new_id = max_id + 1
    while new_id in node_positions:
        new_id += 1
    return new_id


def _insert_point_on_edge(point_geom, graph, node_positions, node_ids_arr, node_coords_arr):
    nearest = _find_nearest_edge(point_geom, graph)
    if not nearest:
        print("[snap] no nearest edge, fallback to nearest node")
        node_id = _nearest_node(point_geom, node_ids_arr, node_coords_arr)
        snapped = Point(node_positions[node_id])
        return node_id, snapped

    u, v, line = nearest
    edge_data = graph[u][v]
    if not math.isfinite(edge_data.get("weight", float("inf"))):
        print(f"[snap] edge ({u},{v}) blocked, fallback to nearest node")
        node_id = _nearest_node(point_geom, node_ids_arr, node_coords_arr)
        snapped = Point(node_positions[node_id])
        return node_id, snapped

    proj = line.project(point_geom)
    if proj <= 1e-9:
        return u, Point(node_positions[u])
    if proj >= line.length - 1e-9:
        return v, Point(node_positions[v])

    split_point = line.interpolate(proj)

    def _split_line(line_obj, new_pt):
        coords = list(line_obj.coords)
        acc = 0.0
        new_coords1 = [coords[0]]
        new_coords2 = []
        for i in range(len(coords) - 1):
            seg_start = coords[i]
            seg_end = coords[i + 1]
            seg_line = LineString([seg_start, seg_end])
            seg_len = seg_line.length
            if acc + seg_len >= proj - 1e-9:
                ratio = 0.0 if seg_len == 0 else (proj - acc) / seg_len
                interp_x = seg_start[0] + ratio * (seg_end[0] - seg_start[0])
                interp_y = seg_start[1] + ratio * (seg_end[1] - seg_start[1])
                new_point = (interp_x, interp_y)
                new_coords1.append(new_point)
                new_coords2 = [new_point, seg_end]
                new_coords2.extend(coords[i + 2 :])
                break
            acc += seg_len
            new_coords1.append(seg_end)
        if len(new_coords2) == 0:
            return None
        return LineString(new_coords1), LineString(new_coords2)

    split_segments = _split_line(line, split_point)
    if not split_segments:
        print("[snap] split result invalid, fallback to nearest node")
        node_id = _nearest_node(point_geom, node_ids_arr, node_coords_arr)
        snapped = Point(node_positions[node_id])
        return node_id, snapped

    new_node_id = _generate_pseudo_node_id(node_positions)
    node_positions[new_node_id] = (split_point.x, split_point.y)

    seg_coords = [list(split_segments[0].coords), list(split_segments[1].coords)]
    graph.remove_edge(u, v)

    _add_edge_with_geometry(graph, u, new_node_id, seg_coords[0])
    _add_edge_with_geometry(graph, new_node_id, v, seg_coords[1])

    return new_node_id, split_point

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
        start_id = _normalize_node_id(start_node_id)
        if start_point is None and start_id in node_positions:
            sx, sy = node_positions[start_id]
            start_point = Point(sx, sy)
    else:
        snapped_id, snapped_point = _insert_point_on_edge(start_point, G, node_positions, node_ids_arr, node_coords_arr)
        node_ids_arr, node_coords_arr = _node_lookup_arrays(node_positions)
        start_anchor_id = _generate_pseudo_node_id(node_positions)
        node_positions[start_anchor_id] = (start_point.x, start_point.y)
        start_id = start_anchor_id
        _add_edge_with_geometry(
            G,
            start_id,
            snapped_id,
            [(start_point.x, start_point.y), (snapped_point.x, snapped_point.y)],
        )

    if goal_node_id is not None:
        if isinstance(goal_node_id, (list, tuple, set)):
            goal_ids = [_normalize_node_id(g) for g in goal_node_id]
        else:
            goal_ids = [_normalize_node_id(goal_node_id)]
    else:
        goal_ids = []

    goal_nodes = []
    goal_metadata = {}
    goal_candidates_payload = []

    for idx, cand in enumerate(goal_candidates):
        point = cand["point"]
        if idx < len(goal_ids):
            node_id = goal_ids[idx]
            snapped_point = Point(node_positions.get(node_id, (point.x, point.y)))
        else:
            node_ids_arr, node_coords_arr = _node_lookup_arrays(node_positions)
            snapped_id, snapped_point = _insert_point_on_edge(
                point, G, node_positions, node_ids_arr, node_coords_arr
            )
            node_ids_arr, node_coords_arr = _node_lookup_arrays(node_positions)
            goal_anchor_id = _generate_pseudo_node_id(node_positions)
            node_positions[goal_anchor_id] = (point.x, point.y)
            _add_edge_with_geometry(
                G,
                snapped_id,
                goal_anchor_id,
                [(snapped_point.x, snapped_point.y), (point.x, point.y)],
            )
            node_id = goal_anchor_id

        cand["node_id"] = node_id

        if node_id not in goal_nodes:
            goal_nodes.append(node_id)
            goal_metadata[node_id] = cand

        goal_candidates_payload.append(
            {
                "node_id": int(node_id),
                "goal_point": _point_to_lonlat(point),
                "shelter_attr": cand.get("shelter_attr", {}),
                "distance_m": cand.get("distance_m"),
            }
        )


    # --- 通行止め適用 ---
    if blocked_edges:
        for u, v in blocked_edges:
            if G.has_edge(u, v):
                G[u][v]["weight"] = float("inf")

    print(f"[D*Lite] ゴール候補ノード: {goal_nodes}")
    best_result = None

    for idx, goal_node in enumerate(goal_nodes):
        goal_meta = goal_metadata.get(goal_node)
        dlite_state = initial_state if idx == 0 and initial_state else None
        dlite = DStarLite(G, start_id, [goal_node], node_positions, initial_state=dlite_state)

        if edge_updates:
            for upd in edge_updates:
                u, v = upd["u"], upd["v"]
                if upd["weight"] is not None:
                    dlite.update_edge_cost(u, v, upd["weight"])
                elif upd["blocked"]:
                    dlite.update_edge_cost(u, v, float("inf"))
                else:
                    dlite.update_edge_cost(u, v, None)

        start_compute = time.time()
        dlite.compute_shortest_path()
        elapsed = time.time() - start_compute
        print(f"[D*Lite] goal {goal_node} compute 実行時間: {elapsed:.2f} 秒")
        print("[D*Lite] extract_path 開始")
        raw_route = dlite.extract_path()
        print(f"[D*Lite] extract_path 結果: {raw_route[:10] if raw_route else raw_route}")
        route = _simplify_route_nodes(raw_route)
        if not route or dlite.get_reached_goal() != goal_node:
            print(f"[D*Lite] goal {goal_node} route not found")
            continue

        total_dist = sum(
            float(G[route[i]][route[i + 1]]["weight"])
            for i in range(len(route) - 1)
            if math.isfinite(G[route[i]][route[i + 1]]["weight"])
        )
        print(f"[D*Lite] goal {goal_node} total_dist={total_dist}")

        if best_result is None or total_dist < best_result["distance"]:
            best_result = {
                "route": route,
                "distance": total_dist,
                "goal_id": goal_node,
                "dlite_state": dlite.export_state(),
                "goal_meta": goal_meta,
            }

    if not best_result:
        raise ValueError("到達可能なゴールがありません。")

    route = best_result["route"]
    total_dist = best_result["distance"]
    reached_goal_id = best_result["goal_id"]
    dlite_state = best_result["dlite_state"]
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
        "start_id": _serialize_node_id(start_id),
        "goal_id": _serialize_node_id(reached_goal_id),
        "goal_node_ids": [_serialize_node_id(g) for g in goal_nodes],
        "goal_candidates": goal_candidates_payload,
        "selected_shelter_attr": selected_goal_attr,
        "blocked_edges": [{"u": int(u), "v": int(v)} for u, v in (blocked_edges or [])],
        "dlite_state": dlite_state,
    }


if __name__ == "__main__":
    result = run_dlite_algorithm()

    # save_route_to_shapefile(result)
