import os
import math
import numpy as np
import networkx as nx
import geopandas as gpd
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import time
from shapely.geometry import Point, LineString
from scripts.QGIS.find_shelter import find_nearest_shelter
from scripts.QGIS.dlite_algorithm import DStarLite

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

EdgeUpdate = Dict[str, Any]

def _ensure_point(point: Any) -> Point:
    if isinstance(point, Point):
        return point
    if isinstance(point, (tuple, list)) and len(point) == 2:
        return Point(float(point[0]), float(point[1]))
    if isinstance(point, dict) and "lon" in point and "lat" in point:
        return Point(float(point["lon"]), float(point["lat"]))

#  ポイントの座標列正規化
def _normalize_point_sequence(points: Any) -> List[Point]:
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

# 通行止め更新を統一
def _normalize_edge_updates(edges: Optional[Iterable[Any]]) -> List[EdgeUpdate]:
    updates: List[EdgeUpdate] = []
    for edge in edges or []:
        blocked = True
        weight = None
        blocked_point = None
        if isinstance(edge, dict):
            u, v = edge.get("u"), edge.get("v")
            if "blocked" in edge:
                blocked = bool(edge["blocked"])
            if "weight" in edge and edge["weight"] is not None:
                weight = float(edge["weight"])
            if "blocked_point" in edge and edge["blocked_point"]:
                try:
                    blocked_point = _ensure_point(edge["blocked_point"])
                except (TypeError, ValueError):
                    blocked_point = None
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
                "point": blocked_point,
            }
        )
    dedup = {}
    for upd in updates:
        dedup[(upd["u"], upd["v"])] = upd
    return list(dedup.values())

#  ポイントの座標を lon/lat タプルに変換
def _point_to_lonlat(point: Point) -> Dict[str, float]:
    return {"lon": float(point.x), "lat": float(point.y)}

#   numpy配列化して高速化
def _node_lookup_arrays(node_positions: Dict[int, Tuple[float, float]]) -> Tuple[np.ndarray, np.ndarray]:
    node_ids = np.array(list(node_positions.keys()), dtype=np.int64)
    node_coords = np.array(list(node_positions.values()), dtype=float)
    return node_ids, node_coords


# ポイントからの最近某ノードを高速に探索
def _nearest_node(point: Point, node_ids: np.ndarray, node_coords: np.ndarray) -> int:
    target = np.array([point.x, point.y], dtype=float)
    deltas = node_coords - target
    dist_sq = np.einsum("ij,ij->i", deltas, deltas)
    min_idx = int(np.argmin(dist_sq))
    return int(node_ids[min_idx])


def _normalize_node_id(value: Any) -> Any:
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


def _serialize_node_id(value: Any) -> Any:
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


def _add_edge_with_geometry(graph: nx.Graph, a: int, b: int, coords: Sequence[Tuple[float, float]]) -> None:
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


def build_route_coords(
    path: Sequence[int], graph: nx.Graph, node_positions: Dict[int, Tuple[float, float]]
) -> List[Tuple[float, float]]:
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


def _simplify_route_nodes(nodes: Optional[Sequence[int]]) -> List[int]:
    if not nodes:
        return []
    simplified: List[int] = []
    index_map: Dict[int, int] = {}
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


def _find_nearest_edge(point_geom: Point, graph: nx.Graph):
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


def _generate_pseudo_node_id(node_positions: Dict[int, Tuple[float, float]]) -> int:
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


def _insert_point_on_edge(
    point_geom: Point,
    graph: nx.Graph,
    node_positions: Dict[int, Tuple[float, float]],
    node_ids_arr: np.ndarray,
    node_coords_arr: np.ndarray,
    split_tracker: Optional[Dict[Tuple[int, int], List[Tuple[int, int]]]] = None,
) -> Tuple[int, Point]:
    nearest = _find_nearest_edge(point_geom, graph)
    if not nearest:
        print("[snap] no nearest edge, fallback to nearest node")
        node_id = _nearest_node(point_geom, node_ids_arr, node_coords_arr)
        snapped = Point(node_positions[node_id])
        return node_id, snapped

    u, v, line = nearest
    u = int(u)
    v = int(v)
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

    def _split_line(line_obj):
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

    split_segments = _split_line(line)
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

    if split_tracker is not None:
        forward_pairs = [(u, new_node_id), (new_node_id, v)]
        reverse_pairs = [(v, new_node_id), (new_node_id, u)]
        split_tracker[(u, v)] = forward_pairs
        split_tracker[(v, u)] = reverse_pairs
        print(f"[SNAP] split edge ({u}, {v}) -> {forward_pairs}")

    return new_node_id, split_point


def _expand_split_edges(u: int, v: int, split_tracker: Dict[Tuple[int, int], List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
    targets = [(u, v)]
    split_pairs = split_tracker.get((u, v)) or split_tracker.get((v, u))
    if split_pairs:
        print(f"[BLOCK DEBUG] original edge ({u}, {v}) has split pairs {split_pairs}")
        targets.extend(split_pairs)
    seen = set()
    ordered = []
    for src, dst in targets:
        key = (int(src), int(dst))
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _edge_distance_to_point(
    graph: nx.Graph, u: int, v: int, point_geom: Optional[Point]
) -> Optional[float]:
    if point_geom is None:
        return None
    data = graph.get(u, {}).get(v) if isinstance(graph, dict) else None
    if data is None:
        try:
            data = graph[u][v]
        except KeyError:
            return None
    coords = data.get("geometry")
    if not coords:
        return None
    line = coords if isinstance(coords, LineString) else LineString(coords)
    return line.distance(point_geom)


def _select_block_targets(
    u: int,
    v: int,
    split_tracker: Dict[Tuple[int, int], List[Tuple[int, int]]],
    point_geom: Optional[Point],
    graph: nx.Graph,
) -> List[Tuple[int, int]]:
    candidates = _expand_split_edges(u, v, split_tracker)
    if point_geom is None or len(candidates) <= 1:
        return candidates
    distances = []
    for src, dst in candidates:
        dist = _edge_distance_to_point(graph, src, dst, point_geom)
        if dist is None:
            dist = float("inf")
        distances.append(dist)
    min_dist = min(distances) if distances else float("inf")
    if not math.isfinite(min_dist):
        return candidates
    filtered = [
        edge for edge, dist in zip(candidates, distances) if math.isfinite(dist) and abs(dist - min_dist) <= 1e-9
    ]
    return filtered or candidates


def _merge_edge_updates(*update_lists: Optional[Iterable[EdgeUpdate]]) -> List[EdgeUpdate]:
    merged: Dict[Tuple[int, int], EdgeUpdate] = {}
    order: List[Tuple[int, int]] = []
    for updates in update_lists:
        for upd in updates or []:
            u = int(upd["u"])
            v = int(upd["v"])
            key = (u, v)
            if key not in order:
                order.append(key)
            merged[key] = {
                "u": u,
                "v": v,
                "blocked": bool(upd.get("blocked", True)),
                "weight": upd.get("weight"),
                "point": upd.get("point"),
            }
    return [merged[key] for key in order]


def _serialize_blocked_edges(updates: Optional[Iterable[EdgeUpdate]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for upd in updates or []:
        if not upd.get("blocked", True):
            continue
        item = {"u": int(upd["u"]), "v": int(upd["v"])}
        point = upd.get("point")
        if point is not None:
            item["blocked_point"] = _point_to_lonlat(point)
        payload.append(item)
    return payload



def run_dlite_algorithm(
    loads_path: str = os.path.join(DATA_DIR, "processed/roads/ube_roads.shp"),
    start_point: Optional[Any] = None,
    goal_point: Optional[Any] = None,
    start_node_id: Optional[Any] = None,
    goal_node_id: Optional[Any] = None,
    initial_state: Optional[Dict[str, Any]] = None,
    blocked_edges: Optional[Iterable[Any]] = None,
    new_blocked_edges: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    candidate_payload = []#避難所候補
    shelter_result = None
    split_edge_tracker = {}

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

    raw_blocked_edges = blocked_edges or []
    base_blocked_updates = _normalize_edge_updates(raw_blocked_edges)
    incremental_updates = _normalize_edge_updates(new_blocked_edges)
    edge_updates = _merge_edge_updates(base_blocked_updates, incremental_updates)
    blocked_edges_payload = _serialize_blocked_edges(edge_updates)

    # --- 道路レイヤ読込＆グラフ生成 ---
    roads = gpd.read_file(loads_path, usecols=["geometry", "u", "v", "length"])

    # --- グラフ構築 ---
    G = nx.Graph()
    node_positions = {}

    for f in roads.itertuples(index=False):
        if f.geometry is None:
            continue
        u_raw, v_raw = getattr(f, "u", None), getattr(f, "v", None)
        u = int(float(u_raw)) if u_raw is not None else None
        v = int(float(v_raw)) if v_raw is not None else None
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

        node_positions[int(u)] = node_positions.get(int(u), coords[0])
        node_positions[int(v)] = node_positions.get(int(v), coords[-1])

        length_attr = getattr(f, "length", None)
        edge_length = float(length_attr) if length_attr is not None else float(geom.length)
        G.add_edge(
            int(u),
            int(v),
            weight=edge_length,
            base_weight=edge_length,
            geometry=coords,
        )

    # --- 出発点・到着点を最寄りノードへスナップ ---
    node_ids_arr, node_coords_arr = _node_lookup_arrays(node_positions)

    # --- START NODE FIX: ensure start_id exists in current graph ---
    start_id = None
    if start_node_id is not None:
        tmp_id = _normalize_node_id(start_node_id)
        # 前回のノードが今回のGに存在しなければ無視してスナップし直す
        if tmp_id in G and tmp_id in node_positions:
            start_id = tmp_id
            if start_point is None:
                sx, sy = node_positions[start_id]
                start_point = Point(sx, sy)
        else:
            # 存在しない場合はスナップを強制
            start_node_id = None

    if start_node_id is None:
        snapped_id, snapped_point = _insert_point_on_edge(
            start_point, G, node_positions, node_ids_arr, node_coords_arr, split_edge_tracker
        )
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

    # --- GOAL NODE FIX: ensure goal IDs exist in current graph ---
    raw_goal_ids = []
    if goal_node_id is not None:
        if isinstance(goal_node_id, (list, tuple, set)):
            raw_goal_ids = [_normalize_node_id(g) for g in goal_node_id]
        else:
            raw_goal_ids = [_normalize_node_id(goal_node_id)]
    else:
        raw_goal_ids = []

    # 存在しないゴール ID を除外
    goal_ids = [gid for gid in raw_goal_ids if gid in G and gid in node_positions]

    goal_nodes = []
    goal_metadata = {}
    goal_candidates_payload = []

    for idx, cand in enumerate(goal_candidates):
        point = cand["point"]

        use_existing = idx < len(goal_ids)
        node_id = None

        if use_existing:
            gid = goal_ids[idx]
            if gid in G and gid in node_positions:
                node_id = gid
                snapped_point = Point(node_positions[gid])
            else:
                use_existing = False

        if not use_existing:
            node_ids_arr, node_coords_arr = _node_lookup_arrays(node_positions)
            snapped_id, snapped_point = _insert_point_on_edge(
                point, G, node_positions, node_ids_arr, node_coords_arr, split_edge_tracker
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

    def _run_single_planner(dlite_instance: DStarLite) -> Tuple[List[int], Optional[List[int]]]:
        if edge_updates:
            for upd in edge_updates:
                u, v = upd["u"], upd["v"]
                point_geom = upd.get("point")
                targets = _select_block_targets(u, v, split_edge_tracker, point_geom, G)
                point_str = (
                    f"({point_geom.x:.6f}, {point_geom.y:.6f})" if point_geom is not None else "None"
                )
                if upd["weight"] is not None:
                    weight_to_apply = upd["weight"]
                elif upd["blocked"]:
                    weight_to_apply = float("inf")
                else:
                    weight_to_apply = None

                for tu, tv in targets:
                    print(
                        f"[DEBUG] update_edge_cost: u={tu}, v={tv}, blocked={upd['blocked']}, weight={upd['weight']}, point={point_str}"
                    )
                    dlite_instance.update_edge_cost(tu, tv, weight_to_apply)

        start_compute = time.time()
        dlite_instance.compute_shortest_path()
        elapsed = time.time() - start_compute
        print(f"[D*Lite] goal {goal_node} compute 実行時間: {elapsed:.2f} 秒")
        print("[D*Lite] extract_path 開始")
        raw_route = dlite_instance.extract_path()
        print(f"[D*Lite] extract_path 結果: {raw_route[:10] if raw_route else raw_route}")
        route = _simplify_route_nodes(raw_route)
        return route, raw_route

    best_result: Optional[Dict[str, Any]] = None

    for idx, goal_node in enumerate(goal_nodes):
        goal_meta = goal_metadata.get(goal_node)
        dlite_state = initial_state if idx == 0 and initial_state else None
        state_candidates = [("incremental", dlite_state)] if dlite_state else []
        state_candidates.append(("fresh", None))

        route = None
        final_dlite = None

        for attempt_label, state_payload in state_candidates:
            dlite = DStarLite(G, start_id, [goal_node], node_positions, initial_state=state_payload)
            route, _ = _run_single_planner(dlite)
            final_dlite = dlite
            if route and dlite.get_reached_goal() == goal_node:
                break

            if attempt_label == "fresh":
                route = None
                break

            print("[ROUTE DEBUG] incremental計算で経路が得られなかったため、初期状態から再計算します。")

        reached_goal = final_dlite.get_reached_goal() if final_dlite else None
        if not route or reached_goal != goal_node:
            print(f"[D*Lite] goal {goal_node} route not found")
            continue

        total_dist = sum(
            float(G[route[i]][route[i + 1]]["weight"])
            for i in range(len(route) - 1)
            if math.isfinite(G[route[i]][route[i + 1]]["weight"])
        )
        print(f"[D*Lite] goal {goal_node} total_dist={total_dist}")

        if best_result is None or total_dist < best_result.get("distance", float("inf")):
            best_result = {
                "route": route,
                "distance": total_dist,
                "goal_id": goal_node,
                "dlite_state": final_dlite.export_state(),
                "goal_meta": goal_meta,
            }

    if not best_result:
        raise ValueError("到達可能なゴールがありません。")

    route = best_result["route"]
    total_dist = best_result["distance"]
    reached_goal_id = best_result["goal_id"]
    dlite_state = best_result["dlite_state"]
    route_coords = build_route_coords(route, G, node_positions)

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
        "blocked_edges": blocked_edges_payload,
        "dlite_state": dlite_state,
    }


if __name__ == "__main__":
    result = run_dlite_algorithm()
