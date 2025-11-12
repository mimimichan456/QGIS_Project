import sys
import os
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
sys.path.append(PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

from scripts.QGIS.find_shelter import find_nearest_shelter
from scripts.QGIS.dlite_algorithm import DStarLite
from scripts.QGIS.save_route import save_route_to_shapefile


def _ensure_point(point):
    if isinstance(point, Point):
        return point
    if isinstance(point, (tuple, list)) and len(point) == 2:
        return Point(float(point[0]), float(point[1]))
    if isinstance(point, dict) and {"lon", "lat"} <= set(point):
        return Point(float(point["lon"]), float(point["lat"]))
    raise TypeError("Point must be shapely Point or (lon, lat).")


def _normalize_edges(edges):
    normalized = []
    seen = set()
    for edge in edges or []:
        if isinstance(edge, dict):
            u, v = edge.get("u"), edge.get("v")
        else:
            u, v = edge
        if u is None or v is None:
            continue
        pair = (int(u), int(v))
        if pair in seen:
            continue
        seen.add(pair)
        normalized.append(pair)
    return normalized


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
    # --- 出発点とゴール ---
    if start_point is None or goal_point is None:
        res = find_nearest_shelter()
        start_point = res["start_point"]
        goal_point = res["goal_point"]
    else:
        start_point = _ensure_point(start_point)
        goal_point = _ensure_point(goal_point)

    blocked_edges = _normalize_edges(blocked_edges)
    new_blocked_edges = _normalize_edges(new_blocked_edges)

    # --- 道路レイヤ ---
    roads = gpd.read_file(loads_path)

    # --- グラフ構築 ---
    G = nx.Graph()
    node_positions = {}
    edge_geom_map = {}

    for _, f in roads.iterrows():
        u, v = f["u"], f["v"]
        if u is None or v is None:
            continue

        geom = f.geometry

        if geom.geom_type == "MultiLineString":
            lines = list(geom.geoms)
            if not lines:
                continue
            line = lines[0].coords[:]
        else:
            line = list(geom.coords)

        if not line or len(line) < 2:
            continue

        length = float(f["length"]) if "length" in f and f["length"] else geom.length

        if u not in node_positions:
            node_positions[u] = line[0]
        if v not in node_positions:
            node_positions[v] = line[-1]

        edge_geom_map[(u, v)] = line
        edge_geom_map[(v, u)] = list(reversed(line))

        G.add_edge(u, v, weight=length)
        G.add_edge(v, u, weight=length)

    if not node_positions:
        raise ValueError("道路レイヤに有効なノードがありません。")

    # --- 出発点・到着点を最寄りノードへスナップ ---
    def nearest_node(point):
        px, py = point.x, point.y
        best, best_id = float("inf"), None
        for node_id, (x, y) in node_positions.items():
            d = (px - x)**2 + (py - y)**2
            if d < best:
                best, best_id = d, node_id
        return best_id

    if start_node_id is not None:
        start_id = int(start_node_id)
        if start_id not in node_positions:
            raise ValueError("指定した start_id が道路ネットワークに存在しません。")
        if start_point is None:
            sx, sy = node_positions[start_id]
            start_point = Point(sx, sy)
    else:
        start_id = nearest_node(start_point)

    if goal_node_id is not None:
        goal_id = int(goal_node_id)
        if goal_id not in node_positions:
            raise ValueError("指定した goal_id が道路ネットワークに存在しません。")
        if goal_point is None:
            gx, gy = node_positions[goal_id]
            goal_point = Point(gx, gy)
    else:
        goal_id = nearest_node(goal_point)

    if start_id is None or goal_id is None:
        raise ValueError("出発点 / 到着点を道路ネットワークにスナップできませんでした。")

    # --- 通行止め適用 ---
    if blocked_edges:
        for u, v in blocked_edges:
            if G.has_edge(u, v):
                G[u][v]["weight"] = float("inf")
            if G.has_edge(v, u):
                G[v][u]["weight"] = float("inf")

    # --- D* Lite 探索 ---
    try:
        dlite = DStarLite(G, start_id, goal_id, node_positions, initial_state=initial_state)
        if new_blocked_edges:
            for u, v in new_blocked_edges:
                dlite.update_vertex(u)
                dlite.update_vertex(v)
        dlite.compute_shortest_path()
        route = dlite.extract_path()
        total_dist = nx.shortest_path_length(G, source=start_id, target=goal_id, weight="weight")
    except nx.NetworkXNoPath:
        print("❌ No Path Found.")
        return None

    route_coords = build_route_coords(route, edge_geom_map)

    print(f"📏 距離: {total_dist:.2f} m")
    print(f"🛣️ ノード数: {len(route)}")

    return {
        "start": start_point,
        "goal": goal_point,
        "distance_m": total_dist,
        "route_nodes": route,
        "graph": G,
        "node_positions": node_positions,
        "edge_geom_map": edge_geom_map,
        "route_coords": route_coords,
        "start_id": start_id,
        "goal_id": goal_id,
        "blocked_edges": [{"u": u, "v": v} for u, v in blocked_edges],
        "dlite_state": dlite.export_state(),
    }


def build_route_coords(path, edge_geom_map):
    """ノード列 path から座標列を構築する"""
    route_coords = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        geom_line = edge_geom_map.get((u, v))
        if not geom_line:
            continue
        if route_coords and (
            route_coords[-1][0] == geom_line[0][0]
            and route_coords[-1][1] == geom_line[0][1]
        ):
            geom_line = geom_line[1:]
        route_coords.extend(geom_line)
    return route_coords


if __name__ == "__main__":
    result = run_dlite_algorithm()
    if not result:
        sys.exit("❌ 経路が見つかりませんでした")

    save_route_to_shapefile(result)