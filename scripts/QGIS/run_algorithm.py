import sys
import os
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

from scripts.QGIS.find_shelter import find_nearest_shelter
from scripts.QGIS.dlite_algorithm import DStarLite
# from scripts.QGIS.save_route import save_route_to_shapefile


def _ensure_point(point):
    """さまざまな形式を shapely.geometry.Point に統一"""
    if isinstance(point, Point):
        return point
    if isinstance(point, (tuple, list)) and len(point) == 2:
        return Point(float(point[0]), float(point[1]))
    if isinstance(point, dict) and {"lon", "lat"} <= set(point):
        return Point(float(point["lon"]), float(point["lat"]))
    raise TypeError("Point must be shapely Point or (lon, lat).")


def _normalize_edges(edges):
    """(u,v)ペアの重複を排除して整形"""
    normalized = []
    for edge in edges or []:
        if isinstance(edge, dict):
            u, v = edge.get("u"), edge.get("v")
        else:
            u, v = edge
        if u is None or v is None:
            continue
        pair = (int(u), int(v))
        if pair not in normalized:
            normalized.append(pair)
    return normalized


def _nearest_node(point, node_positions):
    px, py = point.x, point.y
    ids = np.array(list(node_positions.keys()))
    coords = np.array(list(node_positions.values()))
    dists = np.linalg.norm(coords - np.array([px, py]), axis=1)
    return int(ids[np.argmin(dists)])


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

    # --- 道路レイヤ読込 ---
    try:
        try:
            roads = gpd.read_file(loads_path, usecols=["geometry", "u", "v", "length"])
        except Exception:
            roads = gpd.read_file(loads_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"道路データが存在しません: {loads_path}")
    except Exception as e:
        raise RuntimeError(f"道路データの読込に失敗しました: {e}")

    # --- グラフ構築 ---
    G = nx.Graph()
    node_positions = {}

    for _, f in roads.iterrows():
        if f.geometry is None:
            continue
        u, v = f.get("u"), f.get("v")
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
        G.add_edge(u, v, weight=f.get("length", geom.length), geometry=coords)

    if not node_positions:
        raise ValueError("道路レイヤに有効なノードがありません。")

    # --- 出発点・到着点を最寄りノードへスナップ ---
    if start_node_id is not None:
        start_id = int(start_node_id)
        if start_id not in node_positions:
            raise ValueError("指定した start_id が存在しません。")
        if start_point is None:
            sx, sy = node_positions[start_id]
            start_point = Point(sx, sy)
    else:
        start_id = _nearest_node(start_point, node_positions)

    if goal_node_id is not None:
        goal_id = int(goal_node_id)
        if goal_id not in node_positions:
            raise ValueError("指定した goal_id が存在しません。")
        if goal_point is None:
            gx, gy = node_positions[goal_id]
            goal_point = Point(gx, gy)
    else:
        goal_id = _nearest_node(goal_point, node_positions)

    if start_id is None or goal_id is None:
        raise ValueError("出発点 / 到着点をスナップできませんでした。")

    # --- 通行止め適用 ---
    if blocked_edges:
        for u, v in blocked_edges:
            if G.has_edge(u, v):
                G[u][v]["weight"] = float("inf")

    # --- D* Lite 実行 ---
    try:
        dlite = DStarLite(G, start_id, goal_id, node_positions, initial_state=initial_state)
        if new_blocked_edges:
            for u, v in new_blocked_edges:
                dlite.update_vertex(u)
                dlite.update_vertex(v)
        dlite.compute_shortest_path()
        route = dlite.extract_path() or []
        if not route:
            raise ValueError("経路が見つかりません。")

        total_dist = sum(
            float(G[route[i]][route[i + 1]]["weight"])
            for i in range(len(route) - 1)
            if np.isfinite(G[route[i]][route[i + 1]]["weight"])
        )
    except nx.NetworkXNoPath:
        print("❌ No Path Found.")
        return None
    except Exception as e:
        raise RuntimeError(f"D* Lite 実行中にエラー発生: {e}")

    # --- 座標列を構築 ---
    route_coords = build_route_coords(route, G)

    print(f"📏 距離: {total_dist:.2f} m")
    print(f"🛣️ ノード数: {len(route)}")

    return {
        "start": start_point,
        "goal": goal_point,
        "distance_m": float(total_dist),
        "route_nodes": route,
        "route_coords": route_coords,
        "start_id": int(start_id),
        "goal_id": int(goal_id),
        "blocked_edges": [{"u": int(u), "v": int(v)} for u, v in (blocked_edges or [])],
        "dlite_state": dlite.export_state(),
    }


def build_route_coords(path, graph):
    """ノード列から座標列を構築"""
    route_coords = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        geom_line = graph[u][v].get("geometry")
        if not geom_line:
            continue
        if route_coords and (route_coords[-1] == geom_line[0]):
            geom_line = geom_line[1:]
        route_coords.extend(geom_line)
    return route_coords


if __name__ == "__main__":
    try:
        result = run_dlite_algorithm()
        if not result:
            sys.exit("❌ 経路が見つかりませんでした")
    except Exception as e:
        sys.exit(f"❌ エラー発生: {e}")

    # save_route_to_shapefile(result)