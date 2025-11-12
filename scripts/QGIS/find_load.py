import os
import geopandas as gpd
from typing import Dict, Optional
from shapely.geometry import LineString, MultiLineString, Point

# --- パス設定 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def _ensure_point(point) -> Point:
    """(lon, lat) / dict / Point を shapely.geometry.Point に統一"""
    if isinstance(point, Point):
        return point
    if isinstance(point, (tuple, list)) and len(point) == 2:
        return Point(float(point[0]), float(point[1]))
    if isinstance(point, dict) and {"lon", "lat"} <= set(point):
        return Point(float(point["lon"]), float(point["lat"]))
    raise TypeError("point には (lon, lat) or shapely.geometry.Point を指定してください。")


def _extract_line_coords(geom: LineString) -> list:
    """LineString → [[x, y], ...]"""
    return list(geom.coords)


def _line_for_distance(geom, target_point: Point):
    """MultiLineString対応 — 対象点に最も近い線分を選択"""
    if isinstance(geom, MultiLineString):
        lines = list(geom.geoms)
        if not lines:
            return None
        # 対象点に最も近い線分を選ぶ
        return min(lines, key=lambda g: g.distance(target_point))
    if isinstance(geom, LineString):
        return geom
    return None


def find_nearest_road_edge(
    point,
    loads_path: str = os.path.join(DATA_DIR, "processed/roads/ube_roads.shp")
) -> Dict:
    target_point = _ensure_point(point)

    if not os.path.exists(loads_path):
        raise FileNotFoundError(f"道路データが見つかりません: {loads_path}")

    roads = gpd.read_file(loads_path)
    if roads.empty:
        raise ValueError("道路データが空です。")

    roads = roads[roads.geometry.notnull()].to_crs(epsg=4326)

    best_row = None
    best_geom = None
    best_dist = float("inf")

    for _, row in roads.iterrows():
        geom = _line_for_distance(row.geometry, target_point)
        if geom is None:
            continue

        dist = geom.distance(target_point)
        if dist < best_dist:
            best_dist = dist
            best_row = row
            best_geom = geom

    if best_row is None or best_geom is None:
        raise ValueError("該当する道路エッジが見つかりませんでした。")

    u, v = best_row.get("u"), best_row.get("v")
    if u is None or v is None:
        raise ValueError("ノードID (u, v) が欠落しています。")

    properties = best_row.drop(labels=["geometry"]).to_dict()

    return {
        "edge": (int(u), int(v)),
        "distance": best_dist,
        "geometry": _extract_line_coords(best_geom),
        "properties": properties,
    }