import os
import geopandas as gpd
from typing import Dict, Optional
from shapely.geometry import LineString, MultiLineString, Point

BASE_DIR = os.path.dirname(os.path.abspath(__file__))         
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../")) 
DATA_DIR = os.path.join(PROJECT_ROOT, "data")                   

def _ensure_point(point) -> Point:
    if isinstance(point, Point):
        return point
    if isinstance(point, (tuple, list)) and len(point) == 2:
        return Point(float(point[0]), float(point[1]))
    if isinstance(point, dict) and {"lon", "lat"} <= set(point):
        return Point(float(point["lon"]), float(point["lat"]))
    raise TypeError("point には (lon, lat) or shapely.geometry.Point を指定してください。")

def _extract_line_coords(geom: LineString) -> list:
    return list(geom.coords)

def _line_for_distance(geom, target_point: Point):
    if isinstance(geom, MultiLineString):
        lines = list(geom.geoms)
        if not lines:
            return None
        return min(lines, key=lambda g: g.distance(target_point))
    if isinstance(geom, LineString):
        return geom
    return None

def find_nearest_road_edge(
    point,
    loads_path: str = os.path.join(DATA_DIR, "processed/roads/ube_roads.shp")
) -> Dict:
    """
    指定した点（lon, lat）に最も近い道路エッジを検索する関数。
    絶対パス不要でRender環境でも動作可能。
    """
    target_point = _ensure_point(point)

    # --- データ読込 ---
    if not os.path.exists(loads_path):
        raise FileNotFoundError(f"道路データが見つかりません: {loads_path}")

    roads = gpd.read_file(loads_path)

    best: Optional[Dict] = None
    best_dist = float("inf")

    for _, row in roads.iterrows():
        u, v = row.get("u"), row.get("v")
        if u is None or v is None or row.geometry is None:
            continue

        geom = _line_for_distance(row.geometry, target_point)
        if geom is None:
            continue

        dist = geom.distance(target_point)
        if dist < best_dist:
            best_dist = dist
            properties = row.drop(labels=["geometry"]).to_dict()
            best = {
                "edge": (int(u), int(v)),
                "distance": dist,
                "geometry": _extract_line_coords(geom),
                "properties": properties,
            }

    if not best:
        raise ValueError("該当する道路エッジが見つかりませんでした。")

    return best