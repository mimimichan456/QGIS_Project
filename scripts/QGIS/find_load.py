import os
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Point

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

#LineString → [[x, y], …] の座標配列に変換
def _extract_line_coords(geom: LineString) -> list:
    return list(geom.coords)

#MultiLineString のときに 一番近い道路を選ぶ
def _line_for_distance(geom, target_point: Point):
    if isinstance(geom, MultiLineString):
        lines = list(geom.geoms)
        if not lines:
            return None
        return min(lines, key=lambda g: g.distance(target_point))

    if isinstance(geom, LineString):
        return geom

    return None

# 指定した座標に最も近い道路を返す
def find_nearest_road_edge(
    point,
    loads_path: str = os.path.join(DATA_DIR, "processed/roads/ube_roads.shp")
):
    # --- 入力位置を Point に正規化 ---
    target_point = _ensure_point(point)

    # --- 道路レイヤを読み込み、緯度経度へ統一 ---
    roads = gpd.read_file(loads_path)
    roads = roads[roads.geometry.notnull()].to_crs(epsg=4326)

    best_row = None
    best_geom = None
    best_dist = float("inf")

    # --- 全ての道路について距離計算し、最短の線分を保持 ---
    for _, row in roads.iterrows():
        geom = _line_for_distance(row.geometry, target_point)
        if geom is None:
            continue

        dist = geom.distance(target_point)

        if dist < best_dist:
            best_dist = dist
            best_row = row
            best_geom = geom

    u, v = best_row.get("u"), best_row.get("v")

    properties = {
        k: v
        for k, v in best_row.drop(labels=["geometry"]).to_dict().items()
    }

    # --- 最近接エッジ情報を整形して返却 ---
    return {
        "edge": (int(u), int(v)),
        "distance": float(best_dist),
        "geometry": _extract_line_coords(best_geom),
        "properties": properties,
    }
