import geopandas as gpd
from typing import Dict, Optional
from shapely.geometry import LineString, MultiLineString, Point

DEFAULT_ROAD_PATH = "/Users/segawamizuto/QGIS_Project/data/processed/roads/ube_roads.shp"

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

def find_nearest_road_edge(point, loads_path: str = DEFAULT_ROAD_PATH) -> Dict:
    """
    任意の点から最も近い道路エッジ(u, v)を探索し、属性付きで返却。
    戻り値例:
        {
            "edge": (u, v),
            "distance": float,
            "geometry": [[x, y], ...],
            "properties": {...}
        }
    """
    target_point = _ensure_point(point)
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
