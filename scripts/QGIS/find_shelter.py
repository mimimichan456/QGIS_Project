import os
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data")

def _normalize_point(point):
    if isinstance(point, Point):
        return point.x, point.y
    if isinstance(point, (tuple, list)) and len(point) == 2:
        return float(point[0]), float(point[1])
    raise TypeError("start_point は (lon, lat) のタプル/リスト、または shapely.geometry.Point を指定してください。")


def find_nearest_shelter(
    start_point=None,
    univ_path=os.path.join(DATA_DIR, "raw/university/ube_university.shp"),
    shelter_path=os.path.join(DATA_DIR, "processed/shelters/ube_shelters.shp"),
):
    # --- シェープファイル読込 ---
    gdf_univ = gpd.read_file(univ_path)
    gdf_shelter = gpd.read_file(shelter_path)

    if gdf_univ.empty:
        raise ValueError("大学シェープファイルに地物がありません。")
    if gdf_shelter.empty:
        raise ValueError("避難所シェープファイルに地物がありません。")

    # --- 座標系統一 ---
    gdf_univ = gdf_univ.to_crs(epsg=4326)
    gdf_shelter = gdf_shelter.to_crs(epsg=4326)

    # --- 出発点 ---
    if start_point is None:
        start_geom = gdf_univ.geometry.iloc[0]
        start_point = (start_geom.x, start_geom.y)
    else:
        start_point = _normalize_point(start_point)

    # --- 最も近い避難所を探索 ---
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x0, y0 = transformer.transform(*start_point)

    min_dist = float("inf")
    goal_point = None
    goal_attr = None

    for _, row in gdf_shelter.iterrows():
        x, y = transformer.transform(row.geometry.x, row.geometry.y)
        dist = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            goal_point = (row.geometry.x, row.geometry.y)
            goal_attr = row.to_dict()

    if "geometry" in goal_attr:
        goal_attr.pop("geometry")

    return {
        "start_point": {"lon": start_point[0], "lat": start_point[1]},
        "goal_point": {"lon": goal_point[0], "lat": goal_point[1]},
        "shelter_attr": goal_attr,
    }