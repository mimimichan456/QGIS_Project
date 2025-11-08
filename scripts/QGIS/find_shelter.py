# nearest_shelter.py
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer

def find_nearest_shelter(
    univ_path="/Users/segawamizuto/QGIS_Project/data/raw/university/ube_university.shp",
    shelter_path="/Users/segawamizuto/QGIS_Project/data/processed/shelters/ube_shelters.shp"
):
    # --- シェープファイル読込 ---
    gdf_univ = gpd.read_file(univ_path)
    gdf_shelter = gpd.read_file(shelter_path)

    # --- 座標系統一（EPSG:4326に変換） ---
    gdf_univ = gdf_univ.to_crs(epsg=4326)
    gdf_shelter = gdf_shelter.to_crs(epsg=4326)

    # --- 出発点（大学） ---
    start_geom = gdf_univ.geometry.iloc[0]
    start_point = (start_geom.x, start_geom.y)

    # --- 最も近い避難所を探索 ---
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x0, y0 = transformer.transform(*start_point)

    min_dist = float("inf")
    goal_point = None
    goal_attr = None

    for _, row in gdf_shelter.iterrows():
        x, y = transformer.transform(row.geometry.x, row.geometry.y)
        dist = ((x - x0)**2 + (y - y0)**2)**0.5  # 平面距離（メートル換算）
        if dist < min_dist:
            min_dist = dist
            goal_point = (row.geometry.x, row.geometry.y)
            goal_attr = row.to_dict()

    # --- 結果 ---
    return {
        "start_point": Point(start_point),
        "goal_point": Point(goal_point),
        "distance_m": min_dist,
        "shelter_attr": goal_attr
    }
