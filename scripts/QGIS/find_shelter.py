import os
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
from pyproj import Transformer
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data")

def _normalize_point(point) -> Point:
    """入力を shapely Point に統一"""
    if isinstance(point, Point):
        return point
    if isinstance(point, (tuple, list)) and len(point) == 2:
        return Point(float(point[0]), float(point[1]))
    if isinstance(point, dict) and {"lon", "lat"} <= set(point):
        return Point(float(point["lon"]), float(point["lat"]))
    raise TypeError("start_point は (lon, lat) または {'lon':, 'lat':} 形式で指定してください。")

def find_nearest_shelter(
    start_point=None,
    univ_path=os.path.join(DATA_DIR, "raw/university/ube_university.shp"),
    shelter_path=os.path.join(DATA_DIR, "processed/shelters/ube_shelters.shp"),
):
    # --- ファイル存在チェック ---
    if not os.path.exists(univ_path):
        raise FileNotFoundError(f"大学データが見つかりません: {univ_path}")
    if not os.path.exists(shelter_path):
        raise FileNotFoundError(f"避難所データが見つかりません: {shelter_path}")

    # --- シェープファイル読込 ---
    gdf_univ = gpd.read_file(univ_path)
    gdf_shelter = gpd.read_file(shelter_path)

    if gdf_univ.empty:
        raise ValueError("大学シェープファイルに地物がありません。")
    if gdf_shelter.empty:
        raise ValueError("避難所シェープファイルに地物がありません。")

    # --- 座標系を統一 (EPSG:4326) ---
    gdf_univ = gdf_univ.to_crs(epsg=4326)
    gdf_shelter = gdf_shelter.to_crs(epsg=4326)

    # --- 出発点 ---
    if start_point is None:
        start_point = gdf_univ.geometry.iloc[0]
    else:
        start_point = _normalize_point(start_point)

    # --- 距離計算は平面上で実施（EPSG:3857）---
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    start_x, start_y = transformer.transform(start_point.x, start_point.y)

    # 投影した避難所座標をNumPy配列化
    shelter_coords = gdf_shelter.geometry.apply(
        lambda g: transformer.transform(g.x, g.y)
    ).to_list()

    # --- ベクトル化距離計算（最速） ---
    pts = np.array(shelter_coords)
    if pts.size == 0:
        raise ValueError("有効な避難所座標が存在しません。")

    dists = np.sqrt((pts[:, 0] - start_x) ** 2 + (pts[:, 1] - start_y) ** 2)
    if not np.isfinite(dists).any():
        raise ValueError("距離計算に失敗しました（座標に欠損が含まれる可能性があります）。")

    idx = int(np.argmin(dists))
    nearest_row = gdf_shelter.iloc[idx]
    goal_geom = nearest_row.geometry

    goal_attr = {
        k: (v.item() if isinstance(v, (np.generic,)) else v)
        for k, v in nearest_row.drop(labels=["geometry"]).to_dict().items()
    }

    # --- 結果構築 ---
    return {
        "start_point": {"lon": start_point.x, "lat": start_point.y},
        "goal_point": {"lon": goal_geom.x, "lat": goal_geom.y},
        "shelter_attr": goal_attr,
    }