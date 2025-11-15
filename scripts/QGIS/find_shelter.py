import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point

# プロジェクト内のデータ配置を把握するための基本パス
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


def find_nearest_shelter(
    start_point=None,
    univ_path=os.path.join(DATA_DIR, "raw/university/ube_university.shp"),
    shelter_path=os.path.join(DATA_DIR, "processed/shelters/ube_shelters.shp"),
    metric_crs: str = "EPSG:6677",
    top_k: int = 3,
):
    # --- シェープファイル読込（大学と避難所の位置） ---
    gdf_univ = gpd.read_file(univ_path)
    gdf_shelter = gpd.read_file(shelter_path)

    # --- 距離計算用にメートル系へ変換　---
    gdf_univ_metric = gdf_univ.to_crs(metric_crs)
    gdf_shelter_metric = gdf_shelter.to_crs(metric_crs)

    # --- 出発地点設定（指定がなければ大学の最初の点） ---
    if start_point is None:
        start_point_metric = gdf_univ_metric.geometry.iloc[0] # 大学の点を地物に
    else:
        start_point_metric = (
            gpd.GeoSeries([_normalize_point(start_point)], crs="EPSG:4326")
            .to_crs(metric_crs)
            .iloc[0]
        )

    # --- 避難所からの距離を計算し近い順にソート ---
    shelter_coords = np.column_stack(
        (gdf_shelter_metric.geometry.x.values, gdf_shelter_metric.geometry.y.values)
    )

    target_vec = np.array([start_point_metric.x, start_point_metric.y])
    deltas = shelter_coords - target_vec
    dist_sq = np.einsum("ij,ij->i", deltas, deltas)
    order = np.argsort(dist_sq)

    #top_k個の避難所に絞る
    if top_k > 0:
        order = order[: min(top_k, len(order))]

    # --- 避難所を配列化 ---
    candidates = []
    for idx in order:
        idx = int(idx)
        row_metric = gdf_shelter_metric.iloc[idx]

        row_wgs = (
            gpd.GeoSeries([row_metric.geometry], crs=metric_crs)
            .to_crs("EPSG:4326")
            .iloc[0]
        )

        attrs = {
            k: v
            for k, v in gdf_shelter.iloc[idx].drop(labels=["geometry"]).to_dict().items()
        }

        candidates.append(
            {
                "goal_point": {"lon": row_wgs.x, "lat": row_wgs.y},
                "shelter_attr": attrs,
                "distance_m": float(dist_sq[idx] ** 0.5),
            }
        )

    # --- 出発地点を WGS84 に戻す ---
    start_point_wgs = (
        gpd.GeoSeries([start_point_metric], crs=metric_crs)
        .to_crs("EPSG:4326")
        .iloc[0]
    )

    # --- 結果を最寄り候補 + 候補一覧で返す ---
    return {
        "start_point": {"lon": start_point_wgs.x, "lat": start_point_wgs.y},
        "candidate_shelters": candidates,
    }
