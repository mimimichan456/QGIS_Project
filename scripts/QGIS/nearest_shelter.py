# nearest_shelter.py
from qgis.core import (
    QgsProject,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsDistanceArea,
    QgsPointXY
)


def find_nearest_shelter(project_or_path, univ_name="ube_university", shelter_name="ube_shelters"):
    """
    大学から最寄り避難所を探索する関数

    Args:
        project_or_path (str or QgsProject): 
            QGISプロジェクト(.qgz)のパス もしくは QgsProject インスタンス
        univ_name (str): 大学レイヤ名
        shelter_name (str): 避難所レイヤ名

    Returns:
        dict: {
            'start_point': QgsPointXY,
            'goal_point': QgsPointXY,
            'distance_m': float,
            'shelter_attr': list
        }
    """

    # ✅ もし文字列パスが渡された場合のみプロジェクトを開く
    if isinstance(project_or_path, str):
        project = QgsProject.instance()
        project.read(project_or_path)
    else:
        project = project_or_path  # すでに開かれたプロジェクトを利用

    # --- レイヤ取得 ---
    univ_layer = project.mapLayersByName(univ_name)[0]
    shelter_layer = project.mapLayersByName(shelter_name)[0]

    # --- 座標系をWGS84に統一 ---
    to_wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
    transform_univ = QgsCoordinateTransform(univ_layer.crs(), to_wgs84, project)
    transform_shelter = QgsCoordinateTransform(shelter_layer.crs(), to_wgs84, project)

    # --- 出発点（大学） ---
    univ_feat = next(univ_layer.getFeatures())
    start_point = transform_univ.transform(univ_feat.geometry().asPoint())

    # --- 最も近い避難所を探索 ---
    distance_calc = QgsDistanceArea()
    distance_calc.setEllipsoid('WGS84')

    min_dist = float("inf")
    goal_point = None
    goal_attr = None

    for f in shelter_layer.getFeatures():
        p = transform_shelter.transform(f.geometry().asPoint())
        dist = distance_calc.measureLine(QgsPointXY(start_point), QgsPointXY(p))
        if dist < min_dist:
            min_dist = dist
            goal_point = p
            goal_attr = f.attributes()

    return {
        "start_point": start_point,
        "goal_point": goal_point,
        "distance_m": min_dist,
        "shelter_attr": goal_attr
    }
