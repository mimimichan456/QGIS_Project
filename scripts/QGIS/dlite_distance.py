# dlite_distance.py
import sys, os
sys.path.append("/Users/segawamizuto/QGIS_Project")

from scripts.QGIS.qgis_env import QgisSession
from scripts.QGIS.nearest_shelter import find_nearest_shelter
import networkx as nx
from math import hypot
from qgis.core import (
    QgsVectorLayer,
    QgsFeature,
    QgsGeometry,
    QgsField,
    QgsFields,
    QgsProject,
    QgsPointXY,
    QgsVectorFileWriter,
    QgsCoordinateTransformContext,
    QgsWkbTypes,
    QgsCoordinateReferenceSystem
)
from qgis.PyQt.QtCore import QVariant


def run_dlite_distance_only(project_path: str):
    """
    D* Lite（距離のみ）版（実質 Dijkstra）歩行者専用。
    """
    import networkx as nx
    from math import hypot
    from qgis.core import QgsProject

    with QgisSession() as qgs:
        project = QgsProject.instance()
        project.read(project_path)
        print("✅ QGIS Project Loaded")

        # --- 出発点とゴール ---
        res = find_nearest_shelter(project)
        start_point = res["start_point"]
        goal_point  = res["goal_point"]
        print(f"🏫 Start: {start_point}")
        print(f"🏁 Goal:  {goal_point}")

        # --- 道路レイヤ ---
        roads = project.mapLayersByName("ube_loads")[0]

        # --- グラフ構築 ---
        G = nx.Graph()
        node_positions = {}
        edge_geom_map  = {}

        for f in roads.getFeatures():
            #ノードの取得
            u, v = f["u"], f["v"]
            # ノードが欠けている場合はスキップ
            if u is None or v is None:
                continue

            # 道路の形状取得    
            geom = f.geometry()
            #交差点等の場合
            if geom.isMultipart():
                lines = geom.asMultiPolyline()
                #リンクが欠けている場合はスキップ
                if not lines:
                    continue
                #余計な分岐を排除するため最初の線分のみ取得
                line = lines[0]
            #直線の場合
            else:
                line = geom.asPolyline()
            #ノードが欠けている場合はスキップ    
            if not line or len(line) < 2:
                continue

            # 距離を取得、属性になければ道路の形状から計算
            length = float(f["length"]) if f["length"] else geom.length()

            # u → 始点, v → 終点 としてxy座標に登録
            if u not in node_positions:
                node_positions[u] = (line[0].x(), line[0].y())
            if v not in node_positions:
                node_positions[v] = (line[-1].x(), line[-1].y())

            # 道路の形状登録
            edge_geom_map[(u, v)] = line
            edge_geom_map[(v, u)] = list(reversed(line))

            # 双方向で追加
            G.add_edge(u, v, weight=length)
            G.add_edge(v, u, weight=length)

        # --- 出発点・到着点を最寄りノードへスナップ ---
        def nearest_node(point):
            px, py = point.x(), point.y()
            best, best_id = float("inf"), None
            #各ノードとの平方根距離を計算して最小値を探索
            for node_id, (x, y) in node_positions.items():
                d = (px - x)**2 + (py - y)**2
                if d < best:
                    best, best_id = d, node_id
            return best_id

        start_id = nearest_node(start_point)
        goal_id  = nearest_node(goal_point)

        # --- 最短経路探索 ---
        try:
            #最短ルートのノード順を取得
            route = nx.shortest_path(G, source=start_id, target=goal_id, weight="weight")
            #最短ルートの距離を取得
            total_dist = nx.shortest_path_length(G, source=start_id, target=goal_id, weight="weight")
        #道路がリンクしていない場合終了    
        except nx.NetworkXNoPath:
            print("❌ No Path Found.")
            return None

        print(f"📏 Total Distance: {total_dist:.2f} m")
        print(f"🛣️ Route Node Count: {len(route)}")

        # --- QGISで描ける道路形状に変換 ---
        route_coords = []
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            geom_line = edge_geom_map.get((u, v))
            if not geom_line:
                continue

            #重複除去（前終点＝次始点なら1点削除）
            if route_coords and (
                route_coords[-1].x() == geom_line[0].x() and route_coords[-1].y() == geom_line[0].y()
            ):
                geom_line = geom_line[1:]

            route_coords.extend(geom_line)

        return {
            "start": start_point,
            "goal": goal_point,
            "distance_m": total_dist,
            "route_nodes": route,
            "route_coords": [(p.x(), p.y()) for p in route_coords],
        }
    



if __name__ == "__main__":
    result = run_dlite_distance_only("/Users/segawamizuto/QGIS_Project/Ube_Project.qgz")
    if not result:
        sys.exit("❌ 経路が見つかりませんでした")

    # --- 出力ファイル作成 ---
    route_points = [QgsPointXY(x, y) for x, y in result["route_coords"]]
    route_geom = QgsGeometry.fromPolylineXY(route_points)

    crs = QgsCoordinateReferenceSystem("EPSG:6668")  

    output_path = "/Users/segawamizuto/QGIS_Project/data/route/Dlite_Route.shp"

    # --- 属性定義 ---
    fields = QgsFields()
    f1 = QgsField()
    f1.setName("distance_m")
    f1.setType(QVariant.Double)
    f2 = QgsField()
    f2.setName("node_count")
    f2.setType(QVariant.Int)
    fields.append(f1)
    fields.append(f2)

    feat = QgsFeature()
    feat.setGeometry(route_geom)
    feat.setAttributes([result["distance_m"], len(result["route_nodes"])])

    # --- 出力オプション ---
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "ESRI Shapefile"
    options.fileEncoding = "UTF-8"
    options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile

    # ✅ CRSを明示的に指定
    writer = QgsVectorFileWriter.create(
        output_path,
        fields,
        QgsWkbTypes.LineString,
        crs,  # ← ここで指定
        QgsCoordinateTransformContext(),
        options
    )
    writer.addFeature(feat)
    del writer  # 保存を確定

    print(f"💾 ルートを上書き保存しました（CRS: {crs.authid()}） → {output_path}")
