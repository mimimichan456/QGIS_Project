# dlite_distance.py
import sys, os
sys.path.append("/Users/segawamizuto/QGIS_Project")

from scripts.QGIS.qgis_env import QgisSession
from scripts.QGIS.find_shelter import find_nearest_shelter
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
from scripts.QGIS.dlite_core import DStarLite
from scripts.QGIS.save_route import save_route_to_shapefile

def run_dlite_distance_only(project_path: str):
    with QgisSession() as qgs:
        project = QgsProject.instance()
        project.read(project_path)

        # --- 出発点とゴール ---
        res = find_nearest_shelter(project)
        start_point = res["start_point"]
        goal_point  = res["goal_point"]

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
            dlite = DStarLite(G, start_id, goal_id, node_positions)
            dlite.compute_shortest_path()
            route = dlite.extract_path()
            #最短ルートの距離を取得
            total_dist = nx.shortest_path_length(G, source=start_id, target=goal_id, weight="weight")
        #道路がリンクしていない場合終了    
        except nx.NetworkXNoPath:
            print("❌ No Path Found.")
            return None

        print(f"📏 距離: {total_dist:.2f} m")
        print(f"🛣️ ノード数: {len(route)}")

        # --- 結果を反映 ---
        return {
            "start": start_point,
            "goal": goal_point,
            "distance_m": total_dist,
            "route_nodes": route,
            "graph": G,
            "node_positions": node_positions,
            "edge_geom_map": edge_geom_map,
        }

    



#他のファイルから実行された場合は無視
if __name__ == "__main__":
    project_path = "/Users/segawamizuto/QGIS_Project/Ube_Project.qgz"
    result = run_dlite_distance_only(project_path)
    if not result:
        sys.exit("❌ 経路が見つかりませんでした")

    # グラフ保持
    G = result["graph"]
    node_positions = result["node_positions"]
    edge_geom_map = result["edge_geom_map"]
    start_id = result["route_nodes"][0]
    goal_id = result["route_nodes"][-1]

    # --- 初回D* Lite探索 ---
    dlite = DStarLite(G, start_id, goal_id, node_positions)
    dlite.compute_shortest_path()

    path = dlite.extract_path()
    if not path:
        print("❌ 経路が見つかりません")
        sys.exit()

    # --- 道路形状に沿った座標列を構築 ---
    route_coords = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        geom_line = result["graph"].edges[u, v].get("geom") if "geom" in result["graph"].edges[u, v] else None
        if not geom_line:
            geom_line = edge_geom_map.get((u, v))
        if not geom_line:
            continue

        if route_coords and (
            route_coords[-1][0] == geom_line[0].x() and route_coords[-1][1] == geom_line[0].y()
        ):
            geom_line = geom_line[1:]
        route_coords.extend(geom_line)

    result["route_nodes"] = path
    result["route_coords"] = [(p.x(), p.y()) for p in route_coords]


    # 初回経路保存
    save_route_to_shapefile(result)

    # --- 対話モード ---
    while True:
        cmd = input("\n>>> 通行止め道路を指定 (u v) / q: ").strip().lower()
        if cmd == "q":
            print("👋 終了します")
            break

        try:
            u, v = map(int, cmd.split())
        except ValueError:
            print("⚠️ 'u v' の形式で入力してください")
            continue

        if not G.has_edge(u, v):
            print("⚠️ その道路は存在しません")
            continue

        G[u][v]["weight"] = float("inf")
        print(f"🚧 通行止め設定: {u} → {v}")

        dlite.update_vertex(u)
        dlite.update_vertex(v)
        dlite.compute_shortest_path()

        path = dlite.extract_path()
        if not path:
            print("❌ 経路が見つかりません")
            continue

        # --- 道路形状を再構築して保存 ---
        route_coords = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            geom_line = edge_geom_map.get((u, v))
            if not geom_line:
                continue
            if route_coords and (
                route_coords[-1].x() == geom_line[0].x() and route_coords[-1].y() == geom_line[0].y()
            ):
                geom_line = geom_line[1:]
            route_coords.extend(geom_line)

        result["route_nodes"] = path
        result["route_coords"] = [(p.x(), p.y()) for p in route_coords]
        result["distance_m"] = sum(G[path[i]][path[i + 1]]["weight"] for i in range(len(path) - 1))

        print(f"📏 距離: {result['distance_m']:.2f} m")
        print(f"🛣️ ノード数: {len(path)}")

        save_route_to_shapefile(result)
