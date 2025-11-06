import os
import osmnx as ox

# ======= 設定 =======
place_name = "Ube, Yamaguchi, Japan"
data_dir = os.path.join(os.path.dirname(__file__), "../data")
output_path = os.path.join(data_dir, "ube_pedestrian_roads.geojson")

print(f"📍 保存先: {output_path}")

# ======= OSMデータ取得 =======
G = ox.graph_from_place(place_name, network_type="walk", simplify=True)

# ======= GeoDataFrame化 =======
nodes, edges = ox.graph_to_gdfs(G)

# ======= 保存 =======
edges.to_file(output_path, driver="GeoJSON")

print("✅ 宇部市の歩行者道路データを取得・保存しました。")
print(f"ノード数: {len(nodes)}, エッジ数: {len(edges)}")
