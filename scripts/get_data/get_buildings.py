import os
import osmnx as ox

# ======= 設定 =======
place_name = "Ube, Yamaguchi, Japan"
data_dir = os.path.join(os.path.dirname(__file__), "../data")
os.makedirs(data_dir, exist_ok=True)
output_path = os.path.join(data_dir, "ube_buildings.geojson")

print(f"📍 保存先: {output_path}")

# ======= OSM 建物データ取得 =======
# buildingタグを持つポリゴンを取得
buildings = ox.features_from_place(place_name, tags={"building": True})

# ======= 保存 =======
buildings.to_file(output_path, driver="GeoJSON")

print("✅ 宇部市の建物データを取得・保存しました。")
print(f"建物数: {len(buildings)}")
