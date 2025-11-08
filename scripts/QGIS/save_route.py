# save_route_to_shapefile_gpd.py
import geopandas as gpd
from shapely.geometry import LineString

def save_route_to_shapefile(
    result,
    output_path="/Users/segawamizuto/QGIS_Project/data/route/Dlite_Route.shp",
):
    # --- LineStringジオメトリ作成 ---
    route_line = LineString(result["route_coords"])

    # --- GeoDataFrame作成 ---
    gdf = gpd.GeoDataFrame(
        [{
            "distance_m": result["distance_m"],
            "node_count": len(result["route_nodes"]),
            "geometry": route_line,
        }],
        crs="EPSG:6668"
    )

    # --- Shapefile保存 ---
    gdf.to_file(output_path, driver="ESRI Shapefile", encoding="utf-8")

    print(f"💾 経路を保存しました: {output_path}")
