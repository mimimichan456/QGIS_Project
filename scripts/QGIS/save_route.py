from qgis.core import (
    QgsPointXY, QgsGeometry, QgsFields, QgsField,
    QgsFeature, QgsVectorFileWriter, QgsCoordinateReferenceSystem,
    QgsCoordinateTransformContext, QgsWkbTypes
)
from qgis.PyQt.QtCore import QVariant

def save_route_to_shapefile(result, output_path="/Users/segawamizuto/QGIS_Project/data/route/Dlite_Route.shp"):

    route_points = [QgsPointXY(x, y) for x, y in result["route_coords"]]
    route_geom = QgsGeometry.fromPolylineXY(route_points)

    # --- 座標系 ---
    crs = QgsCoordinateReferenceSystem("EPSG:6668")

    # --- 属性定義 ---
    fields = QgsFields()
    f1 = QgsField("distance_m", QVariant.Double)
    f2 = QgsField("node_count", QVariant.Int)
    fields.append(f1)
    fields.append(f2)

    # --- フィーチャ作成 ---
    feat = QgsFeature()
    feat.setGeometry(route_geom)
    feat.setAttributes([result["distance_m"], len(result["route_nodes"])])

    # --- 出力オプション ---
    options = QgsVectorFileWriter.SaveVectorOptions()
    options.driverName = "ESRI Shapefile"
    options.fileEncoding = "UTF-8"
    options.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile

    writer = QgsVectorFileWriter.create(
        output_path,
        fields,
        QgsWkbTypes.LineString,
        crs,
        QgsCoordinateTransformContext(),
        options
    )
    writer.addFeature(feat)
    del writer  # 保存確定

    print(f"💾 経路を保存しました")

