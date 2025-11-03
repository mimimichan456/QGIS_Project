import sys, os
sys.path.append("/Applications/QGIS.app/Contents/Resources/python")
os.environ["PROJ_LIB"] = "/Applications/QGIS.app/Contents/Resources/proj"

from qgis.core import (
    QgsApplication,
    QgsVectorLayer,
    Qgis
)

QgsApplication.setPrefixPath("/Applications/QGIS.app/Contents/MacOS", True)
qgs = QgsApplication([], False)
qgs.initQgis()

# ✅ 修正版：ここ！
print("✅ QGIS initialized:", Qgis.version())

qgs.exitQgis()
