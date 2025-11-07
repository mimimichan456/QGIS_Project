# qgis_env.py
import sys, os
from qgis.core import QgsApplication

class QgisSession:
    def __enter__(self):
        sys.path.append("/Applications/QGIS.app/Contents/Resources/python")
        os.environ["PROJ_LIB"] = "/Applications/QGIS.app/Contents/Resources/proj"

        QgsApplication.setPrefixPath("/Applications/QGIS.app/Contents/MacOS", True)
        self.qgs = QgsApplication([], False)
        self.qgs.initQgis()
        return self.qgs

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.qgs.exitQgis()
