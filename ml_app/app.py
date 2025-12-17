import sys

from PyQt5.QtWidgets import QApplication

from .ui import DataAnalysisApp


def run_app() -> None:
    app = QApplication(sys.argv)
    window = DataAnalysisApp()
    window.show()
    sys.exit(app.exec_())
