from typing import List, Optional

import pandas as pd
from PyQt5.QtWidgets import (
    QFileDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QComboBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtWidgets import QMainWindow

from .training import train_linear_regression
from .visualization import create_pairplot
from .exporters import export_model_to_excel, export_model_to_pdf


class DescriptionWindow(QWidget):
    def __init__(self, description: str):
        super().__init__()
        self.setWindowTitle("Data Description")
        self.setGeometry(100, 100, 600, 400)
        layout = QVBoxLayout()
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlainText(description)
        layout.addWidget(self.text_edit)
        self.setLayout(layout)


class DataAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML: Data Load Analyze Visualize and Train Model")
        self.setGeometry(100, 100, 800, 600)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        self.layout.addWidget(self.load_button)

        self.describe_button = QPushButton("Show Data Description")
        self.describe_button.clicked.connect(self.show_data_description)
        self.layout.addWidget(self.describe_button)

        self.visualize_button = QPushButton("Visualize Data")
        self.visualize_button.clicked.connect(self.visualize_data)
        self.layout.addWidget(self.visualize_button)

        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QListWidget.MultiSelection)
        self.layout.addWidget(self.feature_list)

        self.target_combo = QComboBox()
        self.layout.addWidget(self.target_combo)

        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.train_model_button)
        self.layout.addWidget(self.train_button)

        self.export_pdf_button = QPushButton("Export to PDF")
        self.export_pdf_button.clicked.connect(self.export_to_pdf_button)
        self.layout.addWidget(self.export_pdf_button)

        self.export_excel_button = QPushButton("Export to Excel")
        self.export_excel_button.clicked.connect(self.export_to_excel_button)
        self.layout.addWidget(self.export_excel_button)

        self.status_label = QLabel("Status: Ready")
        self.layout.addWidget(self.status_label)

        self.stats_label = QLabel("")
        self.layout.addWidget(self.stats_label)

        self.df: Optional[pd.DataFrame] = None
        self.model = None
        self.loss_table = {}
        self.training_stats = {}
        self.selected_feature_names: List[str] = []
        self.plot_path = "pairplot.png"

    def load_data(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "CSV Files (*.csv);;Excel Files (*.xlsx *.xls)"
        )
        if file_path:
            if file_path.endswith(".csv"):
                self.df = pd.read_csv(file_path)
            else:
                self.df = pd.read_excel(file_path)
            self.status_label.setText(f"Status: Loaded data from {file_path}")
            self.update_feature_target_selection()

    def update_feature_target_selection(self) -> None:
        self.feature_list.clear()
        self.target_combo.clear()
        if self.df is not None:
            self.feature_list.addItems(self.df.columns)
            self.target_combo.addItems(self.df.columns)

    def show_data_description(self) -> None:
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        description = self.df.describe().to_string()
        self.description_window = DescriptionWindow(description)
        self.description_window.show()

    def visualize_data(self) -> None:
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        self.plot_path = create_pairplot(self.df)
        self.status_label.setText("Status: Data visualized")

    def selected_features(self) -> List[str]:
        return [item.text() for item in self.feature_list.selectedItems()]

    def train_model_button(self) -> None:
        if self.df is None:
            QMessageBox.warning(self, "Error", "No data loaded")
            return
        features = self.selected_features()
        target = self.target_combo.currentText()
        if not features or not target:
            QMessageBox.warning(self, "Error", "Please select features and target")
            return
        self.selected_feature_names = features
        (
            self.model,
            self.loss_table,
            self.training_stats,
        ) = train_linear_regression(self.df, features, target)
        stats_text = "\n".join(
            [f"{key}: {value}" for key, value in self.training_stats.items()]
        )
        self.stats_label.setText(stats_text)
        self.status_label.setText("Status: Model trained")

    def ensure_model_trained(self) -> bool:
        if self.model is None:
            QMessageBox.warning(self, "Error", "No model trained yet")
            return False
        return True

    def export_to_pdf_button(self) -> None:
        if not self.ensure_model_trained():
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF", "", "PDF Files (*.pdf)"
        )
        if file_path:
            export_model_to_pdf(
                self.model,
                self.loss_table,
                self.training_stats,
                file_path,
                self.plot_path,
            )
            self.status_label.setText(f"Status: Exported to {file_path}")

    def export_to_excel_button(self) -> None:
        if not self.ensure_model_trained():
            return
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Excel", "", "Excel Files (*.xlsx)"
        )
        if file_path:
            export_model_to_excel(
                self.selected_feature_names,
                self.model,
                self.training_stats,
                file_path,
            )
            self.status_label.setText(f"Status: Exported to {file_path}")
