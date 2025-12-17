from typing import Dict, Iterable

import pandas as pd
from fpdf import FPDF
from sklearn.linear_model import LinearRegression


def export_model_to_pdf(
    model: LinearRegression,
    loss_table: Dict[str, float],
    training_stats: Dict[str, float],
    file_path: str,
    plot_path: str,
) -> None:
    """Export model details and plot to a PDF file."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Model Results", ln=True, align="C")

    pdf.cell(200, 10, txt="Coefficients:", ln=True)
    for i, coef in enumerate(model.coef_):
        pdf.cell(200, 10, txt=f"Feature {i}: {coef}", ln=True)

    pdf.cell(200, 10, txt=f"Intercept: {model.intercept_}", ln=True)

    pdf.cell(200, 10, txt="Loss Table:", ln=True)
    for key, value in loss_table.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    pdf.cell(200, 10, txt="Training Stats:", ln=True)
    for key, value in training_stats.items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)

    pdf.image(plot_path, x=10, y=None, w=190)
    pdf.output(file_path)


def export_model_to_excel(
    selected_features: Iterable[str],
    model: LinearRegression,
    training_stats: Dict[str, float],
    file_path: str,
) -> None:
    """Export coefficients and training stats to Excel."""
    coeffs = pd.DataFrame({
        "Feature": list(selected_features),
        "Coefficient": model.coef_,
    })
    coeffs.to_excel(file_path, index=False)

    stats_df = pd.DataFrame(list(training_stats.items()), columns=["Metric", "Value"])
    with pd.ExcelWriter(file_path, mode="a") as writer:
        stats_df.to_excel(writer, sheet_name="Training Stats", index=False)
