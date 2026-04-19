import os
from fpdf import FPDF
import datetime


class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(0, 10, "CardioLink Anomaly Detection Report", ln=True, align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_pdf_report(result_text, graph_paths, hrv_data, spo2_data, bpm_data, output_path=None):

    if output_path is None:
        timestamp = datetime.datetime.now().strftime(f"%Y%m%d_%H%M%S")
        output_path = f"results/anomaly_report_{timestamp}.pdf"

    # Ensure output_path is valid
    if not isinstance(output_path, (str, os.PathLike)):
        raise ValueError(f"Expected a string or PathLike object for output_path, got {type(output_path)}")
    
    # Check if the directory exists, if not, create it
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf = PDF()
    pdf.add_page()

    # Summary
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, result_text.replace("✅", "Normal").replace("⚠️", "Anomaly"))

    # Check if graph_paths is a tuple and handle it
    if isinstance(graph_paths, tuple):
        print("Warning: graph_paths is a tuple, converting to list")
        graph_paths = list(graph_paths)  # Convert to list if it's a tuple
    
    # Add plots if graph_paths is not empty
   # Add plots with titles
    if graph_paths:
        for graph in graph_paths:
            if isinstance(graph, tuple) and len(graph) == 2:
                title, path = graph
                if isinstance(path, (str, os.PathLike)) and os.path.exists(path):
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(0, 10, title, ln=True)
                    pdf.image(path, w=170)
                    pdf.ln(5)
                else:
                    print(f"Warning: Graph path '{path}' is invalid or does not exist.")
            else:
                print(f"Warning: Invalid graph entry {graph}")

    
    # Table title
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Sample Data Summary (First 10 Values)", ln=True)

    # Table headers
    pdf.set_font("Arial", "B", 10)
    pdf.cell(30, 10, "Index", 1)
    pdf.cell(40, 10, "HRV", 1)
    pdf.cell(40, 10, "SpO2", 1)
    pdf.cell(40, 10, "BPM", 1)
    pdf.ln()

    # Table rows (Handle case where data is less than 10)
    pdf.set_font("Arial", "", 10)
    num_rows = min(10, len(hrv_data), len(spo2_data), len(bpm_data))  # Ensure there's enough data
    for i in range(num_rows):
        pdf.cell(30, 10, str(i), 1)
        pdf.cell(40, 10, f"{hrv_data[i]:.2f}", 1)
        pdf.cell(40, 10, f"{spo2_data[i]:.2f}", 1)
        pdf.cell(40, 10, f"{bpm_data[i]:.2f}", 1)
        pdf.ln()

    # Save the PDF to the provided path (after all content is added)
    pdf.output(output_path)
    print(f"📄 PDF report generated: {output_path}")
