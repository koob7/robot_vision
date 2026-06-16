import cv2
import argparse
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from PIL import Image

import config

# --- ARGUMENTY ---
parser = argparse.ArgumentParser(description="Generator markera ArUco do PDF")
parser.add_argument("marker_id", type=int, help="ID markera (0-249)")
args = parser.parse_args()

marker_id = args.marker_id

# --- PARAMETRY ---
marker_size_mm = config.detected_marker_size_mm
margin_mm = 15
border_mm = 5
marker_space_mm = 5
pdf_border_mm = 10  # poprawiona nazwa

# --- WALIDACJA ---
if not (0 <= marker_id <= 249):
    raise ValueError("marker_id musi być w zakresie 0-249")

# --- SŁOWNIK ---
aruco_dict = cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT)

# --- GENEROWANIE ---
dpi = 300
inch_per_meter = 39.3701

marker_size_m = marker_size_mm / 1000
size_px = int(marker_size_m * inch_per_meter * dpi)

# --- PDF ---
pdf_file = f"aruco_markers/aruco_marker_grid_{marker_id}_{marker_size_mm}mm.pdf"

cols = 2
rows = 4
per_page = cols * rows

cell_size_mm = marker_size_mm + 2 * margin_mm + 2 * border_mm

# ✅ uwzględnienie odstępów między markerami
page_width_mm = cols * cell_size_mm + (cols - 1) * marker_space_mm + 2 * pdf_border_mm
page_height_mm = rows * cell_size_mm + (rows - 1) * marker_space_mm + 2 * pdf_border_mm

c = canvas.Canvas(
    pdf_file,
    pagesize=(page_width_mm * mm, page_height_mm * mm)
)

start_id = marker_id
end_id = min(marker_id + 23, 249)

current_id = start_id

while current_id <= end_id:

    for i in range(per_page):

        if current_id > end_id:
            break

        row = i // cols
        col = i % cols

        marker_img = cv2.aruco.generateImageMarker(
            aruco_dict,
            current_id,
            size_px
        )

        marker_img = 255 - marker_img
        pil_img = Image.fromarray(marker_img)

        # ✅ dodany spacing między komórkami
        x_offset = pdf_border_mm + col * (cell_size_mm + marker_space_mm)
        y_offset = pdf_border_mm + (rows - 1 - row) * (cell_size_mm + marker_space_mm)

        # czarne tło
        c.setFillColor(colors.white)
        c.rect(
            x_offset * mm,
            y_offset * mm,
            cell_size_mm * mm,
            cell_size_mm * mm,
            fill=1
        )

        # biały obszar
        c.setFillColor(colors.black)
        c.rect(
            (x_offset + border_mm) * mm,
            (y_offset + border_mm) * mm,
            (marker_size_mm + 2 * margin_mm) * mm,
            (marker_size_mm + 2 * margin_mm) * mm,
            fill=1
        )

        # marker
        c.drawImage(
            ImageReader(pil_img),
            (x_offset + border_mm + margin_mm) * mm,
            (y_offset + border_mm + margin_mm) * mm,
            width=marker_size_mm * mm,
            height=marker_size_mm * mm
        )

        current_id += 1

    c.showPage()

c.save()