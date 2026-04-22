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
margin_mm = 10
border_mm = 5

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

marker_img = cv2.aruco.generateImageMarker(
    aruco_dict,
    marker_id,
    size_px
)

pil_img = Image.fromarray(marker_img)

# --- PDF ---
pdf_file = f"aruco_markers/aruco_marker_{marker_id}_{marker_size_mm}mm.pdf"
page_size_mm = marker_size_mm + 2 * margin_mm + 2 * border_mm

c = canvas.Canvas(
    pdf_file,
    pagesize=(page_size_mm * mm, page_size_mm * mm)
)

c.setFillColor(colors.black)
c.setStrokeColor(colors.black)
c.rect(0, 0, page_size_mm * mm, page_size_mm * mm, fill=1)

c.setFillColor(colors.white)
c.setStrokeColor(colors.white)
c.rect(border_mm * mm, border_mm * mm, (marker_size_mm + 2 * margin_mm) * mm, (marker_size_mm + 2 * margin_mm) * mm, fill=1)

c.drawImage(
    ImageReader(pil_img),
    (margin_mm + border_mm) * mm,
    (margin_mm + border_mm) * mm,
    width=marker_size_mm * mm,
    height=marker_size_mm * mm
)

c.showPage()
c.save()

print(f"Zapisano {pdf_file}")