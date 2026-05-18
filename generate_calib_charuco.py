from PIL import Image

import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

import config

# --- PARAMETRY PLANSZY ---
columnsX = config.columnsX
rowsY = config.rowsY
squareLength = config.squareLength  # metry
markerLength = config.markerLength  # metry

# --- MARGINES (ważne!) ---
margin_mm = 10  # np. 10 mm z każdej strony

# --- ROZMIAR PLANSZY ---
board_width_mm = columnsX * squareLength * 1000
board_height_mm = rowsY * squareLength * 1000

# --- ROZMIAR STRONY PDF ---
page_width_mm = board_width_mm + 2 * margin_mm
page_height_mm = board_height_mm + 2 * margin_mm

# --- GENEROWANIE OBRAZU ---
dpi = 300
inch_per_meter = 39.3701
width_px = int(columnsX * squareLength * inch_per_meter * dpi)
height_px = int(rowsY * squareLength * inch_per_meter * dpi)

aruco_dict = cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT)

board = cv2.aruco.CharucoBoard(
    (columnsX, rowsY),
    squareLength,
    markerLength,
    aruco_dict
)

img = board.generateImage((width_px, height_px))
img = 255 - img  # odwrócenie kolorów (czarne na białe)

pil_img = Image.fromarray(img)

# --- PDF ---
pdf_file = "charuco_calib_board_5x8_35.pdf"

c = canvas.Canvas(
    pdf_file,
    pagesize=(page_width_mm * mm, page_height_mm * mm)
)

# --- POZYCJA (centrowanie) ---
x_offset = margin_mm * mm
y_offset = margin_mm * mm

c.drawImage(
    ImageReader(pil_img),
    x_offset,
    y_offset,
    width=board_width_mm * mm,
    height=board_height_mm * mm
)

c.showPage()
c.save()

print(f"Zapisano {pdf_file}")
print(f"Plansza: {board_width_mm:.1f} mm x {board_height_mm:.1f} mm")
print(f"Strona: {page_width_mm:.1f} mm x {page_height_mm:.1f} mm")

scale = 0.3
resized = cv2.resize(img, None, fx=scale, fy=scale)

cv2.imshow("Charuco board", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()