import cv2

columnsX = 5
rowsY = 8
squareLength = 0.035   # metry
markerLength = 0.026   # metry

detected_marker_size_mm = 40

calib_images_path = "calib_images/"
calib_results_path = "calib_results/"


camera_names = {
    "mx_brio": "MX Brio",
    "creative": "Live! Cam Sync 1080p",
    "integrated_camera": "Integrated Camera",
    "mx_brio_for_business": "MX Brio 705 for Business",
}

ARUCO_DICT = cv2.aruco.DICT_6X6_250
# ARUCO_DICT = cv2.aruco.DICT_4X4_250


#camera config
# --- Otwórz kamerę ---

camera_config_table = [
[cv2.CAP_PROP_AUTOFOCUS, 1],

# [cv2.CAP_PROP_AUTOFOCUS, 0],
# [cv2.CAP_PROP_FOCUS, 30],


# [cv2.CAP_PROP_AUTO_EXPOSURE, 2], # auto mode
[cv2.CAP_PROP_AUTO_EXPOSURE, 1], # manual mode
[cv2.CAP_PROP_EXPOSURE, -7], # krótszy czas naświetlania bardzo poprawił skoki w odczytywanej pozycji
[cv2.CAP_PROP_GAIN, 80], # ISO


[cv2.CAP_PROP_BRIGHTNESS, 10],#10 lub 150
[cv2.CAP_PROP_CONTRAST, 150],
[cv2.CAP_PROP_SATURATION, 0],
[cv2.CAP_PROP_SHARPNESS, 100],
[cv2.CAP_PROP_BACKLIGHT, 0],
[cv2.CAP_PROP_BUFFERSIZE, 1] # --- latency ---
]