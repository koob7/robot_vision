import cv2

columnsX = 5
rowsY = 8
squareLength = 0.035   # metry
markerLength = 0.026   # metry

detected_marker_size_mm = 40

# current_camera = "s21"
# current_camera = "s21_win"
current_camera = "trust_hd720p"

ARUCO_DICT = cv2.aruco.DICT_6X6_250