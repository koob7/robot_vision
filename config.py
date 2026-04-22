import cv2

columnsX = 5
rowsY = 8
squareLength = 0.035   # metry
markerLength = 0.026   # metry

detected_marker_size_mm = 50

current_camera = "s21"
# current_camera = "webcam"

ARUCO_DICT = cv2.aruco.DICT_6X6_250