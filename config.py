import cv2

columnsX = 5
rowsY = 8
squareLength = 0.035   # metry
markerLength = 0.026   # metry

# detected_marker_size_mm = 30
detected_marker_size_mm = 50

# current_camera = "s21"
# current_camera = "s21_win"
# current_camera = "trust_hd720p"
# current_camera = "creative"
# current_camera = "mx_brio"
# current_camera = "mx_brio_2" # MX Brio 480x640
current_camera = "mx_brio_3" # MX Brio 960x1280

ARUCO_DICT = cv2.aruco.DICT_6X6_250
# ARUCO_DICT = cv2.aruco.DICT_4X4_250