import cv2
import numpy as np
import glob

import config

# --- Parametry planszy ChArUco ---
columnsX = config.columnsX
rowsY = config.rowsY
squareLength = config.squareLength # metry
markerLength = config.markerLength # metry

# --- Słownik ArUco ---
aruco_dict = cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT)

# --- Tworzenie planszy ChArUco ---
board = cv2.aruco.CharucoBoard(
    (columnsX, rowsY),
    squareLength,
    markerLength,
    aruco_dict
)

# --- Detektor ---
detector = cv2.aruco.ArucoDetector(aruco_dict)

# --- Listy punktów ---
all_charuco_corners = []
all_charuco_ids = []
image_size = None

# --- Wczytaj zdjęcia ---
images = glob.glob(f"calib_images_8x5x35/{config.current_camera}/*.jpg")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray.shape[::-1]

    print(f"Przetwarzanie {fname}...")

    # --- Detekcja markerów ---
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        # --- Interpolacja narożników ChArUco ---
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray, board
        )

        if retval > 10:  # minimum punktów
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            print(f"wykryto {retval} punktów")
        else:
            print(f"za mało punktów ({retval}), pomijam")
    else:
        print(f"nie wykryto markerów")

if len(all_charuco_corners) == 0:
    print("Nie wykryto żadnych punktów ChArUco. Kalibracja niemożliwa.")
    # pamiętaj że kolejność kwadratów i kodów QR jest ważna
    # przykładowo dla popularnego calib.io szachownica zaczynająca
    # się od kodu QR nie jest rozpoznawana pomimo dobrego słownika
    exit(1)

# --- Kalibracja ---
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    all_charuco_corners,
    all_charuco_ids,
    board,
    image_size,
    None,
    None
)

# --- Wyniki ---
print("Błąd reprojekcji:", ret)
print("Macierz kamery:\n", camera_matrix)
print("Współczynniki dystorsji:\n", dist_coeffs)

# --- Zapis ---
np.save(f"stored_calibrations/{config.current_camera}/camera_matrix.npy", camera_matrix)
np.save(f"stored_calibrations/{config.current_camera}/dist_coeffs.npy", dist_coeffs)