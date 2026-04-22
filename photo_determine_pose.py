import cv2
import numpy as np

import config

# --- PARAMETRY ---
marker_length = config.detected_marker_size_mm / 1000  # metry

# --- Wczytaj kalibrację ---
camera_matrix = np.load(f"stored_calibrations/{config.current_camera}/camera_matrix.npy")
dist_coeffs = np.load(f"stored_calibrations/{config.current_camera}/dist_coeffs.npy")

# --- Słownik ---
aruco_dict = cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT)

# --- Detektor ---
detector = cv2.aruco.ArucoDetector(aruco_dict)

# --- Wczytaj obraz ---
img = cv2.imread("test_2.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- Detekcja markerów ---
corners, ids, _ = detector.detectMarkers(gray)

if ids is not None:
    print(f"Wykryto markery: {ids.flatten()}")

    # --- Estymacja pozycji ---
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        marker_length,
        camera_matrix,
        dist_coeffs
    )

    for i in range(len(ids)):
        rvec = rvecs[i]
        tvec = tvecs[i]

        print("\n=== MARKER ID:", ids[i][0], "===")
        print("Pozycja (tvec) [m]:\n", tvec)
        print("Rotacja (rvec):\n", rvec)

        # --- Rysowanie osi ---
        cv2.drawFrameAxes(
            img,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
            0.03  # długość osi (metry)
        )

        # --- Rysowanie obramowania ---
        cv2.polylines(img, [corners[i].astype(int)], True, (0,255,0), 2)

else:
    print("Nie wykryto markerów.")

# --- Podgląd ---

scale = 0.3
resized = cv2.resize(img, None, fx=scale, fy=scale)

cv2.imshow("Pose estimation", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()