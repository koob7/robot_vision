import cv2
import numpy as np

import config

# --- PARAMETRY ---
marker_length = config.detected_marker_size_mm / 1000  # metry
camera_source = 1  # indeks kamery systemowej

# --- Wczytaj kalibrację ---
camera_matrix = np.load(f"stored_calibrations/{config.current_camera}/camera_matrix.npy")
dist_coeffs = np.load(f"stored_calibrations/{config.current_camera}/dist_coeffs.npy")

# --- Słownik i detektor ---
aruco_dict = cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT)
detector = cv2.aruco.ArucoDetector(aruco_dict)

# --- Otwórz kamerę ---
cap = cv2.VideoCapture(camera_source)

if not cap.isOpened():
    raise RuntimeError("Nie można otworzyć kamery. Sprawdź camera_source.")

print("Uruchomiono podgląd na żywo. Naciśnij 'q', aby zakończyć.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Nie udało się pobrać klatki z kamery.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Detekcja markerów ---
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # --- Estymacja pozycji ---
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            marker_length,
            camera_matrix,
            dist_coeffs,
        )

        for i in range(len(ids)):
            rvec = rvecs[i]
            tvec = tvecs[i]

            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
                0.03,  # długość osi (metry)
            )

            text = f"ID {ids[i][0]} X={tvec[0][0]:.3f} Y={tvec[0][1]:.3f} Z={tvec[0][2]:.3f}m"
            corner = corners[i][0][0].astype(int)
            cv2.putText(
                frame,
                text,
                (corner[0], max(corner[1] - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

    # --- Podgląd (skalowanie) ---
    scale = 0.7
    resized = cv2.resize(frame, None, fx=scale, fy=scale)
    cv2.imshow("Live pose estimation", resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
