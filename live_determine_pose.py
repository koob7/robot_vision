import cv2
import numpy as np
import math
import time
from scipy.spatial.transform import Rotation as Rot

import config

# --- PARAMETRY ---
marker_length = config.detected_marker_size_mm / 1000  # metry

if config.current_camera == "s21":
    camera_source = 1  # indeks kamery s21

elif config.current_camera == "s21_win":
    camera_source = 2  # indeks kamery s21_win

elif config.current_camera == "trust_hd720p":
    camera_source = 1  # indeks kamery Trust HD720p

base_marker_id = 53  # ID markera bazowego (ten, względem którego mierzymy)

def rot_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

def rot_x(theta):
    return np.array([
        [1, 0,              0             ],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta),  np.cos(theta)]
    ])

def rot_y(theta):
    return np.array([
        [ np.cos(theta), 0, np.sin(theta)],
        [ 0,             1, 0            ],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def get_default_correction(qr_id, R):
    R_corr = corrections.get(qr_id, np.eye(3))
    return R @ R_corr

def move_to_base(home_tvec, tvec):
    return  tvec - home_tvec

def rotate_to_base(home_R, R):
    return R @ home_R.T

corrections = {
    42: rot_z(np.deg2rad(90)),     # QR ID 0 → +90° wokół Z
    # 53: rot_z(np.deg2rad(90)) @ rot_x(np.deg2rad(90)),
}

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


if config.current_camera == "s21_win":
    ret, frame = cap.read()
    time.sleep(3)  # krótka pauza, żeby kamera się ustabilizowała

#wyznaczenie markera bazowego


ret, frame = cap.read()
if not ret:
    print("Nie udało się pobrać klatki z kamery.")
    exit(1)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# --- Detekcja markerów ---
corners, ids, _ = detector.detectMarkers(gray)

if ids is None:
    print("Nie wykryto żadnych markerów. Upewnij się, że marker bazowy jest widoczny.")

base_i = np.where(ids.flatten() == base_marker_id)[0]

if len(base_i) != 1:
    print(f"Nie wykryto markera bazowego (ID {base_marker_id}).")
    for i in range(len(ids)):
        print(f"Wykryty marker ID: {ids[i][0]}")
    exit(1)

rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
    corners,
    marker_length,
    camera_matrix,
    dist_coeffs,
)

rvec = rvecs[base_i[0]]
home_tvec = tvecs[base_i[0]]
R, _ = cv2.Rodrigues(rvec)
home_R = get_default_correction(base_marker_id, R)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Nie udało się pobrać klatki z kamery.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # --- Detekcja markerów ---
    corners, ids, _ = detector.detectMarkers(gray)


    # co zrobić z tą pozycją żeby wykorzystać ją jako punkt bazowy

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
            global_rvec = rvecs[i]
            global_tvec = tvecs[i]
            qr_id = ids[i][0]

            global_R, _ = cv2.Rodrigues(global_rvec)

            default_R = get_default_correction(qr_id, global_R)
            default_rvec, _ = cv2.Rodrigues(default_R)

            R = rotate_to_base(home_R, default_R) 
            # R = default_R

            tvec = move_to_base(home_tvec, global_tvec )

            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                dist_coeffs,
                default_rvec,
                global_tvec,
                0.03,  # długość osi (metry)
            )



            # --- wyciąganie kątów ---
            theta_y = np.arctan2(R[0,2], R[2,2])
            theta_z = np.arctan2(R[1,0], R[1,1])


            # konwersja na stopnie (czytelniejsze)
            theta_y_deg = np.degrees(theta_y)
            theta_z_deg = np.degrees(theta_z)

            r = Rot.from_matrix(R)
            angles = r.as_euler('zyx', degrees=True)

            # --- linie tekstu ---
            lines = [
                f"ID {ids[i][0]}",
                f"X={tvec[0][0]:.3f} Y={tvec[0][1]:.3f} Z={tvec[0][2]:.3f} m",
                f"rot X {angles[2]:.1f}, rot Y {angles[1]:.1f}, rot Z {angles[0]:.1f} deg"
            ]

            corner = corners[i][0][0].astype(int)

            for j, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (corner[0], max(corner[1] - 10 - j * 20, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

    # --- Podgląd (skalowanie) ---
    scale = 1.0
    resized = cv2.resize(frame, None, fx=scale, fy=scale)

    cv2.imshow("Live pose estimation", resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
