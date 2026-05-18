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
    camera_source = 0  # indeks kamery s21_win

elif config.current_camera == "trust_hd720p":
    camera_source = 0  # indeks kamery Trust HD720p

elif config.current_camera == "creative":
    camera_source = 1  # indeks kamery Creative

elif config.current_camera == "mx_brio":
    camera_source = 0  # indeks kamery MX Brio

# base_marker_id = 53  # ID markera bazowego (ten, względem którego mierzymy)
base_marker_id = 54  # ID markera bazowego (ten, względem którego mierzymy)

def draw_camera_axes_ui(img):
    h, w = img.shape[:2]

    origin = (60, h - 60)

    x_end = (origin[0] + 60, origin[1])      # prawo
    y_end = (origin[0], origin[1] - 60)      # góra
    z_end = (origin[0] - 40, origin[1] + 40) # pseudo Z (ukośnie)

    cv2.arrowedLine(img, origin, x_end, (0, 0, 255), 3)   # X red
    cv2.arrowedLine(img, origin, y_end, (0, 255, 0), 3)   # Y green
    cv2.arrowedLine(img, origin, z_end, (255, 0, 0), 3)   # Z blue

    cv2.putText(img, "X", x_end, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.putText(img, "Y", y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.putText(img, "Z", z_end, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

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
    return  R @ R_corr

def move_to_base(home_tvec, tvec):
    return  tvec - home_tvec

def rotate_to_base(home_R, R):
    # return  home_R.T @ R // ta kolejność decyduje o kierunku obrotu ale wartości są takie same
    return  R @ home_R.T


corrections = {
    # 42: rot_z(np.deg2rad(90)) ,     # QR ID 0 → +90° wokół Z
    # 53: rot_z(np.deg2rad(90)) @ rot_x(np.deg2rad(90)),
    56: rot_z(np.deg2rad(-90)),
}

# --- Wczytaj kalibrację ---
camera_matrix = np.load(f"stored_calibrations/{config.current_camera}/camera_matrix.npy")
dist_coeffs = np.load(f"stored_calibrations/{config.current_camera}/dist_coeffs.npy")

# --- Słownik i detektor ---
aruco_dict = cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT)
detector = cv2.aruco.ArucoDetector(aruco_dict)

# --- Otwórz kamerę ---
cap = cv2.VideoCapture(camera_source)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual (zależne od kamery)
# cap.set(cv2.CAP_PROP_EXPOSURE, -4)
# cap.set(cv2.CAP_PROP_GAIN, 0)

if not cap.isOpened():
    raise RuntimeError("Nie można otworzyć kamery. Sprawdź camera_source.")

print("Uruchomiono podgląd na żywo. Naciśnij 'q', aby zakończyć.")


if config.current_camera == "s21_win":
    ret, frame = cap.read()
    time.sleep(3)  # krótka pauza, żeby kamera się ustabilizowała

#wyznaczenie markera bazowego


ret, frame = cap.read()

# h, w = frame.shape[:2]

# # tworzymy siatkę punktów
# step = 50
# for y in range(0, h, step):
#     for x in range(0, w, step):
#         cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

# # undistort siatki
# undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs)

# cv2.imshow("Grid Original", cv2.resize(frame, None, fx=0.5, fy=0.5))
# cv2.imshow("Grid Undistorted", cv2.resize(undistorted, None, fx=0.5, fy=0.5))

if not ret:
    print("Nie udało się pobrać klatki z kamery.")
    exit(1)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
print({frame.shape})

inv = cv2.bitwise_not(gray)

# --- Detekcja markerów ---
corners, ids, _ = detector.detectMarkers(inv)

if ids is None:
    print("Nie wykryto żadnych markerów. Upewnij się, że marker bazowy jest widoczny.")
    exit(1)

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
home_rvec, _ = cv2.Rodrigues(home_R)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Nie udało się pobrać klatki z kamery.")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # gray = clahe.apply(gray)
    # gray = cv2.GaussianBlur(gray, (5,5), 0)

    inv = cv2.bitwise_not(gray)
    # cv2.imshow("inv", inv)

    # --- Detekcja markerów ---
    corners, ids, _ = detector.detectMarkers(inv)


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


            # R = rotate_to_base(home_R, default_R)
            R = default_R
            rvec, _ = cv2.Rodrigues(default_R) 

            tvec = move_to_base(home_tvec, global_tvec )
            # tvec = global_tvec
            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                dist_coeffs,
                rvec,
                global_tvec,
                0.03,  # długość osi (metry)
            )

            # OBRÓT YZ - najlepszy rezultat - prawie idealny - policzone w matlabie
            theta_y = np.arctan2(R[0,2], R[2,2])
            theta_z = np.arctan2(R[1,0], R[1,1])
        
            #obrót YZX - to samo co as_euler('xzy')
            # theta_y = np.arctan2(-R[2,0], R[0,0])
            # cos_theta_z = math.sqrt(1 - R[1,0]**2)
            # theta_z = np.arctan2(R[1,0], cos_theta_z)

            #obrót ZY
            # theta_y = np.arctan2(R[1,2], R[1,0])
            # theta_z = np.arctan2(R[1,0], R[0,0])


            # konwersja na stopnie (czytelniejsze)
            theta_y_deg = np.degrees(theta_y)
            theta_z_deg = np.degrees(theta_z)

            r = Rot.from_matrix(R)
            angles = r.as_euler('xzy', degrees=True) # extrinsic
            # angles2 = r.as_euler('yzx', degrees=True) # intrinsic

            # --- linie tekstu ---
            lines = [
                f"ID {ids[i][0]}",
                f"X={tvec[0][0]:.3f} Y={tvec[0][1]:.3f} Z={tvec[0][2]:.3f} m",
                f"rot angles X {angles[0]:.1f}, rot Y {angles[1]:.1f}, rot Z {angles[2]:.1f} deg",
                # f"rot angles2 X {angles2[0]:.1f}, rot Y {angles2[1]:.1f}, rot Z {angles2[2]:.1f} deg",
                f"rot manual Y {theta_y_deg:.1f}, rot Z {theta_z_deg:.1f} deg"
            ]

            print(" | ".join(lines))

            corner = corners[i][0][0].astype(int)

            for j, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (corner[0], max(corner[1] - 10 - j * 50, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

    # --- Podgląd (skalowanie) ---
    draw_camera_axes_ui(frame)

    scale = 0.4
    resized = cv2.resize(frame, None, fx=scale, fy=scale)
    cv2.imshow("Live pose estimation", resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
