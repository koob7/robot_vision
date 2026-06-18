import cv2
import cv2.aruco as aruco
import numpy as np
import time

import config

# =========================
# PARAMETRY
# =========================

MARKER_SIZE = config.detected_marker_size_mm / 1000  # metry

# =========================
# KALIBRACJA
# =========================

camera_matrix_right = np.load("stored_calibrations/mx_brio/camera_matrix.npy")#right
dist_coeffs_right = np.load("stored_calibrations/mx_brio/dist_coeffs.npy")
right_index = 0

camera_matrix_left = np.load("stored_calibrations/mx_brio2/camera_matrix.npy")#left
dist_coeffs_left = np.load("stored_calibrations/mx_brio2/dist_coeffs.npy")
left_index = 2

# Stereo extrinsics
R = np.load("stored_calibrations/stereo/R.npy")
T = np.load("stored_calibrations/stereo/T.npy")

# =========================
# PROJECTION MATRICES
# =========================

P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P1 = camera_matrix_left @ P1

P2 = np.hstack((R, T))
P2 = camera_matrix_right @ P2

# =========================
# ARUCO
# =========================

aruco_dict = aruco.getPredefinedDictionary(config.ARUCO_DICT)
aruco_params = aruco.DetectorParameters()

detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# =========================
# KAMERY
# =========================

cap1 = cv2.VideoCapture(left_index)
cap2 = cv2.VideoCapture(right_index)

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)


# =========================
# PĘTLA
# =========================

while True:

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    gray1_inv = cv2.bitwise_not(gray1)
    gray2_inv = cv2.bitwise_not(gray2)

    corners1, ids1, _ = detector.detectMarkers(gray1_inv)
    corners2, ids2, _ = detector.detectMarkers(gray2_inv)

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    for c in corners1:
        cv2.cornerSubPix(
            gray1_inv,
            c,
            (5,5),
            (-1,-1),
            criteria
        )

    for c in corners2:
        cv2.cornerSubPix(
            gray2_inv,
            c,
            (5,5),
            (-1,-1),
            criteria
        )

    if ids1 is not None and ids2 is not None:

        ids1_flat = ids1.flatten()
        ids2_flat = ids2.flatten()

        common_ids = set(ids1_flat).intersection(set(ids2_flat))

        R54 = np.eye(3)
        R56 = np.eye(3)

        for marker_id in common_ids:

            idx1 = np.where(ids1_flat == marker_id)[0][0]
            idx2 = np.where(ids2_flat == marker_id)[0][0]

            c1 = corners1[idx1][0]
            c2 = corners2[idx2][0]

            # =====================
            # TRIANGULACJA ROGÓW
            # =====================

            points_4d = cv2.triangulatePoints(
                P1,
                P2,
                c1.T,
                c2.T
            )

            points_3d = points_4d[:3] / points_4d[3]

            # Środek markera
            center_3d = np.mean(points_3d, axis=1)

            x, y, z = center_3d

            #rotation matrix

            p0 = points_3d[:, 0]
            p1 = points_3d[:, 1]
            p2 = points_3d[:, 2]
            p3 = points_3d[:, 3]

            # oś X markera
            x_axis = p1 - p0
            x_axis /= np.linalg.norm(x_axis)

            # oś Y markera
            y_axis = p3 - p0
            y_axis /= np.linalg.norm(y_axis)

            # oś Z = normalna do płaszczyzny
            z_axis = np.cross(x_axis, y_axis)
            z_axis /= np.linalg.norm(z_axis)

            # poprawka ortogonalności
            y_axis = np.cross(z_axis, x_axis)

            R_marker = np.column_stack((x_axis, y_axis, z_axis))

            if (marker_id == 56):
                R56 = R_marker

            if (marker_id == 54):
                R54 = R_marker

            theta_y = np.arctan2(R_marker[0,2], R_marker[2,2])
            theta_z = np.arctan2(R_marker[1,0], R_marker[1,1])
            theta_y_deg = np.degrees(theta_y)
            theta_z_deg = np.degrees(theta_z)

            # =====================
            # RYSOWANIE
            # =====================

            aruco.drawDetectedMarkers(frame1, [corners1[idx1]])
            aruco.drawDetectedMarkers(frame2, [corners2[idx2]])

            text = f"ID:{marker_id} X:{x:.4f} Y:{y:.4f} Z:{z:.4f} y:{theta_y_deg:.2f}° z:{theta_z_deg:.2f}°"
            print(text)

            cv2.putText(
                frame1,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame2,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        
        # global_T @ home_T.T
        #R54 - home
        #R56 - target
        R56_diff = R54.T @ R56

        theta_y = np.arctan2(R56_diff[0,2], R56_diff[2,2])
        theta_z = np.arctan2(R56_diff[1,0], R56_diff[1,1])
        theta_y_deg = np.degrees(theta_y)
        theta_z_deg = np.degrees(theta_z)

        text = f"R56_diff y:{theta_y_deg:.2f}° z:{theta_z_deg:.2f}°"
        print(text)

    scaled_frame1 = cv2.resize(frame1, (640, 480))
    scaled_frame2 = cv2.resize(frame2, (640, 480))
    cv2.imshow("Camera 1", scaled_frame1)
    cv2.imshow("Camera 2", scaled_frame2)

    time.sleep(0.10)

    if cv2.waitKey(1) == 27:
        break

# =========================
# CLEANUP
# =========================

cap1.release()
cap2.release()

cv2.destroyAllWindows()