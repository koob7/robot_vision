import cv2
import cv2.aruco as aruco
import numpy as np

# =========================
# PARAMETRY
# =========================

MARKER_SIZE = 0.04

# =========================
# KALIBRACJA
# =========================

camera_matrix_1 = np.load("stored_calibrations/mx_brio/camera_matrix.npy")
dist_coeffs_1 = np.load("stored_calibrations/mx_brio/dist_coeffs.npy")

camera_matrix_2 = np.load("stored_calibrations/mx_brio2/camera_matrix.npy")
dist_coeffs_2 = np.load("stored_calibrations/mx_brio2/dist_coeffs.npy")

# Stereo extrinsics
R = np.load("stored_calibrations/stereo/R.npy")
T = np.load("stored_calibrations/stereo/T.npy")

# =========================
# PROJECTION MATRICES
# =========================

P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P1 = camera_matrix_1 @ P1

P2 = np.hstack((R, T))
P2 = camera_matrix_2 @ P2

# =========================
# ARUCO
# =========================

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
aruco_params = aruco.DetectorParameters()

detector = aruco.ArucoDetector(aruco_dict, aruco_params)

# =========================
# KAMERY
# =========================

cap1 = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(1)

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

    corners1, ids1, _ = detector.detectMarkers(gray1)
    corners2, ids2, _ = detector.detectMarkers(gray2)

    for i in range(len(ids1)):
        aruco.drawDetectedMarkers(frame1, [corners1[i]])

    for i in range(len(ids2)):
        aruco.drawDetectedMarkers(frame2, [corners2[i]])

    if ids1 is not None and ids2 is not None:

        ids1_flat = ids1.flatten()
        ids2_flat = ids2.flatten()

        common_ids = set(ids1_flat).intersection(set(ids2_flat))

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

            # =====================
            # RYSOWANIE
            # =====================

            aruco.drawDetectedMarkers(frame1, [corners1[idx1]])
            aruco.drawDetectedMarkers(frame2, [corners2[idx2]])

            text = f"ID:{marker_id} X:{x:.2f} Y:{y:.2f} Z:{z:.2f}"

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

    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    if cv2.waitKey(1) == 27:
        break

# =========================
# CLEANUP
# =========================

cap1.release()
cap2.release()

cv2.destroyAllWindows()