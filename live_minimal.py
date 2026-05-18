import cv2
import numpy as np
import math
import time
from scipy.spatial.transform import Rotation as Rot
import config

marker_length = config.detected_marker_size_mm / 1000
camera_source = 0
base_marker_id = 54  # base marker ID

def move_to_base(home_tvec, tvec):
    return  tvec - home_tvec

def rotate_to_base(home_R, R):
    return  home_R.T @ R

camera_matrix = np.load(f"stored_calibrations/{config.current_camera}/camera_matrix.npy")
dist_coeffs = np.load(f"stored_calibrations/{config.current_camera}/dist_coeffs.npy")
aruco_dict = cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT)
detector = cv2.aruco.ArucoDetector(aruco_dict)

cap = cv2.VideoCapture(camera_source)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    inv = cv2.bitwise_not(gray)
    cv2.imshow("inv", inv)

    corners, ids, _ = detector.detectMarkers(inv)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            marker_length,
            camera_matrix,
            dist_coeffs,
        )

        for i in range(len(ids)):
            rvec = rvecs[i]
            tvec = tvecs[i]
            qr_id = ids[i][0]

            R, _ = cv2.Rodrigues(rvec)
           
            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
                0.03,
            )

            # I assume marcer can only rotate around Y and then around Z axes
            theta_y_deg = np.degrees(np.arctan2(R[0,2], R[2,2]))
            theta_z_deg = np.degrees(np.arctan2(R[1,0], R[1,1]))



            lines = [
                f"ID {ids[i][0]}",
                f"X={tvec[0][0]:.3f} Y={tvec[0][1]:.3f} Z={tvec[0][2]:.3f} m",
                f"rot manual Y {theta_y_deg:.1f}, rot Z {theta_z_deg:.1f} deg"
            ]

            print( lines[0] + " | " + lines[1] )

            corner = corners[i][0][0].astype(int)
            for j, line in enumerate(lines):
                cv2.putText(
                    frame,
                    line,
                    (corner[0], max(corner[1] - 10 - j * 20, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

    scale = 1.0
    resized = cv2.resize(frame, None, fx=scale, fy=scale)
    cv2.imshow("Live pose estimation", resized)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
