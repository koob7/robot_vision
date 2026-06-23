import cv2
import numpy as np
import math
import time
from scipy.spatial.transform import Rotation as Rot
from sympy import public

import config
import camera
import vision_helper
import determine_pose


class single_determine_pose(determine_pose.determine_pose):
    def __init__(self):
        self.camera = camera.Camera("mx_brio_for_business", 1920, 1080, position=camera.position.SINGLE)

        if not self.camera.is_ready():
            print("Nie można uruchomić kamery. Sprawdź połączenie i konfigurację.")
            return

        self.camera_matrix = self.camera.get_camera_matrix()
        self.dist_coeffs = self.camera.get_dist_coeffs()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict)
        self.marker_length = config.detected_marker_size_mm
        self.scale_factor = self.camera.get_scale_factor()
        self.name = self.camera.get_name()

        self.criteria = (
            cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

    def is_ready(self):
        return self.camera.is_ready()

    def find_markers(self):
        frame = self.camera.get_frame()
        if frame is None:
            print("Nie można pobrać klatki z kamery.")
            return None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv_gray = cv2.bitwise_not(gray)

        corners, ids, _ = self.detector.detectMarkers(inv_gray)

        for c in corners:
            cv2.cornerSubPix(
                inv_gray,
                c,
                (5,5),
                (-1,-1),
                self.criteria
            )

        if ids is None: 
            return None

        found_markers = {}

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self.marker_length,
            self.camera_matrix,
            self.dist_coeffs,
        )

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i in range(len(ids)):

            qr_rvec = rvecs[i]
            qr_tvec = tvecs[i]
            qr_id = ids[i][0]
            qr_R = cv2.Rodrigues(qr_rvec)

            found_markers[qr_id] = {
                "qr_tvec": qr_tvec,
                "qr_R": qr_R[0]
            }

        scaled_frame = cv2.resize(frame, None, fx=self.scale_factor, fy=self.scale_factor)
        cv2.imshow(self.name, scaled_frame)
        cv2.waitKey(1)
        return found_markers