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
import os


class stereo_determine_pose(determine_pose.determine_pose):
    def __init__(self):

        base_dir = os.path.dirname(os.path.abspath(__file__))

        left_camera_name = "mx_brio_for_business"
        right_camera_name = "mx_brio"

        self.camera = camera.Camera(left_camera_name, 1920, 1080, position=camera.position.LEFT)
        self.camera_right = camera.Camera(right_camera_name, 1920, 1080, position=camera.position.RIGHT)

        if not self.camera.is_ready() or not self.camera_right.is_ready():
            print("Nie można uruchomić kamery. Sprawdź połączenie i konfigurację.")
            return

        self.camera_matrix_left = self.camera.get_camera_matrix()
        self.dist_coeffs = self.camera.get_dist_coeffs()

        self.camera_matrix_right = self.camera_right.get_camera_matrix()
        self.dist_coeffs_right = self.camera_right.get_dist_coeffs()

        self.stereo_R = np.load(f"{base_dir}\\{config.calib_results_path}\\stereo\\{vision_helper.merge_camera_name(left_camera_name, right_camera_name)}\\R.npy")
        self.stereo_T = np.load(f"{base_dir}\\{config.calib_results_path}\\stereo\\{vision_helper.merge_camera_name(left_camera_name, right_camera_name)}\\T.npy")

        self.projection_matrix_left = self.camera_matrix_left @ np.hstack((np.eye(3), np.zeros((3, 1))))
        self.projection_matrix_right = self.camera_matrix_right @ np.hstack((self.stereo_R, self.stereo_T))

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
        return self.camera.is_ready() and self.camera_right.is_ready()

    def find_markers(self):
        frame_left = self.camera.get_frame()
        frame_right = self.camera_right.get_frame()
        if frame_left is None or frame_right is None:
            print("Nie można pobrać klatki z kamery.")
            return None
        
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        inv_gray_left = cv2.bitwise_not(gray_left)

        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        inv_gray_right = cv2.bitwise_not(gray_right)

        corners_left, ids_left, _ = self.detector.detectMarkers(inv_gray_left)
        corners_right, ids_right, _ = self.detector.detectMarkers(inv_gray_right)

        for c in corners_left:
            cv2.cornerSubPix(
                inv_gray_left,
                c,
                (5,5),
                (-1,-1),
                self.criteria
            )

        if ids_left is None or ids_right is None: 
            return None

        found_markers = {}

        ids_left_flat = ids_left.flatten()
        ids_right_flat = ids_right.flatten()
        common_ids = set(ids_left_flat) & set(ids_right_flat)

        if not common_ids:
            return None
        
        for marker_id in common_ids:
            left_index = np.where(ids_left_flat == marker_id)[0][0]
            right_index = np.where(ids_right_flat == marker_id)[0][0]

            left_corners = corners_left[left_index][0]
            right_corners = corners_right[right_index][0]

            points_4d = cv2.triangulatePoints(
                self.projection_matrix_left,
                self.projection_matrix_right,
                left_corners.T,
                right_corners.T
            )

            points_3d = points_4d[:3] / points_4d[3]
            center_3d = np.mean(points_3d, axis=1)

            p0 = points_3d[:, 0]
            p1 = points_3d[:, 1]
            p2 = points_3d[:, 2]
            p3 = points_3d[:, 3]

            x_axis = p1 - p0
            x_axis /= np.linalg.norm(x_axis)

            y_axis = p3 - p0
            y_axis /= np.linalg.norm(y_axis)

            z_axis = np.cross(x_axis, y_axis)
            z_axis /= np.linalg.norm(z_axis)

            y_axis = np.cross(z_axis, x_axis)

            R_marker = np.column_stack((x_axis, y_axis, z_axis))
            
            cv2.aruco.drawDetectedMarkers(frame_left, [corners_left[left_index]])
            cv2.aruco.drawDetectedMarkers(frame_right, [corners_right[right_index]])

            found_markers[marker_id] = {
                "qr_tvec": center_3d,
                "qr_R": R_marker
            }

        frame_left_resized = cv2.resize(frame_left, None, fx=self.scale_factor, fy=self.scale_factor)
        frame_right_resized = cv2.resize(frame_right, None, fx=self.scale_factor, fy=self.scale_factor)

        cv2.imshow(f"{self.name}_left", frame_left_resized)
        cv2.imshow(f"{self.name}_right", frame_right_resized)
        cv2.waitKey(1)

        return found_markers