from pathlib import Path
import cv2
import vision_helper

import config
import camera
import calibration
import numpy as np
import cv2.aruco as aruco


import os

camera_left_name = "mx_brio_for_business"
camera_right_name = "mx_brio"
min_corners = 10
base_dir = os.path.dirname(os.path.abspath(__file__))

camera_left = camera.Camera(
    camera_name=camera_left_name,
    width=config.CAMERA_WIDTH,
    height=config.CAMERA_HEIGHT,
    position=camera.position.LEFT
)

camera_right = camera.Camera(
    camera_name=camera_right_name,
    width=config.CAMERA_WIDTH,
    height=config.CAMERA_HEIGHT,
    position=camera.position.RIGHT
)

calibration_left = calibration.calibration(
    name=camera_left_name,
    min_corners=min_corners,
    scale_factor=camera_left.get_scale_factor()
)

calibration_right = calibration.calibration(
    name=camera_right_name,
    min_corners=min_corners,
    scale_factor=camera_right.get_scale_factor()
)

stereo_calibration_dir = f"{Path(base_dir)}\\{config.calib_results_path}\\stereo\\{vision_helper.merge_camera_name(camera_left.get_name(), camera_right.get_name())}"
photo_index = 0

while True:
    frame_left = camera_left.get_frame()
    frame_right = camera_right.get_frame()

    if frame_left is None or frame_right is None:
        continue

    detection_left = calibration_left.display_next_frame(frame_left)
    detection_right = calibration_right.display_next_frame(frame_right)

    valid_detection_left = detection_left[0]
    valid_detection_right = detection_right[0]

    key_left = detection_left[1]
    key_right = detection_right[1]

    if key_left in (27, ord("q")) or key_right in (27, ord("q")):
        break

    if key_left == 32 or key_right == 32: # SPACE
        if not valid_detection_left:
            print("Pominięto zapis: plansza nie jest poprawnie wykryta.")
            continue

        path_left = f"{camera_left.get_photo_dir()}\\{photo_index:03d}.png"
        path_right = f"{camera_right.get_photo_dir()}\\{photo_index:03d}.png"

        Path(path_left).parent.mkdir(parents=True, exist_ok=True)
        Path(path_right).parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(path_left, frame_left)
        cv2.imwrite(path_right, frame_right)

        photo_index += 1
        print(f"Zapisano zdjęcie {str(photo_index)}: {path_left}")
        print(f"Zapisano zdjęcie {str(photo_index)}: {path_right}")

print("Zakończono zapis zdjęć.")

corners_list_left, ids_list_left, image_size_left = calibration_left.load_charuco_measurements(camera_left.get_photo_dir())
ret_left, camera_matrix_left, dist_coeffs_left, rvecs_left, tvecs_left = calibration_left.calibrate_camera(corners_list_left, ids_list_left, image_size_left)

corners_list_right, ids_list_right, image_size_right = calibration_right.load_charuco_measurements(camera_right.get_photo_dir())
ret_right, camera_matrix_right, dist_coeffs_right, rvecs_right, tvecs_right = calibration_right.calibrate_camera(corners_list_right, ids_list_right, image_size_right)

stereo_points = calibration_left.stereo_calibration(camera_left.get_photo_dir(), camera_right.get_photo_dir())

if image_size_left != image_size_right:
    print("Uwaga: rozdzielczości obrazów wejściowych różnią się między kamerami.")
    exit(1)

stereo_ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
    stereo_points[0],
    stereo_points[1],
    stereo_points[2],
    camera_matrix_left,
    dist_coeffs_left,
    camera_matrix_right,
    dist_coeffs_right,
    image_size_left,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    flags=cv2.CALIB_FIX_INTRINSIC,
)

camera_left.save_camera_matrix(camera_matrix_left)
camera_left.save_dist_coeffs(dist_coeffs_left)

camera_right.save_camera_matrix(camera_matrix_right)
camera_right.save_dist_coeffs(dist_coeffs_right)


Path(stereo_calibration_dir).mkdir(parents=True, exist_ok=True) 
np.save(f"{stereo_calibration_dir}\\R.npy", R)
np.save(f"{stereo_calibration_dir}\\T.npy", T)