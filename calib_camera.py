from datetime import date, datetime
from pathlib import Path
import cv2

import config
import camera
import calibration
import numpy as np


import os

camera_name = "mx_brio_for_business"
min_corners = 10
base_dir = os.path.dirname(os.path.abspath(__file__))
path_dir = f"{base_dir}\\{config.calib_images_path}\\single\\{camera_name}\\{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}"


camera = camera.Camera(
    camera_name=camera_name,
    width=1920,
    height=1080,
    position=camera.position.SINGLE
)

calibration = calibration.calibration(
    name=camera_name,
    min_corners=min_corners,
    scale_factor=camera.get_scale_factor()
)

photo_index = 0

while True:
    frame = camera.get_frame()

    if frame is None:
        continue

    detection = calibration.display_next_frame(frame)

    valid_detection = detection[0]

    key = detection[1]

    if key in (27, ord("q")):
        break

    if key == 32: # SPACE
        if not valid_detection:
            print("Pominięto zapis: plansza nie jest poprawnie wykryta.")
            continue

        path = f"{path_dir}\\{photo_index:03d}.png"

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(path, frame)

        photo_index += 1
        print(f"Zapisano zdjęcie {str(photo_index)}: {path}")

print("Zakończono zapis zdjęć.")


corners_list, ids_list, image_size = calibration.load_charuco_measurements(path_dir)
ret, camera_matrix, dist_coeffs, rvecs, tvecs = calibration.calibrate_camera(corners_list, ids_list, image_size)


if not ret:
    print("Kalibracja nie powiodła się.")
else:
    print("Kalibracja zakończona sukcesem.")

    print(f"Zapisano wyniki kalibracji w: {camera.get_calib_path()}")
    camera.save_camera_matrix(camera_matrix)
    camera.save_dist_coeffs(dist_coeffs)

camera.__del__()



