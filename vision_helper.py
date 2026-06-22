def rotation_in_base (R_marker, R_base):
    return R_base.T @ R_marker

def merge_camera_name (left_name, right_name):
    return f"{left_name}-{right_name}"




import numpy as np
import cv2

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

def rotate_to_base(home_T, global_T):
    # return  home_T.T @ global_T # ta kolejność decyduje o kierunku obrotu ale wartości są takie same
    return  global_T @ home_T.T


corrections = {
    # 42: rot_z(np.deg2rad(90)) ,     # QR ID 0 → +90° wokół Z
    # 53: rot_z(np.deg2rad(90)) @ rot_x(np.deg2rad(90)),
    56: rot_z(np.deg2rad(-90)),
}