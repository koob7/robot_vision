import re

def rotation_in_base (R_marker, R_base):
    return R_base.T @ R_marker

def merge_camera_name (left_name, right_name):
    return f"{left_name}-{right_name}"

def split_camera_name (merged_name):
    names = merged_name.split("-")

    if len(names) != 2:
        return None, None
    
    return names[0], names[1]

def get_camera_info (camera_name):

    numbers = [int(n) for n in re.findall(r'\d+', camera_name)]

    if len(numbers) != 2:
        return None, None
    
    return numbers[0], numbers[1]

def get_camera_name (camera_name):
    name, rest = re.split(r'(?=\d)', camera_name, maxsplit=1)
    name = name[:-1]
    return name


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


#uwaga - korekcja w osi X na 99% nie działa w złożeniu rotacji Y-Z
#uwaga - jak zmienimy rotację markera bazowego to wyniki dla kolejnych markerów przestają mieć sens bo układ współrzędnych się obraca i konwencja obrotów Y-Z traci sens
#dla markera bazowego sens ma tylko rotacja w osi w głąb markara czyli w okół osi Z - czy to się zmienia?
corrections = {
    # 42: rot_z(np.deg2rad(90)) ,     # QR ID 0 → +90° wokół Z
    # 53: rot_z(np.deg2rad(90)) @ rot_x(np.deg2rad(90)),
    56: rot_z(np.deg2rad(-90)),
    54: rot_z(np.deg2rad(-90)),
}

#uwaga - kąty podane są w układzie marker_id1 (bazowego)!!!
#pozycja - podana jest w układzie kamer
#return x, y, z, pitch, yaw 
#equations taken from matrix got by multyplying rotation matrix Y by Z (by X(0)) - Y-Z-X

#matlab
# syms phi theta psi real

# Ry = @(a)[ cos(a)   0   sin(a);
#            0        1      0;
#           -sin(a)   0   cos(a) ];

# Rz = @(a)[ cos(a) -sin(a) 0;
#            sin(a)  cos(a) 0;
#            0       0      1 ];

# Rx = @(a)[ 1 0 0;
#            0 cos(a) -sin(a);
#            0 sin(a)  cos(a) ];

# R = Ry(phi) * Rz(theta)
# R = Ry(phi) * Rz(theta)*Rx(0)
# R2 =  Rz(theta) * Ry(phi);
# R3 = Ry(phi) * Rz(theta) * Rx(psi);

#output
# R = 
# [ cos(phi)*cos(theta), -cos(phi)*sin(theta), sin(phi)]
# [          sin(theta),           cos(theta),        0]
# [-cos(theta)*sin(phi),  sin(phi)*sin(theta), cos(phi)]

#uwaga   - kąty  podane są w układzie marker_id1 (bazowego)!!!
#pozycja - podana jest w układzie kamer
def get_marker_mutal_pose_YZ(found_markers, pairs):
    mutual_poses = {}

    for marker_id1, marker_id2 in pairs:
        if marker_id1 not in found_markers or marker_id2 not in found_markers:
            mutual_poses[(marker_id1, marker_id2)] = None
            continue

        home_R = found_markers[marker_id1]["qr_R"]
        home_tvec = found_markers[marker_id1]["qr_tvec"].reshape(3)

        target_R = found_markers[marker_id2]["qr_R"]
        target_tvec = found_markers[marker_id2]["qr_tvec"].reshape(3)

        R_mutual = home_R.T @ target_R
        T_mutual = target_tvec - home_tvec                # pozycja w układzie kamer
        # T_mutual = home_R.T @ (target_tvec - home_tvec) # pozycja w układzie markera_id1

        theta_y = np.arctan2(R_mutual[0,2], R_mutual[2,2])
        theta_z = np.arctan2(R_mutual[1,0], R_mutual[1,1])
        theta_y_deg = np.degrees(theta_y)
        theta_z_deg = np.degrees(theta_z)

        text = f"T_mutual x {T_mutual[0]:.2f} y {T_mutual[1]:.2f} z {T_mutual[2]:.2f} | R_mutual y:{theta_y_deg:.2f}° z:{theta_z_deg:.2f}°"
        print(text)

        mutual_poses[(marker_id1, marker_id2)] = (
            T_mutual[0],
            T_mutual[1],
            T_mutual[2],
            theta_y_deg,
            theta_z_deg,
        )

    return mutual_poses