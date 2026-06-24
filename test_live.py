import single_determine_pose
import stereo_determine_pose
import numpy as np
import time

determine_pose = single_determine_pose.single_determine_pose()

while True:
    found_markers = determine_pose.find_markers()

    if found_markers:
        if 50 in found_markers and 53 in found_markers:
            home_R = found_markers[53]["qr_R"]
            target_R = found_markers[50]["qr_R"]

            R_diff = home_R.T @ target_R

            theta_y = np.arctan2(R_diff[0,2], R_diff[2,2])
            theta_z = np.arctan2(R_diff[1,0], R_diff[1,1])
            theta_y_deg = np.degrees(theta_y)
            theta_z_deg = np.degrees(theta_z)

            text = f"R_diff y:{theta_y_deg:.2f}° z:{theta_z_deg:.2f}°"
            print(text)
            # time.sleep(0.5)
    
    mean_markers = determine_pose.get_mean(10)
    if mean_markers:
        if 50 in mean_markers and 53 in mean_markers:
            home_R = mean_markers[53]["mean_R"]
            target_R = mean_markers[50]["mean_R"]

            R_diff = home_R.T @ target_R

            theta_y = np.arctan2(R_diff[0,2], R_diff[2,2])
            theta_z = np.arctan2(R_diff[1,0], R_diff[1,1])
            theta_y_deg = np.degrees(theta_y)
            theta_z_deg = np.degrees(theta_z)

            text = f"R_diff mean y:{theta_y_deg:.2f}° z:{theta_z_deg:.2f}°"
            print(text)
            # time.sleep(0.5)

