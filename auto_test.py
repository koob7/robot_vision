import serial
import time
import csv
import numpy as np
from pathlib import Path
import stereo_determine_pose
import single_determine_pose


# --------------------------------------------------
# Konfiguracja
# --------------------------------------------------

PORT = 'COM3'
BAUDRATE = 1282000

STABILIZATION_TIME = 1.0  # s


czas_us = 50000

determine_pose = stereo_determine_pose.stereo_determine_pose()




# --------------------------------------------------
# Funkcje
# --------------------------------------------------

def calculate_angles(markers):
    """
    Zwraca (theta_y_deg, theta_z_deg)
    """
    if markers is None:
        return None

    if 50 not in markers or 53 not in markers:
        return None

    home_R = markers[53]
    target_R = markers[50]

    R_diff = home_R.T @ target_R

    theta_y = np.arctan2(R_diff[0, 2], R_diff[2, 2])
    theta_z = np.arctan2(R_diff[1, 0], R_diff[1, 1])

    return (
        np.degrees(theta_y),
        np.degrees(theta_z)
    )


def get_current_angles():
    found_markers = determine_pose.find_markers()

    if not found_markers:
        return None

    if 50 not in found_markers or 53 not in found_markers:
        return None

    home_R = found_markers[53]["qr_R"]
    target_R = found_markers[50]["qr_R"]

    return calculate_angles({
        53: home_R,
        50: target_R
    })


def get_mean_angles():
    mean_markers = determine_pose.get_mean(10)

    if not mean_markers:
        return None

    if 50 not in mean_markers or 53 not in mean_markers:
        return None

    home_R = mean_markers[53]["mean_R"]
    target_R = mean_markers[50]["mean_R"]

    return calculate_angles({
        53: home_R,
        50: target_R
    })


# --------------------------------------------------
# Generowanie zakresu:
# 0 -> 30 -> -30
# --------------------------------------------------

positions1 = list(range(-90, 90, 10))
positions2 = list(range(-40, 40))

previus_i = 0
previus_j = 0

determine_pose.find_markers()
print(f"oczekiwanie na kamerę {determine_pose.get_name()}...")
time.sleep(10)

# --------------------------------------------------
# CSV
# --------------------------------------------------

csv_filename = f"csv\\{determine_pose.get_name()}.csv"
Path(csv_filename).parent.mkdir(parents=True, exist_ok=True)

with serial.Serial(PORT, BAUDRATE, timeout=1) as ser, open(csv_filename, "w", newline="") as csvfile:

    writer = csv.writer(csvfile, delimiter=';')



    writer.writerow([
        "i_deg",
        "j_deg",
        "theta_y",
        "theta_z",
        "mean_theta_y",
        "mean_theta_z"
    ])

    for i in positions1:
        for j in positions2:
            diff = max(abs(i - previus_i), abs(j - previus_j))
            cmd = (
                f"cmd 2 0 89000 89000 "
                f"{j * 1000} "
                f"{i * 1000} "
                f"0 "
                f"{diff*czas_us} "
                f"0x37373737\r\n"
            )

            print(cmd.strip())

            ser.write(cmd.strip().encode("ascii"))

            time.sleep(diff*czas_us/1000000)

            # stabilizacja
            time.sleep(STABILIZATION_TIME)

            # pomiar chwilowy
            current = get_current_angles()

            if current is None:
                theta_y = None
                theta_z = None
            else:
                theta_y, theta_z = current

            # pomiar średni
            mean = get_mean_angles()

            if mean is None:
                mean_theta_y = None
                mean_theta_z = None
            else:
                mean_theta_y, mean_theta_z = mean

            writer.writerow([
                i,
                j,
                theta_y,
                theta_z,
                mean_theta_y,
                mean_theta_z
            ])

            csvfile.flush()

            print(
                f"i={i:3d} "
                f"j={j:3d} "
                f"theta_y={theta_y} "
                f"theta_z={theta_z} "
                f"mean_y={mean_theta_y} "
                f"mean_z={mean_theta_z}"
            )

            previus_i = i
            previus_j = j

    cmd = ( 
        f"cmd 2 0 89000 89000 "
        f"{0 * 1000} "
        f"{0 * 1000} "
        f"0 "
        f"{czas_us*10} "
        f"0x37373737\r\n"
    )

    print(cmd.strip())
    ser.write(cmd.strip().encode("ascii"))

print(f"Zapisano: {csv_filename}")