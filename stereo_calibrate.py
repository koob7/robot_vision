from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import cv2.aruco as aruco
import numpy as np

import config


@dataclass
class CharucoDetection:
    corners: np.ndarray
    ids: np.ndarray


def build_board():
    aruco_dict = cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT)
    return cv2.aruco.CharucoBoard(
        (config.columnsX, config.rowsY),
        config.squareLength,
        config.markerLength,
        aruco_dict,
    )


def get_board_corners(board):
    if hasattr(board, "getChessboardCorners"):
        corners = board.getChessboardCorners()
    else:
        corners = board.chessboardCorners
    return np.asarray(corners, dtype=np.float32)


def detect_charuco(image, detector, board, min_corners):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return None

    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners,
        ids,
        gray,
        board,
    )

    if retval is None or retval < min_corners or charuco_ids is None:
        return None

    return CharucoDetection(corners=charuco_corners, ids=charuco_ids)


def collect_stereo_pairs(left_index, right_index, board, output_dir, min_corners):
    aruco_dict = aruco.getPredefinedDictionary(config.ARUCO_DICT)
    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

    left_dir = output_dir / "left"
    right_dir = output_dir / "right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    cap_left = cv2.VideoCapture(left_index)
    cap_right = cv2.VideoCapture(right_index)

    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)


    if not cap_left.isOpened() or not cap_right.isOpened():
        cap_left.release()
        cap_right.release()
        raise RuntimeError("Nie można otworzyć obu kamer.")

    saved_pairs = []
    pair_index = 0

    print("Sterowanie: spacja = zapis pary, q / Esc = koniec zbierania")

    try:
        while True:
            ret_left, frame_left = cap_left.read()
            ret_right, frame_right = cap_right.read()

            if not ret_left or not ret_right:
                print("Utracono obraz z jednej z kamer.")
                break

            left_detection = detect_charuco(frame_left, detector, board, min_corners)
            right_detection = detect_charuco(frame_right, detector, board, min_corners)

            left_display = frame_left.copy()
            right_display = frame_right.copy()

            if left_detection is not None:
                aruco.drawDetectedCornersCharuco(
                    left_display,
                    left_detection.corners,
                    left_detection.ids,
                )
            if right_detection is not None:
                aruco.drawDetectedCornersCharuco(
                    right_display,
                    right_detection.corners,
                    right_detection.ids,
                )

            status_left = "OK" if left_detection is not None else "BRAK"
            status_right = "OK" if right_detection is not None else "BRAK"

            cv2.putText(
                left_display,
                f"Left camera: {status_left}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if left_detection is not None else (0, 0, 255),
                2,
            )
            cv2.putText(
                right_display,
                f"Right camera: {status_right}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if right_detection is not None else (0, 0, 255),
                2,
            )
            cv2.putText(
                left_display,
                "SPACE = save pair | Q/Esc = finish",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Stereo left", left_display)
            cv2.imshow("Stereo right", right_display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            if key == 32:
                if left_detection is None or right_detection is None:
                    print("Pominięto zapis: plansza nie jest wykryta w obu kamerach.")
                    continue

                left_path = left_dir / f"left_{pair_index:04d}.png"
                right_path = right_dir / f"right_{pair_index:04d}.png"

                cv2.imwrite(str(left_path), frame_left)
                cv2.imwrite(str(right_path), frame_right)

                saved_pairs.append((left_path, right_path))
                pair_index += 1
                print(f"Zapisano parę {pair_index}: {left_path.name}, {right_path.name}")
    finally:
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()

    return saved_pairs


def load_charuco_measurements(image_paths, board, min_corners):
    aruco_dict = aruco.getPredefinedDictionary(config.ARUCO_DICT)
    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

    corners_list = []
    ids_list = []
    image_size = None

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Pomijam {path}: nie można wczytać obrazu.")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])

        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None:
            continue

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners,
            ids,
            gray,
            board,
        )

        if retval is None or retval < min_corners or charuco_ids is None:
            continue

        corners_list.append(charuco_corners)
        ids_list.append(charuco_ids)

    if image_size is None:
        raise RuntimeError("Nie znaleziono żadnych poprawnych obrazów do kalibracji.")

    return corners_list, ids_list, image_size


def calibrate_mono_camera(corners_list, ids_list, board, image_size):
    retval, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(
        corners_list,
        ids_list,
        board,
        image_size,
        None,
        None,
    )
    return retval, camera_matrix, dist_coeffs


def build_stereo_measurements(left_paths, right_paths, board, min_corners):
    aruco_dict = aruco.getPredefinedDictionary(config.ARUCO_DICT)
    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
    board_corners = get_board_corners(board)

    object_points = []
    image_points_left = []
    image_points_right = []

    for left_path, right_path in zip(left_paths, right_paths):
        left_image = cv2.imread(str(left_path))
        right_image = cv2.imread(str(right_path))

        if left_image is None or right_image is None:
            continue

        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        left_corners, left_ids_raw, _ = detector.detectMarkers(left_gray)
        right_corners, right_ids_raw, _ = detector.detectMarkers(right_gray)

        if left_ids_raw is None or right_ids_raw is None:
            continue

        left_retval, left_charuco_corners, left_charuco_ids = cv2.aruco.interpolateCornersCharuco(
            left_corners,
            left_ids_raw,
            left_gray,
            board,
        )
        right_retval, right_charuco_corners, right_charuco_ids = cv2.aruco.interpolateCornersCharuco(
            right_corners,
            right_ids_raw,
            right_gray,
            board,
        )

        if (
            left_retval is None
            or right_retval is None
            or left_retval < min_corners
            or right_retval < min_corners
            or left_charuco_ids is None
            or right_charuco_ids is None
        ):
            continue

        left_ids = left_charuco_ids.flatten()
        right_ids = right_charuco_ids.flatten()
        common_ids = np.intersect1d(left_ids, right_ids)

        if common_ids.size < min_corners:
            continue

        left_lookup = {int(marker_id): idx for idx, marker_id in enumerate(left_ids)}
        right_lookup = {int(marker_id): idx for idx, marker_id in enumerate(right_ids)}

        object_view = []
        left_view = []
        right_view = []

        for marker_id in common_ids:
            left_idx = left_lookup[int(marker_id)]
            right_idx = right_lookup[int(marker_id)]

            object_view.append(board_corners[int(marker_id)])
            left_view.append(left_charuco_corners[left_idx, 0])
            right_view.append(right_charuco_corners[right_idx, 0])

        object_points.append(np.asarray(object_view, dtype=np.float32).reshape(-1, 1, 3))
        image_points_left.append(np.asarray(left_view, dtype=np.float32).reshape(-1, 1, 2))
        image_points_right.append(np.asarray(right_view, dtype=np.float32).reshape(-1, 1, 2))

    if not object_points:
        raise RuntimeError("Brak wspólnych punktów ChArUco do kalibracji stereo.")

    return object_points, image_points_left, image_points_right


def save_calibration(
    output_dir,
    left_name,
    right_name,
    left_camera_matrix,
    left_dist_coeffs,
    right_camera_matrix,
    right_dist_coeffs,
    R,
    T,
):
    left_dir = output_dir / left_name
    right_dir = output_dir / right_name
    stereo_dir = output_dir / "stereo"

    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    stereo_dir.mkdir(parents=True, exist_ok=True)

    np.save(left_dir / "camera_matrix.npy", left_camera_matrix)
    np.save(left_dir / "dist_coeffs.npy", left_dist_coeffs)
    np.save(right_dir / "camera_matrix.npy", right_camera_matrix)
    np.save(right_dir / "dist_coeffs.npy", right_dist_coeffs)
    np.save(stereo_dir / "R.npy", R)
    np.save(stereo_dir / "T.npy", T)


def main():
    parser = argparse.ArgumentParser(
        description="Capture synchronized stereo images, calibrate both cameras, and save stereo extrinsics."
    )
    parser.add_argument("--left-camera", type=int, default=2, help="Index of the left camera")
    parser.add_argument("--right-camera", type=int, default=0, help="Index of the right camera")
    parser.add_argument("--left-name", default="mx_brio2", help="Folder name for left camera calibration")
    parser.add_argument("--right-name", default="mx_brio", help="Folder name for right camera calibration")
    parser.add_argument("--output-root", default="calib_images_stereo", help="Folder for captured stereo pairs")
    parser.add_argument("--min-corners", type=int, default=12, help="Minimum detected ChArUco corners per image")
    parser.add_argument("--min-pairs", type=int, default=10, help="Recommended minimum number of captured pairs")
    args = parser.parse_args()

    board = build_board()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    capture_dir = Path(args.output_root) / timestamp

    saved_pairs = collect_stereo_pairs(
        args.left_camera,
        args.right_camera,
        board,
        capture_dir,
        args.min_corners,
    )

    if len(saved_pairs) < args.min_pairs:
        print(f"Zebrano tylko {len(saved_pairs)} par. Wynik może być słaby.")

    left_paths = [pair[0] for pair in saved_pairs]
    right_paths = [pair[1] for pair in saved_pairs]

    left_corners, left_ids, left_size = load_charuco_measurements(left_paths, board, args.min_corners)
    right_corners, right_ids, right_size = load_charuco_measurements(right_paths, board, args.min_corners)

    if left_size != right_size:
        print("Uwaga: rozdzielczości obrazów wejściowych różnią się między kamerami.")

    left_ret, left_camera_matrix, left_dist_coeffs = calibrate_mono_camera(
        left_corners,
        left_ids,
        board,
        left_size,
    )
    right_ret, right_camera_matrix, right_dist_coeffs = calibrate_mono_camera(
        right_corners,
        right_ids,
        board,
        right_size,
    )

    object_points, image_points_left, image_points_right = build_stereo_measurements(
        left_paths,
        right_paths,
        board,
        args.min_corners,
    )

    stereo_ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        object_points,
        image_points_left,
        image_points_right,
        left_camera_matrix,
        left_dist_coeffs,
        right_camera_matrix,
        right_dist_coeffs,
        left_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    save_calibration(
        Path("stored_calibrations"),
        args.left_name,
        args.right_name,
        left_camera_matrix,
        left_dist_coeffs,
        right_camera_matrix,
        right_dist_coeffs,
        R,
        T,
    )

    print("Kalibracja zakończona.")
    print(f"Left reprojection error: {left_ret}")
    print(f"Right reprojection error: {right_ret}")
    print(f"Stereo reprojection error: {stereo_ret}")
    print("Zapisano:")
    print(f"  stored_calibrations/{args.left_name}/camera_matrix.npy")
    print(f"  stored_calibrations/{args.left_name}/dist_coeffs.npy")
    print(f"  stored_calibrations/{args.right_name}/camera_matrix.npy")
    print(f"  stored_calibrations/{args.right_name}/dist_coeffs.npy")
    print("  stored_calibrations/stereo/R.npy")
    print("  stored_calibrations/stereo/T.npy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())