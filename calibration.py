import config
import cv2
import numpy as np
from pathlib import Path

class CharucoDetection:
    def __init__(self, corners: np.ndarray, ids: np.ndarray, image_size: tuple):
        self.corners = corners
        self.ids = ids
        self.image_size = image_size

class calibration:
    def __init__(self, name, min_corners, scale_factor):

        self.name = name
        self.scale_factor = scale_factor
        self.image_size = None
        self.min_corners = min_corners
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT)
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict)
        self.board = cv2.aruco.CharucoBoard(
            (config.columnsX, config.rowsY),
            config.squareLength,
            config.markerLength,
            self.aruco_dict
        )

    def get_board_corners(self):
        if hasattr(self.board, "getChessboardCorners"):
            corners = self.board.getChessboardCorners()
        else:
            corners = self.board.chessboardCorners
        return np.asarray(corners, dtype=np.float32)

    def detect_charuco(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None:
            return None

        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            corners,
            ids,
            gray,
            self.board,
        )

        if retval is None or retval < self.min_corners or charuco_ids is None:
            return None

        return CharucoDetection(corners=charuco_corners, ids=charuco_ids, image_size=(image.shape[1], image.shape[0]))

    def load_charuco_measurements(self, image_paths):
        image_size = None
        corners_list = []
        ids_list = []

        images_path = list(Path(image_paths).glob("*.png"))

        for path in images_path:
            image = cv2.imread(str(path))
            if image is None:
                print(f"Pomijam {path}: nie można wczytać obrazu.")
                continue

            detection = self.detect_charuco(image)
            if detection is None:
                continue
            
            if image_size is None:
                image_size = detection.image_size

            if image_size != detection.image_size:
                print(f"Pomijam {path}: rozmiar obrazu różni się od poprzednich.")
                continue

            corners_list.append(detection.corners)
            ids_list.append(detection.ids)

        return corners_list, ids_list, image_size
    
    def calibrate_camera(self, corners_list, ids_list, image_size):
        if None in (corners_list, ids_list, image_size):
            print("Brak danych do kalibracji.")
            return False, None, None, None, None
        return cv2.aruco.calibrateCameraCharuco(corners_list, ids_list, self.board, image_size, None, None)
    def display_next_frame (self, frame):
        detection = self.detect_charuco(frame)

        display_frame = frame.copy()

        if detection is not None:
            cv2.aruco.drawDetectedCornersCharuco(
                display_frame,
                detection.corners,
                detection.ids,
                (0, 255, 0),                    # kolor
            )

        status = "OK" if detection is not None else "Brak"

        reversed_scale = 1/self.scale_factor

        cv2.putText(
            display_frame,
            f"Detekcja: {status}",
            (int(10*reversed_scale), int(30*reversed_scale)),                       # pozycja tekstu
            cv2.FONT_HERSHEY_SIMPLEX,
            reversed_scale,                            # skala
            (0, 255, 0) if detection is not None else (0, 0, 255),  # kolor
            int(2*reversed_scale),                              # grubość
            cv2.LINE_AA
        )

        cv2.putText(
            display_frame,
            "SPACE = save pair | Q/Esc = finish",
            (int(20*reversed_scale), int(80*reversed_scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            reversed_scale,
            (255, 255, 255),
            int(2*reversed_scale),
        )

        display_frame = cv2.resize(display_frame, None, fx=self.scale_factor, fy=self.scale_factor)
        cv2.imshow(f"{self.name}_{frame.shape[1]}_{frame.shape[0]}pxq", display_frame)
        key = cv2.waitKey(1) & 0xFF

        return detection is not None, key
            