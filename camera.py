import config
import cv2
import enum
import numpy as np
from pygrabber.dshow_graph import FilterGraph


class camera_position(enum.Enum):
    LEFT = "left/"
    RIGHT = "right/"
    SINGLE = "single/"


class Camera:
    def __init__(self, camera_name: str, width: int, height: int, position: camera_position):

        self.display_width = 680
        self.display_height = 480
        self.frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)

        if None in [camera_name, width, height, position]:
            print("Nie podano wszystkich parametrów kamery")
            return
        
        self.camera_ready = False
        self.camera_id = None
        self.width = width
        self.height = height
        self.position = position
        self.camera_name = camera_name
        
        if camera_name not in config.camera_names:
            print(f"Nieznana kamera: {camera_name}")
            return

        graph = FilterGraph()

        for i, name in enumerate(graph.get_input_devices()):
            if name == config.camera_names[camera_name]:
                self.camera_id = i
                break

        if self.camera_id is None:
            print(f"Nie można znaleźć kamery: {camera_name}")
            return
        
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            print(f"Nie można otworzyć kamery: {camera_name}")
            return

        self.configure_camera()
        self.camera_ready = True

    def configure_camera(self):
        for prop_id, value in config.camera_config_table:
            self.cap.set(prop_id, value)
        

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def get_camera_matrix(self):
        if self.camera_ready:
            calib_path = f"{config.calib_results_path}{self.position.value}camera_matrix.npy"

            camera_matrix = np.load(calib_path)
            if camera_matrix is not None:
                return camera_matrix
            
        return np.eye(3)
    
    def get_dist_coeffs(self):
        if self.camera_ready:
            calib_path = f"{config.calib_results_path}{self.position.value}dist_coeffs.npy"

            dist_coeffs = np.load(calib_path)
            if dist_coeffs is not None:
                return dist_coeffs
            
        return np.zeros((5,))
    
    def get_calib_path(self):
        return f"{config.calib_results_path}{self.position.value}"
    
    def get_frame(self):
        if self.camera_ready:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                return frame
        
        img = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)

        cv2.putText(
            img,
            "brak kamery",
            (50, 240),              # pozycja tekstu
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,                            # skala
            (255, 255, 255),                # biały kolor
            3,                              # grubość
            cv2.LINE_AA
        )
        self.frame = img
        return img

    def release(self):
        if self.camera_ready:
            self.cap.release()

    def __del__(self):
        self.release()

    def display_frame(self):
        scale = self.display_width/self.width
        scaled_frame = cv2.resize(self.frame, None, fx=scale, fy=scale)
        cv2.imshow(self.camera_name+"_"+str(self.width) +"_"+str(self.height)+"px", scaled_frame)