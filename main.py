import camera
import cv2

camera_testowa = camera.Camera(
    camera_name="mx_brio_for_business", 
    width=3840,
    height=2160,
    position=camera.camera_position.LEFT
)

frame = camera_testowa.get_frame()

camera_testowa.display_frame()
wait_key = cv2.waitKey(0)