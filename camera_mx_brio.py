import cv2
import os
import time

import config

# --- KONFIG ---
camera_source = 0
save_dir = f"calib_images_8x5x35/{config.current_camera}"

os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(camera_source)

if not cap.isOpened():
    raise RuntimeError("Nie można otworzyć kamery MX Brio")

# --- USTAWIENIA ---
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, -8)
cap.set(cv2.CAP_PROP_GAIN, 80)

cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)
cap.set(cv2.CAP_PROP_CONTRAST, 150)
cap.set(cv2.CAP_PROP_SATURATION, 0)
cap.set(cv2.CAP_PROP_SHARPNESS, 120)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

img_counter = 0
last_frame = None

# --- CALLBACK MYSZY ---
def mouse_callback(event, x, y, flags, param):
    global img_counter, last_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        if last_frame is not None:
            filename = os.path.join(
                save_dir,
                f"img_{time.strftime('%Y%m%d_%H%M%S')}_{img_counter:03d}.png"
            )
            cv2.imwrite(filename, last_frame)
            print(f"[CLICK] Zapisano: {filename}")
            img_counter += 1

cv2.namedWindow("MX Brio Capture")
cv2.setMouseCallback("MX Brio Capture", mouse_callback)

print("Kliknij LPM aby zrobić zdjęcie | Q aby wyjść")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Błąd odczytu klatki")
        break

    last_frame = frame.copy()

    # scale = 2.0
    # resized = cv2.resize(frame, None, fx=scale, fy=scale)

    cv2.imshow("MX Brio Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()