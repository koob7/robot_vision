from pathlib import Path
import config

import os

base_dir = os.path.dirname(os.path.abspath(__file__))

path = f"{base_dir}\\{config.calib_images_path}\\single\\mx_brio_for_business\\{0:03d}.png"

print("PATH:", path)
print("PARENT:", Path(path).parent)

Path(path).parent.mkdir(parents=True, exist_ok=True)

print("EXISTS:", Path(path).parent.exists())