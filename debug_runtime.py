import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import ctypes

try:
    print("Testing ctypes.cdll.msvcrt.free...")
    try:
        print(f"Address of free: {ctypes.cdll.msvcrt.free}")
    except Exception as e:
        print(f"ctypes.cdll.msvcrt.free failed: {e}")

    print("Creating numpy image...")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    print("Creating MP Image...")
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    print("MP Image created successfully.")

except Exception as e:
    import traceback
    traceback.print_exc()
