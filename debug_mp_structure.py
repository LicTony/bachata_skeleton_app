import mediapipe
import os

mp_path = os.path.dirname(mediapipe.__file__)
print(f"MediaPipe Path: {mp_path}")
try:
    print("Contents:")
    for item in os.listdir(mp_path):
        print(f" - {item}")
except Exception as e:
    print(f"Error: {e}")
