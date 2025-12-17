import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import requests

MODEL_PATH = "pose_landmarker_full.task"

# Ensure model exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    response = requests.get(url)
    if response.status_code == 200:
        with open(MODEL_PATH, 'wb') as f:
            f.write(response.content)
        print("Model downloaded.")
    else:
        print(f"Failed to download model: {response.status_code}")

def run_test():
    try:
        print("Initializing PoseLandmarker with CPU Delegate...")
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH, delegate=python.BaseOptions.Delegate.CPU)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO
        )
        
        landmarker = vision.PoseLandmarker.create_from_options(options)
        print("PoseLandmarker created.")

        # Create dummy image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        
        print("Running detect_for_video...")
        for i in range(5):
            print(f"Frame {i}")
            timestamp_ms = i * 33
            landmarker.detect_for_video(mp_image, timestamp_ms)
            
        print("Success!")
        landmarker.close()

    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()
