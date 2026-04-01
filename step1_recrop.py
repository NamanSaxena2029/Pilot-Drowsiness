"""
STEP 1 — MEDIAPIPE RECROP
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker

MODEL_PATH = "face_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading face_landmarker.task...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)

options = FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.4,
    min_face_presence_confidence=0.4,
    min_tracking_confidence=0.4,
    running_mode=mp_vision.RunningMode.IMAGE
)
detector = FaceLandmarker.create_from_options(options)
print("[INFO] MediaPipe ready.")

INPUT_BASE  = "data/train_data"
OUTPUT_BASE = "data/cropped_mp"

os.makedirs(f"{OUTPUT_BASE}/drowsy",    exist_ok=True)
os.makedirs(f"{OUTPUT_BASE}/notdrowsy", exist_ok=True)

def get_face_crop(img, lms, pad=0.15):
    h, w = img.shape[:2]
    xs = [lm.x * w for lm in lms]
    ys = [lm.y * h for lm in lms]
    x1 = max(0, int(min(xs) - pad * w))
    y1 = max(0, int(min(ys) - pad * h))
    x2 = min(w, int(max(xs) + pad * w))
    y2 = min(h, int(max(ys) + pad * h))
    crop = img[y1:y2, x1:x2]
    return crop if (crop is not None and crop.size > 0) else None

total_saved  = {"drowsy": 0, "notdrowsy": 0}
total_failed = {"drowsy": 0, "notdrowsy": 0}

for label in ["drowsy", "notdrowsy"]:
    input_folder  = os.path.join(INPUT_BASE,  label)
    output_folder = os.path.join(OUTPUT_BASE, label)
    files = [f for f in os.listdir(input_folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    print(f"\n[INFO] Processing {label}: {len(files)} images")

    for fname in tqdm(files, desc=label):
        img_path = os.path.join(input_folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            total_failed[label] += 1
            continue

        rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        if not result.face_landmarks:
            total_failed[label] += 1
            continue

        lms  = result.face_landmarks[0]
        crop = get_face_crop(img, lms, pad=0.15)

        if crop is None:
            total_failed[label] += 1
            continue

        out_path = os.path.join(output_folder, fname)
        cv2.imwrite(out_path, crop)
        total_saved[label] += 1

detector.close()

print("\n" + "="*50)
print("RECROP COMPLETE")
print("="*50)
for label in ["drowsy", "notdrowsy"]:
    saved  = total_saved[label]
    failed = total_failed[label]
    total  = saved + failed
    print(f"{label:12s}: {saved}/{total} saved  ({failed} no-face/failed)")

print(f"\nOutput: data/cropped_mp/")
print("Ab step2_retrain.py chalao")