import os, cv2, numpy as np
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarkerOptions, FaceLandmarker

# ---------------- LOW LIGHT ENHANCEMENT ----------------
def enhance_low_light(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()

    if brightness > 80:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    enhanced = cv2.merge((l,a,b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # extra boost
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.4, beta=20)

    return enhanced

# ---------------- IR SIMULATION ----------------
def simulate_ir(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# ---------------- MEDIAPIPE ----------------
MODEL_PATH = "face_landmarker.task"

options = FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_faces=1,
    min_face_detection_confidence=0.3,
    min_face_presence_confidence=0.3,
    min_tracking_confidence=0.3,
    running_mode=mp_vision.RunningMode.IMAGE
)

detector = FaceLandmarker.create_from_options(options)

# ---------------- PATH ----------------
INPUT_BASE  = "data/train_data"
OUTPUT_BASE = "data/cropped_mp_ll"   # NEW OUTPUT

os.makedirs(f"{OUTPUT_BASE}/drowsy", exist_ok=True)
os.makedirs(f"{OUTPUT_BASE}/notdrowsy", exist_ok=True)

def get_face_crop(img, lms, pad=0.2):
    h,w = img.shape[:2]
    xs = [lm.x*w for lm in lms]
    ys = [lm.y*h for lm in lms]

    x1 = max(0,int(min(xs)-pad*w))
    y1 = max(0,int(min(ys)-pad*h))
    x2 = min(w,int(max(xs)+pad*w))
    y2 = min(h,int(max(ys)+pad*h))

    crop = img[y1:y2, x1:x2]
    return crop if crop.size>0 else None

# ---------------- MAIN LOOP ----------------
for label in ["drowsy","notdrowsy"]:
    inp = os.path.join(INPUT_BASE,label)
    out = os.path.join(OUTPUT_BASE,label)

    files = [f for f in os.listdir(inp) if f.endswith((".jpg",".png",".jpeg"))]

    for fname in tqdm(files, desc=label):
        path = os.path.join(inp,fname)
        img = cv2.imread(path)

        if img is None:
            continue

        # STEP 1: enhance
        enhanced = enhance_low_light(img)

        # TRY 1: normal enhanced
        rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_img)

        # TRY 2: IR simulation fallback
        if not res.face_landmarks:
            ir = simulate_ir(img)
            rgb = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect(mp_img)

        if not res.face_landmarks:
            continue

        crop = get_face_crop(img, res.face_landmarks[0])

        if crop is not None:
            cv2.imwrite(os.path.join(out,fname), crop)

detector.close()

print("✅ LOW LIGHT RECROP DONE → data/cropped_mp_ll/")