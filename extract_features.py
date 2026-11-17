import os, re, sys, argparse, math
import numpy as np, pandas as pd

NAME_RE = re.compile(r"^(?P<driver>\d+)_(?P<rest>.+)_(?P<frame>\d+)_(?:not)?drowsy\.\w+$", re.I)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
COLS = ["detected", "ear_l", "ear_r", "ear", "mar", "yaw", "pitch", "roll", "bl_l", "bl_r", "jaw"]

# MediaPipe FaceMesh landmark indices
L_EYE = [362, 385, 387, 263, 373, 380]   # p1..p6 (p1,p4 = corners)
R_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_V = [(13, 14), (82, 87), (312, 317)]
MOUTH_H = (78, 308)
NOSE, CHIN, FOREHEAD, EYE_L_OUT, EYE_R_OUT = 1, 152, 10, 263, 33

_det = None


def _dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _ear(p):
    return (_dist(p[1], p[5]) + _dist(p[2], p[4])) / (2.0 * _dist(p[0], p[3]) + 1e-9)


def features_from_landmarks(lm, blend):
    """lm: (478,2) array of x,y in PIXELS. blend: dict name->score or {}."""
    ear_l = _ear([lm[i] for i in L_EYE])
    ear_r = _ear([lm[i] for i in R_EYE])
    mv = np.mean([_dist(lm[a], lm[b]) for a, b in MOUTH_V])
    mar = mv / (_dist(lm[MOUTH_H[0]], lm[MOUTH_H[1]]) + 1e-9)

    eye_mid = (lm[EYE_L_OUT] + lm[EYE_R_OUT]) / 2.0
    face_w = _dist(lm[EYE_L_OUT], lm[EYE_R_OUT]) + 1e-9
    face_h = _dist(lm[FOREHEAD], lm[CHIN]) + 1e-9
    yaw = (lm[NOSE][0] - eye_mid[0]) / face_w                 # left/right turn proxy
    pitch = (lm[NOSE][1] - eye_mid[1]) / face_h               # up/down (nodding) proxy
    dx, dy = lm[EYE_L_OUT][0] - lm[EYE_R_OUT][0], lm[EYE_L_OUT][1] - lm[EYE_R_OUT][1]
    roll = math.degrees(math.atan2(dy, dx))                   # head tilt
    return [1, ear_l, ear_r, (ear_l + ear_r) / 2, mar, yaw, pitch, roll,
            blend.get("eyeBlinkLeft", np.nan), blend.get("eyeBlinkRight", np.nan), blend.get("jawOpen", np.nan)]


def _init_worker(model_path):
    global _det
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        num_faces=1, output_face_blendshapes=True,
        min_face_detection_confidence=0.3, min_face_presence_confidence=0.3,
        running_mode=mp_vision.RunningMode.IMAGE)
    _det = mp_vision.FaceLandmarker.create_from_options(opts)


def _process(path):
    import cv2, mediapipe as mp
    nan_row = [0] + [np.nan] * (len(COLS) - 1)
    img = cv2.imread(path)
    if img is None:
        return nan_row
    h, w = img.shape[:2]
    # tiny crops: upscale so the landmark model has enough pixels
    if min(h, w) < 160:
        s = 160.0 / min(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s))); h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = _det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not res.face_landmarks:
        return nan_row
    lm = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
    blend = {b.category_name: b.score for b in res.face_blendshapes[0]} if res.face_blendshapes else {}
    return features_from_landmarks(lm, blend)


def build_index(base):
    rows, bad = [], 0
    for label, folder in enumerate(["notdrowsy", "drowsy"]):
        d = os.path.join(base, folder)
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(IMG_EXT):
                continue
            m = NAME_RE.match(f)
            if not m:
                bad += 1; continue
            rows.append([os.path.join(d, f), label, m.group("driver"),
                         f"{m.group('driver')}_{m.group('rest')}", int(m.group("frame"))])
    if bad:
        print(f"[warn] {bad} files skipped (name did not match pattern)")
    return pd.DataFrame(rows, columns=["image", "label", "driver", "stream", "frame"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/cropped_mp")
    ap.add_argument("--tag", default="normal")
    ap.add_argument("--model", default="face_landmarker.task")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--test", action="store_true", help="only 200 images")
    a = ap.parse_args()

    from tqdm import tqdm
    df = build_index(a.base)
    if a.test:
        df = df.sample(200, random_state=0).reset_index(drop=True)
    print(f"{len(df)} images | drivers: {sorted(df.driver.unique())}")

    paths = df.image.tolist()
    if a.workers > 1:
        import multiprocessing as mp
        with mp.Pool(a.workers, initializer=_init_worker, initargs=(a.model,)) as pool:
            out = list(tqdm(pool.imap(_process, paths, chunksize=64), total=len(paths)))
    else:
        _init_worker(a.model)
        out = [_process(p) for p in tqdm(paths)]

    feat = pd.DataFrame(out, columns=COLS)
    res = pd.concat([df.reset_index(drop=True), feat], axis=1)
    out_csv = f"features_{a.tag}{'_test' if a.test else ''}.csv"
    res.to_csv(out_csv, index=False)

    det = res.detected.mean()
    print(f"\nSaved {out_csv} | face found in {det*100:.1f}% of images")
    print(res.groupby("driver").detected.mean().round(3).to_string())
    print("\nfeature means by label (0=notdrowsy, 1=drowsy):")
    print(res[res.detected == 1].groupby("label")[["ear", "mar", "yaw", "pitch", "bl_l", "bl_r", "jaw"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()