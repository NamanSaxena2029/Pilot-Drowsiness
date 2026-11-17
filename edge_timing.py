import sys, glob, os, time, platform
import numpy as np, cv2, mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import psutil
folder, task = sys.argv[1], sys.argv[2]
paths = sorted(glob.glob(os.path.join(folder, "*.jpg")) + glob.glob(os.path.join(folder, "*.png")))[:60]
assert paths, "no images found"
print("device:", platform.platform(), "| cpu:", platform.processor() or platform.machine(), "| cores:", os.cpu_count(),
      "| RAM total %.1f GB" % (psutil.virtual_memory().total / 1e9))
lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path=task), num_faces=1, output_face_blendshapes=True))
imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
for im in imgs[:5]: lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=im))
t0 = time.perf_counter()
for im in imgs: lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=im))
t_mp = (time.perf_counter() - t0) / len(imgs) * 1000
buf = np.random.rand(60, 6); mu = buf.mean(0); sd = buf.std(0) + 1e-9
def feats(b):
    z = (b - mu) / sd; o = [z[-1]]
    for w in (15, 60): zw = z[-w:]; o += [[(b[-w:, 0] < .2).mean()], zw.mean(0), zw.std(0, ddof=1), [zw[:, 0].min()]]
    return np.concatenate([np.ravel(x) for x in o])
for _ in range(100): feats(buf)
t0 = time.perf_counter()
for _ in range(2000): feats(buf)
t_f = (time.perf_counter() - t0) / 2000 * 1000
X = np.random.randn(500, 26); y = np.arange(500) % 2
m = make_pipeline(StandardScaler(), LogisticRegression(C=.5, max_iter=500)).fit(X, y); x1 = X[:1]
for _ in range(100): m.predict_proba(x1)
t0 = time.perf_counter()
for _ in range(1000): m.predict_proba(x1)
t_l = (time.perf_counter() - t0) / 1000 * 1000
tot = t_mp + t_f + t_l
print(f"MediaPipe {t_mp:.1f} ms | features {t_f:.3f} ms | LR {t_l:.3f} ms | total {tot:.1f} ms/frame = {1000/tot:.1f} frames/s")
print("process RSS: %.0f MB" % (psutil.Process().memory_info().rss / 1e6))