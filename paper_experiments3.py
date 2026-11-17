import os, io, re, glob, time, warnings, platform
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from math import comb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import features_lodo as F
CSV = "features_normal.csv"
META = {"driver","stream","frame","label","image","path","scenario","cond","condition","glasses"}
log = io.StringIO()
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n")
def build(norm="driver", norm_first=None):
    df = F.load(CSV)
    out = F.add_features(df, norm, norm_first=norm_first) if norm_first else F.add_features(df, norm)
    d2 = out[0] if isinstance(out, tuple) else out
    ok = re.compile(r"^(ear|ear_l|ear_r|mar|bl|jaw|yaw|pitch|roll)$|^perclos_\d+$|_(mean|std|min)_\d+$")
    cols = [c for c in d2.columns if c not in META and pd.api.types.is_numeric_dtype(d2[c]) and ok.search(c)]
    d2 = d2.copy(); d2[cols] = d2[cols].fillna(0.0); return d2, cols
def model(): return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))
def per_driver_acc(d2, f):
    r = {}
    for h in sorted(d2["driver"].unique()):
        tr, te = d2[d2["driver"] != h], d2[d2["driver"] == h]
        m = model().fit(tr[f].values, tr["label"].astype(int).values)
        r[h] = float((m.predict(te[f].values) == te["label"].astype(int).values).mean())
    return pd.Series(r)

P("=" * 70, "\nPART C  PER-DRIVER PAIRED DIFFERENCES vs. whole-recording normalisation\n", "=" * 70)
acc = {}
for name, kw in [("whole", dict(norm="driver")), ("K=300", dict(norm="driver", norm_first=300)),
                 ("K=900", dict(norm="driver", norm_first=900)), ("K=1800", dict(norm="driver", norm_first=1800)),
                 ("none", dict(norm="none"))]:
    d2, c2 = build(**kw); acc[name] = per_driver_acc(d2, list(F.pick(c2, "no_pose")))
A = pd.DataFrame(acc); P(A.round(3).to_string())
P("\ndifference vs whole (accuracy points) per driver, wins out of 4, exact two-sided sign test p:")
for n in ["K=300", "K=900", "K=1800", "none"]:
    d = A[n] - A["whole"]; w = int((d > 0).sum()); k = max(w, 4 - w)
    p = min(1.0, 2 * sum(comb(4, i) for i in range(k, 5)) / 2 ** 4)
    P(f"  {n:7s} diffs {[round(100*x,1) for x in d]}  mean {100*d.mean():+.1f}  wins {w}/4  sign-test p = {p:.3f}")
P("note: with n = 4 drivers the smallest possible two-sided sign-test p is 0.125, so no normalisation effect can be")
P("      'significant' at driver level; the table only shows how consistent the direction is across drivers.")

P("\n" + "=" * 70, "\nPART D  SINGLE-CORE CPU TIMING (stand-in for a slow CPU, NOT an edge device)\n", "=" * 70)
try:
    import psutil; psutil.Process().cpu_affinity([0]); P("process pinned to core 0 only")
except Exception as e:
    P("could not pin affinity:", e)
P("machine:", platform.processor() or platform.machine())
d2, c2 = build("driver"); f = list(F.pick(c2, "no_pose")); m = model().fit(d2[f].values, d2["label"].astype(int).values)
x1 = d2[f].values[:1]
for _ in range(100): m.predict_proba(x1)
t0 = time.perf_counter()
for _ in range(2000): m.predict_proba(x1)
t_lr = (time.perf_counter() - t0) / 2000 * 1000; P(f"logistic regression: {t_lr:.3f} ms/frame (1 core)")
raw = F.load(CSV); t_mp = None
try:
    import mediapipe as mp, cv2
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    task = glob.glob("**/*.task", recursive=True)[0]
    col = next(c for c in ("image", "path") if c in raw.columns)
    paths = [p for p in raw[col].astype(str).values[:3000:30] if os.path.exists(p)][:40]
    lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=task), num_faces=1, output_face_blendshapes=True))
    imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
    for im in imgs[:5]: lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=im))
    t0 = time.perf_counter()
    for im in imgs: lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=im))
    t_mp = (time.perf_counter() - t0) / len(imgs) * 1000
    P(f"MediaPipe FaceLandmarker (1 core): {t_mp:.1f} ms/frame -> pipeline about {t_mp + 0.06 + t_lr:.1f} ms/frame = {1000/(t_mp+0.06+t_lr):.0f} frames/s")
except Exception as e:
    P("MediaPipe skipped:", repr(e)[:150])
try:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1); tf.config.threading.set_inter_op_parallelism_threads(1)
    from tensorflow.keras import layers, models
    b = tf.keras.applications.MobileNetV2(input_shape=(160,160,3), include_top=False, pooling="avg", weights=None)
    i = layers.Input((3,160,160,3)); z = layers.LSTM(128)(layers.TimeDistributed(b)(i))
    cnn = models.Model(i, layers.Dense(1, activation="sigmoid")(z)); xb = np.random.rand(1,3,160,160,3).astype("float32")
    for _ in range(3): cnn.predict(xb, verbose=0)
    t0 = time.perf_counter()
    for _ in range(15): cnn.predict(xb, verbose=0)
    P(f"CNN-LSTM (1 core, 1 thread): {(time.perf_counter()-t0)/15*1000:.0f} ms per 3-frame window (random weights, same architecture)")
except Exception as e:
    P("CNN-LSTM skipped:", repr(e)[:150])
open("paper_results3.txt", "w", encoding="utf-8").write(log.getvalue()); P("\nsaved paper_results3.txt -- paste the whole output back to Claude")