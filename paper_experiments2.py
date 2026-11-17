import os, io, re, glob, time, pickle, platform, itertools, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import features_lodo as F

CSV = "features_normal.csv"
META = {"driver", "stream", "frame", "label", "image", "path", "scenario", "cond", "condition", "glasses"}
log = io.StringIO()
def P(*a):
    s = " ".join(str(x) for x in a); print(s); log.write(s + "\n")

def build(norm="driver", norm_first=None):
    df = F.load(CSV)
    out = F.add_features(df, norm, norm_first=norm_first) if norm_first else F.add_features(df, norm)
    d2 = out[0] if isinstance(out, tuple) else out
    ok = re.compile(r"^(ear|ear_l|ear_r|mar|bl|jaw|yaw|pitch|roll)$|^perclos_\d+$|_(mean|std|min)_\d+$")
    cols = [c for c in d2.columns if c not in META and pd.api.types.is_numeric_dtype(d2[c]) and ok.search(c)]
    d2 = d2.copy(); d2[cols] = d2[cols].fillna(0.0)
    return d2, cols

def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))

def oof(d2, feats, n_train=None, seed=0):
    """held-out predictions (0/1) per driver, as dict driver -> (y, pred)"""
    res = {}; drivers = sorted(d2["driver"].unique())
    for h in drivers:
        others = [d for d in drivers if d != h]
        if n_train:
            others = list(np.random.RandomState(seed).choice(others, n_train, replace=False))
        tr = d2[d2["driver"].isin(others)]; te = d2[d2["driver"] == h]
        m = model().fit(tr[feats].values, tr["label"].astype(int).values)
        res[h] = (te["label"].astype(int).values, m.predict(te[feats].values))
    return res

def boot(variants, B=500, block=300, seed=1):
    """paired block bootstrap of mean-over-drivers accuracy. variants: name -> oof dict (same rows)."""
    rng = np.random.RandomState(seed); names = list(variants); drivers = sorted(next(iter(variants.values())))
    corr = {n: {h: (variants[n][h][0] == variants[n][h][1]).astype(float) for h in drivers} for n in names}
    nb = {h: int(np.ceil(len(corr[names[0]][h]) / block)) for h in drivers}
    draws = {n: [] for n in names}
    for _ in range(B):
        acc = {n: [] for n in names}
        for h in drivers:
            ids = rng.randint(0, nb[h], nb[h])
            idx = np.concatenate([np.arange(i * block, min((i + 1) * block, len(corr[names[0]][h]))) for i in ids])
            for n in names: acc[n].append(corr[n][h][idx].mean())
        for n in names: draws[n].append(np.mean(acc[n]))
    return {n: np.array(v) for n, v in draws.items()}

def point(o): return float(np.mean([(y == p).mean() for y, p in o.values()]))
def ci(v): return np.percentile(v, [2.5, 97.5])

# ====================================================================== PART A
P("=" * 70, "\nPART A1  NORMALISATION VARIANTS WITH 95% BLOCK-BOOTSTRAP CIs (mean accuracy over 4 drivers)\n", "=" * 70)
variants = {}; feats_by = {}
for name, kw in [("whole recording", dict(norm="driver")),
                 ("running K=300", dict(norm="driver", norm_first=300)),
                 ("running K=900", dict(norm="driver", norm_first=900)),
                 ("running K=1800", dict(norm="driver", norm_first=1800)),
                 ("none", dict(norm="none"))]:
    try:
        d2, c2 = build(**kw); f = list(F.pick(c2, "no_pose")); variants[name] = oof(d2, f); feats_by[name] = (d2, f)
    except Exception as e:
        P("failed:", name, e)
dr = boot(variants)
base = "whole recording"
P(f"{'variant':20s} {'acc':>6s}  {'95% CI':>15s}   {'diff vs whole':>14s}  {'95% CI of diff':>17s}  P(diff<=0)")
for n in variants:
    a = point(variants[n]); lo, hi = ci(dr[n]); d = dr[n] - dr[base]; dlo, dhi = ci(d)
    P(f"{n:20s} {a:6.3f}  [{lo:.3f}, {hi:.3f}]   {np.mean(d):+14.3f}  [{dlo:+.3f}, {dhi:+.3f}]  {np.mean(d <= 0):.3f}")
P("\nper-driver accuracy of each variant:")
P(pd.DataFrame({n: {h: (y == p).mean() for h, (y, p) in variants[n].items()} for n in variants}).round(3).to_string())

P("\n" + "=" * 70, "\nPART A2  LR WITH 2 vs 3 TRAINING DRIVERS (whole-recording normalisation), 95% CIs\n", "=" * 70)
d2, f = feats_by[base]
drivers = sorted(d2["driver"].unique())
two = {}
for seed in range(3):                         # three random choices of which 2 of the 3 other drivers train
    two[f"2 train drivers (draw {seed})"] = oof(d2, f, n_train=2, seed=seed)
three = {"3 train drivers": oof(d2, f)}
allv = {**three, **two}; dr2 = boot(allv)
for n in allv:
    lo, hi = ci(dr2[n]); P(f"{n:28s} acc {point(allv[n]):.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
m2 = np.mean([point(v) for v in two.values()])
P(f"mean over the three 2-driver draws: {m2:.3f}")

# ====================================================================== PART B
P("\n" + "=" * 70, "\nPART B  END-TO-END CPU TIMING\n", "=" * 70)
P("machine:", platform.processor() or platform.machine(), "| cpu count:", os.cpu_count(), "| python", platform.python_version())
try:
    import psutil; rss = lambda: psutil.Process().memory_info().rss / 1e6
except Exception:
    rss = lambda: float("nan")
m = model().fit(d2[f].values, d2["label"].astype(int).values)

# B1: feature computation from a 60-frame buffer (what a live system must do per frame)
raw = F.load(CSV)
RB = [c for c in ["ear", "ear_l", "ear_r", "mar", "bl", "jaw"] if c in raw.columns]
buf = raw[RB].values[:60].astype(float); mu = buf.mean(0); sd = buf.std(0) + 1e-9
def live_features(b):
    z = (b - mu) / sd
    out = [z[-1]]
    ear_i = RB.index("ear")
    for w in (15, 60):
        zw = z[-w:]; closed = (b[-w:, ear_i] < 0.75 * 0.3).mean()
        out += [[closed], zw.mean(0), zw.std(0, ddof=1), [zw[:, ear_i].min()]]
    return np.concatenate([np.ravel(o) for o in out])
for _ in range(200): live_features(buf)
N = 5000; t0 = time.perf_counter()
for _ in range(N): live_features(buf)
t_feat = (time.perf_counter() - t0) / N * 1000
x1 = d2[f].values[:1]
for _ in range(200): m.predict_proba(x1)
t0 = time.perf_counter()
for _ in range(2000): m.predict_proba(x1)
t_lr = (time.perf_counter() - t0) / 2000 * 1000
P(f"rolling features from a 60-frame buffer (numpy, approximate): {t_feat:.3f} ms/frame")
P(f"logistic regression: {t_lr:.3f} ms/frame; model size {len(pickle.dumps(m)) / 1024:.1f} KB")

# B2: MediaPipe landmarks
t_mp = None
try:
    import mediapipe as mp, cv2
    from mediapipe.tasks import python as mpp
    from mediapipe.tasks.python import vision
    tasks = glob.glob("**/*.task", recursive=True)
    imgcol = next((c for c in ("image", "path") if c in raw.columns), None)
    paths = [p for p in raw[imgcol].astype(str).values[:3000:30] if os.path.exists(p)][:60] if imgcol else []
    if not tasks or not paths:
        P("MediaPipe timing skipped: need a *.task model in this folder tree and readable image paths in the CSV column 'image'/'path'.",
          "found tasks:", tasks[:2], "| images found:", len(paths))
    else:
        opt = vision.FaceLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path=tasks[0]), num_faces=1, output_face_blendshapes=True)
        lm = vision.FaceLandmarker.create_from_options(opt)
        imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
        for im in imgs[:5]: lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=im))
        t0 = time.perf_counter()
        for im in imgs: lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=im))
        t_mp = (time.perf_counter() - t0) / len(imgs) * 1000
        P(f"MediaPipe FaceLandmarker (+blendshapes), {len(imgs)} frames of size {imgs[0].shape[1]}x{imgs[0].shape[0]}: {t_mp:.1f} ms/frame (model: {tasks[0]})")
except Exception as e:
    P("MediaPipe timing skipped:", repr(e)[:200])
if t_mp is not None:
    tot = t_mp + t_feat + t_lr
    P(f"END-TO-END landmark pipeline: {tot:.1f} ms/frame  ->  {1000 / tot:.0f} frames/s on this CPU")

# B3: CNN-LSTM
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    cnn = None
    for pth in glob.glob("**/*.keras", recursive=True) + glob.glob("**/*.h5", recursive=True):
        try: cnn = tf.keras.models.load_model(pth, compile=False); P("loaded trained model:", pth); break
        except Exception: pass
    if cnn is None:
        base_m = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, pooling="avg", weights=None)
        inp = layers.Input((3, 160, 160, 3)); z = layers.TimeDistributed(base_m)(inp); z = layers.LSTM(128)(z)
        cnn = models.Model(inp, layers.Dense(1, activation="sigmoid")(z)); P("no saved model found: timing the same architecture with random weights (speed is identical)")
    xb = np.random.rand(1, 3, 160, 160, 3).astype("float32")
    r0 = rss()
    for _ in range(5): cnn.predict(xb, verbose=0)
    t0 = time.perf_counter()
    for _ in range(40): cnn.predict(xb, verbose=0)
    lc = (time.perf_counter() - t0) / 40 * 1000
    P(f"CNN-LSTM inference: {lc:.1f} ms per 3-frame window (batch=1, CPU) | parameters {cnn.count_params():,} (~{cnn.count_params() * 4 / 1e6:.1f} MB float32)")
    P(f"process memory with TF model loaded: {rss():.0f} MB (before first predict {r0:.0f} MB)")
except Exception as e:
    P("CNN-LSTM timing skipped:", repr(e)[:200], "\n  -> in the venv run:  pip install tensorflow   and run this script again")
P(f"process memory (landmark pipeline only, no TF) can be read from Task Manager; Python RSS now {rss():.0f} MB")

open("paper_results2.txt", "w", encoding="utf-8").write(log.getvalue())
P("\nsaved paper_results2.txt  --  paste the whole console output back to Claude")