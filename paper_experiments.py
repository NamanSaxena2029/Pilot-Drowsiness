import time, sys, io, pickle, warnings, re
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
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
    nn = int(d2[cols].isna().sum().sum())
    if nn:
        print("note: filling", nn, "NaN feature cells with 0")
        d2 = d2.copy(); d2[cols] = d2[cols].fillna(0.0)
    return d2, cols

def model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000))

def metrics(y, p, s):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    try: auc = roc_auc_score(y, s)
    except Exception: auc = np.nan
    return dict(acc=(tp + tn) / len(y), recall=rec, specificity=spec, bal_acc=(rec + spec) / 2, auc=auc)

def lodo(d2, feats, n_train=None, seed=0):
    rows = []
    drivers = sorted(d2["driver"].unique())
    for h in drivers:
        tr = d2[d2["driver"] != h]; te = d2[d2["driver"] == h]
        if n_train:
            others = [d for d in drivers if d != h]
            rng = np.random.RandomState(seed)
            keep = list(rng.choice(others, n_train, replace=False))
            tr = tr[tr["driver"].isin(keep)]
        m = model().fit(tr[feats].values, tr["label"].astype(int).values)
        s = m.predict_proba(te[feats].values)[:, 1]
        r = metrics(te["label"].astype(int).values, (s >= 0.5).astype(int), s); r["driver"] = h; r["n"] = len(te)
        rows.append(r)
    t = pd.DataFrame(rows).set_index("driver")
    t.loc["MEAN"] = t.drop(columns="n").mean().tolist() + [t["n"].sum()]
    return t

def show(t):
    P(t.round(3).to_string())

# ------------------------------------------------------------------ T1
P("=" * 70, "\nT1  PER-DRIVER LODO, logistic regression, per-driver norm, no_pose\n", "=" * 70)
d_drv, cols = build("driver")
feats_final = list(F.pick(cols, "no_pose"))
P("candidate features:", len(cols), "(expect 33) | final no_pose:", len(feats_final), "(expect 26)")
t1 = lodo(d_drv, feats_final); show(t1)

# ------------------------------------------------------------------ T2
P("\n" + "=" * 70, "\nT2  NORMALISATION VARIANTS (mean over 4 held-out drivers)\n", "=" * 70)
rows = []
for name, kw in [("per-driver, whole recording", dict(norm="driver")),
                 ("per-driver, running first 900 frames/clip", dict(norm="driver", norm_first=900)),
                 ("no per-driver normalisation", dict(norm="none"))]:
    try:
        d2, c2 = build(**kw); f2 = list(F.pick(c2, "no_pose"))
        t = lodo(d2, f2); r = t.loc["MEAN"].to_dict(); r["variant"] = name
        r["worst_driver_acc"] = t.drop("MEAN")["acc"].min(); rows.append(r)
    except Exception as e:
        P("variant failed:", name, e)
t2 = pd.DataFrame(rows).set_index("variant")[["acc", "recall", "specificity", "bal_acc", "auc", "worst_driver_acc"]]
P(t2.round(3).to_string())

# ------------------------------------------------------------------ T3
P("\n" + "=" * 70, "\nT3  FEATURE-SET ABLATION (per-driver norm, mean over 4 held-out drivers)\n", "=" * 70)
rows = []
for mode in ["all", "no_pose", "ear_only", "eyes_only", "no_blend"]:
    try:
        f = list(F.pick(cols, mode)); t = lodo(d_drv, f); r = t.loc["MEAN"].to_dict()
        r["feature_set"] = mode; r["n_features"] = len(f); r["worst_driver_acc"] = t.drop("MEAN")["acc"].min(); rows.append(r)
    except Exception as e:
        P("mode failed:", mode, e)
t3 = pd.DataFrame(rows).set_index("feature_set")[["n_features", "acc", "recall", "bal_acc", "auc", "worst_driver_acc"]]
P(t3.round(3).to_string())

# ------------------------------------------------------------------ T4
P("\n" + "=" * 70, "\nT4  RANDOM FRAME SPLIT (5-fold, shuffled) vs LODO, same model/features\n", "=" * 70)
X = d_drv[feats_final].values; y = d_drv["label"].astype(int).values
skf = StratifiedKFold(5, shuffle=True, random_state=0); a = []; r_ = []; au = []
for tr, te in skf.split(X, y):
    m = model().fit(X[tr], y[tr]); s = m.predict_proba(X[te])[:, 1]; mt = metrics(y[te], (s >= .5).astype(int), s)
    a.append(mt["acc"]); r_.append(mt["recall"]); au.append(mt["auc"])
P(f"random 5-fold : acc {np.mean(a):.3f} (sd {np.std(a):.3f}) recall {np.mean(r_):.3f} auc {np.mean(au):.3f}")
P(f"LODO (T1)     : acc {t1.loc['MEAN','acc']:.3f} recall {t1.loc['MEAN','recall']:.3f} auc {t1.loc['MEAN','auc']:.3f}")
# harder, still leaky variant: split each stream in time (first 80% train, last 20% test)
tr_idx, te_idx = [], []
for _, g in d_drv.groupby("stream"):
    k = int(len(g) * 0.8); tr_idx += list(g.index[:k]); te_idx += list(g.index[k:])
m = model().fit(d_drv.loc[tr_idx, feats_final].values, d_drv.loc[tr_idx, "label"].astype(int).values)
s = m.predict_proba(d_drv.loc[te_idx, feats_final].values)[:, 1]
mt = metrics(d_drv.loc[te_idx, "label"].astype(int).values, (s >= .5).astype(int), s)
P(f"same-driver, later part of each clip held out : acc {mt['acc']:.3f} recall {mt['recall']:.3f} auc {mt['auc']:.3f}")

# ------------------------------------------------------------------ T5
P("\n" + "=" * 70, "\nT5  CPU LATENCY / SIZE\n", "=" * 70)
import os, platform
P("machine:", platform.processor() or platform.machine(), "| cpu count:", os.cpu_count(), "| python", platform.python_version())
m = model().fit(X, y)
x1 = X[:1]
for _ in range(200): m.predict_proba(x1)
t0 = time.perf_counter(); N = 3000
for i in range(N): m.predict_proba(X[i:i + 1])
lat = (time.perf_counter() - t0) / N * 1000
P(f"landmark LR: {lat:.3f} ms per frame for classification only (batch=1)  ->  {1000 / lat:.0f} frames/s")
P("landmark LR pickled size: %.1f KB" % (len(pickle.dumps(m)) / 1024))
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    base = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3), include_top=False, pooling="avg", weights=None)
    inp = layers.Input((3, 160, 160, 3)); z = layers.TimeDistributed(base)(inp); z = layers.LSTM(128)(z); out = layers.Dense(1, activation="sigmoid")(z)
    cnn = models.Model(inp, out); xb = np.random.rand(1, 3, 160, 160, 3).astype("float32")
    for _ in range(5): cnn.predict(xb, verbose=0)
    t0 = time.perf_counter(); N2 = 40
    for _ in range(N2): cnn.predict(xb, verbose=0)
    lc = (time.perf_counter() - t0) / N2 * 1000
    P(f"CNN-LSTM (MobileNetV2 + LSTM128, 3x160x160, untrained weights, same architecture): {lc:.1f} ms per 3-frame window (batch=1)")
    P(f"CNN-LSTM parameters: {cnn.count_params():,}  | approx weight size {cnn.count_params() * 4 / 1e6:.1f} MB (float32)")
except Exception as e:
    P("CNN-LSTM timing skipped (tensorflow not available or failed):", e)
try:
    import psutil
    P("process RSS now: %.0f MB" % (psutil.Process().memory_info().rss / 1e6))
except Exception:
    pass

open("paper_results.txt", "w", encoding="utf-8").write(log.getvalue())
P("\nsaved paper_results.txt  --  paste the whole console output back to Claude")