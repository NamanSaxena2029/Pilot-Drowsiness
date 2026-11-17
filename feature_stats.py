import argparse, inspect, re, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="features_normal.csv")
ap.add_argument("--norm", default="driver")
ap.add_argument("--feats", default="no_pose")
a = ap.parse_args()

import features_lodo as F

# ---------- 2. source code of the feature pipeline ----------
print("=" * 70, "\nSOURCE: features_lodo.load / add_features / pick\n", "=" * 70)
for fn in ("load", "add_features", "pick"):
    try:
        print(inspect.getsource(getattr(F, fn)))
    except Exception as e:
        print(f"[could not print {fn}: {e}]")
try:
    txt = open("extract_features.py", encoding="utf-8").read()
    print("=" * 70, "\nSOURCE: extract_features.py (ear / mar / extract_frame / feature names)\n", "=" * 70)
    for name in ("ear", "mar", "extract_frame"):
        m = re.search(rf"^def {name}\(.*?(?=^def |\Z)", txt, re.S | re.M)
        if m:
            print(m.group(0).rstrip(), "\n")
    for kw in ("FaceLandmarker", "num_faces", "output_face_blendshapes", "FPS", "fps", "stride", "COLUMNS", "cols"):
        for line in txt.splitlines():
            if kw in line:
                print("   >", line.strip())
                break
except Exception as e:
    print("[extract_features.py not read:", e, "]")

# ---------- load + build features ----------
df = F.load(a.csv)
out = F.add_features(df, a.norm)
if isinstance(out, tuple):
    d2 = out[0]
    feats = list(out[1]) if len(out) > 1 and not isinstance(out[1], (int, float, str)) else None
else:
    d2, feats = out, None
meta = {"driver", "stream", "frame", "label", "image", "path", "scenario", "cond", "condition", "glasses"}
if feats is None:
    feats = [c for c in d2.columns if c not in meta and pd.api.types.is_numeric_dtype(d2[c])]
feats = list(F.pick(feats, a.feats))
print("\n", "=" * 70, f"\nFINAL FEATURE SET: {len(feats)} features (norm={a.norm}, feats={a.feats})\n", "=" * 70)
print(", ".join(feats))

# ---------- 1. class counts ----------
y = d2["label"].astype(int)
print("\nCLASS COUNTS (label 0 = alert, 1 = drowsy)")
ct = pd.crosstab(d2["driver"], y, margins=True)
ct["drowsy_%"] = (100 * ct[1] / ct["All"]).round(1)
print(ct)
for col in ("scenario", "condition", "cond", "glasses"):
    if col in d2.columns:
        print(f"\nBY {col}:\n", pd.crosstab(d2[col], y, margins=True))
if "stream" in d2.columns:
    print("\nstreams per driver:\n", d2.groupby("driver")["stream"].nunique().to_dict(), "| total frames per stream (min/median/max):",
          d2.groupby("stream").size().agg(["min", "median", "max"]).to_dict())
miss = df.isna().mean().sort_values(ascending=False).head(8)
print("\nraw missing-value rate (top 8 columns):\n", miss.round(4).to_dict())

# ---------- 3. per-feature statistics ----------
raw = F.load(a.csv)
rows = []
for f in feats:
    x = d2[f].astype(float)
    a0, a1 = x[y == 0], x[y == 1]
    sp = np.sqrt((a0.var(ddof=1) * (len(a0) - 1) + a1.var(ddof=1) * (len(a1) - 1)) / (len(x) - 2))
    rows.append(dict(feature=f,
                     alert_mean=a0.mean(), alert_sd=a0.std(), drowsy_mean=a1.mean(), drowsy_sd=a1.std(),
                     cohens_d=(a1.mean() - a0.mean()) / sp if sp > 0 else np.nan,
                     min=x.min(), max=x.max()))
st = pd.DataFrame(rows).round(4)
pd.set_option("display.width", 200, "display.max_rows", 200)
print("\nPER-FEATURE STATISTICS (after the same normalisation the model sees)\n", st.to_string(index=False))
st.to_csv("feature_stats.csv", index=False)

# raw base signals (before normalisation) for the paper's parameter table
base = [c for c in raw.columns if c not in meta and pd.api.types.is_numeric_dtype(raw[c]) and not re.search(r"_(mean|std|min)_\d+$", c)]
rr = raw.copy()
print("\nRAW BASE SIGNALS (before any normalisation): mean +/- SD per class\n")
tab = []
for c in base:
    v = rr[c].astype(float)
    tab.append(dict(signal=c, alert=f"{v[rr.label==0].mean():.3f} +/- {v[rr.label==0].std():.3f}",
                    drowsy=f"{v[rr.label==1].mean():.3f} +/- {v[rr.label==1].std():.3f}",
                    min=round(v.min(), 3), max=round(v.max(), 3)))
print(pd.DataFrame(tab).to_string(index=False))
print("\nsaved feature_stats.csv")