import argparse, itertools
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, recall_score,
                             f1_score, roc_auc_score)

BASE = ["ear", "ear_l", "ear_r", "mar", "yaw", "pitch", "roll", "bl", "jaw"]


def load(path):
    df = pd.read_csv(path, dtype={"driver": str})
    df = df.sort_values(["stream", "frame"]).reset_index(drop=True)
    df["bl"] = df[["bl_l", "bl_r"]].mean(axis=1)
    cols = [c for c in BASE if c in df.columns]
    # fill missed detections inside each clip, then with the driver median
    df[cols] = df.groupby("stream")[cols].transform(lambda s: s.ffill().bfill())
    df[cols] = df.groupby("driver")[cols].transform(lambda s: s.fillna(s.median()))
    return df.dropna(subset=cols).reset_index(drop=True)


def add_features(df, norm, wins=(15, 60), norm_first=None):
    """norm_first=K: driver statistics come only from the first K frames of every clip
    (simulates a running estimate at the start of a drive). None = whole recording."""
    out = df.copy()
    if norm == "driver":
        ref = out[out.groupby("stream").cumcount() < norm_first] if norm_first else out
        ear_ref = out["driver"].map(ref.groupby("driver")["ear"].quantile(0.9))
    else:
        ear_ref = 0.30
    out["closed"] = (out["ear"] < 0.75 * ear_ref).astype(float)       # eye-closed indicator
    if norm == "driver":
        for c in BASE:
            mu = out["driver"].map(ref.groupby("driver")[c].mean())
            sd = out["driver"].map(ref.groupby("driver")[c].std())
            out[c] = (out[c] - mu) / (sd + 1e-9)
    feats = list(BASE)
    for w in wins:
        g = out.groupby("stream")
        out[f"perclos_{w}"] = g["closed"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        feats.append(f"perclos_{w}")
        for c in ["ear", "bl", "mar", "jaw", "pitch"]:
            r = lambda s, w=w: s.rolling(w, min_periods=1)
            out[f"{c}_mean_{w}"] = g[c].transform(lambda s: r(s).mean())
            out[f"{c}_std_{w}"] = g[c].transform(lambda s: r(s).std().fillna(0))
            feats += [f"{c}_mean_{w}", f"{c}_std_{w}"]
        out[f"ear_min_{w}"] = g["ear"].transform(lambda s, w=w: s.rolling(w, min_periods=1).min())
        feats.append(f"ear_min_{w}")
    return out, feats


def pick(feats, mode):
    if mode == "all":
        return feats
    if mode == "ear_only":
        return [f for f in feats if f.startswith(("ear", "perclos"))]
    if mode == "eyes_only":      # eye + blink-blendshape features, no mouth, no head pose
        return [f for f in feats if f.startswith(("ear", "bl", "perclos"))]
    if mode == "no_pose":
        return [f for f in feats if not f.startswith(("yaw", "roll", "pitch"))]
    if mode == "no_blend":
        return [f for f in feats if not (f.startswith("bl") or f.startswith("jaw"))]
    raise SystemExit("unknown --feats")


def make_model(name):
    if name == "gbm":
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                              l2_regularization=1.0, random_state=42)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=0.5, class_weight="balanced"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="features_normal.csv")
    ap.add_argument("--model", default="gbm", choices=["gbm", "logreg"])
    ap.add_argument("--norm", default="driver", choices=["driver", "none"])
    ap.add_argument("--feats", default="all", choices=["all", "ear_only", "no_blend", "no_pose", "eyes_only"])
    ap.add_argument("--test-csv", default=None, help="train on --csv, test on this CSV (same drivers), e.g. features_lowlight.csv")
    ap.add_argument("--wins", default="15,60", help="rolling window sizes in FRAMES, e.g. 5,25 for 5 fps video")
    ap.add_argument("--norm-first", type=int, default=None, help="driver-norm stats from first K frames of each clip only")
    ap.add_argument("--n-train", type=int, default=3, help="train on n of the other drivers (averaged over all combinations); 2 = same as CNN-LSTM LODO")
    ap.add_argument("--train-stride", type=int, default=5)
    a = ap.parse_args()

    df = load(a.csv)
    df, feats = add_features(df, a.norm, [int(x) for x in a.wins.split(',')])
    feats = pick(feats, a.feats)
    drivers = sorted(df.driver.unique())
    tdf = None
    if a.test_csv:
        tdf = load(a.test_csv)
        tdf, _ = add_features(tdf, a.norm, [int(x) for x in a.wins.split(',')])
    print(f"{len(df)} rows | drivers {drivers} | {len(feats)} features | model={a.model} norm={a.norm} feats={a.feats}")

    rows, sig = [], {}
    for te in drivers:
        ts = (tdf if tdf is not None else df)
        ts = ts[ts.driver == te]
        others = [d for d in drivers if d != te]
        per = []
        for combo in itertools.combinations(others, a.n_train):
            tr = df[df.driver.isin(combo)]
            tr = tr[tr.groupby("stream").cumcount() % a.train_stride == 0]
            m = make_model(a.model).fit(tr[feats], tr.label)
            p = m.predict_proba(ts[feats])[:, 1]
            yp = (p >= 0.5).astype(int)
            per.append(dict(acc=accuracy_score(ts.label, yp),
                            bal_acc=balanced_accuracy_score(ts.label, yp),
                            drowsy_recall=recall_score(ts.label, yp, zero_division=0),
                            drowsy_f1=f1_score(ts.label, yp, zero_division=0),
                            auc=roc_auc_score(ts.label, p)))
        rows.append(dict(test_driver=te, n=len(ts), **pd.DataFrame(per).mean().to_dict()))
        # how well does each single feature separate classes on this unseen driver?
        for f in feats:
            if ts[f].nunique() > 1:
                sig.setdefault(f, {})[te] = roc_auc_score(ts.label, ts[f])

    res = pd.DataFrame(rows)
    name = f"results_feat_{a.csv.replace('features_', '').replace('.csv', '')}_{a.model}_{a.norm}_{a.feats}_n{a.n_train}{'_to_' + a.test_csv.replace('features_','').replace('.csv','') if a.test_csv else ''}.csv"
    res.to_csv(name, index=False)
    print("\n===== LODO on geometry features =====")
    print(res.round(4).to_string(index=False))
    print("\nmean +- std")
    for c in ["acc", "bal_acc", "drowsy_recall", "drowsy_f1", "auc"]:
        print(f"  {c:14s} {res[c].mean():.4f} +- {res[c].std(ddof=0):.4f}")

    s = pd.DataFrame(sig).T
    s["mean_dist_from_0.5"] = (s - 0.5).abs().mean(axis=1)
    s["same_sign_all"] = ((s[drivers] > 0.5).all(axis=1)) | ((s[drivers] < 0.5).all(axis=1))
    print("\nSingle-feature AUC per unseen driver (top 12 by consistency+strength; 0.5 = no signal):")
    print(s.sort_values(["same_sign_all", "mean_dist_from_0.5"], ascending=False).head(12).round(3).to_string())
    print(f"\nsaved {name}")


if __name__ == "__main__":
    main()