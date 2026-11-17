import argparse, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, balanced_accuracy_score
from features_lodo import load, add_features, pick, make_model
warnings.filterwarnings("ignore")


def m(y, p):
    yp = (p >= 0.5).astype(int)
    return dict(acc=accuracy_score(y, yp), bal_acc=balanced_accuracy_score(y, yp),
                drowsy_recall=recall_score(y, yp, zero_division=0), auc=roc_auc_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="features_normal.csv")
    ap.add_argument("--train-stride", type=int, default=5)
    ap.add_argument("--T", type=int, default=30)
    a = ap.parse_args()

    raw = load(a.csv)
    df, feats = add_features(raw, "driver")
    feats = pick(feats, "no_pose")
    drivers = sorted(df.driver.unique())

    # ---- A) random frame split vs LODO (same model: logistic regression, same features) ----------
    rows = []
    for name in ["logreg", "gbm"]:
        tr, te = train_test_split(df, test_size=0.3, random_state=0, stratify=df.label)
        mdl = make_model(name).fit(tr[feats], tr.label)
        rows.append(dict(model=name, protocol="random frame split 70/30", **m(te.label, mdl.predict_proba(te[feats])[:, 1])))
        per = []
        for d in drivers:
            tr = df[df.driver != d]; tr = tr[tr.groupby("stream").cumcount() % a.train_stride == 0]
            ts = df[df.driver == d]
            per.append(m(ts.label, make_model(name).fit(tr[feats], tr.label).predict_proba(ts[feats])[:, 1]))
        rows.append(dict(model=name, protocol="leave-one-driver-out (mean of 4)", **pd.DataFrame(per).mean().to_dict()))
    A = pd.DataFrame(rows); A.to_csv("controls_leakage.csv", index=False)
    print("\n=== A) Same features, same model: random split vs LODO ===")
    print(A.round(3).to_string(index=False))

    # ---- B) prior-work style baseline: last T raw EAR values, Random Forest -------------------
    r = raw.sort_values(["stream", "frame"]).reset_index(drop=True)
    g = r.groupby("stream")["ear"]
    lag = pd.concat({f"e{k}": g.shift(k) for k in range(a.T)}, axis=1)
    lag = lag.apply(lambda c: c.fillna(r.groupby("stream")["ear"].transform("first")))
    r = pd.concat([r, lag], axis=1)
    lag_cols = list(lag.columns)
    d2, f2 = add_features(raw, "driver"); d2 = d2.sort_values(["stream", "frame"]).reset_index(drop=True)

    sets = {f"raw EAR, last {a.T} frames, RF (prior-work style)": (r, lag_cols, "rf"),
            "raw EAR, last 30 frames, LogReg": (r, lag_cols, "logreg"),
            "our final features, LogReg": (d2, pick(f2, "no_pose"), "logreg")}
    rows = []
    for name, (D, cols, kind) in sets.items():
        per = []
        for d in drivers:
            tr = D[D.driver != d]; tr = tr[tr.groupby("stream").cumcount() % a.train_stride == 0]
            ts = D[D.driver == d]
            mdl = (RandomForestClassifier(300, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=0)
                   if kind == "rf" else make_model("logreg")).fit(tr[cols], tr.label)
            per.append(dict(driver=d, **m(ts.label, mdl.predict_proba(ts[cols])[:, 1])))
        P = pd.DataFrame(per); mean = P.drop(columns="driver").mean()
        rows.append(dict(setup=name, **mean.to_dict(), worst_driver_auc=P.auc.min(), best_driver_auc=P.auc.max()))
    B = pd.DataFrame(rows); B.to_csv("controls_priorwork.csv", index=False)
    print("\n=== B) LODO, prior-work-style baseline vs our features (mean of 4 drivers) ===")
    print(B.round(3).to_string(index=False))


if __name__ == "__main__":
    main()