import argparse, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from features_lodo import load, add_features, pick, make_model
warnings.filterwarnings("ignore")


def fit_predict(D, cols, drivers, kind, stride):
    out = {}
    for d in drivers:
        tr = D[D.driver != d]; tr = tr[tr.groupby("stream").cumcount() % stride == 0]
        ts = D[D.driver == d]
        mdl = (RandomForestClassifier(300, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=0)
               if kind == "rf" else make_model(kind)).fit(tr[cols], tr.label)
        out[d] = mdl.predict_proba(ts[cols])[:, 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="features_normal.csv")
    ap.add_argument("--train-stride", type=int, default=5)
    ap.add_argument("--block", type=int, default=300)
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    raw = load(a.csv)
    dn, fn = add_features(raw, "driver")
    dz, fz = add_features(raw, "none")
    drivers = sorted(dn.driver.unique())
    lab = {d: dn[dn.driver == d].label.values for d in drivers}

    r = raw.sort_values(["stream", "frame"]).reset_index(drop=True)
    g = r.groupby("stream")["ear"]
    lag = pd.concat({f"e{k}": g.shift(k) for k in range(30)}, axis=1)
    first = r.groupby("stream")["ear"].transform("first")
    lag = lag.apply(lambda c: c.fillna(first))
    r = pd.concat([r, lag], axis=1)

    P = {"FINAL (all but pose, driver-norm, LogReg)": fit_predict(dn, pick(fn, "no_pose"), drivers, "logreg", a.train_stride),
         "EAR + PERCLOS only": fit_predict(dn, pick(fn, "ear_only"), drivers, "logreg", a.train_stride),
         "Eyes only (EAR, blink, PERCLOS)": fit_predict(dn, pick(fn, "eyes_only"), drivers, "logreg", a.train_stride),
         "No normalisation": fit_predict(dz, pick(fz, "no_pose"), drivers, "logreg", a.train_stride),
         "Gradient boosting (same features)": fit_predict(dn, pick(fn, "no_pose"), drivers, "gbm", a.train_stride),
         "Raw EAR 30 frames, LogReg": fit_predict(r, list(lag.columns), drivers, "logreg", a.train_stride),
         "Raw EAR 30 frames, RF": fit_predict(r, list(lag.columns), drivers, "rf", a.train_stride)}
    names = list(P)
    for d in drivers:                                        # same row order for every model
        assert len(P[names[0]][d]) == len(lab[d]) == len(r[r.driver == d])

    blk = {}
    for d in drivers:
        ts = dn[dn.driver == d]
        b = pd.factorize((ts.stream + "_" + (ts.groupby("stream").cumcount() // a.block).astype(str)).values)[0]
        blk[d] = [np.where(b == k)[0] for k in range(b.max() + 1)]

    def score(p, y):
        return accuracy_score(y, p >= 0.5), roc_auc_score(y, p), recall_score(y, p >= 0.5, zero_division=0)

    rng = np.random.default_rng(a.seed)
    res = {n: [] for n in names}
    for _ in range(a.boot):
        idx = {d: np.concatenate([blk[d][i] for i in rng.integers(0, len(blk[d]), len(blk[d]))]) for d in drivers}
        for n in names:
            res[n].append(np.mean([score(P[n][d][idx[d]], lab[d][idx[d]]) for d in drivers], axis=0))
    R = {n: np.array(v) for n, v in res.items()}

    base = names[0]
    rows = []
    for n in names[1:]:
        row = {"alternative": n}
        for j, m in enumerate(["acc", "auc", "recall"]):
            diff = R[base][:, j] - R[n][:, j]               # final minus alternative
            lo, hi = np.percentile(diff, [2.5, 97.5])
            row[f"{m}_gain"] = f"{diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"
            row[f"{m}_p"] = f"{(diff <= 0).mean():.3f}"
        rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv("paired_tests.csv", index=False)
    print(f"\nFinal model minus each alternative, mean over 4 drivers, 95% paired block-bootstrap CI ({a.boot} rounds)")
    print("p = share of rounds in which the final model was NOT better (small = clearly better)\n")
    print(T.to_string(index=False))
    print("\nsaved paired_tests.csv")


if __name__ == "__main__":
    main()