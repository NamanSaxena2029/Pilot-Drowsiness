import argparse, itertools, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score
from features_lodo import load, add_features, pick, make_model

METRICS = ["acc", "bal_acc", "drowsy_recall", "auc"]


def metrics(y, p):
    yp = (p >= 0.5).astype(int)
    return [accuracy_score(y, yp), balanced_accuracy_score(y, yp), recall_score(y, yp, zero_division=0),
            roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="features_normal.csv")
    ap.add_argument("--test-csv", default=None)
    ap.add_argument("--model", default="logreg", choices=["logreg", "gbm"])
    ap.add_argument("--norm", default="driver", choices=["driver", "none"])
    ap.add_argument("--norm-first", type=int, default=None)
    ap.add_argument("--feats", default="no_pose")
    ap.add_argument("--n-train", type=int, default=3)
    ap.add_argument("--train-stride", type=int, default=5)
    ap.add_argument("--block", type=int, default=300)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df, feats = add_features(load(a.csv), a.norm, norm_first=a.norm_first)
    feats = pick(feats, a.feats)
    tdf = None
    if a.test_csv:
        tdf, _ = add_features(load(a.test_csv), a.norm, norm_first=a.norm_first)
    src = tdf if tdf is not None else df
    drivers = sorted(df.driver.unique())
    print(f"{len(df)} rows | {len(feats)} feats | model={a.model} norm={a.norm} first={a.norm_first} "
          f"feats={a.feats} n_train={a.n_train} test={a.test_csv or a.csv}")

    data = {}                                    # driver -> (y, [p per train-combo], block ids)
    for te in drivers:
        ts = src[src.driver == te]
        ps = []
        for combo in itertools.combinations([d for d in drivers if d != te], a.n_train):
            tr = df[df.driver.isin(combo)]
            tr = tr[tr.groupby("stream").cumcount() % a.train_stride == 0]
            m = make_model(a.model).fit(tr[feats], tr.label)
            ps.append(m.predict_proba(ts[feats])[:, 1])
        blk = (ts.stream + "_" + (ts.groupby("stream").cumcount() // a.block).astype(str)).values
        data[te] = (ts.label.values, ps, pd.factorize(blk)[0])

    def point(te, idx=None):
        y, ps, blk = data[te]
        if idx is not None:
            y, ps = y[idx], [p[idx] for p in ps]
        return np.nanmean([metrics(y, p) for p in ps], axis=0)

    pt = {te: point(te) for te in drivers}
    pt["MEAN"] = np.mean([pt[d] for d in drivers], axis=0)

    rng = np.random.default_rng(a.seed)
    members = {te: [np.where(data[te][2] == b)[0] for b in range(data[te][2].max() + 1)] for te in drivers}
    boots = {te: [] for te in drivers}
    for _ in range(a.boot):
        for te in drivers:
            bl = members[te]
            pick_ = rng.integers(0, len(bl), len(bl))
            boots[te].append(point(te, np.concatenate([bl[i] for i in pick_])))
    boots["MEAN"] = [np.mean([boots[d][i] for d in drivers], axis=0) for i in range(a.boot)]

    rows = []
    for k in drivers + ["MEAN"]:
        b = np.array(boots[k])
        row = {"driver": k}
        for j, mname in enumerate(METRICS):
            lo, hi = np.nanpercentile(b[:, j], [2.5, 97.5])
            row[mname] = f"{pt[k][j]:.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(row)
    res = pd.DataFrame(rows)
    print(f"\n95% block-bootstrap CI ({a.boot} rounds, block={a.block} frames)\n")
    print(res.to_string(index=False))
    tag = "_".join(str(x) for x in [a.csv.replace("features_", "").replace(".csv", ""),
                                    ("to_" + a.test_csv.replace("features_", "").replace(".csv", "")) if a.test_csv else "",
                                    a.model, a.norm + (str(a.norm_first) if a.norm_first else ""), a.feats, f"n{a.n_train}"] if x)
    out = f"paper_{tag}.csv"
    res.to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()