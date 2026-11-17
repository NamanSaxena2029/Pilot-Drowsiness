import os, argparse, warnings
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score
from features_lodo import load, add_features, pick, make_model
warnings.filterwarnings("ignore")


def sc(y, p):
    yp = p >= 0.5
    return [accuracy_score(y, yp), recall_score(y, yp, zero_division=0), roc_auc_score(y, p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="normal")
    ap.add_argument("--csv", default="features_normal.csv")
    ap.add_argument("--train-stride", type=int, default=5)
    ap.add_argument("--block", type=int, default=300)
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()

    cnn = pd.read_csv(f"preds_cnn_{a.tag}.csv", dtype={"driver": str})
    df, feats = add_features(load(a.csv), "driver")
    df["key"] = df.image.map(os.path.basename)
    feats = pick(feats, "no_pose")
    drivers = sorted(df.driver.unique())

    data = {}
    for d in drivers:
        tr = df[df.driver != d]; tr = tr[tr.groupby("stream").cumcount() % a.train_stride == 0]
        ts = df[df.driver == d].copy()
        ts["pf"] = make_model("logreg").fit(tr[feats], tr.label).predict_proba(ts[feats])[:, 1]
        c = cnn[cnn.driver == d].rename(columns={"p": "pc"})[["image", "pc"]]
        m = ts.merge(c, left_on="key", right_on="image", suffixes=("", "_c")).sort_values(["stream", "frame"])
        blk = pd.factorize((m.stream + "_" + (m.groupby("stream").cumcount() // a.block).astype(str)).values)[0]
        data[d] = (m.label.values, m.pf.values, m.pc.values, [np.where(blk == k)[0] for k in range(blk.max() + 1)])
        print(f"driver {d}: {len(m)} matched frames")

    def mean_scores(idx):
        s = {"cnn": [], "feat": []}
        for d in drivers:
            y, pf, pc, _ = data[d]; i = idx[d]
            s["cnn"].append(sc(y[i], pc[i])); s["feat"].append(sc(y[i], pf[i]))
        return np.mean(s["cnn"], axis=0), np.mean(s["feat"], axis=0)

    full = {d: np.arange(len(data[d][0])) for d in drivers}
    pc, pf = mean_scores(full)
    rng = np.random.default_rng(0); C, F = [], []
    for _ in range(a.boot):
        idx = {d: np.concatenate([data[d][3][i] for i in rng.integers(0, len(data[d][3]), len(data[d][3]))]) for d in drivers}
        c_, f_ = mean_scores(idx); C.append(c_); F.append(f_)
    C, F = np.array(C), np.array(F)
    ci = lambda v: tuple(np.percentile(v, [2.5, 97.5]))
    print(f"\nCNN-LSTM ({a.tag}) vs final feature model on the same frames (mean over 4 drivers, {a.boot} paired block-bootstrap rounds)\n")
    rows = []
    for j, n in enumerate(["accuracy", "drowsy_recall", "auc"]):
        d = F[:, j] - C[:, j]
        rows.append({"metric": n,
                     "CNN-LSTM": f"{pc[j]:.3f} [{ci(C[:, j])[0]:.3f}, {ci(C[:, j])[1]:.3f}]",
                     "final features": f"{pf[j]:.3f} [{ci(F[:, j])[0]:.3f}, {ci(F[:, j])[1]:.3f}]",
                     "difference": f"{pf[j] - pc[j]:+.3f} [{ci(d)[0]:+.3f}, {ci(d)[1]:+.3f}]"})
    out = pd.DataFrame(rows); out.to_csv(f"paired_cnn_{a.tag}.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()