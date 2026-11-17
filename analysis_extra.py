import argparse, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from features_lodo import load, add_features, pick, make_model
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="features_normal.csv")
    ap.add_argument("--norm-first", type=int, default=None)
    ap.add_argument("--feats", default="no_pose")
    ap.add_argument("--train-stride", type=int, default=5)
    ap.add_argument("--windows", default="1,30,90,150", help="smoothing windows in frames")
    a = ap.parse_args()

    df, feats = add_features(load(a.csv), "driver", norm_first=a.norm_first)
    feats = pick(feats, a.feats)
    parts = df.stream.str.split("_")
    df["glasses"] = parts.str[1]
    df["scenario"] = parts.str[2]
    drivers = sorted(df.driver.unique())

    preds = []
    coefs = {}
    for te in drivers:
        tr = df[df.driver != te]
        tr = tr[tr.groupby("stream").cumcount() % a.train_stride == 0]
        m = make_model("logreg").fit(tr[feats], tr.label)
        ts = df[df.driver == te].copy()
        ts["p"] = m.predict_proba(ts[feats])[:, 1]
        preds.append(ts)
        coefs[te] = pd.Series(m[-1].coef_[0], index=feats)
    P = pd.concat(preds)
    P["pred"] = (P.p >= 0.5).astype(int)
    P["correct"] = (P.pred == P.label).astype(int)

    # 1. scenario / glasses breakdown ------------------------------------------------
    rows = []
    for key in ["scenario", "glasses"]:
        for (val, drv), g in P.groupby([key, "driver"]):
            rows.append(dict(by=key, value=val, driver=drv, n=len(g), frac_drowsy=g.label.mean(),
                             acc=g.correct.mean(),
                             recall=g[g.label == 1].pred.mean() if (g.label == 1).any() else np.nan,
                             specificity=(1 - g[g.label == 0].pred.mean()) if (g.label == 0).any() else np.nan))
    sc = pd.DataFrame(rows)
    sc.to_csv("analysis_scenarios.csv", index=False)
    print("\n=== 1. Mean over drivers, by scenario / glasses ===")
    print(sc.groupby(["by", "value"])[["frac_drowsy", "acc", "recall", "specificity"]].mean().round(3).to_string())
    print("\nIf accuracy is high only in one scenario (e.g. yawning) the model may be recognising the scenario.")

    # 2. smoothing + alarm metrics ---------------------------------------------------
    out = []
    for w in [int(x) for x in a.windows.split(",")]:
        P["ps"] = P.groupby("stream")["p"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        P["alarm"] = (P.ps >= 0.5).astype(int)
        per = []
        for drv, g in P.groupby("driver"):
            delays, missed, n_pos = [], 0, 0
            for _, s in g.groupby("stream"):
                s = s.sort_values("frame")
                if (s.label == 1).any():
                    n_pos += 1
                    first = np.where(s.label.values == 1)[0][0]
                    hit = np.where(s.alarm.values[first:] == 1)[0]
                    if len(hit): delays.append(hit[0])
                    else: missed += 1
            per.append(dict(driver=drv, acc=(g.alarm == g.label).mean(),
                            recall=g[g.label == 1].alarm.mean(), false_alarm_rate=g[g.label == 0].alarm.mean(),
                            auc=roc_auc_score(g.label, g.ps), median_delay_frames=np.median(delays) if delays else np.nan,
                            clips_missed=f"{missed}/{n_pos}"))
        d = pd.DataFrame(per); d.insert(0, "window", w); out.append(d)
    al = pd.concat(out)
    al.to_csv("analysis_alarms.csv", index=False)
    print("\n=== 2. Alarm metrics vs smoothing window (frames) ===")
    print(al.groupby("window")[["acc", "recall", "false_alarm_rate", "auc", "median_delay_frames"]].mean().round(3).to_string())
    print("\n(per driver in analysis_alarms.csv; delay = frames from first drowsy frame of a clip to first alarm)")

    # 3. feature weights -------------------------------------------------------------
    W = pd.DataFrame(coefs)
    W["mean"] = W.mean(axis=1)
    W["same_sign"] = (W[drivers] > 0).all(axis=1) | (W[drivers] < 0).all(axis=1)
    W = W.reindex(W["mean"].abs().sort_values(ascending=False).index)
    W.to_csv("analysis_weights.csv")
    print("\n=== 3. Top LogReg weights (standardised inputs; one model per LODO fold) ===")
    print(W.head(12).round(3).to_string())
    print(f"\nfeatures with the same sign in all 4 folds: {int(W.same_sign.sum())}/{len(W)}")


if __name__ == "__main__":
    main()