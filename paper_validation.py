import argparse, io, itertools, re, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, recall_score, f1_score, roc_auc_score,
                             matthews_corrcoef, cohen_kappa_score, average_precision_score, brier_score_loss, confusion_matrix)
import features_lodo as F
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="features_normal.csv")
ap.add_argument("--lowlight-csv", default="features_lowlight.csv")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--stride", type=int, default=5, help="use every k-th training frame (consecutive frames are near-duplicates)")
a = ap.parse_args()
SEED = a.seed
np.random.seed(SEED)
log = io.StringIO()
def P(*x):
    s = " ".join(str(i) for i in x); print(s, flush=True); log.write(s + "\n")

# ---------------------------------------------------------------- data
raw = F.load(a.csv)
df, feats_all = F.add_features(raw, "driver")
FEATS = F.pick(feats_all, "no_pose")
drivers = sorted(df.driver.unique())
rest = df["stream"].str.split("_", n=1).str[1]
df["condition"] = rest.str.split("_").str[0]; df["scenario"] = rest.str.split("_").str[1]
P(f"seed={SEED} | frames={len(df)} | drivers={drivers} | features={len(FEATS)} | train stride={a.stride}")
P("pandas", pd.__version__, "| numpy", np.__version__)
import sklearn; P("scikit-learn", sklearn.__version__)

def sub(d): return d[d.groupby("stream").cumcount() % a.stride == 0]

def metrics(y, p):
    yp = (p >= 0.5).astype(int); tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    rec, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    o = dict(acc=(tp + tn) / len(y), bal_acc=(rec + spec) / 2, recall=rec, specificity=spec,
             f1=f1_score(y, yp, zero_division=0), mcc=matthews_corrcoef(y, yp), kappa=cohen_kappa_score(y, yp), brier=brier_score_loss(y, p))
    try: o["auc"] = roc_auc_score(y, p); o["pr_auc"] = average_precision_score(y, p)
    except Exception: o["auc"] = o["pr_auc"] = np.nan
    return o

# ---------------------------------------------------------------- models
class Const:
    def fit(self, X, y): return self
    def predict_proba(self, X): return np.tile([0.0, 1.0], (len(X), 1))

class PerclosRule:
    """no learning except one threshold on perclos_60: choose t on the training drivers (max balanced accuracy)."""
    def __init__(self, col): self.col = col
    def fit(self, X, y):
        v = X[:, self.col]; best = (-1, 0.0)
        for t in np.quantile(v, np.linspace(0.05, 0.95, 37)):
            b = balanced_accuracy_score(y, (v >= t).astype(int))
            if b > best[0]: best = (b, t)
        self.t = best[1]; self.s = np.std(v) + 1e-9; return self
    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-(X[:, self.col] - self.t) / (0.25 * self.s))); return np.c_[1 - p, p]

def make(name, hp):
    if name == "constant": return Const()
    if name == "perclos_rule": return PerclosRule(FEATS.index("perclos_60"))
    if name == "logreg": return make_pipeline(StandardScaler(), LogisticRegression(C=hp["C"], class_weight="balanced", max_iter=3000))
    if name == "svm_linear":
        return make_pipeline(StandardScaler(), CalibratedClassifierCV(LinearSVC(C=hp["C"], class_weight="balanced", max_iter=5000, random_state=SEED), cv=3))
    if name == "svm_rbf": return make_pipeline(StandardScaler(), SVC(C=hp["C"], gamma=hp["gamma"], class_weight="balanced", probability=True, random_state=SEED))
    if name == "rf": return RandomForestClassifier(200, max_depth=hp["depth"], min_samples_leaf=20, class_weight="balanced", n_jobs=-1, random_state=SEED)
    if name == "gbm": return HistGradientBoostingClassifier(max_depth=hp["depth"], learning_rate=0.05, max_iter=200, l2_regularization=1.0, class_weight="balanced", random_state=SEED)
GRID = {"constant": [{}], "perclos_rule": [{}],
        "logreg": [{"C": c} for c in (0.01, 0.1, 0.5, 1, 10)],
        "svm_linear": [{"C": c} for c in (0.01, 0.1, 1)],
        "svm_rbf": [{"C": c, "gamma": g} for c in (1, 10) for g in (0.01, 0.05)],
        "rf": [{"depth": d} for d in (6, 12)], "gbm": [{"depth": d} for d in (3, 5)]}
PURPOSE = {"constant": "chance reference (always 'drowsy')", "perclos_rule": "classical rule baseline, one threshold on PERCLOS",
           "logreg": "linear classical ML baseline / our final model", "svm_linear": "classical margin-based baseline (linear)",
           "svm_rbf": "classical non-linear kernel baseline", "rf": "bagged trees (non-linear, interpretable)", "gbm": "boosted trees (non-linear)"}

def fit_select(name, tr_df):
    """nested selection: inner leave-one-driver-out over the training drivers, criterion = balanced accuracy."""
    grid = GRID[name]; tdr = sorted(tr_df.driver.unique())
    if len(grid) == 1 or len(tdr) < 2: best = grid[0]
    else:
        sc = []
        for hp in grid:
            b = []
            for v in tdr:
                tr, va = sub(tr_df[tr_df.driver != v]), tr_df[tr_df.driver == v].iloc[::3]
                m = make(name, hp).fit(tr[FEATS].values, tr.label.values)
                b.append(balanced_accuracy_score(va.label, (m.predict_proba(va[FEATS].values)[:, 1] >= .5).astype(int)))
            sc.append(np.mean(b))
        best = grid[int(np.argmax(sc))]
    m = make(name, best).fit(sub(tr_df)[FEATS].values, sub(tr_df).label.values)
    return m, best

# ---------------------------------------------------------------- V1 / V2
P("\n" + "=" * 72, "\nV1  BASELINES UNDER LODO (3 training drivers, nested selection by balanced accuracy)\n", "=" * 72)
rows, preds, chosen = [], {}, []
for name in GRID:
    t0 = time.time(); per = []
    for te in drivers:
        trd = df[df.driver != te]; ted = df[df.driver == te]
        m, hp = fit_select(name, trd); p = m.predict_proba(ted[FEATS].values)[:, 1]
        preds[(name, te)] = pd.Series(p, index=ted.index)
        per.append(dict(driver=te, **metrics(ted.label.values, p))); chosen.append(dict(model=name, held_out=te, **hp))
    pm = pd.DataFrame(per); mean = pm.drop(columns="driver").mean()
    rows.append(dict(model=name, **mean.to_dict(), worst_driver_bal_acc=pm.bal_acc.min()))
    pm.to_csv(f"paper_validation_perdriver_{name}.csv", index=False)
    P(f"  {name:13s} done in {time.time()-t0:.0f}s")
V1 = pd.DataFrame(rows).set_index("model")
cols = ["acc", "bal_acc", "recall", "specificity", "f1", "mcc", "kappa", "auc", "pr_auc", "brier", "worst_driver_bal_acc"]
P(V1[cols].round(3).to_string()); V1[cols].to_csv("paper_validation_V1.csv")
P("\nselected hyper-parameters per fold:"); P(pd.DataFrame(chosen).fillna("").to_string(index=False))
P("\npurpose of each baseline:"); [P(f"  {k:13s} {v}") for k, v in PURPOSE.items()]

P("\n" + "=" * 72, "\nV2  METRIC RELIABILITY: ranking of models by each metric (1 = best)\n", "=" * 72)
R = V1[["acc", "bal_acc", "recall", "mcc", "auc", "pr_auc"]].rank(ascending=False).astype(int)
P(R.to_string())
P("\nconstant 'always drowsy': accuracy {:.3f} but MCC {:.3f}, balanced accuracy {:.3f}, specificity {:.3f}".format(
    V1.loc["constant", "acc"], V1.loc["constant", "mcc"], V1.loc["constant", "bal_acc"], V1.loc["constant", "specificity"]))
P("accuracy vs balanced accuracy, per model (gap shows how much class mix flatters accuracy):")
P((V1["acc"] - V1["bal_acc"]).round(3).to_string())

# ---------------------------------------------------------------- V3
P("\n" + "=" * 72, "\nV3  NUMBER OF TRAINING DRIVERS (logistic regression, fixed C=0.5, mean over all driver combinations)\n", "=" * 72)
rows = []
for n in (1, 2, 3):
    per = []
    for te in drivers:
        ted = df[df.driver == te]
        for combo in itertools.combinations([d for d in drivers if d != te], n):
            tr = sub(df[df.driver.isin(combo)]); m = make("logreg", {"C": 0.5}).fit(tr[FEATS].values, tr.label.values)
            per.append(dict(held_out=te, **metrics(ted.label.values, m.predict_proba(ted[FEATS].values)[:, 1])))
    pm = pd.DataFrame(per); g = pm.groupby("held_out")[["acc", "bal_acc", "auc", "mcc"]].mean()
    rows.append(dict(n_train_drivers=n, acc=g.acc.mean(), bal_acc=g.bal_acc.mean(), auc=g.auc.mean(), mcc=g.mcc.mean(),
                     worst_driver_bal_acc=g.bal_acc.min(), best_driver_bal_acc=g.bal_acc.max()))
V3 = pd.DataFrame(rows).set_index("n_train_drivers"); P(V3.round(3).to_string()); V3.to_csv("paper_validation_V3.csv")

# ---------------------------------------------------------------- V4 / V5
P("\n" + "=" * 72, "\nV4  PER-CONDITION AND PER-SCENARIO RESULTS (final logistic regression, LODO predictions from V1)\n", "=" * 72)
pp = pd.concat([preds[("logreg", d)] for d in drivers]).rename("p"); dd = df.join(pp)
def grp(col):
    out = []
    for k, g in dd.groupby(col):
        r = metrics(g.label.values, g.p.values); r.update({col: k, "frames": len(g), "drowsy_share": g.label.mean()}); out.append(r)
    return pd.DataFrame(out).set_index(col)[["frames", "drowsy_share", "acc", "bal_acc", "recall", "specificity", "mcc", "auc"]]
for c in ("condition", "scenario"):
    t = grp(c); P(t.round(3).to_string(), "\n"); t.to_csv(f"paper_validation_V4_{c}.csv")
P("=" * 72, "\nV5  PER-STREAM SPREAD (27 clips; accuracy of the held-out driver's predictions)\n", "=" * 72)
ps = dd.groupby("stream").apply(lambda g: pd.Series(dict(frames=len(g), drowsy_share=g.label.mean(), acc=((g.p >= .5) == g.label).mean())))
P(ps.acc.describe().round(3).to_string()); ps.round(3).to_csv("paper_validation_V5_streams.csv")
P("streams with accuracy < 0.5:", int((ps.acc < 0.5).sum()), "of", len(ps), "|", list(ps.index[ps.acc < 0.5]))

# ---------------------------------------------------------------- V6
P("\n" + "=" * 72, "\nV6  ROBUSTNESS: train on normal, test on the simulated low-light version of the same drivers\n", "=" * 72)
try:
    ll, _ = F.add_features(F.load(a.lowlight_csv), "driver"); per = []
    for te in drivers:
        tr = sub(df[df.driver != te]); tl = ll[ll.driver == te]
        m = make("logreg", {"C": 0.5}).fit(tr[FEATS].values, tr.label.values)
        per.append(dict(driver=te, **metrics(tl.label.values, m.predict_proba(tl[FEATS].values)[:, 1])))
    pm = pd.DataFrame(per); P(pm.round(3).to_string(index=False)); P("mean:"); P(pm.drop(columns="driver").mean().round(3).to_string())
    pm.to_csv("paper_validation_V6.csv", index=False)
except Exception as e:
    P("V6 skipped:", repr(e)[:200])

open("paper_validation.txt", "w", encoding="utf-8").write(log.getvalue()); P("\nsaved paper_validation.txt")