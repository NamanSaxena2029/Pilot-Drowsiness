import os, re, argparse, random, json
import numpy as np, pandas as pd

NAME_RE = re.compile(r"^(?P<driver>\d+)_(?P<rest>.+)_(?P<frame>\d+)_(?:not)?drowsy\.\w+$", re.I)
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def build_df(base):
    rows, bad = [], 0
    for label, folder in enumerate(["notdrowsy", "drowsy"]):
        d = os.path.join(base, folder)
        for f in os.listdir(d):
            if not f.lower().endswith(IMG_EXT):
                continue
            m = NAME_RE.match(f)
            if not m:
                bad += 1
                continue
            driver = m.group("driver")
            # stream = one continuous clip: same driver + condition + scenario
            stream = f"{driver}_{m.group('rest')}"
            rows.append([os.path.join(d, f), label, driver, stream, int(m.group("frame"))])
    if bad:
        print(f"[warn] {bad} files did not match the filename pattern and were skipped")
    return pd.DataFrame(rows, columns=["image", "label", "driver", "stream", "frame"])


def make_windows(df, seq_len=3, stride=1):
    """Windows of seq_len consecutive frames inside ONE stream (never across clips).
    Label of a window = label of its last frame."""
    out = []
    for _, g in df.groupby("stream"):
        g = g.sort_values("frame")
        paths, labels = g.image.tolist(), g.label.tolist()
        for i in range(0, len(paths) - seq_len + 1, stride):
            out.append((paths[i:i + seq_len], labels[i + seq_len - 1]))
    return out


def fold_drivers(drivers, test):
    rest = [d for d in drivers if d != test]
    val = rest[(drivers.index(test)) % len(rest)]      # rotate which driver validates
    train = [d for d in rest if d != val]
    return train, val


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/cropped_mp")
    ap.add_argument("--tag", default="normal")
    ap.add_argument("--init-tag", default=None, help="fine-tune from models/<init-tag>_test<driver>.pth (same fold)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arch", default="cnn_lstm", choices=["cnn_lstm", "cnn_single"],
                    help="cnn_lstm = MobileNetV2 + LSTM over seq-len frames; cnn_single = MobileNetV2 on ONE frame (CNN-only baseline)")
    ap.add_argument("--seq-len", type=int, default=3)
    ap.add_argument("--lr-cnn", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--select", default="bal_acc", choices=["bal_acc", "auc", "acc"], help="validation metric that picks the best epoch")
    ap.add_argument("--train-stride", type=int, default=2, help="consecutive frames are near-duplicates; 2-3 saves time")
    ap.add_argument("--inspect", action="store_true")
    return ap.parse_args()


def run(args):
    import torch, torch.nn as nn, torch.optim as optim, cv2
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, recall_score, f1_score
    from tqdm import tqdm

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, average_precision_score, confusion_matrix
    if args.arch == "cnn_single": args.seq_len = 1
    tag = f"{args.tag}_{args.arch}_s{args.seed}"
    os.makedirs("runs", exist_ok=True)
    json.dump(vars(args) | dict(torch=torch.__version__, optimizer="Adam", loss="CrossEntropy class weights [1.0,1.2]", grad_clip=1.0,
              img_size=160, augment="hflip, rotate10, colorjitter0.3", freeze="MobileNetV2 features[:-4] frozen", lstm_hidden=128),
              open(f"runs/config_{tag}.json", "w"), indent=2)
    df = build_df(args.base)
    drivers = sorted(df.driver.unique())
    assert len(drivers) >= 3, f"Need >=3 drivers, found {drivers}"
    os.makedirs("models", exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev, "| drivers:", drivers)

    tf_train = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((160, 160)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), transforms.ColorJitter(0.3, 0.3, 0.3), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    tf_eval = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((160, 160)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    class WinDS(Dataset):
        def __init__(self, windows, tf): self.w, self.tf = windows, tf
        def __len__(self): return len(self.w)
        def __getitem__(self, i):
            paths, y = self.w[i]; fr = []
            for p in paths:
                im = cv2.imread(p)
                im = np.zeros((160, 160, 3), np.uint8) if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                fr.append(self.tf(im))
            return torch.stack(fr), y

    class CNN_LSTM(nn.Module):
        def __init__(self, pretrained):
            super().__init__()
            self.cnn = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
            self.cnn.classifier = nn.Identity()
            for p in self.cnn.features[:-4].parameters(): p.requires_grad = False
            self.single = args.arch == "cnn_single"
            self.lstm = nn.Identity() if self.single else nn.LSTM(1280, 128, num_layers=1, batch_first=True)
            self.fc = nn.Linear(1280 if self.single else 128, 2)
        def forward(self, x):
            B, T, C, H, W = x.size()
            f = self.cnn(x.view(B * T, C, H, W)).view(B, T, -1).contiguous()
            return self.fc(f[:, -1, :] if self.single else self.lstm(f)[0][:, -1, :])

    results = []
    for test_d in drivers:
        train_d, val_d = fold_drivers(drivers, test_d)
        print(f"\n===== FOLD: test={test_d} | val={val_d} | train={train_d} =====")
        sub = lambda ds: df[df.driver.isin(ds)]
        tl = DataLoader(WinDS(make_windows(sub(train_d), seq_len=args.seq_len, stride=args.train_stride), tf_train), batch_size=args.batch, shuffle=True)
        vl = DataLoader(WinDS(make_windows(sub([val_d]), seq_len=args.seq_len, stride=2), tf_eval), batch_size=16)
        sl = DataLoader(WinDS(make_windows(sub([test_d]), seq_len=args.seq_len, stride=1), tf_eval), batch_size=16)

        init = f"models/{args.init_tag}_test{test_d}.pth" if args.init_tag else None
        model = CNN_LSTM(pretrained=init is None).to(dev)
        if init:
            model.load_state_dict(torch.load(init, map_location=dev)); print("fine-tuning from", init)
        sc = 0.5 if init else 1.0
        head = list(model.fc.parameters()) + ([] if args.arch == "cnn_single" else list(model.lstm.parameters()))
        opt = optim.Adam([
            {"params": model.cnn.features[-4:].parameters(), "lr": args.lr_cnn * sc},
            {"params": head, "lr": args.lr_head * sc}])
        crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.2]).to(dev))
        out_path = f"models/{tag}_test{test_d}.pth"
        best, bad = -1.0, 0
        for ep in range(args.epochs):
            model.train(); c = n = 0
            for x, y in tqdm(tl, desc=f"fold {test_d} ep{ep+1}"):
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad(); o = model(x); crit(o, y).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                c += (o.argmax(1) == y).sum().item(); n += y.size(0)
            model.eval(); vt, vp, vs = [], [], []
            with torch.no_grad():
                for x, y in vl:
                    o = model(x.to(dev)); vs += torch.softmax(o, 1)[:, 1].cpu().tolist(); vp += o.argmax(1).cpu().tolist(); vt += y.tolist()
            vm = dict(acc=accuracy_score(vt, vp), bal_acc=balanced_accuracy_score(vt, vp),
                      auc=roc_auc_score(vt, vs) if len(set(vt)) > 1 else float("nan"))
            va = vm[args.select]
            print(f"ep{ep+1} train acc {c/n:.4f} | val(driver {val_d}) acc {vm['acc']:.4f} bal_acc {vm['bal_acc']:.4f} auc {vm['auc']:.4f} | select={args.select}")
            if va > best: best, bad = va, 0; torch.save(model.state_dict(), out_path)
            else:
                bad += 1
                if bad >= args.patience: print("early stop"); break

        model.load_state_dict(torch.load(out_path, map_location=dev)); model.eval()
        yt, yp, ys, last = [], [], [], [w[0][-1] for w in sl.dataset.w]
        with torch.no_grad():
            for x, y in sl:
                o = model(x.to(dev)); ys += torch.softmax(o, 1)[:, 1].cpu().tolist()
                yp += o.argmax(1).cpu().tolist(); yt += y.tolist()
        print(f"\nTEST on UNSEEN driver {test_d}")
        print(classification_report(yt, yp, target_names=["Not Drowsy", "Drowsy"], zero_division=0))
        auc = roc_auc_score(yt, ys) if len(set(yt)) > 1 else float("nan")
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        rec, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
        results.append(dict(test_driver=test_d, n=len(yt), acc=accuracy_score(yt, yp), bal_acc=(rec + spec) / 2,
                            drowsy_recall=rec, specificity=spec, drowsy_f1=f1_score(yt, yp, pos_label=1, zero_division=0),
                            mcc=matthews_corrcoef(yt, yp), auc=auc, pr_auc=average_precision_score(yt, ys)))
        pd.DataFrame(dict(image=last, driver=test_d, label=yt, prob=ys)).to_csv(f"runs/preds_{tag}_test{test_d}.csv", index=False)

    res = pd.DataFrame(results)
    res.to_csv(f"runs/results_{tag}.csv", index=False)
    print("\n===== LODO SUMMARY (", tag, ") =====")
    print(res.round(4).to_string(index=False))
    print("\nmean +- std:")
    for c in ["acc", "bal_acc", "drowsy_recall", "specificity", "mcc", "auc", "pr_auc"]:
        print(f"  {c:14s} {res[c].mean():.4f} +- {res[c].std(ddof=0):.4f}")


if __name__ == "__main__":
    a = parse_args()
    if a.inspect:
        d = build_df(a.base)
        print(d.groupby(["driver", "label"]).size().unstack(fill_value=0))
        d["condition"] = d.stream.str.split("_").str[1]
        d["scenario"] = d.stream.str.split("_").str[2]
        print("\nconditions:", sorted(d.condition.unique()))
        print("scenarios :", sorted(d.scenario.unique()))
        print("streams   :", d.stream.nunique())
    else:
        run(a)