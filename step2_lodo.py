import os, re, argparse, random
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
    ap.add_argument("--train-stride", type=int, default=2, help="consecutive frames are near-duplicates; 2-3 saves time")
    ap.add_argument("--inspect", action="store_true")
    return ap.parse_args()


def run(args):
    import torch, torch.nn as nn, torch.optim as optim, cv2
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, models
    from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, recall_score, f1_score
    from tqdm import tqdm

    random.seed(42); np.random.seed(42); torch.manual_seed(42)
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
            self.lstm = nn.LSTM(1280, 128, num_layers=1, batch_first=True)
            self.fc = nn.Linear(128, 2)
        def forward(self, x):
            B, T, C, H, W = x.size()
            f = self.cnn(x.view(B * T, C, H, W)).view(B, T, -1).contiguous()
            return self.fc(self.lstm(f)[0][:, -1, :])

    results = []
    for test_d in drivers:
        train_d, val_d = fold_drivers(drivers, test_d)
        print(f"\n===== FOLD: test={test_d} | val={val_d} | train={train_d} =====")
        sub = lambda ds: df[df.driver.isin(ds)]
        tl = DataLoader(WinDS(make_windows(sub(train_d), stride=args.train_stride), tf_train), batch_size=8, shuffle=True)
        vl = DataLoader(WinDS(make_windows(sub([val_d]), stride=2), tf_eval), batch_size=16)
        sl = DataLoader(WinDS(make_windows(sub([test_d]), stride=1), tf_eval), batch_size=16)

        init = f"models/{args.init_tag}_test{test_d}.pth" if args.init_tag else None
        model = CNN_LSTM(pretrained=init is None).to(dev)
        if init:
            model.load_state_dict(torch.load(init, map_location=dev)); print("fine-tuning from", init)
        sc = 0.5 if init else 1.0
        opt = optim.Adam([
            {"params": model.cnn.features[-4:].parameters(), "lr": 1e-5 * sc},
            {"params": model.lstm.parameters(), "lr": 1e-4 * sc},
            {"params": model.fc.parameters(), "lr": 1e-4 * sc}])
        crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.2]).to(dev))
        out_path = f"models/{args.tag}_test{test_d}.pth"
        best, bad = -1.0, 0
        for ep in range(args.epochs):
            model.train(); c = n = 0
            for x, y in tqdm(tl, desc=f"fold {test_d} ep{ep+1}"):
                x, y = x.to(dev), y.to(dev)
                opt.zero_grad(); o = model(x); crit(o, y).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                c += (o.argmax(1) == y).sum().item(); n += y.size(0)
            model.eval(); vc = vn = 0
            with torch.no_grad():
                for x, y in vl:
                    vc += (model(x.to(dev)).argmax(1).cpu() == y).sum().item(); vn += y.size(0)
            va = vc / vn
            print(f"ep{ep+1} train {c/n:.4f} | val(driver {val_d}) {va:.4f}")
            if va > best: best, bad = va, 0; torch.save(model.state_dict(), out_path)
            else:
                bad += 1
                if bad >= 3: print("early stop"); break

        model.load_state_dict(torch.load(out_path, map_location=dev)); model.eval()
        yt, yp, ys = [], [], []
        with torch.no_grad():
            for x, y in sl:
                o = model(x.to(dev)); ys += torch.softmax(o, 1)[:, 1].cpu().tolist()
                yp += o.argmax(1).cpu().tolist(); yt += y.tolist()
        print(f"\nTEST on UNSEEN driver {test_d}")
        print(classification_report(yt, yp, target_names=["Not Drowsy", "Drowsy"], zero_division=0))
        auc = roc_auc_score(yt, ys) if len(set(yt)) > 1 else float("nan")
        results.append(dict(test_driver=test_d, n=len(yt), acc=accuracy_score(yt, yp),
                            drowsy_recall=recall_score(yt, yp, pos_label=1, zero_division=0),
                            drowsy_f1=f1_score(yt, yp, pos_label=1, zero_division=0), auc=auc))

    res = pd.DataFrame(results)
    res.to_csv(f"results_{args.tag}.csv", index=False)
    print("\n===== LODO SUMMARY (", args.tag, ") =====")
    print(res.round(4).to_string(index=False))
    print("\nmean +- std:")
    for c in ["acc", "drowsy_recall", "drowsy_f1", "auc"]:
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