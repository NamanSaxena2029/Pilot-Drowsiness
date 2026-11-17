import os, re, argparse, random
import cv2, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="data/cropped_mp")
ap.add_argument("--out", default="best_model_v2_subj.pth")
# Default: first run of digits in the filename = subject id. CHANGE after --inspect.
ap.add_argument("--regex", default=r"(\d+)")
ap.add_argument("--init", default=None, help="optional .pth to fine-tune from (low-light run)")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--inspect", action="store_true")
args = ap.parse_args()

random.seed(42); np.random.seed(42); torch.manual_seed(42)

def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

rows = []
for label, folder in enumerate(["notdrowsy", "drowsy"]):
    d = os.path.join(args.base, folder)
    for f in os.listdir(d):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            m = re.search(args.regex, f)
            rows.append([os.path.join(d, f), label, m.group(1) if m else "UNKNOWN"])
df = pd.DataFrame(rows, columns=["image", "label", "group"])

if args.inspect:
    print(df.sample(15, random_state=0).to_string(index=False))
    print("\nnum groups:", df.group.nunique(), "| UNKNOWN:", (df.group == "UNKNOWN").sum())
    print(df.groupby("group").size().describe())
    raise SystemExit

assert df.group.nunique() >= 6, "Too few groups - fix --regex (run --inspect)"
assert (df.group == "UNKNOWN").sum() == 0, "Some files have no group id - fix --regex"

# ---- group-wise 70/15/15 split ----
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
tr_idx, tmp_idx = next(gss.split(df, df.label, df.group))
train_df, tmp_df = df.iloc[tr_idx], df.iloc[tmp_idx]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
va_idx, te_idx = next(gss2.split(tmp_df, tmp_df.label, tmp_df.group))
val_df, test_df = tmp_df.iloc[va_idx], tmp_df.iloc[te_idx]

for n, s in [("train", train_df), ("val", val_df), ("test", test_df)]:
    print(f"{n}: {len(s)} imgs | {s.group.nunique()} groups | drowsy={int(s.label.sum())}")
assert not (set(train_df.group) & set(test_df.group)), "LEAK: train/test share groups"
assert not (set(train_df.group) & set(val_df.group)), "LEAK: train/val share groups"
print("Leakage check passed: 0 shared groups between splits")

tf_train = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
    transforms.ColorJitter(0.3, 0.3, 0.3), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
tf_eval = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((160, 160)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class WindowDataset(Dataset):
    """seq_len consecutive frames from the SAME group and SAME label run."""
    def __init__(self, df, seq_len=3, transform=None):
        self.seq_len, self.transform, self.windows = seq_len, transform, []
        for _, g in df.groupby("group"):
            g = sorted(zip(g.image, g.label), key=lambda x: natural_key(os.path.basename(x[0])))
            for i in range(len(g) - seq_len + 1):
                w = g[i:i + seq_len]
                self.windows.append(([p for p, _ in w], w[-1][1]))
    def __len__(self): return len(self.windows)
    def __getitem__(self, i):
        paths, label = self.windows[i]
        frames = []
        for p in paths:
            img = cv2.imread(p)
            img = np.zeros((160, 160, 3), np.uint8) if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(self.transform(img))
        return torch.stack(frames), label

class CNN_LSTM(nn.Module):
    def __init__(self, pretrained=True):
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

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tl = DataLoader(WindowDataset(train_df, transform=tf_train), batch_size=8, shuffle=True)
vl = DataLoader(WindowDataset(val_df, transform=tf_eval), batch_size=8)
sl = DataLoader(WindowDataset(test_df, transform=tf_eval), batch_size=8)

model = CNN_LSTM(pretrained=args.init is None).to(dev)
if args.init:
    model.load_state_dict(torch.load(args.init, map_location=dev))
    print("Fine-tuning from", args.init)
lr_scale = 0.5 if args.init else 1.0
opt = optim.Adam([
    {"params": model.cnn.features[-4:].parameters(), "lr": 1e-5 * lr_scale},
    {"params": model.lstm.parameters(), "lr": 1e-4 * lr_scale},
    {"params": model.fc.parameters(), "lr": 1e-4 * lr_scale}])
crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.2]).to(dev))

best, bad = 0.0, 0
for ep in range(args.epochs):
    model.train(); c = n = 0
    for x, y in tqdm(tl, desc=f"Epoch {ep+1}"):
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(); out = model(x); crit(out, y).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        c += (out.argmax(1) == y).sum().item(); n += y.size(0)
    model.eval(); vc = vn = 0
    with torch.no_grad():
        for x, y in vl:
            vc += (model(x.to(dev)).argmax(1).cpu() == y).sum().item(); vn += y.size(0)
    va = vc / vn
    print(f"Epoch {ep+1} | train {c/n:.4f} | val (unseen subjects) {va:.4f}")
    if va > best:
        best, bad = va, 0; torch.save(model.state_dict(), args.out)
    else:
        bad += 1
        if bad >= 3: print("Early stopping"); break

model.load_state_dict(torch.load(args.out, map_location=dev)); model.eval()
yt, yp, ys = [], [], []
with torch.no_grad():
    for x, y in sl:
        o = model(x.to(dev)); pr = torch.softmax(o, 1)[:, 1].cpu().numpy()
        yt += y.tolist(); yp += o.argmax(1).cpu().tolist(); ys += pr.tolist()
print("\nSUBJECT-INDEPENDENT TEST RESULT")
print(classification_report(yt, yp, target_names=["Not Drowsy", "Drowsy"]))
print("AUC:", round(roc_auc_score(yt, ys), 4))