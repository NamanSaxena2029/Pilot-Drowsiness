import os, argparse
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from step2_lodo import build_df, make_windows


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


class CNN_LSTM(nn.Module):                      # identical to step2_lodo.py
    def __init__(self):
        super().__init__()
        self.cnn = models.mobilenet_v2(weights=None)
        self.cnn.classifier = nn.Identity()
        self.lstm = nn.LSTM(1280, 128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 2)
    def forward(self, x):
        B, T, C, H, W = x.size()
        f = self.cnn(x.view(B * T, C, H, W)).view(B, T, -1).contiguous()
        return self.fc(self.lstm(f)[0][:, -1, :])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/cropped_mp")
    ap.add_argument("--tag", default="normal")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize((160, 160)), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    df = build_df(a.base)
    info = df.set_index("image")[["stream", "frame"]]
    rows = []
    for d in sorted(df.driver.unique()):
        path = f"models/{a.tag}_test{d}.pth"
        model = CNN_LSTM().to(dev)
        model.load_state_dict(torch.load(path, map_location=dev, weights_only=True)); model.eval()
        wins = make_windows(df[df.driver == d], stride=1)
        dl = DataLoader(WinDS(wins, tf), batch_size=32)
        ps = []
        with torch.no_grad():
            for x, _ in tqdm(dl, desc=f"driver {d}"):
                ps += torch.softmax(model(x.to(dev)), 1)[:, 1].cpu().tolist()
        for (paths, y), p in zip(wins, ps):
            last = paths[-1]
            rows.append(dict(driver=d, image=os.path.basename(last), stream=info.loc[last, "stream"],
                             frame=int(info.loc[last, "frame"]), label=y, p=p))
    out = pd.DataFrame(rows)
    out.to_csv(f"preds_cnn_{a.tag}.csv", index=False)
    print(f"saved preds_cnn_{a.tag}.csv ({len(out)} windows)")


if __name__ == "__main__":
    main()