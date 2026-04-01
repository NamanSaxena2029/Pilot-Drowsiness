"""
STEP 2 — RETRAIN CNN-LSTM
MediaPipe cropped data se retrain
"""

import os, cv2, random, numpy as np, torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import pandas as pd

# Seeds
random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)

torch.backends.cudnn.enabled = True

# ========================
# DATA
# ========================
data = []
base_path = "data/cropped_mp"

for label, folder in enumerate(["notdrowsy", "drowsy"]):
    path = os.path.join(base_path, folder)
    for img in os.listdir(path):
        if img.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            data.append([os.path.join(path, img), label])

df = pd.DataFrame(data, columns=["image", "label"])
print(f"Total: {len(df)} | Drowsy: {(df.label==1).sum()} | NotDrowsy: {(df.label==0).sum()}")

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

# ========================
# TRANSFORMS
# ========================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ========================
# DATASET
# ========================
class SequenceDataset(Dataset):
    def __init__(self, df, seq_len=3, transform=None):
        self.df = df.sort_values("image").reset_index(drop=True)
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return len(self.df) - self.seq_len

    def __getitem__(self, idx):
        images = []
        for i in range(self.seq_len):
            row = self.df.iloc[idx + i]
            img = cv2.imread(row["image"])
            if img is None:
                img = np.zeros((160,160,3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            images.append(img)
        images = torch.stack(images)
        label = self.df.iloc[idx + self.seq_len - 1]["label"]
        return images, label

train_dataset = SequenceDataset(train_df, transform=train_transform)
val_dataset   = SequenceDataset(val_df,   transform=test_transform)
test_dataset  = SequenceDataset(test_df,  transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=8, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=8, num_workers=0, pin_memory=True)

# ========================
# MODEL
# ========================
class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.mobilenet_v2(weights="IMAGENET1K_V1")
        self.cnn.classifier = nn.Identity()
        for param in self.cnn.features[:-4].parameters():
            param.requires_grad = False
        self.lstm = nn.LSTM(1280, 128, num_layers=1, batch_first=True)
        self.fc   = nn.Linear(128, 2)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B*T, C, H, W)
        features = self.cnn(x)
        features = features.view(B, T, -1).contiguous()
        out, _ = self.lstm(features)
        out = out[:, -1, :]
        return self.fc(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using: {device}")

model     = CNN_LSTM().to(device)
weights   = torch.tensor([1.0, 1.2]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam([
    {'params': model.cnn.features[-4:].parameters(), 'lr': 1e-5},
    {'params': model.lstm.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(),   'lr': 1e-4}
])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
scaler    = torch.amp.GradScaler("cuda")

# ========================
# TRAIN
# ========================
best_val_acc = 0
patience = 3
counter  = 0

for epoch in range(15):
    model.train()
    correct, total = 0, 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            outputs = model(imgs)
            loss    = criterion(outputs, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        preds    = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    train_acc = correct / total

    # VALIDATION
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(imgs)
            preds   = torch.argmax(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")
    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), "best_model_v2.pth")
        print(f"  ✓ Saved best_model_v2.pth (val={val_acc:.4f})")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break

# ========================
# TEST
# ========================
print("\n[INFO] Testing...")
model.load_state_dict(torch.load("best_model_v2.pth"))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs    = imgs.to(device)
        outputs = model(imgs)
        preds   = torch.argmax(outputs, dim=1).cpu().numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)

print(classification_report(y_true, y_pred, target_names=["Not Drowsy", "Drowsy"]))
print("\nbest_model_v2.pth saved! Ab drowsy_detection.py mein use karo.")