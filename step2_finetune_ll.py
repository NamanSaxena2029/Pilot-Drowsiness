import os, cv2, random, numpy as np, torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import pandas as pd

# ========================
# SEED
# ========================
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ========================
# DATA (LOW LIGHT DATASET)
# ========================
data = []
base_path = "data/cropped_mp_ll"

for label, folder in enumerate(["notdrowsy", "drowsy"]):
    path = os.path.join(base_path, folder)
    for img in os.listdir(path):
        if img.lower().endswith((".jpg",".jpeg",".png",".bmp")):
            data.append([os.path.join(path, img), label])

df = pd.DataFrame(data, columns=["image","label"])
print(f"Total: {len(df)}")

train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
val_df, test_df   = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

# ========================
# TRANSFORMS
# ========================
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160,160)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ColorJitter(0.3,0.3),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160,160)),
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
            row = self.df.iloc[idx+i]
            img = cv2.imread(row["image"])

            if img is None:
                img = np.zeros((160,160,3), dtype=np.uint8)

            # IR / grayscale fix
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform:
                img = self.transform(img)

            images.append(img)

        return torch.stack(images), self.df.iloc[idx+self.seq_len-1]["label"]

# ========================
# LOADERS
# ========================
train_loader = DataLoader(SequenceDataset(train_df, transform=train_transform), batch_size=8, shuffle=True)
val_loader   = DataLoader(SequenceDataset(val_df,   transform=test_transform), batch_size=8)
test_loader  = DataLoader(SequenceDataset(test_df,  transform=test_transform), batch_size=8)

# ========================
# MODEL
# ========================
class CNN_LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = models.mobilenet_v2(weights=None)
        self.cnn.classifier = nn.Identity()

        # freeze early layers
        for param in self.cnn.features[:-4].parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(1280,128,batch_first=True)
        self.fc   = nn.Linear(128,2)

    def forward(self,x):
        B,T,C,H,W = x.size()
        x = x.view(B*T,C,H,W)
        f = self.cnn(x)
        f = f.view(B,T,-1)
        out,_ = self.lstm(f)
        return self.fc(out[:,-1,:])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN_LSTM().to(device)

# LOAD PRETRAINED RGB MODEL (IMPORTANT)
model.load_state_dict(torch.load("best_model_v2.pth", map_location=device))
print("Loaded best_model_v2 --> fine-tuning for low light")

# ========================
# LOSS + OPTIMIZER
# ========================
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam([
    {'params': model.cnn.features[-4:].parameters(), 'lr':5e-6},
    {'params': model.lstm.parameters(), 'lr':5e-5},
    {'params': model.fc.parameters(),   'lr':5e-5}
])

# ========================
# TRAIN
# ========================
best_val_acc = 0

for epoch in range(10):
    model.train()
    correct,total = 0,0

    for imgs,labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        imgs,labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs,labels)

        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        correct += (preds==labels).sum().item()
        total += labels.size(0)

    train_acc = correct/total

    # VALIDATION
    model.eval()
    val_correct,val_total = 0,0
    with torch.no_grad():
        for imgs,labels in val_loader:
            imgs,labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(1)
            val_correct += (preds==labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct/val_total
    print(f"Epoch {epoch+1} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(),"best_model_ll.pth")
        print("Saved best_model_ll.pth")

# ========================
# TEST
# ========================
model.load_state_dict(torch.load("best_model_ll.pth"))
model.eval()

y_true,y_pred = [],[]
with torch.no_grad():
    for imgs,labels in test_loader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(1).cpu().numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
 
print("\nFINAL REPORT:")
print(classification_report(y_true,y_pred))