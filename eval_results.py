# eval_results.py
import torch, numpy as np
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import cv2, os, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

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
        return self.fc(out[:, -1, :])

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

# Data
data = []
for label, folder in enumerate(["notdrowsy", "drowsy"]):
    path = os.path.join("data/cropped_mp", folder)
    for img in os.listdir(path):
        if img.lower().endswith((".jpg",".jpeg",".png",".bmp")):
            data.append([os.path.join(path, img), label])

df = pd.DataFrame(data, columns=["image","label"])
_, temp_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
_, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160,160)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

test_loader = DataLoader(
    SequenceDataset(test_df, transform=test_transform),
    batch_size=8, num_workers=0
)

# Load model
device = torch.device("cuda")
model = CNN_LSTM().to(device)
model.load_state_dict(torch.load("best_model_v2.pth", weights_only=True))
model.eval()

# Predict
y_true, y_pred, y_scores = [], [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:,1].cpu().numpy()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
        y_scores.extend(probs)

print(classification_report(y_true, y_pred, target_names=["Not Drowsy","Drowsy"]))

# Confusion Matrix
plt.figure(figsize=(7,5))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Not Drowsy","Drowsy"],
            yticklabels=["Not Drowsy","Drowsy"])
plt.title("Confusion Matrix — best_model_v2")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_v2.png", dpi=150)
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1],[0,1], color='navy', lw=1, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — best_model_v2")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_v2.png", dpi=150)
plt.show()

print("\n✓ confusion_matrix_v2.png saved")
print("✓ roc_curve_v2.png saved")