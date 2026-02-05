import os, random
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from tqdm import tqdm
from collections import defaultdict

# ================= CONFIG =================
DATA_DIR = "faces"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
MIN_WIDTH = 80
MIN_HEIGHT = 80
LR = 1e-4
VAL_SPLIT = 0.2
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= COLLECT BY VIDEO =================
video_dict = defaultdict(list)

for label, folder in enumerate(["real", "fake"]):
    folder_path = os.path.join(DATA_DIR, folder)
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        try:
            img = Image.open(path)
            w, h = img.size
            if w < MIN_WIDTH or h < MIN_HEIGHT:
                continue
            video_name = file.split("_")[0]
            video_dict[video_name].append((path, label))
        except:
            pass

videos = list(video_dict.keys())
random.shuffle(videos)

split_idx = int(len(videos) * (1 - VAL_SPLIT))
train_videos = set(videos[:split_idx])
val_videos = set(videos[split_idx:])

train_samples = []
val_samples = []

for vid, samples in video_dict.items():
    if vid in train_videos:
        train_samples.extend(samples)
    else:
        val_samples.extend(samples)

# print(f"Train images: {len(train_samples)}")
# print(f"Val images: {len(val_samples)}")

# ================= DATASET =================
class FaceDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# ================= TRANSFORMS =================
def main():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                            [0.229,0.224,0.225])
    ])

    train_ds = FaceDataset(train_samples, train_tf)
    val_ds = FaceDataset(val_samples, val_tf)

    # ================= CLASS BALANCING =================
    labels = [label for _, label in train_samples]
    class_count = [labels.count(0), labels.count(1)]
    weights = [1/class_count[l] for l in labels]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ================= MODEL =================
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ================= TRAIN =================
    best_val_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total * 100

        # ===== VALIDATION =====
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [VAL]"):
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                preds = out.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total * 100

        print(f"Epoch {epoch+1}: TrainAcc={train_acc:.2f}% ValAcc={val_acc:.2f}%")

        
        torch.save(model.state_dict(), "deepfake_face_model_best.pth")
        print("✅ Saved model")

    print("Training complete.")
if __name__ == "__main__":
    main()