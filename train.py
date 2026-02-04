import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

DATA_DIR = "faces"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
MIN_WIDTH = 80
MIN_HEIGHT = 80
LR = 1e-4

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for label, folder in enumerate(["real", "fake"]):
            folder_path = os.path.join(root_dir, folder)
            for file in os.listdir(folder_path):
                path = os.path.join(folder_path, file)
                try:
                    img = Image.open(path)
                    w, h = img.size
                    if w >= MIN_WIDTH and h >= MIN_HEIGHT:
                        self.samples.append((path, label))
                except:
                    pass

        print(f"Loaded {len(self.samples)} images after size filtering")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    dataset = FaceDataset(DATA_DIR, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total * 100
        print(f"Epoch {epoch+1}: Loss={total_loss:.4f}, Acc={acc:.2f}%")

    torch.save(model.state_dict(), "deepfake_face_model.pth")
    print("Model saved as deepfake_face_model.pth")


if __name__ == "__main__":
    main()
