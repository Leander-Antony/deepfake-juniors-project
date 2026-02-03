import os
from collections import defaultdict
import torch
from torchvision import models, transforms
from PIL import Image


FRAME_DIR = "frames/real"   
VIDEO_NAME = "atkdltyyen.mp4"  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = torch.nn.Linear(1280, 2)
model = model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


video_frames = defaultdict(list)

for file in os.listdir(FRAME_DIR):
    if file.endswith(".jpg"):
        video_name = file.split("_")[0]   
        video_frames[video_name].append(os.path.join(FRAME_DIR, file))


def predict_video(video_name):
    frames = video_frames[video_name]
    scores = []

    for frame_path in frames:
        img = Image.open(frame_path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img)
            prob = torch.softmax(out, dim=1)

        scores.append(prob[0][1].item())  

    return sum(scores) / len(scores)


score = predict_video(VIDEO_NAME)
print("Fake probability:", round(score, 3))

if score > 0.7:
    verdict = "Likely Fake"
elif score > 0.5:
    verdict = "Suspicious"
else:
    verdict = "Real"

print("Verdict:", verdict)
