import json
import cv2, os

with open("train_sample_videos/metadata.json") as f:
    metadata = json.load(f)

print(list(metadata.items())[:3])


VIDEO_DIR = "train_sample_videos"
OUTPUT_DIR = "frames"
os.makedirs("frames/real", exist_ok=True)
os.makedirs("frames/fake", exist_ok=True)

for video, info in metadata.items():
    if info["split"] != "train":
        continue

    label = info["label"].lower()  
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, video))
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % 30 == 0: 
            cv2.imwrite(f"{OUTPUT_DIR}/{label}/{video}_{count}.jpg", frame)
        count += 1

    cap.release()
