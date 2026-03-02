import os
import json
import shutil

VIDEO_DIR = "train_sample_videos"   # where videos + metadata.json are
REAL_DIR = "real_vid"
FAKE_DIR = "fake_vid"

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

with open(os.path.join(VIDEO_DIR, "metadata.json"), "r") as f:
    metadata = json.load(f)

moved_real = 0
moved_fake = 0

for video, info in metadata.items():
    src = os.path.join(VIDEO_DIR, video)

    if not os.path.exists(src):
        print("❌ Missing:", video)
        continue

    label = info["label"].lower()

    if label == "real":
        dst = os.path.join(REAL_DIR, video)
        shutil.move(src, dst)
        moved_real += 1

    elif label == "fake":
        dst = os.path.join(FAKE_DIR, video)
        shutil.move(src, dst)
        moved_fake += 1

print(f"✅ Done")
print(f"Moved REAL videos: {moved_real}")
print(f"Moved FAKE videos: {moved_fake}")
