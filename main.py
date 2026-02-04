import cv2, os, json
from mtcnn import MTCNN

VIDEO_DIR = "train_sample_videos"
OUTPUT_DIR = "faces"

detector = MTCNN()

os.makedirs("faces/real", exist_ok=True)
os.makedirs("faces/fake", exist_ok=True)

with open("train_sample_videos/metadata.json") as f:
    metadata = json.load(f)

# Collect already processed videos (Option 1)
processed_videos = set()
for label in ["real", "fake"]:
    for file in os.listdir(f"{OUTPUT_DIR}/{label}"):
        if "_" in file:
            processed_videos.add(file.split("_")[0])

for video, info in metadata.items():
    if info["split"] != "train":
        continue

    # OPTION 1: Skip already processed videos
    if video in processed_videos:
        print(f"Skipping already processed: {video}")
        continue

    label = info["label"].lower()
    path = os.path.join(VIDEO_DIR, video)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Could not open:", path)
        continue

    frame_count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 20 == 0:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = detector.detect_faces(rgb)

                if len(faces) == 0:
                    frame_count += 1
                    continue

                for i, face in enumerate(faces):
                    x, y, w, h = face["box"]

                    # Clamp values
                    x = max(0, x)
                    y = max(0, y)
                    w = max(0, w)
                    h = max(0, h)

                    crop = rgb[y:y+h, x:x+w]
                    if crop.size == 0:
                        continue

                    filename = f"{video}_{frame_count}_{i}.jpg"
                    cv2.imwrite(f"{OUTPUT_DIR}/{label}/{filename}",
                                cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                    saved += 1

            except Exception as e:
                print(f" Skipped bad frame in {video}: {e}")

        frame_count += 1

    cap.release()
    print(f"Processed {video}, saved {saved} faces")
