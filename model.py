import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from torchvision import transforms, models
import torch.nn as nn
import tempfile
import os

IMG_SIZE = 224
FRAME_SKIP = 20  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("🕵️ Deepfake Detector (Image + Video)")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("deepfake_face_model.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

detector = MTCNN()

def predict_face(face_img):
    img = Image.fromarray(face_img).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        prob = torch.softmax(out, dim=1)

    return prob[0][1].item()  # fake probability



def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    scores = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_SKIP == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)

            for face in faces:
                x, y, w, h = face["box"]
                x, y = max(0,x), max(0,y)
                crop = rgb[y:y+h, x:x+w]

                if crop.size == 0:
                    continue

                score = predict_face(crop)
                scores.append(score)

        frame_count += 1

    cap.release()

    if len(scores) == 0:
        return None
    return sum(scores) / len(scores)


def process_image(image):
    img = np.array(image)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    scores = []

    for face in faces:
        x, y, w, h = face["box"]
        x, y = max(0,x), max(0,y)
        crop = rgb[y:y+h, x:x+w]

        if crop.size == 0:
            continue

        score = predict_face(crop)
        scores.append(score)

    if len(scores) == 0:
        return None
    return sum(scores) / len(scores)


uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg","png","jpeg","mp4","avi"])

if uploaded_file:
    suffix = uploaded_file.name.split(".")[-1]

    if suffix in ["jpg","png","jpeg"]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        score = process_image(image)

    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        st.video(tfile.name)
        score = process_video(tfile.name)

    if score is None:
        st.error("No face detected.")
    else:
        st.subheader(f"Fake Probability: {round(score*100,2)}%")

        if score > 0.7:
            st.error("Verdict: FAKE")
        elif score > 0.4:
            st.warning("Verdict: Suspicious")
        else:
            st.success("Verdict: REAL")
