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
import requests
from bs4 import BeautifulSoup

IMG_SIZE = 224
FRAME_SKIP = 20  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.title("🕵️ Deepfake Detector (Image + Video + URL)")

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

def download_from_url(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers)

    # If it's an HTML page (like ibb.co)
    if "text/html" in r.headers.get("Content-Type", ""):
        soup = BeautifulSoup(r.text, "html.parser")
        og_image = soup.find("meta", property="og:image")

        if og_image:
            img_url = og_image["content"]
            return download_from_url(img_url)
        else:
            return None

    suffix = url.split(".")[-1].split("?")[0]
    temp = tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix)

    temp.write(r.content)
    temp.close()
    return temp.name


def predict_face(face_img):
    img = Image.fromarray(face_img).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img)
        prob = torch.softmax(out, dim=1)

    return prob[0][1].item()  


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
    return None if len(scores) == 0 else sum(scores)/len(scores)


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

    return None if len(scores) == 0 else sum(scores)/len(scores)


uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg","png","jpeg","mp4","avi"])

url_input = st.text_input("OR paste Image/Video URL")

file_path = None
is_image = False

if uploaded_file:
    suffix = uploaded_file.name.split(".")[-1]
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix="."+suffix)
    tfile.write(uploaded_file.read())
    file_path = tfile.name

elif url_input:
    st.info("Downloading from URL...")
    file_path = download_from_url(url_input)
    if file_path is None:
        st.error("Failed to download file from URL.")

if file_path:
    ext = file_path.split(".")[-1].lower()

    if ext in ["jpg","jpeg","png"]:
        image = Image.open(file_path)
        st.image(image, caption="Input Image")
        score = process_image(image)

    else:
        st.video(file_path)
        score = process_video(file_path)

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
