import streamlit as st
import cv2
import os
import tempfile
from PIL import Image
import torch
from torchvision import models, transforms
import numpy as np


st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("🎥 Deepfake Video Detector")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.write("Using device:", device)


@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = torch.nn.Linear(1280, 2)
    model = model.to(device)
    model.eval()
    return model

model = load_model()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def extract_frames(video_path, every_n_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n_frames == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        count += 1

    cap.release()
    return frames

def predict_frames(frames):
    scores = []

    for frame in frames:
        img = Image.fromarray(frame).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img)
            prob = torch.softmax(out, dim=1)

        scores.append(prob[0][1].item())

    return float(np.mean(scores))


video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    st.video(video_file)

    st.write("Extracting frames...")
    frames = extract_frames(video_path)

    if len(frames) == 0:
        st.error("No frames extracted.")
    else:
        st.write(f"Frames sampled: {len(frames)}")

        st.write("Running detection...")
        score = predict_frames(frames)

        st.subheader("Result")
        st.write(f"Fake probability: {round(score,3)}")

        if score > 0.7:
            st.error("🔴 Likely Fake")
        elif score > 0.5:
            st.warning("🟡 Suspicious")
        else:
            st.success("🟢 Likely Real")

        st.subheader("Sample Frames")
        st.image(frames[:3], width=200)

    os.remove(video_path)
