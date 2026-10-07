import os

# Streamlit Cloud does not provide a CUDA GPU for this app.
# Force PyTorch/Ultralytics to use CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import threading
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from tensorflow.keras.models import load_model
from ultralytics import YOLO


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Helmet Detection",
    page_icon="🪖",
    layout="wide"
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

HELMET_MODEL_PATH = BASE_DIR / "model" / "helmet_model.keras"
PERSON_MODEL_PATH = BASE_DIR / "model" / "yolo26n.pt"


# ---------------------------------------------------------
# Load models once
# ---------------------------------------------------------

@st.cache_resource
def load_models():
    helmet_model = load_model(
        HELMET_MODEL_PATH,
        compile=False
    )

    person_model = YOLO(
        str(PERSON_MODEL_PATH)
    )

    return helmet_model, person_model


helmet_model, person_model = load_models()


# ---------------------------------------------------------
# Thread safety
# ---------------------------------------------------------

model_lock = threading.Lock()


# ---------------------------------------------------------
# Helmet classification
# ---------------------------------------------------------

def classify_helmet(image):
    """
    Classify a cropped person/head region.

    The saved MobileNetV2 model already contains
    MobileNetV2 preprocessing internally.
    Therefore preprocess_input() must NOT be called here.
    """

    image = cv2.resize(image, (224, 224))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32)

    image = np.expand_dims(
        image,
        axis=0
    )

    with model_lock:
        prediction = float(
            helmet_model.predict(
                image,
                verbose=0
            )[0][0]
        )

    if prediction >= 0.5:
        return "WITHOUT HELMET", prediction

    return "WITH HELMET", 1.0 - prediction


# ---------------------------------------------------------
# Video processing callback
# ---------------------------------------------------------

def process_frame(frame):
    """
    Process one browser webcam frame.
    """

    img = frame.to_ndarray(
        format="bgr24"
    )

    output = img.copy()

    # Detect only the COCO "person" class (class 0)
    with model_lock:
        results = person_model(
                img,
                classes=[0],
                conf=0.45,
                device="cpu",
                verbose=False
        )

    person_found = False

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            x1, y1, x2, y2 = coordinates

            # Keep coordinates inside the frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(output.shape[1], x2)
            y2 = min(output.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            person_found = True

            # Use the upper 45% of the person's bounding box
            # as the head/helmet region.
            person_height = y2 - y1

            upper_y2 = y1 + int(
                person_height * 0.45
            )

            head_region = img[
                y1:upper_y2,
                x1:x2
            ]

            if head_region.size == 0:
                continue

            label, confidence = classify_helmet(
                head_region
            )

            # Green = helmet
            # Red = no helmet
            if label == "WITH HELMET":
                color = (0, 220, 0)
            else:
                color = (0, 0, 255)

            # Person bounding box
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Upper/head region
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, upper_y2),
                (255, 255, 0),
                1
            )

            text = (
                f"{label}: "
                f"{confidence * 100:.1f}%"
            )

            text_y = max(
                30,
                y1 - 10
            )

            cv2.putText(
                output,
                text,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA
            )

    # If YOLO doesn't find a person,
    # do not classify an isolated helmet.
    if not person_found:

        cv2.putText(
            output,
            "NO PERSON DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    return av.VideoFrame.from_ndarray(
        output,
        format="bgr24"
    )


# ---------------------------------------------------------
# User interface
# ---------------------------------------------------------

st.title("🪖 Real-Time Helmet Detection")

st.markdown(
    """
    ### 👤 How it works

    **YOLO** first detects a person in the camera frame.
    The upper portion of the detected person is then passed
    to the **MobileNetV2 helmet classifier**.

    This prevents an isolated helmet from automatically
    being classified as someone wearing a helmet.
    """
)

st.info(
    "Click START below and allow camera access when your "
    "browser asks for permission."
)


# ---------------------------------------------------------
# WebRTC configuration
# ---------------------------------------------------------

RTC_CONFIGURATION = {
    "iceServers": [
        {
            "urls": [
                "stun:stun.l.google.com:19302"
            ]
        }
    ]
}


# ---------------------------------------------------------
# Start webcam
# ---------------------------------------------------------

webrtc_streamer(
    key="helmet-detection",
    video_frame_callback=process_frame,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration=RTC_CONFIGURATION,
    media_toggle_controls=False
)


st.markdown("---")

st.caption(
    "MobileNetV2 + YOLO + OpenCV | "
    "Helmet Detection Project"
)