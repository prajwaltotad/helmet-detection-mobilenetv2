import cv2
import numpy as np
from tensorflow.keras.models import load_model

try:
    from config import MODEL_PATH, IMG_SIZE
except ImportError:
    from src.config import MODEL_PATH, IMG_SIZE


def predict_image(image_path):
    print("Loading model...")
    model = load_model(MODEL_PATH, compile=False)

    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not read image: {image_path}")
        return

    # Resize
    img = cv2.resize(img, IMG_SIZE)

    # OpenCV BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to float
    img = img.astype(np.float32)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # IMPORTANT:
    # Do NOT call preprocess_input here.
    # The saved model already performs MobileNetV2 preprocessing.

    prediction = float(model.predict(img, verbose=0)[0][0])

    if prediction >= 0.5:
        label = "WITHOUT HELMET"
        confidence = prediction
    else:
        label = "WITH HELMET"
        confidence = 1 - prediction

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <image_path>")
    else:
        predict_image(sys.argv[1])