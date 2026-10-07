import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# Model & Training
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 1e-4
NUM_CLASSES = 2

# IMPORTANT: must match the order used during training
CLASS_NAMES = ["with_helmet", "without_helmet"]

# Trained model
MODEL_PATH = os.path.join(MODEL_DIR, "helmet_model.keras")
PERSON_MODEL_PATH = os.path.join(MODEL_DIR, "yolo26n.pt")