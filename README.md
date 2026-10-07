````markdown
# 🪖 Helmet Detection using MobileNetV2 + YOLO + OpenCV

A real-time helmet detection system built using **MobileNetV2 transfer learning**, a pretrained **YOLO person detector**, **TensorFlow/Keras**, and **OpenCV**.

The system is designed to classify whether a detected person is:

- 🟢 **Wearing a Helmet**
- 🔴 **Not Wearing a Helmet**

The project uses a two-stage computer vision pipeline: YOLO first detects whether a person is present, and MobileNetV2 then classifies the upper region of the detected person for helmet usage.

---

## ✨ Features

- 🧠 MobileNetV2 transfer learning for helmet classification
- 👤 YOLO-based person detection
- 🎥 Real-time webcam inference
- 🖼️ Single-image prediction
- 📦 Pascal VOC XML annotation preprocessing
- 🔄 Data augmentation during training
- 📊 Accuracy, precision, recall, F1-score and confusion matrix evaluation
- 💾 Trained model included in the repository
- ☁️ Model training performed using Google Colab GPU
- 🖥️ Local inference using Python and OpenCV

---

## 🧩 System Architecture

The project uses a two-stage inference pipeline:

```text
                    📷 Webcam Frame
                           │
                           ▼
                  👤 YOLO Person Detector
                           │
                    Person detected?
                      /           \
                    No             Yes
                    │               │
                    ▼               ▼
               Ignore Frame    Extract Upper
                               Person Region
                                    │
                                    ▼
                           🧠 MobileNetV2
                              Classifier
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                  🟢 WITH HELMET       🔴 WITHOUT HELMET
````

This approach prevents the system from directly classifying an isolated helmet as a person wearing a helmet.

---

## 📂 Dataset

The project uses the **Bike Helmets Detection** dataset available on Kaggle.

🔗 **Kaggle Dataset:**
[https://www.kaggle.com/datasets/brendan45774/bike-helmets-detection](https://www.kaggle.com/datasets/brendan45774/bike-helmets-detection)

The original dataset contains:

* 🖼️ Images
* 📝 Pascal VOC XML annotations

The XML annotation files provide:

* Object class
* Bounding-box coordinates

The relevant classes are:

```text
With helmet
Without helmet
```

### 🔄 Dataset Preprocessing

The raw Kaggle dataset is **not stored in this repository**.

Instead, the training notebook performs the preprocessing pipeline:

```text
Kaggle Dataset
      ↓
Images + XML Annotations
      ↓
Read XML Files
      ↓
Extract Class Labels
      ↓
Extract Bounding-Box Coordinates
      ↓
Crop Annotated Objects
      ↓
with_helmet/
without_helmet/
      ↓
Train / Validation / Test Split
```

This converts the original annotated dataset into a format suitable for image classification with MobileNetV2.

---

## 🧠 MobileNetV2 Model

The classifier uses **MobileNetV2 pretrained on ImageNet** as the feature extractor.

### Model Architecture

```text
Input Image
224 × 224 × 3
      ↓
Data Augmentation
      ↓
MobileNetV2
      ↓
Global Average Pooling
      ↓
Dense Layer (128 units)
      ↓
Dropout
      ↓
Sigmoid Output
      ↓
Helmet / No Helmet
```

### Model Configuration

| Parameter          | Value                 |
| ------------------ | --------------------- |
| Base Model         | MobileNetV2           |
| Pretrained Weights | ImageNet              |
| Input Size         | 224 × 224             |
| Output             | Binary Classification |
| Activation         | Sigmoid               |
| Loss Function      | Binary Cross-Entropy  |
| Optimizer          | Adam                  |
| Learning Rate      | 1e-4                  |
| Batch Size         | 32                    |

The MobileNetV2 preprocessing step is included **inside the trained model**, so the same preprocessing is automatically used during inference.

---

## 👤 Person Detection

To reduce false predictions caused by isolated helmets, the webcam pipeline uses a pretrained **YOLO person detector** before running the helmet classifier.

```text
Webcam Frame
     ↓
YOLO Person Detection
     ↓
Person Found
     ↓
Upper Region of Person
     ↓
MobileNetV2
     ↓
Helmet Classification
```

The person detection model used in the project is:

```text
yolo26n.pt
```

It is stored locally in:

```text
model/yolo26n.pt
```

---

## 📊 Model Performance

The MobileNetV2 classifier was evaluated on a held-out test set.

### Test Results

| Metric                   |  Result |
| ------------------------ | ------: |
| Test Accuracy            | ~89.35% |
| With Helmet Precision    |    0.95 |
| With Helmet Recall       |    0.88 |
| With Helmet F1-Score     |    0.92 |
| Without Helmet Precision |    0.80 |
| Without Helmet Recall    |    0.92 |
| Without Helmet F1-Score  |    0.85 |

### Confusion Matrix

```text
                    Predicted
                 With    Without

Actual With       126       17

Actual Without      6       67
```

The model correctly classified:

```text
193 / 216 test samples
```

for an overall accuracy of approximately:

```text
89.35%
```

---

## 📁 Project Structure

```text
helmet-detection-mobilenetv2/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .gitignore
│
├── 📂 data/
│   └── .gitkeep
│
├── 📂 model/
│   ├── helmet_model.keras
│   └── yolo26n.pt
│
├── 📂 notebooks/
│   └── helmet_training.ipynb
│
└── 📂 src/
    ├── __init__.py
    ├── config.py
    ├── predict.py
    ├── train.py
    └── webcam.py
```

### 📌 Repository Notes

The following are intentionally **not included** in GitHub:

* Raw Kaggle images
* XML annotation files
* Generated training/validation/test images
* `.venv`
* Python cache files
* IDE-specific files

The `data/` directory is kept with `.gitkeep` so the project structure remains visible.

---

# 🚀 Getting Started

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/prajwaltotad/helmet-detection-mobilenetv2.git
cd helmet-detection-mobilenetv2
```

---

## 2️⃣ Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### Main Dependencies

```text
tensorflow>=2.12
opencv-python
numpy
matplotlib
scikit-learn
Pillow
tqdm
ultralytics
```

---

# 🖼️ Image Prediction

The project includes an image prediction script.

Run:

```bash
python src/predict.py path/to/image.jpg
```

Example:

```bash
python src/predict.py helmet_test.jpg
```

Example output:

```text
Prediction: WITH HELMET
Confidence: 98.56%
```

or:

```text
Prediction: WITHOUT HELMET
Confidence: 75.21%
```

---

# 🎥 Real-Time Webcam Detection

Start the webcam application:

```bash
python src/webcam.py
```

The application:

1. 📷 Opens the webcam
2. 👤 Detects a person using YOLO
3. 🔍 Extracts the upper region of the detected person
4. 🧠 Passes the region to MobileNetV2
5. 🪖 Classifies helmet usage
6. 📊 Displays the prediction and confidence
7. 🟩 Uses green for helmet detection
8. 🟥 Uses red for no-helmet detection

Press:

```text
Q
```

to exit the webcam application.

---

# ☁️ Model Training

The model was trained using **Google Colab** with GPU acceleration.

The complete training workflow is documented in:

```text
notebooks/helmet_training.ipynb
```

The notebook covers:

```text
Kaggle Dataset Download
        ↓
XML Annotation Processing
        ↓
Image Cropping
        ↓
Class Organisation
        ↓
Train / Validation / Test Split
        ↓
MobileNetV2 Transfer Learning
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Model Export
```

The trained MobileNetV2 model is saved as:

```text
model/helmet_model.keras
```

---

# 🧪 Training Strategy

The project uses **transfer learning** rather than training a convolutional neural network completely from scratch.

MobileNetV2 is used as the pretrained feature extractor, while a custom classification head is trained for the two project classes.

### Why MobileNetV2?

MobileNetV2 provides a useful balance between:

* ⚡ Inference speed
* 🧠 Feature extraction capability
* 💻 Computational efficiency
* 📦 Lightweight deployment

This makes it suitable for real-time computer vision prototypes.

---

# ⚙️ Configuration

Core project settings are stored in:

```text
src/config.py
```

Current configuration includes:

```python
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 1e-4

CLASS_NAMES = [
    "with_helmet",
    "without_helmet"
]
```

The trained models are referenced using:

```python
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "helmet_model.keras"
)

PERSON_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "yolo26n.pt"
)
```

---

# 🧪 Local Testing

The project was tested locally using:

* 💻 Windows
* 🐍 Python virtual environment
* 🧠 TensorFlow/Keras
* 🎥 OpenCV webcam
* 👤 YOLO person detection
* 🪖 MobileNetV2 classification

The trained model was transferred from Google Colab to the local repository and successfully used for webcam inference.

---

# ⚠️ Limitations

This project is primarily a **classification-based helmet detection prototype**.

The MobileNetV2 model itself does not directly perform object detection. YOLO is used to first establish the presence of a person, after which the upper region of the detected person is passed to the classifier.

Some challenging situations may still produce incorrect predictions, including:

* 👥 Multiple people overlapping
* 📐 Extreme camera angles
* 🌑 Poor lighting
* 🫥 Heavy occlusion
* 📏 Very small people in the frame
* 🎥 Motion blur
* 🪖 Helmets held very close to a person's head
* 🧍 Partially visible people

Therefore, the system should be considered a **computer vision prototype** rather than a production-grade road-safety enforcement system.

---

# 🔮 Future Improvements

Potential future improvements include:

* 🎯 Dedicated helmet object detection
* 👥 Multi-person helmet detection
* 🧠 Larger and more diverse training datasets
* 🚫 More hard-negative samples, such as helmets being held or placed on objects
* 📈 Further model fine-tuning
* 📱 Edge-device optimization
* 🌐 Web-based deployment
* 🎥 Video-file inference
* 🚦 Integration with traffic monitoring systems

---

# 📌 Project Highlights

✅ MobileNetV2 transfer learning
✅ Pascal VOC XML annotation processing
✅ Binary helmet classification
✅ YOLO-based person detection
✅ Real-time OpenCV webcam inference
✅ Model evaluation with standard classification metrics
✅ GPU-based training using Google Colab
✅ Local model deployment and testing

---

# 👨‍💻 Author

**Prajwal J Totad**

GitHub:

```text
https://github.com/prajwaltotad
```

---

# 📜 License

This repository contains code and trained model files created for educational and project purposes.

The original dataset and pretrained model components retain their respective licenses and attribution requirements.

Please refer to the source dataset and model documentation before using this project commercially.

