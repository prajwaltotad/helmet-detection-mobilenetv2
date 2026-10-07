# 🪖 Helmet Detection using MobileNetV2 + YOLO + OpenCV

A computer vision project for detecting whether a person is wearing a helmet using **MobileNetV2 transfer learning**, a pretrained **YOLO person detector**, **TensorFlow/Keras**, and **OpenCV**.

The project supports:

- 🖼️ Single-image helmet classification
- 🎥 Real-time local webcam detection
- 👤 Person detection using YOLO
- 🧠 Helmet classification using MobileNetV2
- 📊 Model evaluation using accuracy, precision, recall and F1-score
- ☁️ Model training using Google Colab GPU

---

## ✨ Features

- 🧠 MobileNetV2 transfer learning
- 👤 YOLO-based person detection
- 🪖 Helmet / no-helmet classification
- 🎥 Real-time webcam inference
- 🖼️ Single-image prediction
- 📦 Pascal VOC XML annotation preprocessing
- 🔄 Data augmentation during training
- 📊 Accuracy, precision, recall and F1-score evaluation
- 📈 Confusion matrix evaluation
- ☁️ Training using Google Colab
- 💾 Trained models included in the repository

---

# 🧩 System Architecture

The project uses a two-stage computer vision pipeline.

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
```

The YOLO stage is used to determine whether a person is present before the helmet classifier is applied.

This helps reduce false detections from isolated objects such as a helmet appearing without a person.

---

# 📂 Dataset

The project uses the **Bike Helmets Detection** dataset from Kaggle.

🔗 **Kaggle Dataset:**
[https://www.kaggle.com/datasets/brendan45774/bike-helmets-detection](https://www.kaggle.com/datasets/brendan45774/bike-helmets-detection)

The original dataset contains:

* 🖼️ Images
* 📝 Pascal VOC XML annotations

The XML files provide:

* Object class labels
* Bounding-box coordinates

The relevant classes are:

```text
With helmet
Without helmet
```

---

## 🔄 Dataset Preprocessing

The raw Kaggle dataset is **not included in this GitHub repository**.

Instead, the training notebook downloads and processes the dataset.

The preprocessing workflow is:

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

The resulting cropped images are used to train the MobileNetV2 classifier.

The complete training workflow is documented in:

```text
notebooks/helmet_training.ipynb
```

---

# 🧠 MobileNetV2 Model

The helmet classifier uses **MobileNetV2 pretrained on ImageNet**.

## Model Architecture

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

## Model Configuration

| Parameter           | Value                |
| ------------------- | -------------------- |
| Base Model          | MobileNetV2          |
| Pretrained Weights  | ImageNet             |
| Input Size          | 224 × 224            |
| Classification Type | Binary               |
| Output Activation   | Sigmoid              |
| Loss Function       | Binary Cross-Entropy |
| Optimizer           | Adam                 |
| Learning Rate       | 1e-4                 |
| Batch Size          | 32                   |

The MobileNetV2 preprocessing operation is included **inside the trained model**, ensuring that the same preprocessing is used during inference.

---

# 👤 YOLO Person Detection

A pretrained YOLO model is used before the MobileNetV2 classifier.

The detector identifies whether a **person** is present in the frame.

The current model is:

```text
yolo26n.pt
```

It is stored in:

```text
model/yolo26n.pt
```

The inference pipeline is:

```text
📷 Webcam
     ↓
👤 YOLO Person Detection
     ↓
Person Bounding Box
     ↓
Upper Person Region
     ↓
🧠 MobileNetV2
     ↓
🪖 Helmet Classification
```

This is intended to reduce cases where an isolated helmet is detected as someone wearing a helmet.

---

# 📊 Model Performance

The MobileNetV2 classifier was evaluated on a held-out test set.

## Test Results

| Metric                   |  Result |
| ------------------------ | ------: |
| Test Accuracy            | ~89.35% |
| With Helmet Precision    |    0.95 |
| With Helmet Recall       |    0.88 |
| With Helmet F1-Score     |    0.92 |
| Without Helmet Precision |    0.80 |
| Without Helmet Recall    |    0.92 |
| Without Helmet F1-Score  |    0.85 |

## Confusion Matrix

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

giving an overall accuracy of approximately:

```text
89.35%
```

---

# 📁 Project Structure

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

The main dependencies are:

```text
tensorflow
opencv-python
numpy
matplotlib
scikit-learn
Pillow
tqdm
ultralytics
```

---

# 🖼️ Single Image Prediction

The project includes a script for classifying a single image.

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

Start the local webcam application:

```bash
python src/webcam.py
```

The application performs the following steps:

1. 📷 Opens the webcam
2. 👤 Detects a person using YOLO
3. 🔍 Extracts the upper region of the person
4. 🧠 Passes the region to MobileNetV2
5. 🪖 Classifies helmet usage
6. 📊 Displays the predicted class and confidence

### Keyboard Control

Press:

```text
Q
```

to close the application.

---

# ☁️ Model Training

The model was trained in **Google Colab** using GPU acceleration.

The training notebook is included in:

```text
notebooks/helmet_training.ipynb
```

The training workflow includes:

```text
Kaggle Dataset Download
        ↓
XML Annotation Processing
        ↓
Bounding-Box Cropping
        ↓
Class Organisation
        ↓
Train / Validation / Test Split
        ↓
MobileNetV2 Transfer Learning
        ↓
Model Training
        ↓
Evaluation
        ↓
Model Export
```

The trained models are:

```text
model/helmet_model.keras
model/yolo26n.pt
```

The MobileNetV2 model was trained in Colab and then transferred to the local project for inference.

---

# 🧪 Local Testing

The project was tested locally using:

* 💻 Windows
* 🐍 Python virtual environment
* 🧠 TensorFlow / Keras
* 🎥 OpenCV
* 👤 YOLO
* 🪖 MobileNetV2

The local webcam pipeline was successfully tested using the trained models.

---

# ⚙️ Configuration

Project settings are stored in:

```text
src/config.py
```

Important configuration values include:

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

Model paths are configured as:

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

# ⚠️ Limitations

This project is a **classification-based helmet detection prototype**.

MobileNetV2 performs image classification and does not independently locate helmets.

The current pipeline uses YOLO to detect a person and then classifies the upper region of the detected person.

Some challenging situations may still produce incorrect predictions, including:

* 👥 Multiple people overlapping
* 📐 Extreme camera angles
* 🌑 Poor lighting
* 🫥 Heavy occlusion
* 📏 Very small people in the frame
* 🎥 Motion blur
* 🪖 Helmets held very close to a person's head
* 🧍 Partially visible people
* Unusual poses or camera perspectives

The project should therefore be considered a **prototype / educational computer vision project**, not a production-grade safety enforcement system.

---

# 🔮 Future Improvements

Possible future improvements include:

* 🎯 Dedicated helmet object detection
* 👥 Multi-person helmet detection
* 🧠 Larger and more diverse datasets
* 🚫 Additional hard-negative training samples
* 🪖 Better distinction between worn and non-worn helmets
* 📈 Further model fine-tuning
* 📱 Edge-device optimization
* 🎥 Video-file inference
* 🚦 Integration with traffic monitoring systems

Examples of useful hard-negative samples include:

```text
Helmet being held in a hand
Helmet placed on a table
Helmet near a person's head but not worn
Helmet appearing without a rider
```

---

# 📌 Project Highlights

✅ MobileNetV2 transfer learning
✅ Pascal VOC XML annotation processing
✅ Binary helmet classification
✅ YOLO-based person detection
✅ Real-time OpenCV webcam inference
✅ Image prediction
✅ Model evaluation using standard classification metrics
✅ GPU-based training using Google Colab
✅ Local deployment and testing

---

# 📜 Files and Their Purpose

| File / Folder                     | Purpose                                    |
| --------------------------------- | ------------------------------------------ |
| `src/config.py`                   | Project paths and configuration            |
| `src/train.py`                    | MobileNetV2 training pipeline              |
| `src/predict.py`                  | Single-image prediction                    |
| `src/webcam.py`                   | Real-time webcam inference                 |
| `model/helmet_model.keras`        | Trained MobileNetV2 classifier             |
| `model/yolo26n.pt`                | YOLO person detector                       |
| `notebooks/helmet_training.ipynb` | Training and preprocessing notebook        |
| `requirements.txt`                | Python dependencies                        |
| `data/`                           | Reserved for local/generated dataset files |

---

# 👨‍💻 Author

**Prajwal J Totad**

GitHub:

[https://github.com/prajwaltotad](https://github.com/prajwaltotad)

Project Repository:

[https://github.com/prajwaltotad/helmet-detection-mobilenetv2](https://github.com/prajwaltotad/helmet-detection-mobilenetv2)

---

# 📜 License & Attribution

This project was created for educational, portfolio, and project demonstration purposes.

The original dataset and pretrained model components retain their respective licenses and attribution requirements.

Dataset:

[https://www.kaggle.com/datasets/brendan45774/bike-helmets-detection](https://www.kaggle.com/datasets/brendan45774/bike-helmets-detection)

The project also uses a pretrained YOLO model through Ultralytics. Users should review the applicable licensing terms for third-party models and libraries before using this project commercially.