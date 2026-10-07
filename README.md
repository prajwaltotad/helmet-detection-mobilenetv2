# 🪖 Helmet Detection using MobileNetV2 + YOLO + OpenCV

A real-time helmet detection system built using **MobileNetV2 transfer learning**, a pretrained **YOLO person detector**, **TensorFlow/Keras**, **OpenCV**, and **Streamlit**.

The project is designed to classify whether a detected person is:

- 🟢 **Wearing a Helmet**
- 🔴 **Not Wearing a Helmet**

The system uses a two-stage computer vision pipeline:

1. 👤 **YOLO** detects whether a person is present.
2. 🧠 **MobileNetV2** classifies the upper region of the detected person as `with_helmet` or `without_helmet`.

The project supports both:

- 🖥️ Local real-time webcam inference using OpenCV
- 🌐 Browser-based webcam inference using Streamlit and WebRTC

---

## ✨ Features

- 🧠 MobileNetV2 transfer learning
- 👤 YOLO-based person detection
- 🪖 Helmet / no-helmet classification
- 🎥 Real-time local webcam detection
- 🌐 Browser-based webcam application
- 🖼️ Single-image prediction
- 📦 Pascal VOC XML annotation preprocessing
- 🔄 Data augmentation during training
- 📊 Accuracy, precision, recall and F1-score evaluation
- 📈 Confusion matrix evaluation
- ☁️ Training using Google Colab GPU
- 💾 Trained model included in the repository
- 🚀 Streamlit deployment support

---

# 🧩 System Architecture

The project uses a two-stage inference pipeline.

```text
                    📷 Input Frame
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

This approach prevents an isolated object from being directly classified as helmet usage when no person is detected.

---

# 📂 Dataset

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

---

## 🔄 Dataset Preprocessing

The raw Kaggle dataset is **not stored in this repository**.

Instead, the training notebook performs the preprocessing workflow:

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

This converts the original annotated dataset into a format suitable for binary image classification with MobileNetV2.

The complete preprocessing and training workflow is documented in:

```text
notebooks/helmet_training.ipynb
```

---

# 🧠 MobileNetV2 Model

The helmet classifier uses **MobileNetV2 pretrained on ImageNet** as the feature extractor.

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

### Preprocessing

MobileNetV2 preprocessing is included **inside the trained model**.

Therefore, the local prediction and webcam applications do **not** apply `preprocess_input()` a second time.

---

# 👤 Person Detection with YOLO

To reduce false detections caused by isolated helmets, the application uses a pretrained **YOLO person detector** before running the helmet classifier.

The pipeline is:

```text
📷 Webcam Frame
       ↓
👤 YOLO Person Detection
       ↓
Person Found
       ↓
Upper Region of Person
       ↓
🧠 MobileNetV2
       ↓
Helmet Classification
```

The person detection model used in this project is:

```text
yolo26n.pt
```

It is stored in:

```text
model/yolo26n.pt
```

The YOLO model is used to detect the presence of a person before running helmet classification.

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

# 📁 Repository Structure

```text
helmet-detection-mobilenetv2/
│
├── 📄 app.py
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 packages.txt
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

The following are intentionally **not included** in the repository:

* Raw Kaggle images
* XML annotation files
* Generated train/validation/test image folders
* `.venv`
* Python cache files
* IDE-specific configuration files

The `data/` directory is retained using `.gitkeep` so the project structure remains visible.

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

The project uses the following major dependencies:

```text
tensorflow>=2.12
opencv-python
numpy
matplotlib
scikit-learn
Pillow
tqdm
ultralytics
streamlit
streamlit-webrtc
```

---

# 🖼️ Image Prediction

The project includes a script for classifying individual images.

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

# 🎥 Local Real-Time Webcam Detection

The project includes a local OpenCV webcam application.

Run:

```bash
python src/webcam.py
```

The application:

1. 📷 Opens the webcam
2. 👤 Detects people using YOLO
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

### Example Local Pipeline

```text
Webcam
  ↓
YOLO Person Detection
  ↓
Upper Person Region
  ↓
MobileNetV2
  ↓
Helmet / No Helmet
```

---

# 🌐 Web Application

The project also includes a browser-based application using:

* 🌐 Streamlit
* 🎥 streamlit-webrtc
* 👤 YOLO
* 🧠 MobileNetV2
* 🖼️ OpenCV

The web application allows users to use their **own webcam directly from the browser**.

Unlike the local `webcam.py` application, the web application does not use:

```python
cv2.VideoCapture(0)
```

Instead, the browser provides the camera stream through WebRTC.

---

## ▶️ Run the Web Application Locally

From the project root:

```bash
streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

Allow camera access when prompted by the browser.

---

## 🌐 Web Application Pipeline

```text
🌐 Browser
     ↓
📷 User Webcam
     ↓
🎥 WebRTC Stream
     ↓
👤 YOLO Person Detection
     ↓
🔍 Upper Person Region
     ↓
🧠 MobileNetV2
     ↓
🟢 WITH HELMET
or
🔴 WITHOUT HELMET
```

---

## 🚀 Online Deployment

The web application is designed to be deployed using **Streamlit Community Cloud**.

Deployment flow:

```text
GitHub Repository
       ↓
Streamlit Community Cloud
       ↓
app.py
       ↓
Public Web Application
       ↓
Users access through browser
       ↓
Browser Camera Permission
       ↓
Real-Time Prediction
```

After deployment, the application will be available through a public `streamlit.app` URL.

### Deployment Requirements

The repository must contain:

```text
app.py
requirements.txt
model/helmet_model.keras
model/yolo26n.pt
src/
```

The deployed application uses the same trained models included in the repository.

### ⚠️ Camera Permission

The user must allow camera access in the browser before the application can process webcam frames.

---

# ☁️ Model Training

The model was trained using **Google Colab** with GPU acceleration.

A T4 GPU was used for training.

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

The trained classifier is saved as:

```text
model/helmet_model.keras
```

The person detection model is:

```text
model/yolo26n.pt
```

---

# 🧪 Training Strategy

The project uses **transfer learning** instead of training a convolutional neural network entirely from scratch.

MobileNetV2 is used as the pretrained feature extractor, while a custom classification head is trained for the two project classes:

```text
with_helmet
without_helmet
```

## Why MobileNetV2?

MobileNetV2 provides a useful balance between:

* ⚡ Inference speed
* 🧠 Feature extraction capability
* 💻 Computational efficiency
* 📦 Lightweight deployment

This makes it suitable for real-time computer vision applications and prototypes.

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

The model paths are defined using:

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
* 🎥 OpenCV
* 👤 YOLO
* 🪖 MobileNetV2
* 🌐 Streamlit
* 📡 WebRTC

The trained MobileNetV2 model was transferred from Google Colab to the local project and successfully used for image and webcam inference.

---

# 🔍 Example Predictions

### 🪖 With Helmet

```text
Prediction: WITH HELMET
Confidence: 98.56%
```

### 🚫 Without Helmet

```text
Prediction: WITHOUT HELMET
Confidence: 75.21%
```

The confidence values depend on the input image and lighting conditions.

---

# ⚠️ Limitations

This project is primarily a **classification-based helmet detection prototype**.

The MobileNetV2 model itself does not perform object detection. YOLO is used to first detect the presence of a person, after which the upper region of the detected person is passed to the classifier.

Some challenging situations may still produce incorrect predictions, including:

* 👥 Multiple people overlapping
* 📐 Extreme camera angles
* 🌑 Poor lighting
* 🫥 Heavy occlusion
* 📏 Very small people in the frame
* 🎥 Motion blur
* 🪖 Helmets held very close to a person's head
* 🧍 Partially visible people
* 👤 Unusual poses or camera perspectives

The current system should therefore be considered a **computer vision prototype** rather than a production-grade road-safety enforcement system.

---

# 🔮 Future Improvements

Possible future improvements include:

* 🎯 Dedicated helmet object detection
* 👥 Multi-person helmet detection
* 🧠 Larger and more diverse training datasets
* 🚫 Addition of hard-negative samples such as:

  * Helmet being held in a hand
  * Helmet placed on a table
  * Helmet near a person's head but not worn
* 📈 Further model fine-tuning
* 📱 Edge-device optimization
* 🌐 Improved web deployment
* 🎥 Video-file inference
* 🚦 Integration with traffic monitoring systems
* 📊 More extensive benchmarking across different environments

---

# 📌 Project Highlights

✅ MobileNetV2 transfer learning
✅ Pascal VOC XML annotation processing
✅ Binary helmet classification
✅ YOLO-based person detection
✅ Real-time OpenCV webcam inference
✅ Browser-based webcam inference
✅ Streamlit web application
✅ WebRTC camera streaming
✅ Model evaluation using standard classification metrics
✅ GPU-based training using Google Colab
✅ Local model deployment and testing

---

# 📜 Project Files

| File / Folder                     | Purpose                                  |
| --------------------------------- | ---------------------------------------- |
| `app.py`                          | Streamlit browser-based application      |
| `src/config.py`                   | Project paths and training configuration |
| `src/train.py`                    | MobileNetV2 training pipeline            |
| `src/predict.py`                  | Single-image prediction                  |
| `src/webcam.py`                   | Local OpenCV webcam application          |
| `model/helmet_model.keras`        | Trained MobileNetV2 classifier           |
| `model/yolo26n.pt`                | YOLO person detection model              |
| `notebooks/helmet_training.ipynb` | Google Colab training notebook           |
| `requirements.txt`                | Python dependencies                      |

---

# 👨‍💻 Author

**Prajwal J Totad**

GitHub:

[https://github.com/prajwaltotad](https://github.com/prajwaltotad)

Project Repository:

[https://github.com/prajwaltotad/helmet-detection-mobilenetv2](https://github.com/prajwaltotad/helmet-detection-mobilenetv2)

---

# 📜 License & Attribution

This project is created for educational, portfolio, and project demonstration purposes.

The original Kaggle dataset and pretrained model components retain their respective licenses and attribution requirements.

The dataset used in this project is obtained from Kaggle:

[https://www.kaggle.com/datasets/brendan45774/bike-helmets-detection](https://www.kaggle.com/datasets/brendan45774/bike-helmets-detection)

The project also uses a pretrained YOLO model through Ultralytics. Users should review the applicable Ultralytics licensing terms before using the project commercially.

Please verify the licenses of all third-party datasets, models, and libraries before commercial deployment.

````

### One last thing before you push this README

When your Streamlit app is **actually deployed successfully**, add this directly under the title:

```markdown
🌐 **Live Demo:** https://YOUR-APP-NAME.streamlit.app
````

