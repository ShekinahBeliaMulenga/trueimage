<div align="center">

# TrueImage

### A Deep Learning System for Detecting AI-Generated Face Images

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Forensic%20Vision-green.svg)
![ONNX](https://img.shields.io/badge/ONNX-Face%20Detection-orange.svg)
![Data Retention](https://img.shields.io/badge/Data%20Retention-45s-red.svg)

</div>

---

## 📖 Overview

AI models can now generate images of human faces that look convincingly real, making it hard to tell a photograph from a synthetic image. **TrueImage** is a deep learning system built to make that distinction.

Developed at the **Copperbelt University (CBU)**, the system combines face detection, geometric checks, and content screening around a trained classifier, and is designed to run on ordinary CPU hardware without needing a GPU.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Backend** | Python / Flask | Core logic and routing |
| **Face Detection** | YuNet (ONNX, via OpenCV) | Fast, CPU-based face localization |
| **Deep Learning** | TensorFlow / Keras, EfficientNetV2-S | Trained classifier for detecting synthetic images |
| **Content Moderation** | NudeNet | Screens uploads before they reach the classifier |
| **Image Processing** | OpenCV, Pillow | Orientation handling, cropping, and geometric transforms |
| **Orchestration** | APScheduler | Automatic deletion of processed files |

Note: only face *detection* runs on the ONNX runtime (YuNet). The classifier that decides real vs. AI-generated runs on TensorFlow/Keras, loading a trained `.keras` model file.

---

## ✨ Key Features

### 🧠 Detecting Synthetic Faces
TrueImage uses a custom-trained **EfficientNetV2-S** model to detect the small inconsistencies that AI-generated images tend to have: unnatural pixel patterns, checkerboard artifacts, and unusually smooth textures. Classification runs on a padded crop of just the detected face rather than the full uploaded image, so the model focuses on the part of the image where these patterns actually appear.

### 📐 Face Geometry Check
Before classification, the system checks whether a detected face is geometrically plausible, comparing the position of the eyes, nose, and mouth. This rejects side-on photos and false detections where a non-face object (such as a wall socket) is mistaken for a face.

### 🔄 Orientation Handling
The system does not assume an uploaded image is right-side up. It tests all four rotations and keeps whichever produces the most confident, most geometrically plausible face detection.

### 🛡️ Content Moderation with a Second Chance
Every upload is checked for inappropriate content before analysis. Clearly inappropriate images are rejected outright. Borderline images get one more chance: the system crops to the detected face and re-checks the crop alone, so an otherwise normal portrait is not rejected because of something unrelated elsewhere in the frame.

### ⚡ Fast, CPU-Only Face Detection
Face detection runs on the YuNet model through OpenCV, which is fast enough on ordinary CPU hardware that no GPU is needed for this step.

### 🔒 Automatic Deletion (45 Second Window)
Uploaded images are not stored permanently. Each file is deleted at the end of its own request, and a background job independently checks every 15 seconds and deletes anything left over that is older than 45 seconds. This means files are still removed even if a request is interrupted partway through.

---

## 🧪 The Analysis Pipeline

1. **Upload**: The user uploads an image. File format and size are checked first.
2. **Content Check**: The image is screened for inappropriate content. Clear violations are rejected. Borderline images get a second check on just the detected face before being rejected.
3. **Face Detection**: YuNet looks for a human face, testing all four rotations, and checks that the result is geometrically a real face rather than a false match.
4. **Classification**: A padded crop of the detected face is passed to the EfficientNetV2-S classifier, which produces a confidence score.
5. **Result and Deletion**: The result is shown to the user, and all related files are deleted within 45 seconds.

---

## 📁 Project Structure

```plaintext
trueimage/
├── app/
│   ├── controllers/
│   │   ├── upload_controller.py   # /predict route: the full request pipeline
│   │   └── result_controller.py   # Result display route
│   ├── services/
│   │   ├── explicit_detector.py       # NudeNet-based content moderation
│   │   ├── face_detector.py           # YuNet, orientation handling, geometry checks
│   │   ├── face_detection_result.py   # FaceBox / FaceDetectionResult dataclasses
│   │   ├── face_visualizer.py         # Draws detection overlays for display
│   │   ├── ai_inference_engine.py     # EfficientNetV2-S classifier (TensorFlow/Keras)
│   │   ├── result_interpreter.py      # Converts a probability into a labelled result
│   │   └── image_processor.py         # Image loading/preprocessing helpers
│   ├── utils/             # Logging and input validation helpers
│   ├── static/             # CSS (Tailwind), JS (purge timers, UI logic)
│   ├── templates/          # Jinja2 templates (index.html, result.html)
│   ├── config.py
│   └── __init__.py         # Flask application factory
├── training/               # Dataset preparation, training, and evaluation scripts
├── scheduler.py             # Background job that deletes old files
├── run.py                   # Application entry point
├── requirements.txt         # Production dependencies
└── README.md                 # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Clone & Initialize
```bash
git clone https://github.com/ShekinahBeliaMulenga/trueimage.git
cd trueimage
python -m venv venv
# Windows: venv\Scripts\activate | Unix: source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
This includes TensorFlow, which is required to load the classifier. On a fresh environment, confirm it installs correctly for your platform before proceeding, since GPU and CPU builds can differ.

### 3. Run the Application
```bash
python run.py
```
*Open `http://localhost:5000` in a browser.*

---

## 🔬 Methodology & Constraints

### Why EfficientNetV2-S?
EfficientNetV2-S was chosen for its Fused-MBConv blocks, which give a good balance of accuracy and speed on CPU hardware. This lets TrueImage run without needing a GPU.

### Scope & Limitations
* **Portrait Focus**: Built for clear, high-resolution human face images.
* **Environmental Sensitivity**: Very low lighting or heavy motion blur can reduce detection accuracy.
* **Still Images Only**: The system analyses still images and does not process video.
* **Generator Coverage**: Detection accuracy reflects the generators represented in the training data and may be lower for image generators released after training.

---

## 🤝 Author

**Shekinah B. Mulenga**
Computer Science Student
*Copperbelt University, School of ICT*

---

## 📜 License

This project is licensed under the MIT License.

<div align="center">

*Bringing Truth to Digital Imagery*

</div>
