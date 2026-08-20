
<div align="center">

# TrueImage

### Strategic Deep Learning Framework for the Surgical Detection of AI-Generated Facial Synthesis

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Forensic%20Vision-green.svg)
![ONNX](https://img.shields.io/badge/ONNX-Face%20Detection-orange.svg)
![Data Retention](https://img.shields.io/badge/Data%20Retention-45s-red.svg)

</div>

---

## 📖 Overview

In an era where **Generative Adversarial Networks (GANs)** and latent diffusion models can synthesize hyper-realistic human faces with a single prompt, the boundary between biological reality and digital fabrication is dissolving. **TrueImage** is a forensic deep learning pipeline engineered to restore that boundary.

Developed at the **Copperbelt University (CBU)**, this project moves beyond simple classification. It utilizes a multi-stage verification architecture, incorporating high-speed facial localization, geometric symmetry analysis, and automated content screening, to provide a robust, **CPU-optimized solution** for real-time deepfake diagnostics.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Backend** | Python / Flask | Core logic and RESTful routing |
| **Face Detection** | YuNet (ONNX, via OpenCV) | Ultra-fast CNN for CPU-based facial localization |
| **Deep Learning** | TensorFlow / Keras, EfficientNetV2-S | Custom-trained classifier for synthesis artifact detection |
| **Content Moderation** | NudeNet | Screens uploads before they reach the classifier |
| **Image Processing** | OpenCV, Pillow | Orientation handling, cropping, and geometric transforms |
| **Orchestration** | APScheduler | Automated forensic data purging |

Note: only face *detection* runs on the ONNX runtime (YuNet). The classifier that actually decides real vs. AI-generated runs on TensorFlow/Keras, loading a trained `.keras` model file.

---

## ✨ Key Features

### 🧠 Synthesis Signature Analysis
TrueImage utilizes a custom-trained **EfficientNetV2-S** model to detect "synthesis signatures": micro-level pixel inconsistencies, checkerboard artifacts, and unnatural texture smoothing that characterize AI-generated imagery. Classification runs on a padded crop of the detected face rather than the full uploaded image, keeping the model's fixed input focused on the region where these artifacts actually concentrate.

### 📐 Geometric Validation & Side-View Rejection
Before classification, the system performs a **Symmetry Check**. By calculating the ratio between 5-point facial landmarks, it automatically rejects side-profile shots and pareidolia-driven false positives (face-like patterns in non-face objects) to prevent unreliable results.

### 🔄 Robust Orientation Handling
Uploads are not assumed to be upright. The system tests all four right-angle rotations and keeps whichever produces the highest-confidence, most geometrically plausible detection, rather than trusting the as-uploaded orientation outright.

### 🛡️ Content Moderation with Recovery
Every upload is screened by a NudeNet-based check before analysis. Clearly inappropriate uploads are rejected outright. Borderline (SUGGESTIVE) uploads are given one automatic second chance: the system crops to the detected face and re-screens the crop alone, so a false positive triggered by something elsewhere in frame doesn't need to block an otherwise legitimate portrait.

### ⚡ Millisecond-Level CPU Inference
By leveraging the **YuNet ONNX** runtime and OpenCV's C++ backend for face detection, the system keeps localization overhead low even without GPU hardware.

### 🔒 Automated Forensic Scrubbing (45s Window)
To ensure absolute user privacy, TrueImage implements a **Zero-Retention Policy** with two layers: uploaded files are deleted at the end of each request regardless of outcome, and a background scheduler independently sweeps and deletes any remaining result image older than 45 seconds, so the guarantee holds even if a request is interrupted.

---

## 🧪 The Analysis Pipeline

1. **Secure Ingestion**: User uploads an image via the dashboard; format and size are validated before anything else runs.
2. **Content Screening**: The image is checked for inappropriate content. Clear violations are rejected; borderline cases get a second, face-only check before being rejected.
3. **Face Localization**: YuNet scans for a human face, testing all four rotations to find the correct orientation, and verifies the result is geometrically face-shaped.
4. **Diagnostic Inference**: A padded crop of the detected face is passed through the EfficientNetV2-S classifier to calculate an authenticity confidence score.
5. **Lockdown & Purge**: Results are displayed with a live countdown; the server scrubs all data within 45 seconds.

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
├── scheduler.py             # Zero-retention background purge job
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

### 2. Deploy Dependencies
```bash
pip install -r requirements.txt
```
This includes TensorFlow, which is required to load the classifier. On a fresh environment, confirm it installs correctly for your platform before proceeding (GPU vs CPU builds can differ).

### 3. Launch Suite
```bash
python run.py
```
*Access the dashboard at `http://localhost:5000`*

---

## 🔬 Methodology & Constraints

### Why EfficientNetV2-S?
We selected EfficientNetV2-S for its **Fused-MBConv blocks**, which provide a superior balance between depth and width for CPU-bound environments. This allows TrueImage to perform deep forensic analysis without requiring high-end GPUs.

### Scope & Limitations
* **Portrait Focus**: Optimized strictly for high-resolution human face images.
* **Environmental Sensitivity**: Extremely low lighting or heavy motion blur may impact detection accuracy.
* **Static Analysis**: Currently limited to still images; does not process video synthesis.
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

*"Bringing Truth to Digital Imagery"*

</div>
