
<div align="center">

# TrueImage

### Strategic Deep Learning Framework for the Surgical Detection of AI-Generated Facial Synthesis

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Forensic%20Vision-green.svg)
![ONNX](https://img.shields.io/badge/ONNX-Inference-orange.svg)
![Data Retention](https://img.shields.io/badge/Data%20Retention-45s-red.svg)

</div>

---

## 📖 Overview

In an era where **Generative Adversarial Networks (GANs)** and latent diffusion models can synthesize hyper-realistic human faces with a single prompt, the boundary between biological reality and digital fabrication is dissolving. **TrueImage** is a forensic deep learning pipeline engineered to restore that boundary.

Developed at the **Copperbelt University (CBU)**, this project moves beyond simple classification. It utilizes a multi-stage verification architecture—incorporating high-speed facial localization and geometric symmetry analysis—to provide a robust, **CPU-optimized solution** for real-time deepfake diagnostics.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|:---|:---|:---|
| **Backend** | Python / Flask | Core logic and RESTful routing |
| **Face Detection** | YuNet ONNX | Ultra-fast CNN for CPU-based facial localization |
| **Inference Engine** | ONNX Runtime | High-performance execution of trained weights |
| **Deep Learning** | EfficientNetV2-S | Custom-trained classifier for synthesis artifact detection |
| **Image Processing** | OpenCV | CLAHE enhancement and geometric transforms |
| **Orchestration** | APScheduler | Automated forensic data purging |

---

## ✨ Key Features

### 🧠 Synthesis Signature Analysis
TrueImage utilizes a custom-trained **EfficientNetV2-S** model to detect "synthesis signatures"—micro-level pixel inconsistencies, checkerboard artifacts, and unnatural texture smoothing that characterize AI-generated imagery.

### 📐 Geometric Validation & Side-View Rejection
Before classification, the system performs a **Symmetry Check**. By calculating the ratio between 5-point facial landmarks, it automatically rejects side-profile shots to prevent false negatives caused by extreme occlusions.

### ⚡ Millisecond-Level CPU Inference
By leveraging the **YuNet ONNX** runtime and OpenCV’s C++ backend, the system delivers diagnostics in milliseconds, bypassing the massive hardware requirements of traditional GPU-dependent models.

### 🛡️ Automated Forensic Scrubbing (45s Window)
To ensure absolute user privacy, TrueImage implements a **Zero-Retention Policy**. All uploaded assets and analysis artifacts are permanently deleted by a background scheduler exactly 45 seconds after the analysis is complete.

---

## 🧪 The Analysis Pipeline

1. **Secure Ingestion**: User uploads an image via the cyber-fluid dashboard.
2. **Face Localization**: YuNet scans for a human face and applies **4-way auto-rotation** to ensure upright alignment.
3. **Geometric Check**: System validates facial symmetry. If the ratio indicates a side-view, the analysis is safely aborted.
4. **Diagnostic Inference**: The cropped face is passed through the EfficientNetV2-S engine to calculate an authenticity confidence score.
5. **Lockdown & Purge**: Results are displayed with a live countdown; the server scrubs all data 45 seconds later.

---

## 📁 Project Structure

```plaintext
trueimage/
├── app/                  # Main Flask application package
│   ├── static/           # CSS (Tailwind), JS (Purge timers, UI logic)
│   ├── templates/        # Jinja2 templates (index.html, result.html)
│   └── __init__.py       # Flask application factory
├── core/                 # Forensic & AI Logic
│   ├── face_detector.py  # YuNet implementation & Symmetry checks
│   ├── classifier.py     # ONNX-based detection logic
│   └── scheduler.py      # Zero-Retention purge engine
├── training/             # CNN training scripts and research notebooks
├── run.py                # Application entry point
├── requirements.txt      # Production-ready dependencies
└── README.md             # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Clone & Initialize
```bash
git clone [https://github.com/ShekinahBeliaMulenga/trueimage.git](https://github.com/ShekinahBeliaMulenga/trueimage.git)
cd trueimage
python -m venv venv
# Windows: venv\Scripts\activate | Unix: source venv/bin/activate
```

### 2. Deploy Dependencies
```bash
pip install -r requirements.txt
```

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
