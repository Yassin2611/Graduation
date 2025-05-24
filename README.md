```markdown
# Fabric Defect Detection and local dataset quality control

> Benchmarking state-of-the-art computer-vision models for textile defect detection on the Tianchi dataset and a custom Roboflow dataset, plus a Flask deployment demo.

---

## 📖 Overview

This academic project explores and compares several leading computer vision architectures for detecting defects in fabric imagery. We:

1. Evaluated **YOLOv11**, **YOLOv12**, **RT-DETR**, and **RF-DETR** on the challenging **Guangzhou Tianchi** dataset.
2. Collected and annotated a custom **Roboflow** dataset (558 images, ~1 100 annotations) of local fabric defects, capturing real-world variations observed in our partner production line.
3. Transferred Tianchi-trained weights to our local dataset, demonstrating significant performance gains via ensemble finetuning.
4. Provided a minimal **Flask** application for real-time defect detection demonstration.

---

## 🧩 Models & Architectures

### YOLOv11
- **Backbone**: CSPDarknet with cross-stage partial connections for efficient feature propagation.
- **Neck**: PANet (Path Aggregation Network) for multi-scale feature fusion.
- **Head**: Anchor-based prediction at three scales with improved loss functions for small-object performance.

### YOLOv12
- **Backbone**: Enhanced CSPDarknet with deeper layers and Mish activations for better gradient flow.
- **Neck**: BiFPN (Bi-directional Feature Pyramid Network) providing weighted feature fusion across scales.
- **Head**: Anchor-free detection head with decoupled classification and regression branches.

### RT-DETR
- **Core**: Real-time variant of DETR optimized for speed.
- **Components**:
  - **Backbone**: Lightweight CNN (e.g., MobileNet) for faster inference.
  - **Transformer Modules**: Reduced layers and simplified attention blocks.
  - **Head**: Efficient set-based prediction; trades slight accuracy for higher FPS.

### RF-DETR
- **Core**: Regional Feature DETR, combining region proposal strategies with transformer detection.
- **Components**:
  - **Backbone**: ResNet-50 with FPN for multi-scale features.
  - **Region Proposals**: RPN-like module for candidate regions.
  - **Transformer Decoder**: Processes proposals with self- and cross-attention.
  - **Head**: Predicts class and bounding boxes for refined proposals.

---

## 🗂️ Repository Structure

```

├── data/
│   ├── tianchi/                 # Tianchi dataset splits & annotations
│   └── roboflow\_local/          # Custom dataset images & labels
│
├── notebooks/                   # EDA, benchmarking, and finetuning scripts
│   ├── 01\_benchmark.ipynb
│   └── 02\_finetune\_local.ipynb
│
├── src/                         # Source code
│   ├── train.py                 # Unified training for all models
│   ├── evaluate.py              # mAP calculation & logging
│   └── utils/                   # Data loaders, augmentations, metrics
│
├── flask\_app/                   # Deployment demo
│   ├── app.py                   # Flask server for inference
│   └── templates/               # HTML/CSS assets
│
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container configuration for demo
└── README.md                    # Project documentation

````

---

## ⚙️ Installation

1. **Clone repository**
   ```bash
   git clone https://github.com/your-username/fabric-defect-benchmark.git
   cd fabric-defect-benchmark
````

2. **Set up virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 💡 Usage Examples

### Training

```bash
python src/train.py \
  --dataset data/tianchi \
  --model yolov12 \
  --output runs/tianchi_yolov12
```

### Finetuning

```bash
python src/train.py \
  --dataset data/roboflow_local \
  --pretrained runs/tianchi_yolov12/weights/best.pt \
  --model yolov12 \
  --output runs/local_yolov12
```

### Inference Demo

```bash
cd flask_app
python app.py
# Open http://localhost:5000
```

---


