# Water Level Detection with YOLO

A machine learning project exploring two separate techniques for water level detection and segmentation:

1. **Visual Gauging** - YOLOv8-based object detection with staff gauge
2. **Virtual Gauging** - Segmentation-based approach without staff gauge

This project implements both approaches with different datasets, models, and use cases.

## What This Project Does

This repository contains two independent technical approaches for water level measurement:

### Part 1: Visual Gauging (YOLOv8 Object Detection)

Uses **YOLOv8** object detection for visual water level measurement with staff gauge reference:

- Detects water level markings (0.2-2.3 meter equivalents)
- Identifies staff gauge for calibration
- Three model variants: YOLOv8 **Nano** (CPU), **Small** (CPU/GPU), and **Large** (GPU-optimized)
- 12-class object detection with local or cloud-based training
- Requires labeled bounding boxes with staff gauge present in images

**Use case**: Measuring water levels in images where a staff gauge is visible and can be used as a reference.

**Training Options**:

- Local: `YOLO_n+s_local/` - Nano/Small models for CPU or local GPU
- Cloud: `YOLOv8l_colab/` - Large model optimized for Google Colab with free GPU/TPU

### Part 2: Virtual Gauging (Segmentation Models)

Uses **SegFormer**, **SAM (Segment Anything)**, and **Mobile SAM** for pixel-level water segmentation without staff gauge:

- Pixel-level semantic segmentation of water bodies
- No dependency on staff gauge or reference objects
- Multiple segmentation architectures for different accuracy/speed tradeoffs
- Separate dataset with pixel-level masks instead of bounding boxes

**Use case**: Segmenting water regions in images where staff gauge is unavailable or not needed.

### Key Features

- 🎯 **Two Independent Approaches**: Choose between object detection or segmentation
- 🏷️ **Visual Gauging**: 12-class YOLO detection with staff gauge calibration
- 🌊 **Virtual Gauging**: Pixel-level segmentation without reference objects
- 📊 **Separate Datasets**: Each approach has optimized training data
- 🚀 **Easy Inference**: Simple Jupyter notebooks for training and predictions
- 📈 **Performance Tracking**: Built-in validation and results logging

## Prerequisites

- Python 3.8+
- pip or conda
- GPU (optional, but recommended for faster training)

## Installation

Clone the repository:

```bash
git clone https://github.com/talhashahzad12345/waterLevel_YOLO.git
cd waterLevel_YOLO
```

Install dependencies:

```bash
pip install ultralytics opencv-python pyyaml torch torchvision
```

For GPU support:

```bash
pip install ultralytics opencv-python pyyaml torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Part 1: Visual Gauging with YOLOv8

### Overview

Visual gauging uses YOLOv8 object detection to identify water level markings and staff gauge in images. The model detects 12 classes including water level measurements (0.2-2.3) and staff gauge reference objects.

**Location**: `visualGuage`

### Dataset Organization

The Visual Gauging datasets are organized in the `visualGuage` directory with three variants:

```text
visualGuage/
├── rawData/
│   └── (Unlabeled raw images from water level monitoring)
│
├── labeledData/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
├── YOLO_n+s_local/
│   ├── augmentedDataSet/
│   │   ├── data.yaml
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── valid/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── test/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── visualizeLabeledData.py
│   ├── yolov8.ipynb
│   ├── yolov8n.pt
│   ├── yolov8s.pt
│   ├── runs/
│   ├── runs_waterlevel/
│   └── viz_images/
│
└── YOLOv8l_colab/
    ├── YOLOV8l_Colab.ipynb
    └── augmentedDataSet/
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/
        │   ├── images/
        │   └── labels/
        ├── test/
        │   ├── images/
        │   └── labels/
        └── visualizeLabeledData.py
```

**Dataset Descriptions**:

- **rawData**: Original unlabeled images from water level monitoring cameras. Use these for initial exploration or to create custom annotations.

- **labeledData**: Manually labeled water level images with bounding box annotations. Organized in train/valid/test splits. Each image has corresponding `.txt` label files in YOLO format (class_id x_center y_center width height). This is the original dataset without augmentation.

- **augmentedDataSet**: Enhanced version of labeledData with data augmentation techniques (rotation, scaling, brightness adjustments, etc.). Located in both YOLO_n+s_local and YOLOv8l_colab directories with their own copies. Contains more training samples through augmentation. **Recommended for training** as it provides better generalization.

**Recommendation**: Use `augmentedDataSet` for training your models to benefit from increased dataset size and improved generalization through augmentation techniques. Each training setup (local and Colab) has its own copy of augmentedDataSet.

### Quick Start: Local Training (YOLOv8 Nano/Small)

Navigate to the local YOLO directory:

```bash
cd visualGuage/YOLO_n+s_local
```

Open the training notebook:

```bash
jupyter notebook yolov8.ipynb
```

Steps in the notebook:

- Update `DATA_YAML` path to point to your dataset configuration
- Run the training cell to train the model (YOLOv8n or YOLOv8s)
- Use validation/inference cells to test predictions

### Quick Start: Cloud Training (YOLOv8 Large on Colab)

For the best accuracy with free GPU/TPU resources, use the YOLOv8 Large model on Google Colab:

Navigate to the Colab directory:

```bash
cd visualGuage/YOLOv8l_colab
```

Open the Colab notebook:

```bash
jupyter notebook YOLOV8l_Colab.ipynb
```

Or open directly in Google Colab:

```bash
https://colab.research.google.com
```

**In Google Colab**:

- Upload the notebook or open from GitHub
- Click "Runtime" → "Change runtime type"
- Select "GPU" or "TPU" for acceleration
- Run the notebook cells sequentially
- Model uses optimized batch sizes for Colab hardware
- **Training is 2-3x faster than local GPU**

**Why YOLOv8 Large on Colab**:

- Takes advantage of free NVIDIA T4/A100 GPUs or TPU
- Achieves highest accuracy (5-10% better than nano/small)
- Completes in 4-8 hours on Colab GPU
- No local GPU requirements
- Automatic checkpoint saving to Google Drive

### Training Configuration

**Local Training Example** (YOLOv8 Nano/Small):

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or yolov8s.pt for better accuracy

results = model.train(
    data="path/to/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    patience=20,
    project="runs_waterlevel",
    name="yolov8n_waterlevel"
)
```

### Quick Start: Inference

```python
from ultralytics import YOLO

# Load trained model from local training
model = YOLO("visualGuage/YOLO_n+s_local/runs_waterlevel/yolov8n_waterlevel/weights/best.pt")

# Or load model trained on Colab
# model = YOLO("path/to/colab/trained/best.pt")

# Run inference on images
results = model.predict(source="path/to/images", save=True, conf=0.25)

# Access detection results
for result in results:
    print(f"Image: {result.path}")
    for box in result.boxes:
        print(f"  Class: {box.cls}, Confidence: {box.conf}")
```

### Available YOLOv8 Models

#### Local Training (CPU/Local GPU)

- **YOLOv8 Nano** (`yolov8n.pt`):
  - Fast, lightweight inference
  - Best for real-time applications
  - Minimal memory requirements

- **YOLOv8 Small** (`yolov8s.pt`):
  - Balanced accuracy and speed
  - Good for most applications
  - Moderate memory requirements

#### Cloud Training (Google Colab GPU/TPU)

- **YOLOv8 Large** (`YOLOv8l_colab/`):
  - **GPU-optimized** for cloud environments
  - Highest accuracy and performance
  - Designed for Google Colab's free GPU/TPU
  - Better handling of complex water level scenes
  - Optimal batch sizes for Colab hardware

## Part 2: Virtual Gauging with Segmentation

### Introduction

Virtual gauging uses segmentation models to identify water bodies at the pixel level **without requiring a staff gauge reference**. Three models are implemented: SegFormer, SAM, and Mobile SAM.

**Location**: `virtualGuage/`

**Key Difference**: Virtual gauging segments water regions directly, making it usable when staff gauge is unavailable.

### Data Format

Segmentation models require pixel-level masks:

```text
Dataset/
├── train/
│   ├── images/       (RGB images)
│   └── masks/        (Binary or multi-class masks)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Mask format:

- Binary masks: 0 = non-water, 255 = water
- Multi-class masks: Different pixel values for different classes

### Quick Start: SegFormer

Open and run the SegFormer notebook:

```bash
cd virtualGuage
jupyter notebook SegFormer.ipynb
```

SegFormer offers:

- Fast training and inference
- Good accuracy/speed balance
- Lightweight model suitable for deployment

### Quick Start: SAM (Segment Anything Model)

Open and run the SAM notebook:

```bash
cd virtualGuage
jupyter notebook SAM.ipynb
```

SAM advantages:

- Zero-shot segmentation capability
- No task-specific training required
- Foundation model trained on 1B+ masks

### Quick Start: Mobile SAM

Open and run the Mobile SAM notebook:

```bash
cd virtualGuage
jupyter notebook Mobile_SAM.ipynb
```

Mobile SAM benefits:

- Optimized for mobile and edge devices
- 60x smaller than SAM
- Real-time inference on CPU

## Project Structure

```text
waterLevel_YOLO/
├── README.md
├── Talha_Atif_Presentation.pptx     # Project presentation
├── visualGuage/                     # Visual gauging with YOLOv8
│   ├── rawData/                     # Unlabeled raw images
│   ├── labeledData/                 # Manually labeled images
│   │   ├── data.yaml
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── valid/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   └── test/
│   │       ├── images/
│   │       └── labels/
│   │
│   ├── YOLO_n+s_local/              # Local training (CPU/GPU)
│   │   ├── yolov8.ipynb             # Training and inference notebook
│   │   ├── yolov8n.pt               # YOLOv8 Nano pre-trained weights
│   │   ├── yolov8s.pt               # YOLOv8 Small pre-trained weights
│   │   ├── augmentedDataSet/        # Labeled + augmented training data
│   │   │   ├── data.yaml
│   │   │   ├── train/
│   │   │   ├── valid/
│   │   │   ├── test/
│   │   │   ├── visualizeLabeledData.py
│   │   │   └── viz_images/
│   │   ├── runs/                    # Training results
│   │   ├── runs_waterlevel/         # Water level detection results
│   │   └── viz_images/
│   │
│   └── YOLOv8l_colab/               # Cloud training (Google Colab GPU/TPU)
│       ├── YOLOV8l_Colab.ipynb      # Colab notebook for YOLOv8 Large
│       └── augmentedDataSet/        # Labeled + augmented training data
│           ├── data.yaml
│           ├── train/
│           ├── valid/
│           ├── test/
│           ├── visualizeLabeledData.py
│           └── viz_images/
│
└── virtualGuage/                    # Virtual gauging with segmentation
    ├── SegFormer.ipynb              # SegFormer segmentation model
    ├── SAM.ipynb                    # Segment Anything Model
    ├── Mobile_SAM.ipynb             # Mobile-optimized SAM
    └── Dataset/                     # Segmentation dataset with masks
        ├── train/
        │   ├── images/
        │   └── masks/
        ├── valid/
        │   ├── images/
        │   └── masks/
        └── test/
            ├── images/
            └── masks/
```

## Comparison: Visual vs Virtual Gauging

| Feature | Visual Gauging (YOLOv8) | Virtual Gauging (Segmentation) |
|---------|-------------------------|-------------------------------|
| **Technique** | Object Detection (Bounding Boxes) | Semantic Segmentation (Pixel-level) |
| **Requires Staff Gauge** | Yes (calibration reference) | No |
| **Data Format** | Bounding box annotations | Pixel-level masks |
| **Models** | YOLOv8n (CPU), YOLOv8s (CPU/GPU), YOLOv8l (GPU-optimized Colab) | SegFormer, SAM, Mobile SAM |
| **Output** | Class predictions with confidence | Segmentation maps |
| **Use Case** | Structured gauges with reference | Arbitrary water bodies |
| **Training Data** | Augmented/labeled image dataset | Segmentation mask dataset |

## Usage Examples

### Visualizing YOLOv8 Dataset

Use the provided visualization script to view labeled detections:

```bash
cd visualGuage/YOLO_n+s_local
python augmentedDataSet/visualizeLabeledData.py
# Output saved to viz_images/
```

### Batch Prediction with YOLOv8

```python
from ultralytics import YOLO
from pathlib import Path

model = YOLO("visualGuage/YOLO_n+s_local/runs_waterlevel/yolov8n_waterlevel/weights/best.pt")

# Predict on folder
results = model.predict(
    source="path/to/test/images",
    save=True,
    project="results",
    name="predictions",
    conf=0.25,
    imgsz=640
)

print(f"Processed {len(results)} images")
```

### Running Segmentation Inference

Each segmentation model notebook includes inference examples. For example with SegFormer:

```python
# Load model from trained checkpoint
model = load_model("path/to/checkpoint")

# Segment image
mask = model.predict(image)

# Visualize result
visualize_mask(mask)
```

## Getting Help

### Documentation

- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [YOLO GitHub Repository](https://github.com/ultralytics/ultralytics)
- [Roboflow Dataset](https://universe.roboflow.com/watersegmentation-zsbtl/my-first-project-q9p6z/dataset/1)
- [SAM Documentation](https://github.com/facebookresearch/segment-anything)
- [SegFormer Paper](https://arxiv.org/abs/2105.03722)

### Common Issues

**Dataset not found error**: Ensure paths in notebooks point to the correct dataset location on your system.

**CUDA out of memory**: Reduce batch size or image size in training parameters.

**Slow training on CPU or local machine**: Use the YOLOv8 Large Colab notebook (`visualGuage/YOLOv8l_colab/YOLOV8l_Colab.ipynb`) for free GPU/TPU access. This is optimized for high-accuracy training with significantly faster training times.

**Segmentation model inference is slow**: Use Mobile SAM for faster inference on CPU or edge devices.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Maintainers & Authors

- **Talha Shahzad** - [GitHub](https://github.com/talhashahzad12345)
- **Atif Latif** - [GitHub](https://github.com/atiflatif3452)

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics) for YOLOv8
- [Roboflow](https://roboflow.com/) for dataset management and augmentation
- [Meta AI](https://ai.facebook.com/) for SAM and Mobile SAM models
- [NVIDIA](https://www.nvidia.com/) for SegFormer architecture

## Citation

If you use this project in your research, please cite:

```bibtex
@software{waterLevel_YOLO,
  author = {Shahzad, Talha and Latif, Atif},
  title = {Water Level Detection with YOLO},
  url = {https://github.com/talhashahzad12345/waterLevel_YOLO},
  year = {2025}
}
```
