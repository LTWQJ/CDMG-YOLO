# CDMG-YOLO: A Lightweight and Structurally Optimized YOLO-Based Model for Small Object Detection in Open-Pit Mining UAV Imagery
This repository provides the implementation of CDMG-YOLO, a structurally optimized, lightweight object detection model tailored for heterogeneous mining machinery detection in UAV-based remote sensing imagery. This work is proposed in our paper:

> **Authors**: [ltw]

## 🔍 Overview

CDMG-YOLO is based on the YOLOv11n architecture and introduces several novel modules to improve performance in small-object detection tasks under complex open-pit mining environments.

### 🚧 Key Challenges Addressed

* Small object scale variation
* Background clutter and occlusion
* Low detection recall for mining machinery in aerial imagery

### ✅ Key Contributions

* **P2 Detection Head**: Enhances shallow feature utilization for small targets
* **D-C3k2 Module**: Uses large-kernel deformable convolution to boost spatial modeling
* **MCAM + CAFM Modules**: Improve global and local attention mechanisms
* **GHM Loss Integration**: Strengthens learning on hard and sparse samples
* **Custom UAV Dataset**: 1905 annotated orthophoto images from 39 open-pit mines in Xiangyang, China (2022–2024)

## 📊 Performance

| Model         | Precision | Recall    | mAP\@0.5  | Params (M) | GFLOPs   |
| ------------- | --------- | --------- | --------- | ---------- | -------- |
| YOLOv5n       | 0.475     | 0.561     | 0.311     | 1.8        | 4.5      |
| YOLOv8n       | 0.687     | 0.506     | 0.612     | 3.1        | 8.3      |
| YOLOv11n      | 0.748     | 0.574     | 0.488     | 2.6        | 6.4      |
|BGF-YOLOv10	  | 0.685     |0.547      |0.351	    |2.0	       |8.5	    |
|FFCA-YOLO	    |0.769	      |0.689	      |0.559    |4.5	     |17.4	    |
|ATBHC-YOLO	    |0.835     |0.618    |0.587    |36.8	     |167.2    |
| **CDMG-YOLO** | **0.859** | **0.764** | **0.697** | **3.1**    | **12.7** |

## 📁 Repository Structure
```
CDMG-YOLO/
├── CDMG-ultralytics/             # ultralytics code containing improved code (D-C3k2, CAFM, MCAM, etc.)
├── Improvement_module/           # Improve the module code (D-C3k2, CAFM, MCAM, etc.)
├── Run_model/            # Training and evaluation 
├── utils/              # Helper functions and metrics
└── README.md
🚀 Getting Started
Requirements
Python 3.9+

PyTorch >= 1.10

CUDA 11.x

OpenCV, torchvision, tqdm
```

### Training
```
After setting up the basic environment, use the provided ymal file to run the ltw_train.py file
```


