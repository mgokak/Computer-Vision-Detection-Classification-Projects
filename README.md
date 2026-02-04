# Computer Vision Detection & Classification Projects

## Overview

This repository contains **end-to-end computer vision projects** that cover both **image classification** and **object detection** using deep learning.
The projects focus on applying CNN-based models to solve **real-world visual recognition and safety problems**.

---

## Projects Included

```
Computer_Vision_Projects/
│
├── Animals_ImageClassification.ipynb
├── PPE_Object_Detection.ipynb
└── README.md
```

---

## 1) Animal Image Classification  
**Notebook:** `Animals_ImageClassification.ipynb`

This project builds a **Convolutional Neural Network (CNN)** to classify animal images into different categories.

### Problem Type
- **Multi-class image classification**
- One label per image

### Typical Workflow
- Load images from directory structure
- Resize and normalize images
- Train CNN using convolution and pooling layers
- Evaluate classification accuracy

### Core Concepts Used
- Convolutional layers (Conv2D)
- Max pooling
- Fully connected layers
- Softmax-based class prediction

### Example Code Pattern
```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(224, 224),
    batch_size=32
)
```

```python
model = tf.keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
```

---

## 2) PPE Object Detection  
**Notebook:** `PPE_Object_Detection.ipynb`

This project focuses on **object detection for workplace safety**, identifying Personal Protective Equipment (PPE) such as helmets and vests in images.

### Problem Type
- **Object detection**
- Multiple objects per image
- Bounding box + class prediction

### Typical Workflow
- Load a pre-trained detection model
- Perform inference on images or video frames
- Detect and localize PPE items
- Visualize bounding boxes and labels

### Core Concepts Used
- Object detection vs classification
- Bounding boxes
- Confidence scores
- Pre-trained detection models

### Example Code Pattern
```python
from ultralytics import YOLO
model = YOLO("best.pt")
results = model.predict(source=image_path)
```

---

## Classification vs Object Detection (Key Difference)

| Aspect | Image Classification | Object Detection |
|------|----------------------|------------------|
| Output | Single label | Multiple objects |
| Location info | ❌ No | ✅ Yes |
| Complexity | Lower | Higher |
| Use cases | Image tagging | Surveillance, safety |

---

## Real-World Applications

- Wildlife monitoring and animal recognition
- Safety compliance in construction sites
- Automated inspection systems
- Smart surveillance and monitoring

---


## Requirements

```bash
pip install tensorflow keras opencv-python ultralytics numpy matplotlib
```

---


## Author

**Manasa Vijayendra Gokak**  
Graduate Student – Data Science  

