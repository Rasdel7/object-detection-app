# Object Detection App 🎯

Detects 80 types of objects in any image
using YOLOv8 — the world's fastest object detector.

## Live Demo
[Click here](https://object-detection-app-6dlbswb22mj8qfve68z7ji.streamlit.app)

## Features
- Detects 80 COCO object categories
- Adjustable confidence threshold
- Bounding boxes with labels
- Detection statistics and charts
- Confidence score per object
- Object count summary

## What It Detects
Person, Car, Bus, Bicycle, Dog, Cat,
Phone, Laptop, Chair, Food, Airplane,
Boat and 68 more categories.

## Model
YOLOv8 Nano — fastest variant, runs on CPU
Trained on COCO dataset (80 classes)

## Tools Used
- Python, Ultralytics YOLOv8, Streamlit
- Pillow, NumPy, Pandas, Matplotlib

## How to Run Locally
pip install streamlit ultralytics Pillow numpy pandas matplotlib
streamlit run app.py
