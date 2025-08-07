# Anomalies Detection in Crops using Deep Learning

This project focuses on detecting anomalies in Sugarcane crop using drone-captured imagery and deep learning techniques. The implementation compares and utilizes different object detection models including CNN, R-CNN, and YOLOv8 to identify disease-affected or anomalous regions in crop fields.

## 📁 Repository Structure

.
├── CNN # Traditional Convolutional Neural Network implementation
├── RCNN # Region-based CNN for object detection
├── YOLO-v8 # YOLOv8 implementation using Ultralytics
├── README.md # Project documentation
├── Readme.txt # (Legacy/backup) readme file

## 🚁 Project Objective

Use drone imagery combined with deep learning models to:
- Detect unhealthy or anomalous sections of crops
- Compare performance between multiple object detection architectures
- Build a pipeline that could be integrated with drone-based monitoring systems

## 🧠 Models Used

### 1. CNN
- A basic convolutional neural network for image classification.
- Useful as a baseline model.

### 2. R-CNN
- Region-based Convolutional Neural Networks for object detection.
- Implements selective search and classification of proposed regions.

### 3. YOLOv8
- Ultralytics YOLOv8 object detection model.
- Fast and accurate real-time detection.
- Suitable for drone-based real-time inference.

## 📷 Dataset

> **Note:** The dataset is assumed to be custom drone-captured images of crop fields with anomalies (disease, dryness, pest attack, etc.).  
> If you are using a public dataset, please update this section with a citation or source.

## 🛠️ Requirements

Each folder may have its own dependencies. Broadly, the following packages are needed:

- Python 3.8+
- PyTorch
- TensorFlow (if used in CNN)
- OpenCV
- Ultralytics (for YOLOv8)
- scikit-learn
- matplotlib
- ultralytics


📈 Results & Evaluation

Evaluation metrics include mAP, precision, recall, and inference speed.

Comparisons between models are documented within each subfolder.

🧑‍💻 Author
ETS
UAV and Deep Learning Enthusiast

📄 License
This project is for academic and research purposes. You may modify or use the code with proper citation.
