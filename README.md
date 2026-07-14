# ♻️ Waste Detection & Segregation using YOLOv8

An AI-based waste segregation system that uses **YOLOv8** for real-time waste detection and a **robotic arm** for automated sorting. The system identifies waste through a webcam and classifies it into different categories, enabling efficient and accurate waste management.

## 🚀 Features

- Real-time waste detection
- YOLOv8 object detection model
- Webcam-based inference
- Robotic arm integration for waste sorting
- ONNX model support for deployment

## 🛠️ Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch
- ONNX

## 📁 Project Structure

```
waste_detection/
├── garbage_yolo/
├── best.pt
├── best.onnx
├── last.pt
├── webcam_detect.py
└── README.md
```

## ▶️ Installation

```bash
git clone https://github.com/Malavikavinod7/waste_detection.git
cd waste_detection

pip install ultralytics opencv-python torch torchvision numpy
```

## ▶️ Run

```bash
python webcam_detect.py
```

## 🔄 Workflow

```
Webcam → YOLOv8 Detection → Waste Classification → Robotic Arm → Correct Bin
```

## 📌 Future Enhancements

- More waste categories
- Raspberry Pi/Jetson deployment
- Conveyor belt integration
- Cloud monitoring dashboard

## 👩‍💻 Author

**Malavika Vinod**

GitHub: https://github.com/Malavikavinod7
