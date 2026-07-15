# ♻️ Waste Detection & Segregation using YOLOv8

An AI-based waste segregation system that uses YOLOv8 for real-time waste detection and can be extended to robotic sorting workflows. The project provides a simple webcam-based inference entrypoint and a Python package layout suitable for GitHub collaboration.

## 🚀 Features

- Real-time waste detection with YOLOv8
- Webcam-based inference
- Reusable Python package structure
- ONNX model support for deployment workflows
- GitHub-ready project metadata and CI setup

## 🛠️ Tech Stack

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- PyTorch
- ONNX

## 📁 Project Structure

```text
waste_detection/
├── garbage_yolo/
│   ├── inference.py
│   └── webcam_detect.py
├── .github/workflows/ci.yml
├── pyproject.toml
├── requirements.txt
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── best.pt
├── best.onnx
└── webcam_detect.py
```

## ▶️ Installation

```bash
git clone https://github.com/Malavikavinod7/waste_detection.git
cd waste_detection
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## ▶️ Run

```bash
python webcam_detect.py
```

You can also pass optional arguments:

```bash
python webcam_detect.py --model best.pt --source 0 --conf 0.25
```

## 🔄 Workflow

```text
Webcam → YOLOv8 Detection → Waste Classification → Sorting Workflow
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and pull request guidelines.

## 📌 Future Enhancements

- More waste categories
- Raspberry Pi or Jetson deployment
- Conveyor belt integration
- Cloud monitoring dashboard

## 👩‍💻 Author

Malavika Vinod

GitHub: https://github.com/Malavikavinod7
