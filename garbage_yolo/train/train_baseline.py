from ultralytics import YOLO

model = YOLO("runs/detect/baseline10/weights/best.pt")

model.train(
    data="garbage_yolo/data/garbage.yaml",
    imgsz=640,
    epochs=30,
    lr0=1e-4,
    mosaic=0.0,
    mixup=0.0
)

