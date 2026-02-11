from ultralytics import YOLO

model = YOLO("runs/detect/baseline10/weights/best.pt")

results = model(source=0, show=True, stream=True)

for result in results:
    print(result.boxes.xyxy)
    print(result.boxes.conf)
    print(result.boxes.cls)
