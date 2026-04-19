from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="visdrone_tiled.yaml",
    epochs=15,
    imgsz=960,
    batch=8,
    workers=2,
    device=0,
    cache=True,
    close_mosaic=10,
    mosaic=0.7,
    mixup=0.1,
    degrees=10.0,
    translate=0.1,
    scale=0.5
)