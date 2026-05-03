from ultralytics import YOLO
from pathlib import Path

model = YOLO("runs/detect/data/models/chess_nano_v1/weights/best.pt")
metrics = model.val(data="data/raw/chess-pieces/dataset.yaml")

print(f"\nmAP50:     {metrics.box.map50:.3f}")
print(f"mAP50-95:  {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.p.mean():.3f}")
print(f"Recall:    {metrics.box.r.mean():.3f}")