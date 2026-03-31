from ultralytics import YOLO
from pathlib import Path

# Paths
DATASET_YAML = Path("data/raw/chess-pieces/dataset.yaml")
MODEL_SAVE_DIR = Path("data/models")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("Starting YOLOv8-Nano training on chess pieces dataset...")
print(f"Dataset config: {DATASET_YAML}")

# Load YOLOv8-Nano — smallest and fastest, good for CPU
model = YOLO("yolov8n.pt")

# Train
results = model.train(
    data=str(DATASET_YAML),
    epochs=50,
    imgsz=640,
    batch=8,          # safe for 8GB RAM on CPU
    workers=2,
    device="cpu",
    project="data/models",
    name="chess_nano_v1",
    exist_ok=True,
    patience=15,      # stop early if no improvement for 15 epochs
    save=True,
    plots=True,
    verbose=True,
)

print("\nTraining complete.")
print(f"Best weights saved to: data/models/chess_nano_v1/weights/best.pt")