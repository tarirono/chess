from ultralytics import YOLO
from ultralytics.utils import ASSETS
from pathlib import Path

Path("data/raw").mkdir(parents=True, exist_ok=True)

print("Loading YOLOv8-Nano weights...")
model = YOLO("yolov8n.pt")

# Use ultralytics' own bundled test image (already on your machine)
test_img = str(ASSETS / "bus.jpg")
print(f"Test image: {test_img}")

print("Running inference...")
results = model(
    source=test_img,
    conf=0.25,
    save=True,
    project="data/raw",
    name="test_run"
)

r = results[0]
print(f"\nInference complete.")
print(f"Detected {len(r.boxes)} objects")
print(f"Result saved to: data/raw/test_run/")

names = model.names
print("\nDetected objects:")
for box in r.boxes:
    cid  = int(box.cls)
    conf = float(box.conf)
    print(f"  {names[cid]:<15} confidence: {conf:.2f}")

print("\nYOLO pipeline works end to end.")