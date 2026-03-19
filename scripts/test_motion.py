import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from src.vision.motion_detector import MotionDetector

detector = MotionDetector()

# Simulate: 10 identical frames (stable), then 5 different (motion), then 10 stable
base_frame = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(base_frame, (100, 100), (300, 300), (200, 200, 200), -1)

moved_frame = base_frame.copy()
cv2.rectangle(moved_frame, (200, 200), (400, 400), (200, 200, 200), -1)

print("Simulating stable → motion → stable sequence:\n")

frames = (
    [("stable", base_frame)] * 10 +
    [("motion", moved_frame)] * 5 +
    [("stable", base_frame)] * 12
)

for i, (label, frame) in enumerate(frames):
    status = detector.update(frame)
    trigger_mark = " <<< TRIGGER — run YOLO now!" if status["trigger"] else ""
    print(f"  Frame {i+1:02d} [{label}]  "
          f"motion={str(status['motion']):<5}  "
          f"diff={status['diff_ratio']:.5f}"
          f"{trigger_mark}")

