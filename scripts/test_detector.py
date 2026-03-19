import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.piece_detector import PieceDetector

detector = PieceDetector()

test_images = list(Path("data/raw/chess-pieces/valid/images").glob("*.jpg"))
if not test_images:
    print("No validation images found.")
else:
    img = test_images[0]
    print(f"Testing on: {img.name}\n")
    detections = detector.detect(img)
    print(f"Detected {len(detections)} pieces:\n")
    for d in sorted(detections, key=lambda x: x["label"]):
        print(f"  {d['label']:<16} conf={d['conf']:.3f}  box={d['box']}")
