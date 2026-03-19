from ultralytics import YOLO
from pathlib import Path

WEIGHTS_PATH = Path("data/models/chess_nano_v1/best.pt")

CLASS_NAMES = [
    "bishop", "black-bishop", "black-king", "black-knight",
    "black-pawn", "black-queen", "black-rook", "white-bishop",
    "white-king", "white-knight", "white-pawn", "white-queen", "white-rook"
]

class PieceDetector:
    def __init__(self, weights: Path = WEIGHTS_PATH, conf: float = 0.4):
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")
        self.model = YOLO(str(weights))
        self.conf = conf
        print(f"PieceDetector loaded — weights: {weights}")

    def detect(self, image_path) -> list[dict]:
        """
        Run inference on an image.
        Returns list of dicts:
          { "label": str, "conf": float, "box": [x1,y1,x2,y2] }
        """
        results = self.model(str(image_path), conf=self.conf, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "label": CLASS_NAMES[int(box.cls)],
                    "conf":  round(float(box.conf), 3),
                    "box":   [round(float(v)) for v in box.xyxy[0].tolist()]
                })
        return detections