from ultralytics import YOLO
from pathlib import Path
import numpy as np

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

    def detect(self, source) -> list[dict]:
        """
        Run inference on an image.

        Args:
            source: file path (str | Path) OR a numpy BGR frame from OpenCV.
                    Passing a numpy array avoids any disk I/O, which was the
                    cause of the race condition in VisionLoop.

        Returns:
            List of dicts: { "label": str, "conf": float, "box": [x1,y1,x2,y2] }
        """
        # BUG FIX: accept numpy arrays directly so VisionLoop never has to
        # write a temp file. Ultralytics YOLO supports np.ndarray as source.
        if isinstance(source, (str, Path)):
            inp = str(source)
        elif isinstance(source, np.ndarray):
            inp = source          # passed straight to Ultralytics
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

        results = self.model(inp, conf=self.conf, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "label": CLASS_NAMES[int(box.cls)],
                    "conf":  round(float(box.conf), 3),
                    "box":   [round(float(v)) for v in box.xyxy[0].tolist()]
                })
        return detections