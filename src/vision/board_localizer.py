import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import json
import numpy as np


CONFIG_PATH = Path("data/models/board_config.json")


class BoardLocalizer:
    """
    Finds the chessboard region within a camera frame.

    Two modes:
      1. Manual calibration - click 4 corners once, saved to JSON config.
      2. Auto detection    - OpenCV contour/line detection fallback.
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config_path = config_path
        self.corners = None          # (x1, y1, x2, y2) board bounding box
        self.points = None           # [(x, y), ...] in TL, TR, BR, BL order
        self._load_config()

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path, encoding="utf-8") as f:
                cfg = json.load(f)

            raw_points = cfg.get("points")
            raw_corners = cfg.get("corners")

            if raw_points:
                self.points = [tuple(pt) for pt in raw_points]
                self.corners = self._points_to_bbox(self.points)
            elif raw_corners:
                self.corners = tuple(raw_corners)

            print(f"BoardLocalizer: loaded corners from {self.config_path}")
        else:
            print("BoardLocalizer: no config found - run calibrate() first.")

    def _save_config(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"corners": list(self.corners) if self.corners else None}
        if self.points:
            payload["points"] = [list(pt) for pt in self.points]

        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"BoardLocalizer: corners saved to {self.config_path}")

    def _points_to_bbox(self, points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        return (min(xs), min(ys), max(xs), max(ys))

    # ------------------------------------------------------------------
    # Mode 1 - Manual calibration (click 4 corners)
    # ------------------------------------------------------------------

    def calibrate(self, image_path: str | Path):
        """
        Opens a window showing the image.
        Click TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT.
        Press 'r' to reset clicks, 'q' to quit without saving.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")

        display_scale = 1.0
        h, w = img.shape[:2]
        max_dim = 1000
        if max(h, w) > max_dim:
            display_scale = max_dim / max(h, w)
            img_display = cv2.resize(img, None, fx=display_scale, fy=display_scale)
        else:
            img_display = img.copy()

        clicks = []
        clone = img_display.copy()
        labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]

        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
                orig_x = int(x / display_scale)
                orig_y = int(y / display_scale)
                clicks.append((orig_x, orig_y))

                cv2.circle(img_display, (x, y), 6, (0, 255, 0), -1)
                label = labels[len(clicks) - 1]
                cv2.putText(
                    img_display,
                    f"{label} ({orig_x},{orig_y})",
                    (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                cv2.imshow("Board Calibration", img_display)

        cv2.namedWindow("Board Calibration", cv2.WINDOW_NORMAL)
        cv2.imshow("Board Calibration", img_display)
        cv2.setMouseCallback("Board Calibration", on_click)

        print("\nCalibration instructions:")
        print("  1. Click the TOP-LEFT corner of the board")
        print("  2. Click the TOP-RIGHT corner of the board")
        print("  3. Click the BOTTOM-RIGHT corner of the board")
        print("  4. Click the BOTTOM-LEFT corner of the board")
        print("  Press 'r' to reset  |  's' to save  |  'q' to quit\n")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord("r"):
                clicks.clear()
                img_display[:] = clone[:]
                cv2.imshow("Board Calibration", img_display)
                print("Reset - click again.")
            elif key == ord("s") and len(clicks) == 4:
                self.points = list(clicks)
                self.corners = self._points_to_bbox(self.points)
                self._save_config()
                print(f"Saved points: {self.points}")
                break
            elif key == ord("q"):
                print("Quit without saving.")
                break

        cv2.destroyAllWindows()
        return self.points

    # ------------------------------------------------------------------
    # Mode 2 - Auto detection via OpenCV contours
    # ------------------------------------------------------------------

    def auto_detect(self, image_path: str | Path) -> tuple | None:
        """
        Attempts to auto-detect the chessboard bounding box using
        contour detection.
        Returns (x1, y1, x2, y2) or None if detection fails.
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        best = None
        best_score = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (h * w * 0.1):
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            squareness = min(cw, ch) / max(cw, ch)
            score = area * squareness
            if score > best_score:
                best_score = score
                best = (x, y, x + cw, y + ch)

        if best:
            self.corners = best
            self.points = None
            print(f"BoardLocalizer auto-detected corners: {self.corners}")
            return self.corners

        print("BoardLocalizer: auto-detection failed - using full image.")
        return (0, 0, w, h)

    # ------------------------------------------------------------------
    # Main interface
    # ------------------------------------------------------------------

    def get_corners(self, image_path: str | Path = None) -> tuple:
        """
        Returns (x1, y1, x2, y2) board corners.
        Priority: saved config -> auto-detect -> full image fallback.
        """
        if self.corners:
            return self.corners
        if image_path:
            result = self.auto_detect(image_path)
            if result:
                return result
        return None

    def get_points(self) -> list[tuple[int, int]] | None:
        return self.points

    def perspective_matrix(self, board_size: int = 800) -> np.ndarray | None:
        if not self.points or len(self.points) != 4:
            return None

        src = np.array(self.points, dtype=np.float32)
        dst = np.array(
            [
                [0, 0],
                [board_size - 1, 0],
                [board_size - 1, board_size - 1],
                [0, board_size - 1],
            ],
            dtype=np.float32,
        )
        return cv2.getPerspectiveTransform(src, dst)

    def warp_image(self, image_path: str | Path, board_size: int = 800) -> np.ndarray | None:
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        matrix = self.perspective_matrix(board_size=board_size)
        if matrix is None:
            return None

        return cv2.warpPerspective(img, matrix, (board_size, board_size))

    def transform_point(self, x: float, y: float, board_size: int = 800) -> tuple[float, float] | None:
        matrix = self.perspective_matrix(board_size=board_size)
        if matrix is None:
            return None

        src = np.array([[[x, y]]], dtype=np.float32)
        dst = cv2.perspectiveTransform(src, matrix)[0][0]
        return float(dst[0]), float(dst[1])
