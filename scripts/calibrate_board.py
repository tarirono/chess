"""
calibrate_board.py  — Live camera version (fixed click detection)
==================================================================
Usage:
    python scripts/calibrate_board.py
    python scripts/calibrate_board.py --camera 1
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import os
import numpy as np
from dotenv import load_dotenv
from src.vision.board_localizer import BoardLocalizer
from src.vision.fen_extractor import BOARD_SIZE, PRIMARY_ORIENTATION, image_to_fen

load_dotenv()

DEBUG_WARP_PATH = Path("data/raw/_debug_warped_board.jpg")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--camera", type=int, default=None)
args = parser.parse_args()
camera_index = args.camera if args.camera is not None else int(os.getenv("CAMERA_INDEX", "0"))

# ── State ─────────────────────────────────────────────────────────────────────
clicks = []
frozen_frame = None
labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
COLORS = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (0, 0, 255)]
WINDOW = "Calibration"

def draw_state(img):
    """Redraw all clicks and lines onto a fresh copy of the image."""
    out = img.copy()
    for i, (px, py) in enumerate(clicks):
        cv2.circle(out, (px, py), 10, COLORS[i], -1)
        cv2.circle(out, (px, py), 12, (255, 255, 255), 2)
        cv2.putText(out, labels[i], (px + 14, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLORS[i], 2)
        if i > 0:
            cv2.line(out, clicks[i - 1], clicks[i], (0, 200, 255), 2)
    if len(clicks) == 4:
        cv2.line(out, clicks[3], clicks[0], (0, 200, 255), 2)
        cv2.putText(out, "4/4 corners — press S to save, R to redo",
                    (10, out.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
    else:
        cv2.putText(out,
                    f"Click corner {len(clicks)+1}/4: {labels[len(clicks)]}  |  R=reset  Q=quit",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out

def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
        clicks.append((x, y))
        print(f"  [{len(clicks)}/4] {labels[len(clicks)-1]}: ({x}, {y})")
        cv2.imshow(WINDOW, draw_state(frozen_frame))

# ── Step 1: open camera and freeze a frame ───────────────────────────────────
print(f"\nOpening camera {camera_index}...")
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"ERROR: Cannot open camera {camera_index}. Check DroidCam + CAMERA_INDEX in .env")
    sys.exit(1)

print("Camera open. Aim at the board, then press SPACE to freeze. Q = quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Cannot read from camera.")
        break

    preview = frame.copy()
    cv2.putText(preview, "Aim camera at board — press SPACE to freeze | Q to quit",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow(WINDOW, preview)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        frozen_frame = frame.copy()
        print("Frame frozen.")
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

cap.release()

# ── Step 2: register callback AFTER window already exists ────────────────────
print("\nSTEP 2: Click the 4 corners of the playable 8x8 area:")
print("        Top-Left → Top-Right → Bottom-Right → Bottom-Left")
print("        S = save  |  R = reset  |  Q = quit\n")

# Show the frozen frame first, THEN attach the callback
cv2.imshow(WINDOW, draw_state(frozen_frame))
cv2.waitKey(1)                          # let the window actually render
cv2.setMouseCallback(WINDOW, mouse_cb)  # attach AFTER window is visible

while True:
    key = cv2.waitKey(20) & 0xFF

    if key == ord('r'):
        clicks.clear()
        print("Reset — click the 4 corners again.")
        cv2.imshow(WINDOW, draw_state(frozen_frame))

    elif key == ord('s'):
        if len(clicks) < 4:
            print(f"  Need 4 corners, only have {len(clicks)}. Keep clicking.")
            continue

        # Save calibration
        localizer = BoardLocalizer()
        localizer.points = list(clicks)
        localizer.corners = localizer._points_to_bbox(localizer.points)
        localizer._save_config()
        print(f"\nSaved to data/models/board_config.json")
        print(f"Points: {clicks}")

        # Save the frozen frame and verify with FEN extraction
        tmp_path = Path("data/raw/_calibration_frame.jpg")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp_path), frozen_frame)

        print("\nVerifying calibration with FEN extraction...")
        try:
            result = image_to_fen(
                image_path=tmp_path,
                localizer=localizer,
                board_size=BOARD_SIZE,
                orientation=PRIMARY_ORIENTATION,
            )
            print(f"Detected {len(result['detections'])} pieces")
            print(f"FEN: {result['fen']}")
            warped = result["warped_image"]
            if warped is not None:
                DEBUG_WARP_PATH.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(DEBUG_WARP_PATH), warped)
                print(f"Warped preview saved to: {DEBUG_WARP_PATH}")
                cv2.imshow("Warped Board Preview", warped)
                cv2.waitKey(0)
        except Exception as e:
            print(f"FEN check skipped (model not trained yet): {e}")

        print("\nDone. Run the system with:")
        print("  python src/dashboard/app.py")
        print("  python scripts/run_vision.py")
        break

    elif key == ord('q'):
        print("Quit without saving.")
        break

cv2.destroyAllWindows()