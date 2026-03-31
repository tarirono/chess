import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from src.vision.board_localizer import BoardLocalizer
from src.vision.fen_extractor import (
    BOARD_SIZE,
    PRIMARY_ORIENTATION,
    image_to_fen,
)

DEBUG_WARP_PATH = Path("data/raw/_debug_warped_board.jpg")

# Use a validation image as calibration target
test_images = list(Path("data/raw/chess-pieces/valid/images").glob("*.jpg"))
img_path = test_images[0]

print(f"Calibration image: {img_path.name}")
print("A window will open. Click the board corners in this order:")
print("TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT.\n")

localizer = BoardLocalizer()
points = localizer.calibrate(img_path)
print(f"Selected points: {points}")

result = image_to_fen(
    image_path=img_path,
    localizer=localizer,
    board_size=BOARD_SIZE,
    orientation=PRIMARY_ORIENTATION,
)

warped = result["warped_image"]
if warped is not None:
    DEBUG_WARP_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DEBUG_WARP_PATH), warped)

print(f"\nImage size: {result['image_width']}x{result['image_height']}")
print(f"Detected {len(result['detections'])} pieces")
if warped is not None:
    print(f"Warped board image saved to: {DEBUG_WARP_PATH}")

print("\nRaw detections:")
for i, det in enumerate(result["detections"], start=1):
    print(
        f"  {i:02d}. {det['label']:<13} "
        f"conf={det['conf']:.3f} box={det['box']}"
    )

print(f"\nBoard state after perspective correction [{PRIMARY_ORIENTATION}]:")
for sq, piece in result["board"].items():
    print(f"  {sq}  {piece}")

print(f"\nFEN placement: {result['fen']}")
print(f"Full FEN (assumed): {result['fen']} w KQkq - 0 1")
