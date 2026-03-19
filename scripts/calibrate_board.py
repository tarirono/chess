import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.board_localizer import BoardLocalizer
from src.vision.piece_detector import PieceDetector
from src.vision.board_mapper import BoardMapper
from PIL import Image

# Use a validation image as calibration target
test_images = list(Path("data/raw/chess-pieces/valid/images").glob("*.jpg"))
img_path = test_images[0]

print(f"Calibration image: {img_path.name}")
print("A window will open. Click TOP-LEFT then BOTTOM-RIGHT of the board.\n")

localizer = BoardLocalizer()

# Try auto-detect first
corners = localizer.auto_detect(img_path)
print(f"Auto-detected corners: {corners}")

# Then run full pipeline with those corners
with Image.open(img_path) as im:
    w, h = im.size

detector  = PieceDetector()
detections = detector.detect(img_path)

x1, y1, x2, y2 = corners
mapper = BoardMapper(
    image_width=w, image_height=h,
    board_x1=x1, board_y1=y1,
    board_x2=x2, board_y2=y2
)

board_state = mapper.detections_to_board(detections)
fen = mapper.board_to_fen_placement(board_state)

print(f"\nBoard state with localized corners:")
for sq, piece in board_state.items():
    print(f"  {sq}  {piece}")

print(f"\nFEN: {fen} w KQkq - 0 1")
