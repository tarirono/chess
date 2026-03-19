import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.piece_detector import PieceDetector
from src.vision.board_mapper import BoardMapper
from PIL import Image

detector = PieceDetector()

test_images = list(Path("data/raw/chess-pieces/valid/images").glob("*.jpg"))
img_path = test_images[0]

# Get image dimensions
with Image.open(img_path) as im:
    w, h = im.size

print(f"Image: {img_path.name}  ({w}x{h})\n")

# Detect pieces
detections = detector.detect(img_path)
print(f"Detected {len(detections)} pieces\n")

# Map to board
mapper = BoardMapper(image_width=w, image_height=h)
board_state = mapper.detections_to_board(detections)

print("Board state (square → piece):")
for square, piece in board_state.items():
    print(f"  {square}  {piece}")

# Generate FEN placement
fen = mapper.board_to_fen_placement(board_state)
print(f"\nFEN placement string:")
print(f"  {fen}")
print(f"\nFull FEN (assume white to move):")
print(f"  {fen} w KQkq - 0 1")

