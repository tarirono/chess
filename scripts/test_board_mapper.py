import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.board_localizer import BoardLocalizer
from src.vision.fen_extractor import PRIMARY_ORIENTATION, image_to_fen


test_images = list(Path("data/raw/chess-pieces/valid/images").glob("*.jpg"))
img_path = test_images[0]

print(f"Image: {img_path.name}\n")

result = image_to_fen(
    image_path=img_path,
    localizer=BoardLocalizer(),
    orientation=PRIMARY_ORIENTATION,
)

print(f"Detected {len(result['detections'])} pieces\n")
print(f"Board state [{PRIMARY_ORIENTATION}]:")
for square, piece in result["board"].items():
    print(f"  {square}  {piece}")

print("\nFEN placement string:")
print(f"  {result['fen']}")
print("\nFull FEN (assume white to move):")
print(f"  {result['fen']} w KQkq - 0 1")
