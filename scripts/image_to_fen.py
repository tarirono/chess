import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from src.vision.board_localizer import BoardLocalizer
from src.vision.fen_extractor import BOARD_SIZE, PRIMARY_ORIENTATION, image_to_fen


def main():
    parser = argparse.ArgumentParser(description="Convert a chessboard image into FEN.")
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument(
        "--save-warp",
        default="data/raw/_debug_warped_board.jpg",
        help="Where to save the warped board preview.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    result = image_to_fen(
        image_path=image_path,
        localizer=BoardLocalizer(),
        board_size=BOARD_SIZE,
        orientation=PRIMARY_ORIENTATION,
    )

    warped = result["warped_image"]
    if warped is not None:
        out_path = Path(args.save_warp)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), warped)
        print(f"Warped board image saved to: {out_path}")

    print(f"Image: {image_path.name}")
    print(f"Orientation: {result['orientation']}")
    print(f"Detected pieces: {len(result['detections'])}")
    print(f"FEN placement: {result['fen']}")
    print(f"Full FEN (assumed): {result['fen']} w KQkq - 0 1")


if __name__ == "__main__":
    main()
