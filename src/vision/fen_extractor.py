import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2

from src.vision.board_localizer import BoardLocalizer
from src.vision.board_mapper import BoardMapper
from src.vision.piece_detector import PieceDetector


BOARD_SIZE = 800
PRIMARY_ORIENTATION = "transpose_vertical_flip"


def orientation_transform(name: str, x: float, y: float, board_size: int) -> tuple[float, float]:
    if name == "normal":
        return x, y
    if name == "horizontal_flip":
        return (board_size - 1) - x, y
    if name == "vertical_flip":
        return x, (board_size - 1) - y
    if name == "both_flips":
        return (board_size - 1) - x, (board_size - 1) - y
    if name == "transpose":
        return y, x
    if name == "transpose_horizontal_flip":
        return (board_size - 1) - y, x
    if name == "transpose_vertical_flip":
        return y, (board_size - 1) - x
    if name == "transpose_both_flips":
        return (board_size - 1) - y, (board_size - 1) - x
    raise ValueError(f"Unknown orientation: {name}")


def detections_to_fen(
    detections: list[dict],
    localizer: BoardLocalizer | None = None,
    board_size: int = BOARD_SIZE,
    orientation: str = PRIMARY_ORIENTATION,
    image_width: int | None = None,
    image_height: int | None = None,
) -> dict:
    """
    Convert detections into a board-state dict and FEN placement.

    If a localizer with 4 saved points is available, detections are projected
    through the perspective transform and interpreted on a normalized board.
    Otherwise, falls back to the older rectangular board mapping.
    """
    if not detections:
        return {"board": {}, "fen": None}

    if localizer and localizer.get_points():
        mapper = BoardMapper(image_width=board_size, image_height=board_size)
        board_state = {}

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cx = (x1 + x2) / 2
            cy = y1 + (y2 - y1) * 0.75

            transformed = localizer.transform_point(cx, cy, board_size=board_size)
            if transformed is None:
                continue

            tx, ty = orientation_transform(orientation, transformed[0], transformed[1], board_size)
            if not (0 <= tx < board_size and 0 <= ty < board_size):
                continue

            square = mapper.pixel_to_square(tx, ty)
            if square not in board_state or det["conf"] > board_state[square]["conf"]:
                board_state[square] = {"piece": det["label"], "conf": det["conf"]}

        fen_board = {sq: info["piece"] for sq, info in sorted(board_state.items())}
        fen = mapper.board_to_fen_placement(fen_board)
        return {"board": fen_board, "fen": fen}

    if image_width is None or image_height is None:
        raise ValueError("image_width and image_height are required without perspective points.")

    corners = localizer.get_corners() if localizer else None
    if corners:
        x1, y1, x2, y2 = corners
        mapper = BoardMapper(
            image_width=image_width,
            image_height=image_height,
            board_x1=x1,
            board_y1=y1,
            board_x2=x2,
            board_y2=y2,
        )
    else:
        mapper = BoardMapper(image_width=image_width, image_height=image_height)

    fen_board = mapper.detections_to_board(detections)
    fen = mapper.board_to_fen_placement(fen_board)
    return {"board": fen_board, "fen": fen}


def image_to_fen(
    image_path: str | Path,
    detector: PieceDetector | None = None,
    localizer: BoardLocalizer | None = None,
    board_size: int = BOARD_SIZE,
    orientation: str = PRIMARY_ORIENTATION,
) -> dict:
    """
    Full image -> detections -> board -> FEN pipeline.
    """
    image_path = Path(image_path)
    if detector is None:
        detector = PieceDetector()
    if localizer is None:
        localizer = BoardLocalizer()

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    h, w = img.shape[:2]
    detections = detector.detect(image_path)
    result = detections_to_fen(
        detections=detections,
        localizer=localizer,
        board_size=board_size,
        orientation=orientation,
        image_width=w,
        image_height=h,
    )

    warped = None
    if localizer.get_points():
        warped = localizer.warp_image(image_path, board_size=board_size)

    return {
        "image_width": w,
        "image_height": h,
        "detections": detections,
        "board": result["board"],
        "fen": result["fen"],
        "warped_image": warped,
        "orientation": orientation,
    }
