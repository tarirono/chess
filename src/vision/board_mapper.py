import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np


class BoardMapper:
    """
    Maps pixel bounding boxes to chess square notation (a1-h8).
    
    Assumes a top-down camera with the board filling most of the image.
    White pieces start at the bottom (ranks 1-2), black at top (ranks 7-8).
    """

    def __init__(self, image_width: int, image_height: int,
                 board_x1: int = None, board_y1: int = None,
                 board_x2: int = None, board_y2: int = None):
        """
        Args:
            image_width, image_height: full image dimensions
            board_x1/y1/x2/y2: bounding box of the board within the image.
                                If None, assumes board fills the whole image.
        """
        self.img_w = image_width
        self.img_h = image_height

        # Board region — default to full image
        self.bx1 = board_x1 if board_x1 is not None else 0
        self.by1 = board_y1 if board_y1 is not None else 0
        self.bx2 = board_x2 if board_x2 is not None else image_width
        self.by2 = board_y2 if board_y2 is not None else image_height

        self.board_w = self.bx2 - self.bx1
        self.board_h = self.by2 - self.by1
        self.square_w = self.board_w / 8
        self.square_h = self.board_h / 8

    def pixel_to_square(self, x: float, y: float) -> str:
        """
        Convert a pixel coordinate (centre of a bounding box) to
        a chess square like 'e4' or 'a1'.

        Files:  a=left ... h=right  (x axis)
        Ranks:  8=top  ... 1=bottom (y axis, image top = rank 8)
        """
        # Clamp to board region
        x = max(self.bx1, min(self.bx2 - 1, x))
        y = max(self.by1, min(self.by2 - 1, y))

        # Relative position within board (0.0 – 1.0)
        rel_x = (x - self.bx1) / self.board_w
        rel_y = (y - self.by1) / self.board_h

        file_idx = int(rel_x * 8)   # 0=a ... 7=h
        rank_idx = int(rel_y * 8)   # 0=rank8 ... 7=rank1

        # Clamp indices to valid range
        file_idx = max(0, min(7, file_idx))
        rank_idx = max(0, min(7, rank_idx))

        file_char = chr(ord('a') + file_idx)
        rank_num  = 8 - rank_idx          # top of image = rank 8

        return f"{file_char}{rank_num}"

    def detections_to_board(self, detections: list[dict]) -> dict[str, str]:
        """
        Convert a list of PieceDetector detections to a board state dict.

        Returns:
            { "e4": "white-pawn", "e7": "black-pawn", ... }
        """
        board = {}
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            # Use centre-bottom of box (more accurate for piece base)
            cx = (x1 + x2) / 2
            cy = y1 + (y2 - y1) * 0.75   # 75% down = near base of piece

            square = self.pixel_to_square(cx, cy)

            # If two pieces map to same square, keep higher confidence one
            if square not in board or det["conf"] > board[square]["conf"]:
                board[square] = {
                    "piece": det["label"],
                    "conf":  det["conf"]
                }

        return {sq: info["piece"] for sq, info in sorted(board.items())}

    def board_to_fen_placement(self, board: dict[str, str]) -> str:
        """
        Convert board state dict to FEN piece placement string.
        e.g. 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
        """
        PIECE_TO_FEN = {
            "white-king":   "K", "white-queen":  "Q",
            "white-rook":   "R", "white-bishop": "B",
            "white-knight": "N", "white-pawn":   "P",
            "black-king":   "k", "black-queen":  "q",
            "black-rook":   "r", "black-bishop": "b",
            "black-knight": "n", "black-pawn":   "p",
            "bishop":       "B",  # fallback for ambiguous class
        }

        rows = []
        for rank in range(8, 0, -1):          # rank 8 down to 1
            empty = 0
            row_str = ""
            for file in "abcdefgh":
                square = f"{file}{rank}"
                if square in board:
                    if empty:
                        row_str += str(empty)
                        empty = 0
                    piece_label = board[square]
                    row_str += PIECE_TO_FEN.get(piece_label, "?")
                else:
                    empty += 1
            if empty:
                row_str += str(empty)
            rows.append(row_str)

        return "/".join(rows)