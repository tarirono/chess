import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chess
import numpy as np
import torch

# ── Constants ─────────────────────────────────────────────────────────

# 12 piece planes: 6 piece types × 2 colors
# Order: P N B R Q K p n b r q k  (uppercase=white, lowercase=black)
PIECE_PLANES = [
    (chess.PAWN,   chess.WHITE),
    (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE),
    (chess.ROOK,   chess.WHITE),
    (chess.QUEEN,  chess.WHITE),
    (chess.KING,   chess.WHITE),
    (chess.PAWN,   chess.BLACK),
    (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK),
    (chess.ROOK,   chess.BLACK),
    (chess.QUEEN,  chess.BLACK),
    (chess.KING,   chess.BLACK),
]

NUM_PLANES      = 12   # piece planes
BOARD_SIZE      = 8
INPUT_CHANNELS  = 13   # 12 pieces + 1 side-to-move plane
NUM_MOVES       = 64 * 64  # 4096 possible from-to combinations


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Encode a chess.Board into a (13, 8, 8) float32 tensor.

    Planes 0–11 : one-hot piece occupancy per piece type per color.
    Plane 12    : side to move (all 1s = white to move, all 0s = black).

    Returns shape: (13, 8, 8)
    """
    tensor = np.zeros((INPUT_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    for plane_idx, (piece_type, color) in enumerate(PIECE_PLANES):
        for sq in board.pieces(piece_type, color):
            # chess squares: 0=a1 (bottom-left) → 63=h8 (top-right)
            rank = sq // 8   # 0–7
            file = sq  % 8   # 0–7
            tensor[plane_idx, rank, file] = 1.0

    # Side-to-move plane
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    return torch.tensor(tensor, dtype=torch.float32)


def move_to_index(move: chess.Move) -> int:
    """
    Encode a chess.Move as an integer index in [0, 4095].
    Index = from_square * 64 + to_square
    """
    return move.from_square * 64 + move.to_square


def index_to_move(index: int) -> chess.Move:
    """Decode an integer index back to a chess.Move (no promotion info)."""
    from_sq = index // 64
    to_sq   = index  % 64
    return chess.Move(from_sq, to_sq)


def moves_to_game_samples(moves_uci: list[str]) -> list[tuple]:
    """
    Replay a game from a list of UCI move strings.
    Returns a list of (board_tensor, move_index) training samples —
    one per position in the game.
    """
    board   = chess.Board()
    samples = []

    for uci in moves_uci:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break

        if move not in board.legal_moves:
            break

        # Encode position BEFORE the move is made
        tensor     = board_to_tensor(board)
        move_index = move_to_index(move)
        samples.append((tensor, move_index))

        board.push(move)

    return samples