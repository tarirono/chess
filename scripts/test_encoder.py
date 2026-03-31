import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chess
import torch
from src.behavioral.encoder import (
    board_to_tensor, move_to_index, index_to_move, moves_to_game_samples
)
from src.behavioral.dataset import ChessDataset

print("=" * 50)
print("Test 1 — board_to_tensor")
print("=" * 50)
board = chess.Board()
t = board_to_tensor(board)
print(f"Tensor shape : {t.shape}   (expected: torch.Size([13, 8, 8]))")
print(f"White pawns  : {t[0].sum().int().item()}   (expected: 8)")
print(f"Black pawns  : {t[6].sum().int().item()}   (expected: 8)")
print(f"Side-to-move : {t[12, 0, 0].item():.0f}   (expected: 1 = white)")

print("\n" + "=" * 50)
print("Test 2 — move encoding round-trip")
print("=" * 50)
move = chess.Move.from_uci("e2e4")
idx  = move_to_index(move)
back = index_to_move(idx)
print(f"Move         : e2e4")
print(f"Encoded idx  : {idx}")
print(f"Decoded back : {back.uci()}   (expected: e2e4)")

print("\n" + "=" * 50)
print("Test 3 — game to samples")
print("=" * 50)
sample_moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
samples = moves_to_game_samples(sample_moves)
print(f"Moves in    : {len(sample_moves)}")
print(f"Samples out : {len(samples)}   (one per position)")
print(f"Sample[0]   : tensor {samples[0][0].shape}, label {samples[0][1]}")

print("\n" + "=" * 50)
print("Test 4 — ChessDataset (bracket 1200, first 100 games)")
print("=" * 50)
ds = ChessDataset(bracket="1200", max_games=100)
x, y = ds[0]
print(f"Sample tensor shape : {x.shape}")
print(f"Sample label        : {y.item()}  (move index 0–4095)")
print(f"Dataset size        : {len(ds):,} positions")

