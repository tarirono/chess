# scripts/validate_style.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chess
import json
from src.api.move_service import MoveService

service = MoveService()
BRACKETS = ["1200", "1400", "1600"]
N_GAMES = 20

def play_game(bracket):
    board = chess.Board()
    stats = {"castled": False, "center_moves": 0, "moves": 0}
    center = {chess.D4, chess.D5, chess.E4, chess.E5}

    for _ in range(60):
        if board.is_game_over():
            break
        result = service.get_move(board.fen(), elo=int(bracket))
        if not result or not result.get("uci"):
            break
        move = chess.Move.from_uci(result["uci"])
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.KING:
            if abs(move.from_square - move.to_square) == 2:
                stats["castled"] = True
        if move.to_square in center:
            stats["center_moves"] += 1
        stats["moves"] += 1
        board.push(move)
    return stats

results = {}
for bracket in BRACKETS:
    print(f"Analyzing bracket {bracket}...")
    castled, center_total, total_moves = 0, 0, 0
    for _ in range(N_GAMES):
        s = play_game(bracket)
        castled += s["castled"]
        center_total += s["center_moves"]
        total_moves += s["moves"]
    results[bracket] = {
        "castling_rate_pct": round(castled / N_GAMES * 100, 1),
        "center_moves_per_game": round(center_total / N_GAMES, 2),
    }
    print(f"  Castling rate:      {results[bracket]['castling_rate_pct']}%")
    print(f"  Center moves/game:  {results[bracket]['center_moves_per_game']}")

out = Path("data/models/behavioral/style_validation.json")
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {out}")
