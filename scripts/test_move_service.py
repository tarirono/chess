import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.move_service import MoveService
import chess

print("Loading move service...")
service = MoveService()

# Test positions
positions = [
    {
        "name":  "Starting position",
        "fen":   chess.STARTING_FEN,
        "elo":   1200,
    },
    {
        "name":  "After 1.e4",
        "fen":   "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "elo":   1400,
    },
    {
        "name":  "Sicilian after 1.e4 c5",
        "fen":   "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq c6 0 2",
        "elo":   1600,
    },
    {
        "name":  "Complex middlegame",
        "fen":   "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 4",
        "elo":   1400,
    },
]

print(f"\n{'='*55}")
for pos in positions:
    result = service.get_move(pos["fen"], elo=pos["elo"])
    board  = chess.Board(pos["fen"])
    move   = chess.Move.from_uci(result["uci"])
    san    = board.san(move)
    print(f"\n  {pos['name']} (Elo target: {pos['elo']})")
    print(f"  Bot plays : {result['uci']} ({san})")
    print(f"  Bracket   : {result['bracket']}")
    print(f"  Confidence: {result['conf']:.4f}")

print(f"\n{'='*55}")
print("Move service working correctly.")

# ```

# Everything is ready and waiting. When training finishes, just run:
# ```
# python scripts/test_move_service.py
# ```

# And to start the API server:
# ```
# uvicorn src.api.app:app --reload --port 8000