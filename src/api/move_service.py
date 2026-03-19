import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chess
import torch
from src.behavioral.model   import ChessResNet
from src.behavioral.encoder import board_to_tensor, move_to_index, index_to_move, NUM_MOVES

MODELS_DIR = Path("data/models/behavioral")

BRACKETS = ["1200", "1400", "1600"]


class MoveService:
    """
    Wraps trained behavioral cloning models.
    Given a FEN string and a target Elo bracket,
    returns the best human-like UCI move.
    """

    def __init__(self):
        self.models: dict[str, ChessResNet] = {}
        self._load_models()

    def _load_models(self):
        for bracket in BRACKETS:
            path = MODELS_DIR / f"chess_bot_{bracket}.pt"
            if not path.exists():
                print(f"  [MoveService] No model for bracket {bracket} — skipping.")
                continue
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            model = ChessResNet()
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.models[bracket] = model
            print(f"  [MoveService] Loaded Elo {bracket} "
                  f"(val_acc={checkpoint.get('val_acc', '?'):.1f}%)")

        if not self.models:
            raise RuntimeError("No trained models found. Run train_behavioral.py first.")

    def _closest_bracket(self, elo: int) -> str:
        """Return the bracket name closest to the given Elo."""
        bracket_centers = {"1200": 1200, "1400": 1400, "1600": 1600}
        available = {k: v for k, v in bracket_centers.items() if k in self.models}
        return min(available, key=lambda k: abs(available[k] - elo))

    def _legal_mask(self, board: chess.Board) -> torch.Tensor:
        """Build a boolean mask of legal move indices."""
        mask = torch.zeros(NUM_MOVES, dtype=torch.bool)
        for move in board.legal_moves:
            mask[move_to_index(move)] = True
        return mask

    def get_move(self, fen: str, elo: int = 1400,
                 temperature: float = 1.0) -> dict:
        """
        Given a FEN position and target Elo, return a move dict:
          {
            "uci":     str,   e.g. "e2e4"
            "bracket": str,   e.g. "1400"
            "conf":    float, top move probability after masking
          }

        temperature > 1.0 = more random (weaker play)
        temperature < 1.0 = more deterministic (stronger play)
        """
        bracket = self._closest_bracket(elo)
        model   = self.models[bracket]

        try:
            board = chess.Board(fen)
        except ValueError as e:
            raise ValueError(f"Invalid FEN: {e}")

        if board.is_game_over():
            return {"uci": None, "bracket": bracket, "conf": 0.0,
                    "reason": "game over"}

        tensor     = board_to_tensor(board)
        legal_mask = self._legal_mask(board)

        model.eval()
        with torch.no_grad():
            logits = model(tensor.unsqueeze(0))[0]

            # Mask illegal moves
            logits[~legal_mask] = float("-inf")

            # Apply temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            probs      = torch.softmax(logits, dim=0)
            move_index = probs.argmax().item()
            conf       = probs[move_index].item()

        move = index_to_move(move_index)

        # Ensure move is legal (promotion fallback)
        if move not in board.legal_moves:
            # Try with queen promotion
            move_q = chess.Move(move.from_square, move.to_square,
                                promotion=chess.QUEEN)
            if move_q in board.legal_moves:
                move = move_q
            else:
                # Fallback to first legal move
                move = next(iter(board.legal_moves))

        return {
            "uci":     move.uci(),
            "bracket": bracket,
            "conf":    round(conf, 4),
        }