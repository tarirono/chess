import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chess
import torch
from src.behavioral.model   import ChessResNet
from src.behavioral.encoder import board_to_tensor, move_to_index, index_to_move, NUM_MOVES

MODELS_DIR = Path("data/models/behavioral")
BRACKETS   = ["1200", "1400", "1600"]


def _load_state_dict(model: ChessResNet, checkpoint: dict, path: Path) -> None:
    """
    Load weights robustly regardless of whether the checkpoint was saved
    with 'policy.' or 'policy_head.' key prefix (Colab vs local training).

    BUG FIX: the previous str.replace approach silently loaded wrong weights
    if the prefixes already matched (the replace became a no-op).  We now:
      1. Try strict loading first (both environments aligned).
      2. If that fails with a key error, remap 'policy.' → 'policy_head.'
         only when those keys are actually present, then verify no keys are
         missing or unexpected after remapping.
    """
    state_dict = checkpoint["state_dict"]

    # --- attempt 1: strict load (preferred, covers locally trained models) ---
    try:
        model.load_state_dict(state_dict, strict=True)
        return
    except RuntimeError:
        pass

    # --- attempt 2: remap Colab 'policy.' prefix → local 'policy_head.' ---
    needs_remap = any(k.startswith("policy.") for k in state_dict)
    if needs_remap:
        remapped = {
            k.replace("policy.", "policy_head.", 1): v
            for k, v in state_dict.items()
        }
    else:
        remapped = state_dict

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        raise RuntimeError(
            f"[MoveService] Cannot load {path.name} — "
            f"missing keys after remap: {missing}"
        )
    if unexpected:
        raise RuntimeError(
            f"[MoveService] Cannot load {path.name} — "
            f"unexpected keys after remap: {unexpected}"
        )


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
            _load_state_dict(model, checkpoint, path)
            model.eval()
            self.models[bracket] = model
            print(f"  [MoveService] Loaded Elo {bracket} "
                  f"(val_acc={checkpoint.get('val_acc', '?'):.1f}%)")

        if not self.models:
            raise RuntimeError("No trained models found. Run train_behavioral.py first.")

    def _closest_bracket(self, elo: int) -> str:
        bracket_centers = {"1200": 1200, "1400": 1400, "1600": 1600}
        available = {k: v for k, v in bracket_centers.items() if k in self.models}
        return min(available, key=lambda k: abs(available[k] - elo))

    def _legal_mask(self, board: chess.Board) -> torch.Tensor:
        mask = torch.zeros(NUM_MOVES, dtype=torch.bool)
        for move in board.legal_moves:
            mask[move_to_index(move)] = True
        return mask

    def get_move(self, fen: str, elo: int = 1400,
                 temperature: float = 1.0) -> dict:
        """
        Given a FEN position and target Elo, return a move dict:
          { "uci": str, "bracket": str, "conf": float }
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
            logits[~legal_mask] = float("-inf")

            if temperature != 1.0:
                logits = logits / temperature

            probs      = torch.softmax(logits, dim=0)
            move_index = probs.argmax().item()
            conf       = probs[move_index].item()

        move = index_to_move(move_index)

        if move not in board.legal_moves:
            move_q = chess.Move(move.from_square, move.to_square,
                                promotion=chess.QUEEN)
            if move_q in board.legal_moves:
                move = move_q
            else:
                move = next(iter(board.legal_moves))

        return {
            "uci":     move.uci(),
            "bracket": bracket,
            "conf":    round(conf, 4),
        }