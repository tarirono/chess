import json
from pathlib import Path

DIFFICULTY_PRIORS = {
    "Opening": 0.2, "Development": 0.2, "Centralization": 0.3,
    "Pawn_structure": 0.35, "Piece_activity": 0.35,
    "Doubled_pawns": 0.4, "Isolated_pawn": 0.4,
    "Fork": 0.45, "Pin": 0.5, "Passed_pawn": 0.5,
    "Endgame": 0.55, "Checkmate_pattern": 0.55, "Open_file_rook": 0.55,
    "Outpost_control": 0.6, "Skewer": 0.6, "Discovery": 0.65,
    "Inaccuracy": 0.3, "Mistake": 0.45, "Blunder": 0.6,
}

out = Path("data/models/skill_difficulties.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(DIFFICULTY_PRIORS, indent=2))
print("Saved skill difficulties:")
for s, d in sorted(DIFFICULTY_PRIORS.items(), key=lambda x: x[1]):
    print(f"  {s:<22} {d}")