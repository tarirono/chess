import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from src.behavioral.encoder import moves_to_game_samples

PROCESSED_DIR = Path("data/processed/lichess")
CACHE_DIR     = Path("data/processed/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class ChessDataset(Dataset):
    """
    PyTorch Dataset built from Lichess JSONL game files.
    Each sample: (board_tensor [13×8×8], move_index [int])
    """

    def __init__(self, bracket: str, max_games: int = None):
        """
        Args:
            bracket:   one of '1200', '1400', '1600'
            max_games: limit number of games (None = all)
        """
        self.bracket   = bracket
        cache_path     = CACHE_DIR / f"dataset_{bracket}.pt"

        if cache_path.exists():
            print(f"Loading cached dataset: {cache_path}")
            data = torch.load(cache_path, weights_only=False)
            self.tensors = data["tensors"]
            self.labels  = data["labels"]
        else:
            self.tensors, self.labels = self._build(bracket, max_games, cache_path)

        print(f"Dataset [{bracket}]: {len(self.tensors):,} positions loaded.")

    def _build(self, bracket: str, max_games: int, cache_path: Path):
        jsonl_path = PROCESSED_DIR / f"games_{bracket}.jsonl"
        if not jsonl_path.exists():
            raise FileNotFoundError(
                f"No game file found: {jsonl_path}\n"
                f"Run scripts/download_lichess.py first."
            )

        tensors, labels = [], []
        games_loaded    = 0

        print(f"Building dataset from {jsonl_path}...")
        with open(jsonl_path, encoding="utf-8") as f:
            lines = f.readlines()

        if max_games:
            lines = lines[:max_games]

        for line in tqdm(lines, desc=f"  Encoding [{bracket}]"):
            record = json.loads(line)
            samples = moves_to_game_samples(record["moves"])
            for tensor, move_idx in samples:
                tensors.append(tensor)
                labels.append(move_idx)
            games_loaded += 1

        tensors_t = torch.stack(tensors)
        labels_t  = torch.tensor(labels, dtype=torch.long)

        torch.save({"tensors": tensors_t, "labels": labels_t}, cache_path)
        print(f"  Cached to {cache_path}")
        print(f"  {games_loaded:,} games → {len(tensors_t):,} positions")

        return tensors_t, labels_t

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]