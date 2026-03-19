import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import chess.pgn
import io
import json
import zstandard as zstd
from tqdm import tqdm

OUT_DIR = Path("data/processed/lichess")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ELO_BRACKETS = {
    "1200": (1100, 1300),
    "1400": (1300, 1500),
    "1600": (1500, 1700),
}
GAMES_PER_BRACKET = 3000

# Lichess open database — small monthly file (Jan 2013, ~20MB compressed)
# This is a real .zst file, no redirect
DATASET_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"
RAW_ZST = Path("data/raw/lichess_2013_01.pgn.zst")


def download_dataset():
    if RAW_ZST.exists():
        size_mb = RAW_ZST.stat().st_size / (1024 * 1024)
        print(f"Already downloaded: {RAW_ZST} ({size_mb:.1f} MB)")
        return

    print("Downloading Lichess database (Jan 2013, ~20MB)...")
    headers = {"User-Agent": "chess-ecosystem-pfe/1.0"}
    r = requests.get(DATASET_URL, stream=True, timeout=60, headers=headers)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    with open(RAW_ZST, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="Downloading"
    ) as bar:
        for chunk in r.iter_content(1024 * 256):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"Downloaded: {RAW_ZST}")


def extract_by_bracket():
    counts  = {k: 0 for k in ELO_BRACKETS}
    writers = {
        k: open(OUT_DIR / f"games_{k}.jsonl", "w", encoding="utf-8")
        for k in ELO_BRACKETS
    }

    print("\nFiltering games by Elo bracket (streaming decompression)...")

    dctx = zstd.ZstdDecompressor()

    with open(RAW_ZST, "rb") as fh, \
         dctx.stream_reader(fh) as reader, \
         tqdm(desc="Games read", unit=" games") as bar:

        text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")

        while True:
            if all(v >= GAMES_PER_BRACKET for v in counts.values()):
                break

            game = chess.pgn.read_game(text_stream)
            if game is None:
                break

            bar.update(1)

            try:
                w_elo = int(game.headers.get("WhiteElo", 0))
                b_elo = int(game.headers.get("BlackElo", 0))
            except (ValueError, TypeError):
                continue

            avg_elo = (w_elo + b_elo) // 2
            result  = game.headers.get("Result", "")

            if result not in ("1-0", "0-1", "1/2-1/2"):
                continue

            moves = [m.uci() for m in game.mainline_moves()]
            if len(moves) < 20:
                continue

            for bracket_name, (lo, hi) in ELO_BRACKETS.items():
                if counts[bracket_name] >= GAMES_PER_BRACKET:
                    continue
                if lo <= avg_elo <= hi:
                    record = {
                        "white_elo": w_elo,
                        "black_elo": b_elo,
                        "avg_elo":   avg_elo,
                        "bracket":   bracket_name,
                        "result":    result,
                        "moves":     moves,
                    }
                    writers[bracket_name].write(json.dumps(record) + "\n")
                    counts[bracket_name] += 1
                    break

    for w in writers.values():
        w.close()

    print("\nExtracted per bracket:")
    for bracket, count in counts.items():
        path = OUT_DIR / f"games_{bracket}.jsonl"
        print(f"  Elo {bracket}: {count:,} games → {path}")

    return counts


if __name__ == "__main__":
    download_dataset()
    counts = extract_by_bracket()
    total = sum(counts.values())
    print(f"\nTotal training games: {total:,}")
    print("Ready for position encoding.")