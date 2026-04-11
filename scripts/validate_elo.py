"""
Elo Validation Script — Phase B
================================
Tests each behavioral cloning bot bracket against Stockfish at calibrated
depths that correspond to approximate Elo ratings:

  Stockfish depth 1  ≈  800–1000 Elo
  Stockfish depth 3  ≈  1200–1400 Elo
  Stockfish depth 5  ≈  1600–1800 Elo
  Stockfish depth 8  ≈  2000+     Elo

For each bracket we play N games (alternating colors) and report:
  - Win / Draw / Loss rates
  - Average centipawn loss per move
  - Expected Elo range based on W/D/L

Results are saved to data/models/behavioral/elo_validation.json
and printed as a summary table.

Usage:
    python scripts/validate_elo.py
    python scripts/validate_elo.py --games 20 --depth 3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import chess
import chess.engine
import json
import argparse
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

from src.api.move_service import MoveService

# ── Load Stockfish path from .env (same source as engine_analyzer.py) ──
load_dotenv()

def _resolve_stockfish() -> str:
    """
    Resolve Stockfish path using the same priority as engine_analyzer.py:
      1. STOCKFISH_PATH env var (from .env)
      2. Common system locations
      3. 'stockfish' on PATH (last resort)
    """
    env_path = os.getenv("STOCKFISH_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    candidates = [
        "stockfish",
        "/usr/games/stockfish",
        "/usr/local/bin/stockfish",
        "/opt/homebrew/bin/stockfish",
        r"C:\stockfish\stockfish.exe",
        r"C:\Program Files\Stockfish\stockfish.exe",
    ]
    for c in candidates:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(c)
            engine.quit()
            print(f"Stockfish found at: {c}")
            return c
        except Exception:
            continue

    return "stockfish"   # will raise a clear error if missing

STOCKFISH_PATH = _resolve_stockfish()

# Opponent depth per bracket
BRACKET_VS_DEPTH = {
    "1200": 3,
    "1400": 5,
    "1600": 8,
}

BLUNDER_CP = 200
MISTAKE_CP = 100


def play_game(
    service:         MoveService,
    engine:          chess.engine.SimpleEngine,
    bracket:         str,
    engine_depth:    int,
    bot_plays_white: bool,
) -> dict:
    board = chess.Board()
    moves_played = 0
    bot_cp_losses = []
    move_classes = {"best": 0, "good": 0, "inaccuracy": 0, "mistake": 0, "blunder": 0}

    while not board.is_game_over() and moves_played < 200:
        is_bot_turn = (board.turn == chess.WHITE) == bot_plays_white

        if is_bot_turn:
            result = service.get_move(board.fen(), elo=int(bracket))
            uci    = result.get("uci")
            if not uci:
                break
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break

            pre      = engine.analyse(board, chess.engine.Limit(depth=engine_depth))
            score_before = pre["score"].white()
            cp_before    = score_before.score(mate_score=10000) or 0

            board.push(move)
            post     = engine.analyse(board, chess.engine.Limit(depth=engine_depth))
            score_after = post["score"].white()
            cp_after    = score_after.score(mate_score=10000) or 0

            cp_loss = cp_before - cp_after if bot_plays_white else cp_after - cp_before
            cp_loss = max(0, cp_loss)
            bot_cp_losses.append(cp_loss)

            if cp_loss >= BLUNDER_CP:
                move_classes["blunder"] += 1
            elif cp_loss >= MISTAKE_CP:
                move_classes["mistake"] += 1
            elif cp_loss >= 50:
                move_classes["inaccuracy"] += 1
            elif cp_loss <= 10:
                move_classes["best"] += 1
            else:
                move_classes["good"] += 1
        else:
            result = engine.play(board, chess.engine.Limit(depth=engine_depth))
            board.push(result.move)

        moves_played += 1

    outcome = board.outcome()
    if outcome is None:
        result_str = "draw"
    elif outcome.winner is None:
        result_str = "draw"
    elif (outcome.winner == chess.WHITE) == bot_plays_white:
        result_str = "win"
    else:
        result_str = "loss"

    avg_cp_loss = round(sum(bot_cp_losses) / len(bot_cp_losses), 1) if bot_cp_losses else 0
    return {
        "result":       result_str,
        "moves":        moves_played,
        "avg_cp_loss":  avg_cp_loss,
        "move_classes": move_classes,
    }


def performance_to_elo(win_rate: float, draw_rate: float) -> int:
    import math
    score = win_rate + 0.5 * draw_rate
    score = max(0.01, min(0.99, score))
    return int(-400 * math.log10(1 / score - 1))


def validate_bracket(
    service:      MoveService,
    engine:       chess.engine.SimpleEngine,
    bracket:      str,
    n_games:      int,
    engine_depth: int,
) -> dict:
    print(f"\n  Bracket {bracket} vs Stockfish depth {engine_depth} ({n_games} games)")

    wins = draws = losses = 0
    all_cp_losses = []
    all_move_classes = {"best": 0, "good": 0, "inaccuracy": 0, "mistake": 0, "blunder": 0}

    for i in tqdm(range(n_games), desc=f"    Elo {bracket}"):
        stats = play_game(service, engine, bracket, engine_depth, i % 2 == 0)

        if stats["result"] == "win":
            wins += 1
        elif stats["result"] == "draw":
            draws += 1
        else:
            losses += 1

        all_cp_losses.append(stats["avg_cp_loss"])
        for k, v in stats["move_classes"].items():
            all_move_classes[k] += v

    total     = wins + draws + losses
    win_rate  = wins / total
    draw_rate = draws / total
    loss_rate = losses / total
    avg_cp    = round(sum(all_cp_losses) / len(all_cp_losses), 1)
    elo_diff  = performance_to_elo(win_rate, draw_rate)

    total_moves = sum(all_move_classes.values())
    blunder_pct = round(all_move_classes["blunder"] / max(1, total_moves) * 100, 1)
    mistake_pct = round(all_move_classes["mistake"]  / max(1, total_moves) * 100, 1)
    best_pct    = round(all_move_classes["best"]     / max(1, total_moves) * 100, 1)

    summary = {
        "bracket":              bracket,
        "opponent_depth":       engine_depth,
        "n_games":              n_games,
        "wins":                 wins,
        "draws":                draws,
        "losses":               losses,
        "win_rate":             round(win_rate,  3),
        "draw_rate":            round(draw_rate, 3),
        "loss_rate":            round(loss_rate, 3),
        "avg_cp_loss":          avg_cp,
        "elo_diff_vs_opponent": elo_diff,
        "blunder_pct":          blunder_pct,
        "mistake_pct":          mistake_pct,
        "best_move_pct":        best_pct,
        "move_classes":         all_move_classes,
    }

    print(f"    W/D/L: {wins}/{draws}/{losses}  "
          f"avg_cp_loss={avg_cp}  "
          f"elo_diff={elo_diff:+d}  "
          f"blunders={blunder_pct}%")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Validate bot Elo via Stockfish games")
    parser.add_argument("--games",   type=int, default=10)
    parser.add_argument("--depth",   type=int, default=None)
    parser.add_argument("--bracket", type=str, default=None)
    args = parser.parse_args()

    print("Phase B — Elo Validation")
    print(f"Stockfish path: {STOCKFISH_PATH}")
    print("=" * 50)

    print("Loading behavioral cloning models...")
    service = MoveService()

    print(f"Starting Stockfish...")
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        print(f"\nERROR: Cannot start Stockfish: {e}")
        print("Fix: install Stockfish and set STOCKFISH_PATH in your .env file.")
        sys.exit(1)

    brackets = [args.bracket] if args.bracket else ["1200", "1400", "1600"]
    results  = {}

    try:
        for bracket in brackets:
            if bracket not in service.models:
                print(f"  No model for bracket {bracket} — skipping.")
                continue
            depth  = args.depth or BRACKET_VS_DEPTH.get(bracket, 5)
            result = validate_bracket(service, engine, bracket, args.games, depth)
            results[bracket] = result
    finally:
        engine.quit()

    out_path = Path("data/models/behavioral/elo_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": results}, f, indent=2)

    print(f"\n{'='*60}")
    print("Elo Validation Summary")
    print(f"{'='*60}")
    print(f"{'Bracket':<10} {'W/D/L':<14} {'Avg CP Loss':<14} {'Elo Δ':<10} {'Blunders'}")
    print("-" * 60)
    for b, r in results.items():
        wdl = f"{r['wins']}/{r['draws']}/{r['losses']}"
        print(f"  {b:<8} {wdl:<14} {r['avg_cp_loss']:<14} "
              f"{r['elo_diff_vs_opponent']:>+6}     {r['blunder_pct']}%")

    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()