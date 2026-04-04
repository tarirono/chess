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

import chess
import chess.engine
import json
import argparse
from datetime import datetime
from tqdm import tqdm

from src.api.move_service import MoveService


STOCKFISH_PATH = "stockfish"   # update to your path if needed

# Opponent depth per bracket — calibrated to approximate Elo range
BRACKET_VS_DEPTH = {
    "1200": 3,   # SF depth 3 ≈ 1200–1400 Elo
    "1400": 5,   # SF depth 5 ≈ 1600–1800 Elo (intentionally harder = stress test)
    "1600": 8,   # SF depth 8 ≈ 2000+ Elo   (stress test for strongest bracket)
}

# Centipawn loss thresholds
BLUNDER_CP  = 200
MISTAKE_CP  = 100


def play_game(
    service:        MoveService,
    engine:         chess.engine.SimpleEngine,
    bracket:        str,
    engine_depth:   int,
    bot_plays_white: bool,
) -> dict:
    """
    Play one game between the bot and Stockfish.
    Returns game statistics.
    """
    board = chess.Board()
    moves_played = 0
    bot_cp_losses = []
    move_classes = {"best": 0, "good": 0, "inaccuracy": 0, "mistake": 0, "blunder": 0}

    while not board.is_game_over() and moves_played < 200:
        is_bot_turn = (board.turn == chess.WHITE) == bot_plays_white

        if is_bot_turn:
            # Bot move
            result = service.get_move(board.fen(), elo=int(bracket))
            uci    = result.get("uci")
            if not uci:
                break
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break

            # Evaluate bot's move quality with Stockfish
            pre = engine.analyse(board, chess.engine.Limit(depth=engine_depth))
            score_before = pre["score"].white()
            cp_before    = score_before.score(mate_score=10000) or 0

            board.push(move)
            post = engine.analyse(board, chess.engine.Limit(depth=engine_depth))
            score_after = post["score"].white()
            cp_after    = score_after.score(mate_score=10000) or 0

            # CP loss from bot's perspective
            if bot_plays_white:
                cp_loss = cp_before - cp_after
            else:
                cp_loss = cp_after - cp_before
            cp_loss = max(0, cp_loss)
            bot_cp_losses.append(cp_loss)

            # Classify
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
            # Stockfish move
            result = engine.play(board, chess.engine.Limit(depth=engine_depth))
            board.push(result.move)

        moves_played += 1

    # Game result from bot's perspective
    outcome = board.outcome()
    if outcome is None:
        result_str = "draw"   # game ended by move limit
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


def performance_to_elo(win_rate: float, draw_rate: float) -> tuple[int, int]:
    """
    Approximate Elo difference from win/draw/loss rates.
    Returns (estimated_elo_min, estimated_elo_max).

    Uses the FIDE Elo formula: expected_score = 1/(1+10^(-diff/400))
    We invert: diff = -400 * log10(1/score - 1)
    """
    import math
    score = win_rate + 0.5 * draw_rate
    score = max(0.01, min(0.99, score))  # clamp to avoid log(0)
    elo_diff = int(-400 * math.log10(1 / score - 1))
    # Stockfish depth calibration offsets
    return elo_diff


def validate_bracket(
    service:      MoveService,
    engine:       chess.engine.SimpleEngine,
    bracket:      str,
    n_games:      int,
    engine_depth: int,
) -> dict:
    """Run n_games for one bracket and aggregate results."""
    print(f"\n  Bracket {bracket} vs Stockfish depth {engine_depth} ({n_games} games)")

    wins = draws = losses = 0
    all_cp_losses = []
    all_move_classes = {"best": 0, "good": 0, "inaccuracy": 0, "mistake": 0, "blunder": 0}

    for i in tqdm(range(n_games), desc=f"    Elo {bracket}"):
        bot_white = (i % 2 == 0)  # alternate colors
        stats = play_game(service, engine, bracket, engine_depth, bot_white)

        if stats["result"] == "win":
            wins += 1
        elif stats["result"] == "draw":
            draws += 1
        else:
            losses += 1

        all_cp_losses.append(stats["avg_cp_loss"])
        for k, v in stats["move_classes"].items():
            all_move_classes[k] += v

    total    = wins + draws + losses
    win_rate = wins / total
    draw_rate= draws / total
    loss_rate= losses / total
    avg_cp   = round(sum(all_cp_losses) / len(all_cp_losses), 1)

    elo_diff = performance_to_elo(win_rate, draw_rate)

    total_moves = sum(all_move_classes.values())
    blunder_pct = round(all_move_classes["blunder"] / max(1, total_moves) * 100, 1)
    mistake_pct = round(all_move_classes["mistake"]  / max(1, total_moves) * 100, 1)
    best_pct    = round(all_move_classes["best"]     / max(1, total_moves) * 100, 1)

    summary = {
        "bracket":          bracket,
        "opponent_depth":   engine_depth,
        "n_games":          n_games,
        "wins":             wins,
        "draws":            draws,
        "losses":           losses,
        "win_rate":         round(win_rate,  3),
        "draw_rate":        round(draw_rate, 3),
        "loss_rate":        round(loss_rate, 3),
        "avg_cp_loss":      avg_cp,
        "elo_diff_vs_opponent": elo_diff,
        "blunder_pct":      blunder_pct,
        "mistake_pct":      mistake_pct,
        "best_move_pct":    best_pct,
        "move_classes":     all_move_classes,
    }

    print(f"    W/D/L: {wins}/{draws}/{losses}  "
          f"avg_cp_loss={avg_cp}  "
          f"elo_diff={elo_diff:+d}  "
          f"blunders={blunder_pct}%")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Validate bot Elo via Stockfish games")
    parser.add_argument("--games",  type=int, default=10,
                        help="Number of games per bracket (default: 10)")
    parser.add_argument("--depth",  type=int, default=None,
                        help="Override Stockfish depth for all brackets")
    parser.add_argument("--bracket", type=str, default=None,
                        help="Test only this bracket (e.g. '1400')")
    args = parser.parse_args()

    print("Phase B — Elo Validation")
    print("=" * 50)

    # Load bot service
    print("Loading behavioral cloning models...")
    service = MoveService()

    # Start Stockfish
    print(f"Starting Stockfish from '{STOCKFISH_PATH}'...")
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    except Exception as e:
        print(f"ERROR: Cannot start Stockfish: {e}")
        print("Install Stockfish and update STOCKFISH_PATH in this script.")
        sys.exit(1)

    brackets = [args.bracket] if args.bracket else ["1200", "1400", "1600"]
    results  = {}

    try:
        for bracket in brackets:
            if bracket not in service.models:
                print(f"  No model for bracket {bracket} — skipping.")
                continue

            depth = args.depth or BRACKET_VS_DEPTH.get(bracket, 5)
            result = validate_bracket(service, engine, bracket, args.games, depth)
            results[bracket] = result

    finally:
        engine.quit()

    # Save results
    out_path = Path("data/models/behavioral/elo_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results":   results,
        }, f, indent=2)

    # Print summary table
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
