import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chess
import chess.engine
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ── Stockfish path ────────────────────────────────────────────────────
# Priority: .env STOCKFISH_PATH → common system locations → "stockfish" (PATH)
def _find_stockfish() -> str:
    # 1. Explicit env var
    env_path = os.getenv("STOCKFISH_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # 2. Common installation paths
    candidates = [
        "stockfish",                             # on PATH (Linux/Mac/Windows)
        "/usr/games/stockfish",                  # Debian/Ubuntu apt install
        "/usr/local/bin/stockfish",              # Homebrew / manual Linux
        "/opt/homebrew/bin/stockfish",           # Homebrew Apple Silicon
        r"C:\stockfish\stockfish.exe",           # Windows common location
        r"C:\Program Files\Stockfish\stockfish.exe",
    ]
    for c in candidates:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(c)
            engine.quit()
            return c
        except Exception:
            continue

    return "stockfish"   # last resort — will raise a clear error on use


STOCKFISH_PATH  = _find_stockfish()
DEFAULT_DEPTH   = 15

# Centipawn thresholds
BLUNDER_CP      = 200
MISTAKE_CP      = 100
INACCURACY_CP   = 50


@dataclass
class MoveAnalysis:
    """
    Result of analysing one move with Stockfish.

    Attributes:
        uci             The move that was played (UCI string).
        best_move_uci   The engine's best move for this position.
        cp_loss         Centipawn loss vs best move (0 = best move played).
        classification  'best' | 'good' | 'inaccuracy' | 'mistake' | 'blunder' | 'unknown'
        tactical_hints  Set of engine-detected tactical motifs, e.g.
                        {'missed_mate', 'missed_check', 'hanging_piece'}.
        available       False when Stockfish could not be reached.
    """
    uci:            str
    best_move_uci:  str | None      = None
    cp_loss:        int             = 0
    classification: str             = "unknown"
    tactical_hints: set             = field(default_factory=set)
    available:      bool            = True


class EngineAnalyzer:
    """
    Thin wrapper around python-chess's SimpleEngine (UCI protocol).

    Provides per-move centipawn loss, best-move comparison, and a set of
    tactical hints.  Falls back gracefully to a stub MoveAnalysis when
    Stockfish is not installed.

    Usage:
        analyzer = EngineAnalyzer()                 # auto-detects Stockfish
        analyzer = EngineAnalyzer(path="stockfish") # explicit path
        analyzer = EngineAnalyzer(depth=18)         # deeper search

        analysis = analyzer.analyze_move(board, move)
        print(analysis.classification, analysis.cp_loss)

        analyzer.close()
    """

    def __init__(self,
                 path:  str = STOCKFISH_PATH,
                 depth: int = DEFAULT_DEPTH):
        self._depth  = depth
        self._engine = None
        self._ok     = False
        self._try_init(path)

    # ------------------------------------------------------------------
    # Init / teardown
    # ------------------------------------------------------------------

    def _try_init(self, path: str) -> None:
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(path)
            self._ok     = True
            print(f"EngineAnalyzer: Stockfish ready (path='{path}', depth={self._depth})")
        except Exception as e:
            print(
                f"EngineAnalyzer: Stockfish unavailable ({e}). "
                "Phase C will fall back to heuristic skill tagging.\n"
                "To enable engine analysis, install Stockfish and set "
                "STOCKFISH_PATH in your .env file."
            )

    def close(self) -> None:
        if self._engine:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None
            self._ok     = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_move(self,
                     board: chess.Board,
                     move:  chess.Move) -> MoveAnalysis:
        """
        Analyse the quality of `move` in `board` (position before move).

        Returns a MoveAnalysis.  If Stockfish is unavailable, returns a
        stub with available=False so callers can fall back gracefully.
        """
        if not self._ok or self._engine is None:
            return MoveAnalysis(uci=move.uci(), available=False,
                                classification="unknown")

        try:
            return self._analyse(board, move)
        except Exception as e:
            print(f"EngineAnalyzer.analyze_move error: {e}")
            return MoveAnalysis(uci=move.uci(), available=False,
                                classification="unknown")

    # ------------------------------------------------------------------
    # Internal analysis
    # ------------------------------------------------------------------

    def _analyse(self,
                 board: chess.Board,
                 move:  chess.Move) -> MoveAnalysis:
        limit = chess.engine.Limit(depth=self._depth)

        # Score before the move (with best move)
        info_pre = self._engine.analyse(board, limit, multipv=1)
        best_move_obj = info_pre.get("pv", [None])[0]
        best_move_uci = best_move_obj.uci() if best_move_obj else None

        score_pre = info_pre["score"].white()
        cp_pre    = self._score_to_cp(score_pre)

        # Score after the move
        board_after = board.copy()
        board_after.push(move)
        info_post = self._engine.analyse(board_after, limit)
        score_post = info_post["score"].white()
        cp_post    = self._score_to_cp(score_post)

        # CP loss from the moving player's perspective
        if board.turn == chess.WHITE:
            cp_loss = cp_pre - cp_post
        else:
            cp_loss = cp_post - cp_pre
        cp_loss = max(0, cp_loss)

        # Classification
        classification = self._classify(cp_loss)

        # Tactical hints
        hints = self._tactical_hints(board, move, board_after,
                                     best_move_obj, info_pre)

        return MoveAnalysis(
            uci            = move.uci(),
            best_move_uci  = best_move_uci,
            cp_loss        = cp_loss,
            classification = classification,
            tactical_hints = hints,
            available      = True,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_cp(score: chess.engine.PovScore) -> int:
        if score.is_mate():
            return 10000 if (score.mate() or 0) > 0 else -10000
        return score.score() or 0

    @staticmethod
    def _classify(cp_loss: int) -> str:
        if cp_loss >= BLUNDER_CP:
            return "blunder"
        if cp_loss >= MISTAKE_CP:
            return "mistake"
        if cp_loss >= INACCURACY_CP:
            return "inaccuracy"
        if cp_loss <= 10:
            return "best"
        return "good"

    @staticmethod
    def _tactical_hints(board:        chess.Board,
                         move:         chess.Move,
                         board_after:  chess.Board,
                         best_move:    chess.Move | None,
                         info_pre:     dict) -> set:
        hints = set()

        # Missed forced mate
        score_pre = info_pre["score"].white()
        if score_pre.is_mate() and (score_pre.mate() or 0) > 0:
            if best_move and move != best_move:
                hints.add("missed_mate")

        # Missed check (best move gave check, player didn't)
        if best_move:
            test = board.copy()
            test.push(best_move)
            if test.is_check() and not board_after.is_check():
                hints.add("missed_check")

        # Hanging piece left en prise
        mover = board.turn
        if board_after.is_attacked_by(not mover, move.to_square):
            if not board_after.is_attacked_by(mover, move.to_square):
                p = board.piece_at(move.from_square)
                if p and p.piece_type in {chess.QUEEN, chess.ROOK,
                                           chess.BISHOP, chess.KNIGHT}:
                    hints.add("hanging_piece")

        return hints
