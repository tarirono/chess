import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chess
from src.graph.engine_analyzer import MoveAnalysis


class SkillTagger:
    """
    Detects chess tactical concepts in a position.

    When a MoveAnalysis object is supplied (from EngineAnalyzer), engine
    centipawn data is used directly for blunder classification and to
    confirm tactical patterns.  Falls back to pure python-chess heuristics
    when no analysis is available (e.g. during tests without Stockfish).
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag_position(self,
                     board: chess.Board,
                     move: chess.Move,
                     analysis: MoveAnalysis | None = None) -> list[str]:
        """
        Return a list of skill concepts present in this position.

        Args:
            board:    Board state BEFORE the move.
            move:     The move played.
            analysis: Optional Stockfish MoveAnalysis.  When supplied the
                      blunder / mistake / inaccuracy classification is
                      engine-backed; otherwise pure heuristics are used.
        """
        tags = set()
        board_after = board.copy()
        board_after.push(move)

        # ── Tactical patterns ─────────────────────────────────────────

        if self._is_fork(board, move, board_after):
            tags.add("Fork")

        if self._is_pin(board, move, board_after):
            tags.add("Pin")

        if self._is_discovery(board, move, board_after):
            tags.add("Discovery")

        if self._is_skewer(board, move, board_after):
            tags.add("Skewer")

        if board_after.is_check() or board_after.is_checkmate():
            tags.add("Checkmate_pattern")

        # ── Game-phase tags ───────────────────────────────────────────

        if self._is_endgame(board):
            tags.add("Endgame")

        if self._is_opening(board):
            tags.add("Opening")

        # ── Pawn structure ────────────────────────────────────────────

        if self._involves_pawn_structure(board, move, board_after):
            tags.add("Pawn_structure")

        # ── Blunder / mistake / inaccuracy ────────────────────────────

        blunder_tag = self._classify_quality(board, move, board_after, analysis)
        if blunder_tag:
            tags.add(blunder_tag)

        # ── Piece activity (only when nothing more specific fires) ────

        if self._is_piece_activity(board, move, board_after, tags):
            tags.add("Piece_activity")

        # ── Missed tactics (engine only) ──────────────────────────────

        if analysis:
            if "missed_mate" in analysis.tactical_hints:
                tags.add("Checkmate_pattern")   # missed a forced mate
            if "missed_check" in analysis.tactical_hints and not tags & {
                "Fork", "Pin", "Skewer", "Discovery"
            }:
                tags.add("Piece_activity")      # at least missed an active move

        return sorted(tags)

    # ------------------------------------------------------------------
    # Quality classification
    # ------------------------------------------------------------------

    def _classify_quality(self,
                          board: chess.Board,
                          move: chess.Move,
                          board_after: chess.Board,
                          analysis: MoveAnalysis | None) -> str | None:
        """
        Return "Blunder", "Mistake", "Inaccuracy", or None.
        Engine-backed when analysis is present, heuristic otherwise.
        """
        if analysis:
            mapping = {
                "blunder":    "Blunder",
                "mistake":    "Mistake",
                "inaccuracy": "Inaccuracy",
            }
            return mapping.get(analysis.classification)

        # ── Heuristic fallback ────────────────────────────────────────
        if self._is_heuristic_blunder(board, move, board_after):
            return "Blunder"
        return None

    # ------------------------------------------------------------------
    # Tactical pattern detectors
    # ------------------------------------------------------------------

    def _is_fork(self,
                 board: chess.Board,
                 move: chess.Move,
                 board_after: chess.Board) -> bool:
        """Moved piece now attacks ≥2 valuable enemy pieces."""
        piece = board_after.piece_at(move.to_square)
        if not piece:
            return False
        mover = board.turn
        valuable = {chess.QUEEN, chess.ROOK, chess.KING, chess.BISHOP, chess.KNIGHT}
        attacks = board_after.attacks(move.to_square)
        attacked = sum(
            1 for sq in attacks
            if (p := board_after.piece_at(sq))
            and p.color != mover
            and p.piece_type in valuable
        )
        return attacked >= 2

    def _is_pin(self,
                board: chess.Board,
                move: chess.Move,
                board_after: chess.Board) -> bool:
        """A move creates or exploits an absolute pin on the opponent."""
        opponent = not board.turn
        king_sq = board_after.king(opponent)
        if king_sq is None:
            return False
        for sq in chess.SQUARES:
            piece = board_after.piece_at(sq)
            if piece and piece.color == opponent and sq != king_sq:
                if board_after.is_pinned(opponent, sq):
                    return True
        return False

    def _is_discovery(self,
                      board: chess.Board,
                      move: chess.Move,
                      board_after: chess.Board) -> bool:
        """Moving a piece reveals a sliding piece attack on the enemy king."""
        opponent = not board.turn
        king_sq = board_after.king(opponent)
        if king_sq is None:
            return False
        mover = board.turn
        for sq in chess.SQUARES:
            p = board_after.piece_at(sq)
            if (p and p.color == mover and sq != move.to_square
                    and p.piece_type in {chess.BISHOP, chess.ROOK, chess.QUEEN}):
                if king_sq in board_after.attacks(sq):
                    # Make sure this line was BLOCKED before the move
                    if not (king_sq in board.attacks(sq)):
                        return True
        return False

    def _is_skewer(self,
                   board: chess.Board,
                   move: chess.Move,
                   board_after: chess.Board) -> bool:
        """Piece attacks a valuable piece with a lesser piece sitting behind it."""
        piece = board_after.piece_at(move.to_square)
        if not piece:
            return False
        if piece.piece_type not in {chess.BISHOP, chess.ROOK, chess.QUEEN}:
            return False
        mover = board.turn
        attacks = board_after.attacks(move.to_square)
        for sq in attacks:
            target = board_after.piece_at(sq)
            if target and target.color != mover:
                if target.piece_type in {chess.QUEEN, chess.KING}:
                    return True
        return False

    def _is_heuristic_blunder(self,
                               board: chess.Board,
                               move: chess.Move,
                               board_after: chess.Board) -> bool:
        """Hangs a piece with no recapture (pure pattern, no engine)."""
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
        mover = board.turn
        if piece.piece_type not in {
            chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT
        }:
            return False
        attacked = board_after.is_attacked_by(not mover, move.to_square)
        defended = board_after.is_attacked_by(mover, move.to_square)
        return attacked and not defended

    # ------------------------------------------------------------------
    # Game-phase detectors
    # ------------------------------------------------------------------

    @staticmethod
    def _is_endgame(board: chess.Board) -> bool:
        """Endgame: ≤12 pieces remain or queens are off the board."""
        pieces = board.piece_map()
        if len(pieces) <= 12:
            return True
        queens = [p for p in pieces.values() if p.piece_type == chess.QUEEN]
        return len(queens) == 0

    @staticmethod
    def _is_opening(board: chess.Board) -> bool:
        """
        Opening: first 6 full moves AND enough pieces still on the board
        (prevents isolated FEN positions or endgames from being mis-tagged).
        """
        if board.fullmove_number > 6:
            return False
        # Don't tag as opening if there are ≤12 pieces (looks like an endgame FEN)
        if len(board.piece_map()) <= 12:
            return False
        return True

    # ------------------------------------------------------------------
    # Pawn structure detector
    # ------------------------------------------------------------------

    @staticmethod
    def _involves_pawn_structure(board: chess.Board,
                                 move: chess.Move,
                                 board_after: chess.Board) -> bool:
        """
        True when the move meaningfully changes pawn structure:
        pawn advance, capture creating passed/isolated/doubled pawns,
        or en-passant.
        """
        piece = board.piece_at(move.from_square)
        if not piece or piece.piece_type != chess.PAWN:
            return False

        # Any pawn capture changes structure
        if board.is_capture(move):
            return True

        # En passant
        if board.is_en_passant(move):
            return True

        # Central pawn advance (d/e files)
        file = chess.square_file(move.to_square)
        if file in (3, 4):   # d=3, e=4
            return True

        # Pawn promotion
        if move.promotion:
            return True

        # Check if the advance creates a passed pawn
        mover = board.turn
        pawns_after = board_after.pieces(chess.PAWN, mover)
        opp_pawns   = board_after.pieces(chess.PAWN, not mover)

        def is_passed(sq: chess.Square, color: chess.Color) -> bool:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            adjacent_files = [f - 1, f, f + 1]
            if color == chess.WHITE:
                blocking_ranks = range(r + 1, 8)
            else:
                blocking_ranks = range(0, r)
            opp = board_after.pieces(chess.PAWN, not color)
            for opp_sq in opp:
                if chess.square_file(opp_sq) in adjacent_files:
                    if chess.square_rank(opp_sq) in blocking_ranks:
                        return False
            return True

        if is_passed(move.to_square, mover):
            return True

        return False

    # ------------------------------------------------------------------
    # Piece activity detector
    # ------------------------------------------------------------------

    @staticmethod
    def _is_piece_activity(board: chess.Board,
                            move: chess.Move,
                            board_after: chess.Board,
                            existing_tags: set) -> bool:
        """
        Fires only when no other specific tag was assigned.
        True for developing moves, centralisation, or active rook lifts.
        This prevents it from being a meaningless catch-all.
        """
        # Already have a meaningful tag
        if existing_tags - {"Piece_activity"}:
            return False

        piece = board.piece_at(move.from_square)
        if not piece:
            return False

        # Development: minor piece moving off back rank
        if piece.piece_type in {chess.KNIGHT, chess.BISHOP}:
            back_rank = 0 if board.turn == chess.WHITE else 7
            if chess.square_rank(move.from_square) == back_rank:
                return True

        # Centralisation: piece moving to central squares
        central = {chess.D4, chess.D5, chess.E4, chess.E5,
                   chess.C3, chess.C6, chess.F3, chess.F6}
        if move.to_square in central:
            return True

        # Rook moving to open file
        if piece.piece_type == chess.ROOK:
            file = chess.square_file(move.to_square)
            pawns_on_file = [
                sq for sq in chess.SQUARES
                if chess.square_file(sq) == file
                and board_after.piece_at(sq)
                and board_after.piece_at(sq).piece_type == chess.PAWN
            ]
            if not pawns_on_file:
                return True

        return False
