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

        pawn_tags = self._pawn_structure_tags(board, move, board_after)
        if pawn_tags:
            tags.update(pawn_tags)
            tags.add("Pawn_structure")

        # ── Blunder / mistake / inaccuracy ────────────────────────────

        blunder_tag = self._classify_quality(board, move, board_after, analysis)
        if blunder_tag:
            tags.add(blunder_tag)

        # ── Piece activity (only when nothing more specific fires) ────

        activity_tags = self._piece_activity_tags(board, move, board_after)
        if activity_tags:
            tags.update(activity_tags)
            tags.add("Piece_activity")
        elif self._is_piece_activity(board, move, board_after, tags):
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

    def _classify_quality(self, board, move, board_after, analysis):
        if analysis and analysis.available:
            mapping = {"blunder": "Blunder", "mistake": "Mistake", "inaccuracy": "Inaccuracy"}
            return mapping.get(analysis.classification)

        # Heuristic fallback when Stockfish is unavailable
        if self._is_heuristic_blunder(board, move, board_after):
            return "Blunder"

        # Mistake: moving queen or rook onto a pawn-attacked square
        piece = board.piece_at(move.from_square)
        if piece and piece.piece_type in {chess.QUEEN, chess.ROOK}:
            mover = board.turn
            for sq in board_after.pieces(chess.PAWN, not mover):
                if move.to_square in board_after.attacks(sq):
                    return "Mistake"

        # Inaccuracy: moving same piece twice in opening (tempo loss)
        if board.fullmove_number <= 10 and len(board.move_stack) >= 2:
            prev = board.peek()
            if move.from_square == prev.to_square:
                from_rank = chess.square_rank(move.from_square)
                to_rank = chess.square_rank(move.to_square)
                going_back = (
                    (board.turn == chess.WHITE and to_rank < from_rank) or
                    (board.turn == chess.BLACK and to_rank > from_rank)
                )
                if going_back:
                    return "Inaccuracy"

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

    def _pawn_structure_tags(self,
                             board: chess.Board,
                             move: chess.Move,
                             board_after: chess.Board) -> set[str]:
        piece = board.piece_at(move.from_square)
        if not piece or piece.piece_type != chess.PAWN:
            return set()

        tags = set()
        mover = board.turn

        if self._is_passed_pawn(board_after, move.to_square, mover):
            tags.add("Passed_pawn")

        if self._is_doubled_pawn(board_after, move.to_square, mover):
            tags.add("Doubled_pawns")

        if self._is_isolated_pawn(board_after, move.to_square, mover):
            tags.add("Isolated_pawn")

        return tags

    @staticmethod
    def _is_passed_pawn(board: chess.Board,
                        sq: chess.Square,
                        color: chess.Color) -> bool:
        piece = board.piece_at(sq)
        if not piece or piece.piece_type != chess.PAWN or piece.color != color:
            return False

        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)
        adjacent_files = {f for f in (file_idx - 1, file_idx, file_idx + 1)
                          if 0 <= f <= 7}
        for opp_sq in board.pieces(chess.PAWN, not color):
            if chess.square_file(opp_sq) not in adjacent_files:
                continue
            opp_rank = chess.square_rank(opp_sq)
            if color == chess.WHITE and opp_rank > rank_idx:
                return False
            if color == chess.BLACK and opp_rank < rank_idx:
                return False
        return True

    @staticmethod
    def _is_doubled_pawn(board: chess.Board,
                         sq: chess.Square,
                         color: chess.Color) -> bool:
        file_idx = chess.square_file(sq)
        pawns_on_file = [
            pawn_sq for pawn_sq in board.pieces(chess.PAWN, color)
            if chess.square_file(pawn_sq) == file_idx
        ]
        return len(pawns_on_file) >= 2

    @staticmethod
    def _is_isolated_pawn(board: chess.Board,
                          sq: chess.Square,
                          color: chess.Color) -> bool:
        file_idx = chess.square_file(sq)
        adjacent_files = {file_idx - 1, file_idx + 1}
        for pawn_sq in board.pieces(chess.PAWN, color):
            if pawn_sq == sq:
                continue
            if chess.square_file(pawn_sq) in adjacent_files:
                return False
        return True

    # ------------------------------------------------------------------
    # Piece activity detector
    # ------------------------------------------------------------------

    def _piece_activity_tags(self,
                             board: chess.Board,
                             move: chess.Move,
                             board_after: chess.Board) -> set[str]:
        piece = board.piece_at(move.from_square)
        if not piece:
            return set()

        tags = set()

        if piece.piece_type in {chess.KNIGHT, chess.BISHOP}:
            back_rank = 0 if board.turn == chess.WHITE else 7
            if chess.square_rank(move.from_square) == back_rank:
                tags.add("Development")

        central = {chess.C3, chess.C6, chess.D4, chess.D5,
                   chess.E4, chess.E5, chess.F3, chess.F6}
        if piece.piece_type != chess.PAWN and move.to_square in central:
            tags.add("Centralization")

        if piece.piece_type in {chess.KNIGHT, chess.BISHOP}:
            if self._is_outpost_square(board_after, move.to_square, board.turn):
                tags.add("Outpost_control")

        if piece.piece_type == chess.ROOK:
            file_idx = chess.square_file(move.to_square)
            pawns_on_file = [
                sq for sq in chess.SQUARES
                if chess.square_file(sq) == file_idx
                and (p := board_after.piece_at(sq))
                and p.piece_type == chess.PAWN
            ]
            if not pawns_on_file:
                tags.add("Open_file_rook")

        return tags

    def _is_outpost_square(self,
                           board: chess.Board,
                           sq: chess.Square,
                           color: chess.Color) -> bool:
        rank = chess.square_rank(sq)
        if color == chess.WHITE and rank < 4:
            return False
        if color == chess.BLACK and rank > 3:
            return False
        return (
            self._is_attacked_by_pawn(board, color, sq)
            and not self._is_attacked_by_pawn(board, not color, sq)
        )

    @staticmethod
    def _is_attacked_by_pawn(board: chess.Board,
                             color: chess.Color,
                             sq: chess.Square) -> bool:
        for pawn_sq in board.pieces(chess.PAWN, color):
            if sq in board.attacks(pawn_sq):
                return True
        return False

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
