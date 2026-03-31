import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chess


class SkillTagger:
    """
    Detects chess tactical concepts in a position using python-chess.
    No Stockfish required — pure pattern recognition.
    """

    def tag_position(self, board: chess.Board,
                     move: chess.Move) -> list[str]:
        """
        Given a board state and the move played, return a list of
        skill concepts present in this position.
        """
        tags = []

        if self._is_fork(board, move):
            tags.append("Fork")
        if self._is_pin(board, move):
            tags.append("Pin")
        if self._is_discovery(board, move):
            tags.append("Discovery")
        if self._is_skewer(board, move):
            tags.append("Skewer")
        if self._is_checkmate_pattern(board, move):
            tags.append("Checkmate_pattern")
        if self._is_endgame(board):
            tags.append("Endgame")
        if self._is_opening(board):
            tags.append("Opening")
        if self._is_blunder(board, move):
            tags.append("Blunder")

        if not tags:
            tags.append("Piece_activity")

        return tags

    def _is_fork(self, board: chess.Board, move: chess.Move) -> bool:
        """A move attacks two or more valuable pieces simultaneously."""
        test = board.copy()
        test.push(move)
        piece = test.piece_at(move.to_square)
        if not piece:
            return False
        attacks = test.attacks(move.to_square)
        valuable = {chess.QUEEN, chess.ROOK, chess.KING}
        attacked_valuable = sum(
            1 for sq in attacks
            if test.piece_at(sq) and
            test.piece_at(sq).color != piece.color and
            test.piece_at(sq).piece_type in valuable
        )
        return attacked_valuable >= 2

    def _is_pin(self, board: chess.Board, move: chess.Move) -> bool:
        """A piece is pinned to the king after the move."""
        test = board.copy()
        test.push(move)
        # Check if any opponent piece is pinned
        opponent = not board.turn
        king_sq = test.king(opponent)
        if king_sq is None:
            return False
        for sq in chess.SQUARES:
            piece = test.piece_at(sq)
            if piece and piece.color == opponent:
                if test.is_pinned(opponent, sq):
                    return True
        return False

    def _is_discovery(self, board: chess.Board,
                       move: chess.Move) -> bool:
        """Moving a piece reveals an attack from a piece behind it."""
        test = board.copy()
        # Check if after moving, a sliding piece behind gains new attacks
        piece = board.piece_at(move.from_square)
        if not piece:
            return False
        test.push(move)
        # See if any of our sliding pieces now attack the opponent king
        opponent = not board.turn
        king_sq = test.king(opponent)
        if king_sq is None:
            return False
        for sq in chess.SQUARES:
            p = test.piece_at(sq)
            if p and p.color == board.turn and sq != move.to_square:
                if p.piece_type in {chess.BISHOP, chess.ROOK, chess.QUEEN}:
                    if king_sq in test.attacks(sq):
                        return True
        return False

    def _is_skewer(self, board: chess.Board, move: chess.Move) -> bool:
        """A valuable piece is attacked and a lesser piece is behind it."""
        test = board.copy()
        test.push(move)
        piece = test.piece_at(move.to_square)
        if not piece:
            return False
        if piece.piece_type not in {chess.BISHOP, chess.ROOK, chess.QUEEN}:
            return False
        attacks = test.attacks(move.to_square)
        for sq in attacks:
            target = test.piece_at(sq)
            if target and target.color != piece.color:
                if target.piece_type in {chess.QUEEN, chess.KING}:
                    return True
        return False

    def _is_checkmate_pattern(self, board: chess.Board,
                               move: chess.Move) -> bool:
        """The move gives check or is near a mating pattern."""
        test = board.copy()
        test.push(move)
        return test.is_check() or test.is_checkmate()

    def _is_endgame(self, board: chess.Board) -> bool:
        """Endgame: few pieces remain on the board."""
        pieces = len(board.piece_map())
        return pieces <= 12

    def _is_opening(self, board: chess.Board) -> bool:
        """Opening: first 10 moves."""
        return board.fullmove_number <= 10

    def _is_blunder(self, board: chess.Board,
                     move: chess.Move) -> bool:
        """A blunder: hangs a piece with no compensation."""
        test = board.copy()
        piece_moved = board.piece_at(move.from_square)
        if not piece_moved:
            return False
        test.push(move)
        # If the piece we just moved is now attacked and not defended
        if test.is_attacked_by(not board.turn, move.to_square):
            if not test.is_attacked_by(board.turn, move.to_square):
                if piece_moved.piece_type in {
                    chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT
                }:
                    return True
        return False