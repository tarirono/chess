import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chess
import requests
from datetime import datetime
from src.graph.skill_tree import SkillTree

BOT_API_URL = "http://127.0.0.1:8087"


class GameManager:
    """
    Central integration of Phase A + B + C.
    Manages a full chess game session:
      - Tracks board state
      - Calls bot API for bot moves (Phase B)
      - Records skill performance in Neo4j (Phase C)
      - Ready to receive moves from VisionLoop (Phase A)
    """

    def __init__(self, player_id: str, player_elo: int = 1400):
        self.player_id  = player_id
        self.player_elo = player_elo
        self.board      = chess.Board()
        self.move_count = 0
        self.pgn_moves  = []
        self.game_id    = None
        self.status     = "not_started"

        # Phase C
        self.skill_tree = SkillTree()
        self.skill_tree.get_or_create_player(player_id, player_elo)

        # Determine bot bracket
        if player_elo < 1300:
            self.bot_bracket = "1200"
        elif player_elo < 1500:
            self.bot_bracket = "1400"
        else:
            self.bot_bracket = "1600"

        print(f"GameManager ready — player: {player_id} "
              f"(Elo {player_elo}), bot bracket: {self.bot_bracket}")

    def start_game(self) -> dict:
        """Start a new game. Returns full state dict."""
        self.board      = chess.Board()
        self.move_count = 0
        self.pgn_moves  = []
        self.status     = "in_progress"
        self.game_id    = self.skill_tree.start_game(
            self.player_id, self.player_elo, self.bot_bracket
        )
        print(f"Game started — ID: {self.game_id}")
        return self._state()

    def player_move(self, uci: str) -> dict:
        """
        Process a player move (from camera or manual input).
        Returns game state dict.
        """
        if self.status != "in_progress":
            return {"error": "Game not in progress"}

        # Validate move
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return {"error": f"Invalid UCI: {uci}"}

        if move not in self.board.legal_moves:
            return {"error": f"Illegal move: {uci}"}

        # Record in Phase C
        self.move_count += 1
        board_before = self.board.copy()
        skills = self.skill_tree.record_player_move(
            game_id=self.game_id,
            player_id=self.player_id,
            move_number=self.move_count,
            move=move,
            board_before=board_before
        )

        # Push move to board
        san = self.board.san(move)
        self.board.push(move)
        self.pgn_moves.append(san)

        print(f"  Player move {self.move_count}: {uci} ({san}) "
              f"— skills: {skills}")

        # Check game over
        if self.board.is_game_over():
            return self._finish_game()

        # Get bot response (Phase B)
        bot_result = self._get_bot_move()
        if "error" in bot_result:
            return bot_result

        return self._state(
            last_player_move=uci,
            last_bot_move=bot_result.get("uci"),
            skills_detected=skills
        )

    def _get_bot_move(self) -> dict:
        """Call Phase B API for bot move."""
        try:
            resp = requests.post(
                f"{BOT_API_URL}/move",
                json={
                    "fen":         self.board.fen(),
                    "elo":         self.player_elo,
                    "temperature": 1.1  # slight randomness
                },
                timeout=10
            )
            resp.raise_for_status()
            result = resp.json()
            bot_uci = result["uci"]

            # Validate and push bot move
            bot_move = chess.Move.from_uci(bot_uci)
            if bot_move not in self.board.legal_moves:
                # Fallback to first legal move
                bot_move = next(iter(self.board.legal_moves))
                bot_uci  = bot_move.uci()

            bot_san = self.board.san(bot_move)
            self.board.push(bot_move)
            self.pgn_moves.append(bot_san)
            self.move_count += 1

            print(f"  Bot move {self.move_count}: {bot_uci} ({bot_san})")

            if self.board.is_game_over():
                return self._finish_game()

            return {"uci": bot_uci, "san": bot_san}

        except Exception as e:
            print(f"  Bot API error: {e}")
            return {"error": str(e)}

    def _finish_game(self) -> dict:
        """Handle game over."""
        outcome = self.board.outcome()
        if outcome is None:
            result = "draw"
        elif outcome.winner == chess.WHITE:
            result = "win"
        else:
            result = "loss"

        self.status = "finished"
        self.skill_tree.db.finish_game(
            self.game_id, result, self.move_count
        )
        print(f"Game over — result: {result}")
        return self._state(game_over=True, result=result)

    def get_skill_summary(self) -> dict:
        """Get ZPD skill summary for this player."""
        return self.skill_tree.get_skill_summary(self.player_id)

    def _state(self, last_player_move: str = None,last_bot_move: str = None,skills_detected: list = None,game_over: bool = False,result: str = None) -> dict:
        """Build current game state dict."""
        
        return {
            "game_id":          self.game_id,
            "fen":              self.board.fen(),
            "pgn":              " ".join(self.pgn_moves),
            "move_count":       self.move_count,
            "turn":             "white" if self.board.turn else "black",
            "last_player_move": last_player_move,
            "last_bot_move":    last_bot_move,
            "skills_detected":  skills_detected or [],
            "game_over":        game_over,
            "result":           result,
            "status":           self.status,
            # Frontend expects these keys
            "last_entry": {
                "skills":      skills_detected or [],
                "player_move": last_player_move,
                "bot_move":    last_bot_move,
            } if last_player_move else None,
            "history":    [],
            "zpd":        [],
        }

    def get_state(self) -> dict:
        return self._state()

    def close(self):
        self.skill_tree.close()