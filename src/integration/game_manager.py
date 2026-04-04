import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chess
import requests
import threading
from src.graph.skill_tree import SkillTree

BOT_API_URL = "http://127.0.0.1:8087"


class GameManager:
    """
    Central integration of Phase A + B + C.
    Manages a full chess game session:
      - Tracks board state
      - Calls bot API for bot moves (Phase B)
      - Records skill performance in Neo4j (Phase C)
      - Accepts moves from VisionLoop (Phase A) via start_vision_thread()
        OR from manual UCI input via player_move()

    Phase A integration:
        manager = GameManager(player_id="adam", player_elo=1400)
        manager.start_game()
        manager.start_vision_thread()   # starts camera in background
        # camera moves now flow automatically into the game
    """

    def __init__(self, player_id: str, player_elo: int = 1400):
        self.player_id  = player_id
        self.player_elo = player_elo
        self.board      = chess.Board()
        self.move_count = 0
        self.pgn_moves  = []
        self.history    = []
        self.game_id    = None
        self.status     = "not_started"

        self._vision_thread: threading.Thread | None = None
        self._vision_loop   = None  # set by start_vision_thread
        self._vision_lock   = threading.Lock()

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

        print(
            f"GameManager ready — player: {player_id} "
            f"(Elo {player_elo}), bot bracket: {self.bot_bracket}"
        )

    # ------------------------------------------------------------------
    # Game lifecycle
    # ------------------------------------------------------------------

    def start_game(self) -> dict:
        """Start a new game. Returns full state dict."""
        self.board      = chess.Board()
        self.move_count = 0
        self.pgn_moves  = []
        self.history    = []
        self.status     = "in_progress"
        self.game_id    = self.skill_tree.start_game(
            self.player_id, self.player_elo, self.bot_bracket
        )
        print(f"Game started — ID: {self.game_id}")
        return self._state()

    def player_move(self, uci: str) -> dict:
        """
        Process a player move from manual input OR from VisionLoop.
        Thread-safe via _vision_lock.
        Returns game state dict.
        """
        with self._vision_lock:
            return self._apply_player_move(uci)

    def _apply_player_move(self, uci: str) -> dict:
        """Internal (already under lock)."""
        if self.status != "in_progress":
            return {"error": "Game not in progress"}

        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return {"error": f"Invalid UCI: {uci}"}

        if move not in self.board.legal_moves:
            return {"error": f"Illegal move: {uci}"}

        # Record in Phase C (now returns dict with cp_loss, move_class)
        self.move_count += 1
        board_before = self.board.copy()
        analysis = self.skill_tree.record_player_move(
            game_id=self.game_id,
            player_id=self.player_id,
            move_number=self.move_count,
            move=move,
            board_before=board_before,
        )
        skills     = analysis["skills"]
        cp_loss    = analysis["cp_loss"]
        move_class = analysis["move_class"]

        # Push player move
        san = self.board.san(move)
        self.board.push(move)
        self.pgn_moves.append(san)

        print(
            f"  Player move {self.move_count}: {uci} ({san}) "
            f"— {move_class} (cp_loss={cp_loss}) — skills: {skills}"
        )

        if self.board.is_game_over():
            self.history.append({
                "move_number": len(self.history) + 1,
                "player_move": uci,
                "bot_move":    None,
                "skills":      skills,
                "move_class":  move_class,
                "cp_loss":     cp_loss,
            })
            return self._finish_game(
                last_player_move=uci,
                last_bot_move=None,
                skills_detected=skills,
                move_class=move_class,
                cp_loss=cp_loss,
            )

        # Get bot response
        bot_result = self._get_bot_move()
        if "error" in bot_result:
            return bot_result

        self.history.append({
            "move_number": len(self.history) + 1,
            "player_move": uci,
            "bot_move":    bot_result.get("uci"),
            "skills":      skills,
            "move_class":  move_class,
            "cp_loss":     cp_loss,
        })

        if bot_result.get("game_over"):
            return self._finish_game(
                last_player_move=uci,
                last_bot_move=bot_result.get("uci"),
                skills_detected=skills,
                move_class=move_class,
                cp_loss=cp_loss,
            )

        return self._state(
            last_player_move=uci,
            last_bot_move=bot_result.get("uci"),
            skills_detected=skills,
            move_class=move_class,
            cp_loss=cp_loss,
        )

    # ------------------------------------------------------------------
    # Phase A integration — VisionLoop in background thread
    # ------------------------------------------------------------------

    def start_vision_thread(self, camera_index: int = 0) -> None:
        """
        Start Phase A (camera pipeline) in a background thread.
        Detected moves are automatically forwarded to player_move().

        Call this AFTER start_game().
        """
        if self._vision_thread and self._vision_thread.is_alive():
            print("Vision thread already running.")
            return

        # Import here to avoid breaking the class when OpenCV is missing
        from src.vision.vision_loop1 import VisionLoop

        def _vision_callback(uci: str):
            """Called by VisionLoop when a move is detected."""
            print(f"[VisionLoop] Move detected: {uci}")
            result = self.player_move(uci)
            if result.get("error"):
                print(f"[VisionLoop] Move rejected: {result['error']}")
            elif result.get("game_over"):
                print(f"[VisionLoop] Game over — stopping vision thread")
                if self._vision_loop:
                    self._vision_loop.stop()

        loop = VisionLoop(
            camera_index=camera_index,
            show_preview=True,
            on_move_detected=_vision_callback,  # NEW callback
        )
        self._vision_loop = loop

        self._vision_thread = threading.Thread(
            target=loop.run,
            daemon=True,
            name="vision-loop",
        )
        self._vision_thread.start()
        print(f"Vision thread started (camera {camera_index}).")

    def stop_vision_thread(self) -> None:
        """Gracefully stop the vision thread."""
        if self._vision_loop:
            self._vision_loop.stop()
        if self._vision_thread:
            self._vision_thread.join(timeout=3)
        print("Vision thread stopped.")

    # ------------------------------------------------------------------
    # Bot
    # ------------------------------------------------------------------

    def _get_bot_move(self) -> dict:
        """Call Phase B API for bot move."""
        try:
            resp = requests.post(
                f"{BOT_API_URL}/move",
                json={
                    "fen":         self.board.fen(),
                    "elo":         self.player_elo,
                    "temperature": 1.1,
                },
                timeout=10,
            )
            resp.raise_for_status()
            result  = resp.json()
            bot_uci = result["uci"]

            bot_move = chess.Move.from_uci(bot_uci)
            if bot_move not in self.board.legal_moves:
                bot_move = next(iter(self.board.legal_moves))
                bot_uci  = bot_move.uci()

            bot_san = self.board.san(bot_move)
            self.board.push(bot_move)
            self.pgn_moves.append(bot_san)
            self.move_count += 1

            print(f"  Bot move {self.move_count}: {bot_uci} ({bot_san})")

            return {
                "uci":      bot_uci,
                "san":      bot_san,
                "game_over": self.board.is_game_over(),
            }

        except Exception as e:
            print(f"  Bot API error: {e}")
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Game over
    # ------------------------------------------------------------------

    def _finish_game(
        self,
        last_player_move: str = None,
        last_bot_move:    str = None,
        skills_detected:  list = None,
        move_class:       str = "unknown",
        cp_loss:          int | None = None,
    ) -> dict:
        outcome = self.board.outcome()
        if outcome is None:
            result = "draw"
        elif outcome.winner == chess.WHITE:
            result = "win"
        else:
            result = "loss"

        self.status = "finished"
        self.skill_tree.db.finish_game(self.game_id, result, self.move_count)
        print(f"Game over — result: {result}")
        return self._state(
            last_player_move=last_player_move,
            last_bot_move=last_bot_move,
            skills_detected=skills_detected,
            move_class=move_class,
            cp_loss=cp_loss,
            game_over=True,
            result=result,
        )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_skill_summary(self) -> dict:
        return self.skill_tree.get_skill_summary(self.player_id)

    def _state(
        self,
        last_player_move: str       = None,
        last_bot_move:    str       = None,
        skills_detected:  list      = None,
        move_class:       str       = "unknown",
        cp_loss:          int | None = None,
        game_over:        bool      = False,
        result:           str       = None,
    ) -> dict:
        return {
            "game_id":          self.game_id,
            "fen":              self.board.fen(),
            "pgn":              " ".join(self.pgn_moves),
            "move_count":       self.move_count,
            "turn":             "white" if self.board.turn else "black",
            "last_player_move": last_player_move,
            "last_bot_move":    last_bot_move,
            "skills_detected":  skills_detected or [],
            "move_class":       move_class,
            "cp_loss":          cp_loss,
            "game_over":        game_over,
            "result":           result,
            "status":           self.status,
            "last_entry": {
                "skills":      skills_detected or [],
                "player_move": last_player_move,
                "bot_move":    last_bot_move,
                "move_class":  move_class,
                "cp_loss":     cp_loss,
            } if last_player_move else None,
            "history": self.history,
            "zpd":     self.skill_tree.get_zpd_recommendations(self.player_id),
        }

    def get_state(self) -> dict:
        return self._state()

    def close(self):
        self.stop_vision_thread()
        self.skill_tree.close()
