import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import chess
import requests
import threading
from typing import Callable
from dotenv import load_dotenv
from src.graph.skill_tree import SkillTree

load_dotenv()

# Read from .env so the URL is not hardcoded — change BOT_API_URL in .env
# to point at a remote host or a different port if needed.
BOT_API_URL = os.getenv("BOT_API_URL", "http://127.0.0.1:8087")

# temperature > 1.0 widens the softmax distribution over legal moves so the
# bot occasionally picks slightly sub-optimal moves — intentional human-variance
# simulation.  1.1 keeps the bot feeling natural without becoming random.
BOT_TEMPERATURE = float(os.getenv("BOT_TEMPERATURE", "1.1"))


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

    def __init__(self,
                 player_id: str,
                 player_elo: int = 1400,
                 state_callback: Callable[[dict], None] | None = None):
        self.player_id  = player_id
        self.player_elo = player_elo
        self.board      = chess.Board()
        self.move_count = 0
        self.pgn_moves  = []
        self.history    = []
        self.game_id    = None
        self.status     = "not_started"

        self._vision_thread: threading.Thread | None = None
        self._vision_loop   = None
        self._vision_lock   = threading.Lock()
        self._state_callback = state_callback

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
        self.bot_elo = player_elo

        print(
            f"GameManager ready — player: {player_id} "
            f"(Elo {player_elo}), bot bracket: {self.bot_bracket}\n"
            f"  Bot API: {BOT_API_URL}  temperature: {BOT_TEMPERATURE}"
        )

    # ------------------------------------------------------------------
    # Game lifecycle
    # ------------------------------------------------------------------

    def start_game(self) -> dict:
        self.board      = chess.Board()
        self.move_count = 0
        self.pgn_moves  = []
        self.history    = []
        self.status     = "in_progress"
        self.game_id    = self.skill_tree.start_game(
            self.player_id, self.player_elo, self.bot_bracket
        )
        print(f"Game started — ID: {self.game_id}")
        state = self._state()
        self._notify_state(state)
        return state

    def player_move(self, uci: str) -> dict:
        with self._vision_lock:
            return self._apply_player_move(uci)

    def _apply_player_move(self, uci: str) -> dict:
        if self.status != "in_progress":
            return {"error": "Game not in progress"}

        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return {"error": f"Invalid UCI: {uci}"}

        if move not in self.board.legal_moves:
            return {"error": f"Illegal move: {uci}"}

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

        state = self._state(
            last_player_move=uci,
            last_bot_move=bot_result.get("uci"),
            skills_detected=skills,
            move_class=move_class,
            cp_loss=cp_loss,
        )
        self._notify_state(state)
        return state

    # ------------------------------------------------------------------
    # Phase A — VisionLoop in background thread
    # ------------------------------------------------------------------

    def start_vision_thread(self, camera_index: int | None = None) -> None:
        """
        Start Phase A camera pipeline in a background thread.
        camera_index defaults to CAMERA_INDEX from .env (fallback: 0).
        """
        if self._vision_thread and self._vision_thread.is_alive():
            print("Vision thread already running.")
            return

        if camera_index is None:
            camera_index = int(os.getenv("CAMERA_INDEX", "0"))

        from src.vision.vision_loop import VisionLoop

        def _vision_callback(uci: str):
            print(f"[VisionLoop] Move detected: {uci}")
            result = self.player_move(uci)
            if result.get("error"):
                print(f"[VisionLoop] Move rejected: {result['error']}")
            elif result.get("game_over"):
                print("[VisionLoop] Game over — stopping vision thread")
                if self._vision_loop:
                    self._vision_loop.stop()

        loop = VisionLoop(
            camera_index=camera_index,
            show_preview=True,
            on_move_detected=_vision_callback,
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
        if self._vision_loop:
            self._vision_loop.stop()
        if self._vision_thread:
            self._vision_thread.join(timeout=3)
        print("Vision thread stopped.")

    def is_vision_running(self) -> bool:
        return bool(self._vision_thread and self._vision_thread.is_alive())

    # ------------------------------------------------------------------
    # Bot
    # ------------------------------------------------------------------

    def _refresh_bot_profile(self) -> tuple[int, str]:
        adaptive_elo, adaptive_bracket = self.skill_tree.recommend_bot_bracket(
            self.player_id, self.player_elo
        )
        self.bot_elo = adaptive_elo
        self.bot_bracket = adaptive_bracket
        return adaptive_elo, adaptive_bracket

    def _get_bot_move(self) -> dict:
        try:
            adaptive_elo, adaptive_bracket = self._refresh_bot_profile()
            resp = requests.post(
                f"{BOT_API_URL}/move",
                json={
                    "fen":         self.board.fen(),
                    "elo":         adaptive_elo,
                    "temperature": BOT_TEMPERATURE,
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

            board_before_bot = self.board.copy()  # save before push
            bot_san = self.board.san(bot_move)
            self.board.push(bot_move)
            self.pgn_moves.append(bot_san)
            self.move_count += 1

            # Record bot move in Phase C graph
            try:
                bot_skills = self.skill_tree.tagger.tag_position(board_before_bot, bot_move)
                self.skill_tree.db.record_move(
                    game_id=self.game_id,
                    move_number=self.move_count,
                    uci=bot_uci,
                    fen_before=board_before_bot.fen(),
                    skills_present=bot_skills,
                    player_found_best=False,
                    cp_loss=None,
                    move_class="bot",
                )
            except Exception as e:
                print(f"  [warn] Bot move not recorded in graph: {e}")

            print(f"  Bot move {self.move_count}: {bot_uci} ({bot_san})")

            return {
                "uci":         bot_uci,
                "san":         bot_san,
                "game_over":   self.board.is_game_over(),
                "bot_elo":     adaptive_elo,
                "bot_bracket": adaptive_bracket,
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
        state = self._state(
            last_player_move=last_player_move,
            last_bot_move=last_bot_move,
            skills_detected=skills_detected,
            move_class=move_class,
            cp_loss=cp_loss,
            game_over=True,
            result=result,
        )
        self._notify_state(state)
        return state

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_skill_summary(self) -> dict:
        return self.skill_tree.get_skill_summary(self.player_id)

    def _notify_state(self, state: dict) -> None:
        if self._state_callback:
            try:
                self._state_callback(state)
            except Exception as e:
                print(f"State callback error: {e}")

    def _state(
        self,
        last_player_move: str        = None,
        last_bot_move:    str        = None,
        skills_detected:  list       = None,
        move_class:       str        = "unknown",
        cp_loss:          int | None = None,
        game_over:        bool       = False,
        result:           str        = None,
    ) -> dict:
        return {
            "game_id":          self.game_id,
            "fen":              self.board.fen(),
            "pgn":              " ".join(self.pgn_moves),
            "move_count":       self.move_count,
            "bot_elo":          self.bot_elo,
            "bot_bracket":      self.bot_bracket,
            "turn":             "white" if self.board.turn else "black",
            "last_player_move": last_player_move,
            "last_bot_move":    last_bot_move,
            "skills_detected":  skills_detected or [],
            "move_class":       move_class,
            "cp_loss":          cp_loss,
            "game_over":        game_over,
            "result":           result,
            "status":           self.status,
            "camera_active":    self.is_vision_running(),
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
