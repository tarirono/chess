import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import chess
import chess.pgn
from datetime import datetime
from typing import Callable

from src.vision.piece_detector import PieceDetector
from src.vision.board_localizer import BoardLocalizer
from src.vision.fen_extractor import detections_to_fen
from src.vision.motion_detector import MotionDetector

# Directory where live PGN files are written
PGN_OUTPUT_DIR = Path("data/pgn")


class VisionLoop:
    """
    Main Phase A loop:
      Camera → Motion detection → YOLO inference → FEN → PGN

    Every detected move is immediately appended to a live PGN file so the
    game is always on disk (spec requirement).  The file is named with the
    session timestamp, e.g. data/pgn/game_143022.pgn.

    Phase A+C integration:
        Provide on_move_detected callback to wire directly into GameManager.

        def handle_move(uci: str):
            manager.player_move(uci)

        loop = VisionLoop(camera_index=0, on_move_detected=handle_move)
        loop.run()

    Or simply use GameManager.start_vision_thread() which does this for you.
    """

    def __init__(
        self,
        camera_index:      int = 0,
        show_preview:      bool = True,
        on_move_detected:  Callable[[str], None] | None = None,
    ):
        self.camera_index     = camera_index
        self.show_preview     = show_preview
        self.on_move_detected = on_move_detected

        self.localizer  = BoardLocalizer()
        self.detector   = PieceDetector(conf=0.45)
        self.motion     = MotionDetector(
            motion_threshold=0.003,
            stability_frames=8
        )

        self.board      = chess.Board()
        self.prev_fen   = None
        self.move_count = 0
        self._running   = False

        # PGN auto-write setup
        PGN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._session_ts  = datetime.now().strftime("%H%M%S")
        self._pgn_path    = PGN_OUTPUT_DIR / f"game_{self._session_ts}.pgn"
        self._pgn_game    = chess.pgn.Game()
        self._pgn_game.headers["Event"] = "Chess Ecosystem — Vision Session"
        self._pgn_game.headers["Site"]  = "Local"
        self._pgn_game.headers["Date"]  = datetime.now().strftime("%Y.%m.%d")
        self._pgn_game.headers["White"] = "Player"
        self._pgn_game.headers["Black"] = "Bot"
        self._pgn_node    = self._pgn_game   # current node in the PGN tree

        print("VisionLoop initialised.")
        print(f"Live PGN output: {self._pgn_path}")

    # ------------------------------------------------------------------
    # Stop signal
    # ------------------------------------------------------------------

    def stop(self):
        """Signal the run loop to exit cleanly."""
        self._running = False

    # ------------------------------------------------------------------
    # PGN auto-write
    # ------------------------------------------------------------------

    def _append_move_to_pgn(self, move: chess.Move) -> None:
        """
        Add move to the in-memory PGN tree and immediately overwrite the
        PGN file on disk.  This ensures the file is always up to date even
        if the session ends unexpectedly.
        """
        self._pgn_node = self._pgn_node.add_variation(move)
        self._write_pgn()

    def _write_pgn(self) -> None:
        """Overwrite the PGN file with the current game state."""
        with open(self._pgn_path, "w") as f:
            print(self._pgn_game, file=f, end="\n\n")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _frame_to_fen(self, frame) -> str | None:
        h, w = frame.shape[:2]
        detections = self.detector.detect(frame)
        if not detections:
            return None
        result = detections_to_fen(
            detections=detections,
            localizer=self.localizer,
            image_width=w,
            image_height=h,
        )
        return result["fen"]

    def _fen_to_move(self, new_fen: str) -> str | None:
        if self.prev_fen is None:
            return None
        if new_fen == self.prev_fen:
            return None
        for move in self.board.legal_moves:
            test_board = self.board.copy()
            test_board.push(move)
            if test_board.board_fen() == new_fen:
                return move.uci()
        # Promotion fallback: try all legal moves with queen promotion
        for move in self.board.legal_moves:
            if move.promotion:
                continue  # already tried above
            # Try queen promotion variant
            promo = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            if promo in self.board.legal_moves:
                test_board = self.board.copy()
                test_board.push(promo)
                if test_board.board_fen() == new_fen:
                    return promo.uci()
        return None

    def _save_snapshot(self):
        """Manual snapshot triggered by pressing 's'."""
        timestamp = datetime.now().strftime("%H%M%S")
        out_dir   = Path("data/raw")
        out_dir.mkdir(parents=True, exist_ok=True)

        if self.prev_fen:
            fen_path = out_dir / f"fen_{timestamp}.txt"
            fen_path.write_text(self.prev_fen)
            print(f"FEN saved to {fen_path}")

        # PGN is already being written continuously — just report path
        print(f"Live PGN is at: {self._pgn_path}  ({self.move_count} moves)")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Start the live camera loop."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")

        self._running = True
        print("Camera opened. Press 'q' to quit, 's' to print FEN + PGN path.\n")
        if self.on_move_detected:
            print("Phase A→C integration active: detected moves → GameManager\n")

        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read from camera.")

        h, w = first_frame.shape[:2]
        corners = self.localizer.get_corners() or (0, 0, w, h)
        print(f"Using board corners: {corners}\n")

        # Write empty PGN header so the file exists from the start
        self._write_pgn()

        inference_count = 0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            status = self.motion.update(frame)

            if self.show_preview:
                color = (0, 0, 255) if status["motion"] else (0, 255, 0)
                label = "MOTION" if status["motion"] else "STABLE"
                cv2.putText(frame, f"{label}  diff={status['diff_ratio']:.4f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Moves detected: {self.move_count}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"PGN: {self._pgn_path.name}",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)
                if self.on_move_detected:
                    cv2.putText(frame, "A+C INTEGRATED",
                                (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                x1, y1, x2, y2 = corners
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.imshow("Chess Vision", frame)

            if status["trigger"]:
                inference_count += 1
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Motion stopped — running inference #{inference_count}...")

                new_fen = self._frame_to_fen(frame)
                if new_fen:
                    move_uci = self._fen_to_move(new_fen)
                    if move_uci:
                        self.move_count += 1
                        move_obj = chess.Move.from_uci(move_uci)
                        print(f"  Move detected: {move_uci} (move #{self.move_count})")

                        # ── Auto-write PGN immediately ───────────────
                        self._append_move_to_pgn(move_obj)
                        print(f"  PGN updated: {self._pgn_path}")

                        # ── Phase A + C integration ──────────────────
                        if self.on_move_detected:
                            self.on_move_detected(move_uci)
                        else:
                            try:
                                self.board.push_uci(move_uci)
                                print(f"  PGN so far: "
                                      f"{self.board.variation_san(self.board.move_stack)}")
                            except Exception as e:
                                print(f"  Invalid move: {e}")
                    else:
                        print("  FEN changed but no legal move matched.")

                    self.prev_fen = new_fen
                    print(f"  FEN: {new_fen}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self._running = False
            elif key == ord('s'):
                self._save_snapshot()

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession ended. Total moves detected: {self.move_count}")
        print(f"Final PGN saved at: {self._pgn_path}")
        return self.board