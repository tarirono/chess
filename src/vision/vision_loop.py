import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import chess
import chess.pgn
from datetime import datetime

from src.vision.piece_detector import PieceDetector
from src.vision.board_localizer import BoardLocalizer
from src.vision.fen_extractor import detections_to_fen
from src.vision.motion_detector import MotionDetector


class VisionLoop:
    """
    Main Phase A loop:
      Camera → Motion detection → YOLO inference → FEN → PGN
    """

    def __init__(self, camera_index: int = 0, show_preview: bool = True):
        self.camera_index = camera_index
        self.show_preview = show_preview

        self.localizer  = BoardLocalizer()
        self.detector   = PieceDetector(conf=0.45)
        self.motion     = MotionDetector(
            motion_threshold=0.003,
            stability_frames=8
        )

        self.board      = chess.Board()
        self.prev_fen   = None
        self.move_count = 0

        print("VisionLoop initialised.")

    def _frame_to_fen(self, frame) -> str | None:
        """
        Run full detection pipeline on a single frame.
        Passes the numpy array directly to YOLO — no temp file written.
        """
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
        """
        Compare new FEN placement with previous to infer the move made.
        Returns UCI move string (e.g. 'e2e4') or None.
        """
        if self.prev_fen is None:
            return None
        if new_fen == self.prev_fen:
            return None

        for move in self.board.legal_moves:
            test_board = self.board.copy()
            test_board.push(move)
            if test_board.board_fen() == new_fen:
                return move.uci()

        return None

    def _save_snapshot(self):
        """
        Save both a FEN file and a PGN file for the current board state.
        Called when the user presses 's'.
        """
        timestamp = datetime.now().strftime("%H%M%S")
        out_dir   = Path("data/raw")
        out_dir.mkdir(parents=True, exist_ok=True)

        # --- FEN ---
        if self.prev_fen:
            fen_path = out_dir / f"fen_{timestamp}.txt"
            fen_path.write_text(self.prev_fen)
            print(f"FEN saved to {fen_path}")

        # --- PGN ---
        if self.board.move_stack:
            game = chess.pgn.Game()
            game.headers["Event"] = "Chess Ecosystem — Vision Session"
            game.headers["Date"]  = datetime.now().strftime("%Y.%m.%d")
            game.headers["White"] = "Player"
            game.headers["Black"] = "Bot"

            node = game
            tmp  = chess.Board()
            for move in self.board.move_stack:
                node = node.add_variation(move)
                tmp.push(move)

            pgn_path = out_dir / f"game_{timestamp}.pgn"
            with open(pgn_path, "w") as f:
                print(game, file=f, end="\n\n")
            print(f"PGN saved to {pgn_path}")
        else:
            print("No moves recorded yet — PGN not saved.")

    def run(self):
        """Start the live camera loop."""
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")

        print("Camera opened. Press 'q' to quit, 's' to save FEN + PGN.\n")

        ret, first_frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read from camera.")

        h, w = first_frame.shape[:2]
        corners = self.localizer.get_corners() or (0, 0, w, h)
        print(f"Using board corners: {corners}\n")

        inference_count = 0

        while True:
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
                        print(f"  Move detected: {move_uci}  "
                              f"(move #{self.move_count})")
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
                break
            elif key == ord('s'):
                self._save_snapshot()

        cap.release()
        cv2.destroyAllWindows()
        print(f"\nSession ended. Total moves detected: {self.move_count}")
        return self.board