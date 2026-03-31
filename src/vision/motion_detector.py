import cv2
import numpy as np


class MotionDetector:
    """
    Detects meaningful motion in a camera feed using frame differencing.
    Only triggers inference when a chess piece has likely been moved.
    """

    def __init__(self,
                 motion_threshold: float = 0.003,
                 stability_frames: int = 8,
                 blur_size: int = 21):
        """
        Args:
            motion_threshold:  fraction of pixels that must change to count as motion (0.3%)
            stability_frames:  how many consecutive still frames before we accept a new position
            blur_size:         gaussian blur kernel size for noise reduction
        """
        self.motion_threshold = motion_threshold
        self.stability_frames = stability_frames
        self.blur_size = blur_size

        self.prev_frame = None
        self.still_count = 0
        self.in_motion = False

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)

    def update(self, frame: np.ndarray) -> dict:
        """
        Feed a new frame. Returns a status dict:
          {
            "motion":    bool  — is there currently motion?
            "trigger":   bool  — True on the FIRST still frame after motion stops
                                 (this is when you should run YOLO)
            "diff_ratio": float — fraction of pixels that changed
          }
        """
        processed = self._preprocess(frame)

        if self.prev_frame is None:
            self.prev_frame = processed
            return {"motion": False, "trigger": False, "diff_ratio": 0.0}

        # Frame difference
        diff = cv2.absdiff(self.prev_frame, processed)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        diff_ratio = np.count_nonzero(thresh) / thresh.size

        self.prev_frame = processed
        trigger = False

        if diff_ratio > self.motion_threshold:
            # Motion detected
            self.in_motion = True
            self.still_count = 0
        else:
            # Frame is still
            if self.in_motion:
                self.still_count += 1
                if self.still_count >= self.stability_frames:
                    # Board has settled after a move — fire inference
                    trigger = True
                    self.in_motion = False
                    self.still_count = 0

        return {
            "motion":     self.in_motion,
            "trigger":    trigger,
            "diff_ratio": round(diff_ratio, 5)
        }

    def reset(self):
        self.prev_frame = None
        self.still_count = 0
        self.in_motion = False