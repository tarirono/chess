import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv
from src.vision.vision_loop import VisionLoop

load_dotenv()

# Read camera index from .env (CAMERA_INDEX=0).
# 0 = default/built-in webcam.  Set to 1 or 2 in .env for external cameras.
camera_index = int(os.getenv("CAMERA_INDEX", "0"))

print("Chess Vision System — Phase A")
print("=" * 40)
print(f"Camera index: {camera_index}  (change CAMERA_INDEX in .env to switch cameras)")
print("Make sure your camera is connected and pointing at the board.\n")

loop = VisionLoop(
    camera_index=camera_index,
    show_preview=True,
)
loop.run()