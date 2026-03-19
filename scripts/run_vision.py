import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vision.vision_loop import VisionLoop

print("Chess Vision System — Phase A")
print("=" * 40)
print("Make sure your camera is connected and pointing at the board.\n")

loop = VisionLoop(
    camera_index=0,    # change to 1 or 2 if your webcam is not default
    show_preview=True
)
loop.run()