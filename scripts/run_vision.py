import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import requests
from dotenv import load_dotenv
from src.vision.vision_loop import VisionLoop

load_dotenv()

camera_index = int(os.getenv("CAMERA_INDEX", "0"))
DASHBOARD_URL = "http://127.0.0.1:5000"

def on_move_detected(uci: str):
    """Send detected move directly to dashboard API"""
    try:
        resp = requests.post(
            f"{DASHBOARD_URL}/api/move",
            json={"uci": uci},
            timeout=5
        )
        data = resp.json()
        if data.get("error"):
            print(f"  Dashboard rejected move: {data['error']}")
        else:
            print(f"  Move sent to dashboard: {uci} ✓")
            bot = data.get("last_bot_move")
            if bot:
                print(f"  Bot responded: {bot}")
    except Exception as e:
        print(f"  Could not reach dashboard: {e}")

print("Chess Vision System — Phase A")
print("=" * 40)
print(f"Camera index: {camera_index}")
print(f"Dashboard: {DASHBOARD_URL}")
print("Make sure dashboard is running and a game is started.\n")

loop = VisionLoop(
    camera_index=camera_index,
    show_preview=True,
    on_move_detected=on_move_detected,
)
loop.run()