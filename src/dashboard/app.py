import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import json
from flask import Flask, render_template, request, jsonify
from src.integration.game_manager1 import GameManager

app = Flask(
    __name__,
    template_folder=str(ROOT / "src" / "dashboard" / "templates"),
    static_folder=str(ROOT / "src" / "dashboard" / "static")
)

manager: GameManager = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/new_game", methods=["POST"])
def new_game():
    global manager
    data      = request.get_json(force=True) or {}
    player_id = data.get("player_id", "player_1")
    elo       = int(data.get("elo", 1400))
    if manager:
        manager.close()
    manager = GameManager(player_id=player_id, player_elo=elo)
    manager.start_game()
    return jsonify(manager.get_state())


@app.route("/api/move", methods=["POST"])
def make_move():
    global manager
    if manager is None:
        return jsonify({"error": "No active game"}), 400
    data  = request.get_json(force=True) or {}
    uci   = data.get("uci", "")
    state = manager.player_move(uci)
    return jsonify(state)


@app.route("/api/state")
def get_state():
    global manager
    if manager is None:
        return jsonify({"error": "No active game"}), 400
    return jsonify(manager.get_state())


@app.route("/api/skills")
def get_skills():
    global manager
    if manager is None:
        return jsonify({"error": "No active game"}), 400
    return jsonify(manager.get_skill_summary())


# ------------------------------------------------------------------
# Phase A camera endpoints
# ------------------------------------------------------------------

@app.route("/api/camera/start", methods=["POST"])
def camera_start():
    global manager
    if manager is None:
        return jsonify({"error": "Start a game first"}), 400
    if manager.status != "in_progress":
        return jsonify({"error": "Game not in progress"}), 400
    try:
        data         = request.get_json(force=True) or {}
        camera_index = int(data.get("camera_index", 0))
        manager.start_vision_thread(camera_index=camera_index)
        return jsonify({"status": "camera started", "camera_index": camera_index})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/camera/stop", methods=["POST"])
def camera_stop():
    global manager
    if manager is None:
        return jsonify({"error": "No active game"}), 400
    manager.stop_vision_thread()
    return jsonify({"status": "camera stopped"})


# ------------------------------------------------------------------
# Elo validation results endpoint
# ------------------------------------------------------------------

@app.route("/api/elo_validation")
def elo_validation():
    path = ROOT / "data" / "models" / "behavioral" / "elo_validation.json"
    if not path.exists():
        return jsonify({"error": "No validation results found. Run scripts/validate_elo.py"}), 404
    with open(path) as f:
        return jsonify(json.load(f))


if __name__ == "__main__":
    print(f"Dashboard running — templates: {ROOT / 'src' / 'dashboard' / 'templates'}")
    app.run(debug=True, port=5000, use_reloader=False)
