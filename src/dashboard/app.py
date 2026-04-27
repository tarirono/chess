import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import json
import os
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from src.integration.game_manager import GameManager

app = Flask(
    __name__,
    template_folder=str(ROOT / "src" / "dashboard" / "templates"),
    static_folder=str(ROOT / "src" / "dashboard" / "static")
)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

manager: GameManager = None


def _emit_state_update(state: dict) -> None:
    socketio.emit("state_update", state)


@app.route("/")
def index():
    return render_template("index.html")


# ------------------------------------------------------------------
# Health / status — used by the dashboard Neo4j pill
# ------------------------------------------------------------------

@app.route("/api/status")
def status():
    """Liveness check + Neo4j connectivity used by the DB status pill."""
    try:
        from neo4j import GraphDatabase
        from dotenv import load_dotenv
        load_dotenv()
        driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687"),
            auth=(
                os.getenv("NEO4J_USER",     "neo4j"),
                os.getenv("NEO4J_PASSWORD", "chess123"),
            ),
        )
        driver.verify_connectivity()
        driver.close()
        neo4j_ok  = True
        neo4j_msg = "connected"
    except Exception as e:
        neo4j_ok  = False
        neo4j_msg = str(e)

    return jsonify({
        "neo4j":   {"ok": neo4j_ok, "msg": neo4j_msg},
        "version": "1.0.0",
    })


# ------------------------------------------------------------------
# Game
# ------------------------------------------------------------------

@app.route("/api/new_game", methods=["POST"])
def new_game():
    global manager
    data      = request.get_json(force=True) or {}
    player_id = data.get("player_id", "player_1")
    elo       = int(data.get("elo", 1400))
    if manager:
        manager.close()
    manager = GameManager(
        player_id=player_id,
        player_elo=elo,
        state_callback=_emit_state_update,
    )
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
    socketio.run(app, debug=True, port=5000, use_reloader=False)
