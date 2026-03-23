import sys
from pathlib import Path

# Must be set before any src imports
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, render_template, request, jsonify
from src.integration.game_manager import GameManager

# Tell Flask where templates are
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


if __name__ == "__main__":
    print(f"Dashboard running — templates: {ROOT / 'src' / 'dashboard' / 'templates'}")
    app.run(debug=True, port=5000, use_reloader=False)

