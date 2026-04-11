"""
mock_neo4j_client.py
====================
In-memory drop-in replacement for Neo4jClient.
Used automatically when Neo4j is not reachable (demo / CI mode).

All data is stored in plain Python dicts — nothing persists between runs,
but the dashboard and game flow work exactly as normal.
"""

from datetime import datetime
import uuid


class MockNeo4jClient:
    """
    Mirrors the public API of Neo4jClient without any database dependency.
    Activate by setting NEO4J_MOCK=true in your .env file.
    """

    def __init__(self):
        self._players:  dict = {}
        self._games:    dict = {}
        self._moves:    list = []
        self._perf:     dict = {}   # {(player_id, skill): {attempts, successes, irt_ability}}
        self._skills:   list = [
            "Pin", "Fork", "Discovery", "Skewer",
            "Checkmate_pattern", "Endgame", "Opening",
            "Pawn_structure", "Piece_activity", "Blunder", "Mistake", "Inaccuracy",
        ]
        print("MockNeo4jClient: running in-memory (Neo4j not connected).")

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Player
    # ------------------------------------------------------------------

    def get_or_create_player(self, player_id: str, elo: int = 1200) -> dict:
        if player_id not in self._players:
            self._players[player_id] = {
                "id":           player_id,
                "elo":          elo,
                "games_played": 0,
                "created_at":   datetime.now().isoformat(),
            }
        return self._players[player_id]

    # ------------------------------------------------------------------
    # Game
    # ------------------------------------------------------------------

    def create_game(self, game_id: str, player_id: str,
                    player_elo: int, bot_bracket: str) -> dict:
        game = {
            "id":          game_id,
            "player_id":   player_id,
            "player_elo":  player_elo,
            "bot_bracket": bot_bracket,
            "played_at":   datetime.now().isoformat(),
            "result":      "in_progress",
        }
        self._games[game_id] = game
        return game

    def finish_game(self, game_id: str, result: str, total_moves: int):
        if game_id in self._games:
            self._games[game_id]["result"]      = result
            self._games[game_id]["total_moves"] = total_moves
            self._games[game_id]["finished_at"] = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Moves + Skills
    # ------------------------------------------------------------------

    def record_move(self, game_id, move_number, uci, fen_before,
                    skills_present, player_found_best,
                    cp_loss=None, move_class="unknown"):
        self._moves.append({
            "game_id":          game_id,
            "move_number":      move_number,
            "uci":              uci,
            "fen_before":       fen_before,
            "skills_present":   skills_present,
            "player_found_best": player_found_best,
            "cp_loss":          cp_loss,
            "move_class":       move_class,
            "recorded_at":      datetime.now().isoformat(),
        })

    def update_player_skill(self, player_id: str, skill_name: str, success: bool):
        key = (player_id, skill_name)
        if key not in self._perf:
            self._perf[key] = {"attempts": 0, "successes": 0, "irt_ability": 0.0}
        self._perf[key]["attempts"]  += 1
        self._perf[key]["successes"] += 1 if success else 0

    def get_single_skill_profile(self, player_id: str, skill_name: str) -> dict | None:
        key = (player_id, skill_name)
        if key not in self._perf:
            return None
        p = self._perf[key]
        return {
            "skill":       skill_name,
            "attempts":    p["attempts"],
            "successes":   p["successes"],
            "irt_ability": p["irt_ability"],
            "difficulty":  0.5,
        }

    def update_irt_params(self, player_id: str, skill_name: str,
                           new_ability: float, new_difficulty: float):
        key = (player_id, skill_name)
        if key in self._perf:
            self._perf[key]["irt_ability"] = new_ability

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_player_skill_profile(self, player_id: str) -> list[dict]:
        results = []
        for (pid, skill), p in self._perf.items():
            if pid != player_id:
                continue
            results.append({
                "skill":       skill,
                "attempts":    p["attempts"],
                "successes":   p["successes"],
                "irt_ability": p["irt_ability"],
                "difficulty":  0.5,
            })
        return sorted(results, key=lambda x: -x["attempts"])

    def get_player_move_history(self, player_id: str, limit: int = 50) -> list[dict]:
        player_games = {gid for gid, g in self._games.items()
                        if g.get("player_id") == player_id}
        history = [m for m in self._moves if m["game_id"] in player_games]
        return history[-limit:]

    def get_all_skills(self) -> list[str]:
        return list(self._skills)