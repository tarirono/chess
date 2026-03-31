import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neo4j import GraphDatabase
from datetime import datetime

URI      = "neo4j://127.0.0.1:7687"
USER     = "neo4j"
PASSWORD = "chess123"


class Neo4jClient:
    """
    Handles all Neo4j operations for the Chess Ecosystem.
    Graph schema:
      (Player)-[:PLAYED]->(Game)-[:HAS_MOVE]->(Move)-[:INVOLVES]->(Skill)
      (Player)-[:PERFORMANCE {attempts, successes, irt_ability}]->(Skill)
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
        self.driver.verify_connectivity()
        print("Neo4jClient connected.")
        self._create_constraints()

    def close(self):
        self.driver.close()

    def _create_constraints(self):
        with self.driver.session() as s:
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Player) REQUIRE p.id IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (g:Game)   REQUIRE g.id IS UNIQUE")
            s.run("CREATE CONSTRAINT IF NOT EXISTS FOR (sk:Skill)  REQUIRE sk.name IS UNIQUE")
        self._seed_skills()

    def _seed_skills(self):
        skills = [
            "Pin", "Fork", "Discovery", "Skewer",
            "Checkmate_pattern", "Endgame", "Opening",
            "Pawn_structure", "Piece_activity", "Blunder"
        ]
        with self.driver.session() as s:
            for skill in skills:
                s.run(
                    "MERGE (sk:Skill {name: $name}) "
                    "ON CREATE SET sk.difficulty = 0.5, sk.created_at = $ts",
                    name=skill, ts=datetime.now().isoformat()
                )

    # ------------------------------------------------------------------
    # Player
    # ------------------------------------------------------------------

    def get_or_create_player(self, player_id: str, elo: int = 1200) -> dict:
        with self.driver.session() as s:
            result = s.run(
                """
                MERGE (p:Player {id: $id})
                ON CREATE SET p.elo = $elo, p.games_played = 0,
                              p.created_at = $ts
                RETURN p
                """,
                id=player_id, elo=elo, ts=datetime.now().isoformat()
            )
            return dict(result.single()["p"])

    # ------------------------------------------------------------------
    # Game
    # ------------------------------------------------------------------

    def create_game(self, game_id: str, player_id: str,
                    player_elo: int, bot_bracket: str) -> dict:
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH (p:Player {id: $player_id})
                CREATE (g:Game {
                    id: $game_id,
                    player_elo: $player_elo,
                    bot_bracket: $bot_bracket,
                    played_at: $ts,
                    result: 'in_progress'
                })
                CREATE (p)-[:PLAYED]->(g)
                RETURN g
                """,
                game_id=game_id, player_id=player_id,
                player_elo=player_elo, bot_bracket=bot_bracket,
                ts=datetime.now().isoformat()
            )
            return dict(result.single()["g"])

    def finish_game(self, game_id: str, result: str, total_moves: int):
        with self.driver.session() as s:
            s.run(
                """
                MATCH (g:Game {id: $game_id})
                SET g.result = $result, g.total_moves = $total_moves,
                    g.finished_at = $ts
                """,
                game_id=game_id, result=result,
                total_moves=total_moves, ts=datetime.now().isoformat()
            )

    # ------------------------------------------------------------------
    # Moves + Skills
    # ------------------------------------------------------------------

    def record_move(self, game_id: str, move_number: int,
                    uci: str, fen_before: str,
                    skills_present: list[str],
                    player_found_best: bool):
        with self.driver.session() as s:
            s.run(
                """
                MATCH (g:Game {id: $game_id})
                CREATE (m:Move {
                    move_number: $move_num,
                    uci: $uci,
                    fen_before: $fen,
                    player_found_best: $best,
                    recorded_at: $ts
                })
                CREATE (g)-[:HAS_MOVE]->(m)
                """,
                game_id=game_id, move_num=move_number,
                uci=uci, fen=fen_before,
                best=player_found_best,
                ts=datetime.now().isoformat()
            )
            for skill in skills_present:
                s.run(
                    """
                    MATCH (g:Game {id: $game_id})
                    MATCH (g)-[:HAS_MOVE]->(m:Move {move_number: $move_num})
                    MATCH (sk:Skill {name: $skill})
                    MERGE (m)-[:INVOLVES]->(sk)
                    """,
                    game_id=game_id, move_num=move_number, skill=skill
                )

    def update_player_skill(self, player_id: str, skill_name: str,
                             success: bool):
        with self.driver.session() as s:
            s.run(
                """
                MATCH (p:Player {id: $pid})
                MATCH (sk:Skill {name: $skill})
                MERGE (p)-[r:PERFORMANCE]->(sk)
                ON CREATE SET r.attempts = 0, r.successes = 0,
                              r.irt_ability = 0.0
                SET r.attempts  = r.attempts + 1,
                    r.successes = r.successes + CASE WHEN $success THEN 1 ELSE 0 END,
                    r.updated_at = $ts
                """,
                pid=player_id, skill=skill_name,
                success=success, ts=datetime.now().isoformat()
            )

    # ------------------------------------------------------------------
    # BUG FIX: targeted single-skill query
    # ------------------------------------------------------------------

    def get_single_skill_profile(self, player_id: str,
                                  skill_name: str) -> dict | None:
        """
        Fetch IRT fields for ONE skill in a single Cypher round-trip.

        BUG FIX (N+1): the old code called get_player_skill_profile() — which
        returns ALL skills — just to extract one. With 3+ tags per move that
        means 3+ full profile queries per move.  This targeted query costs
        exactly one round-trip regardless of how many skills the player has.
        """
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH (p:Player {id: $pid})-[r:PERFORMANCE]->(sk:Skill {name: $skill})
                RETURN sk.name      AS skill,
                       r.attempts   AS attempts,
                       r.successes  AS successes,
                       r.irt_ability AS irt_ability,
                       sk.difficulty AS difficulty
                """,
                pid=player_id, skill=skill_name
            )
            record = result.single()
            return dict(record) if record else None

    # ------------------------------------------------------------------
    # BUG FIX: persist updated skill difficulty back to Neo4j
    # ------------------------------------------------------------------

    def update_irt_params(self, player_id: str, skill_name: str,
                           new_ability: float, new_difficulty: float):
        """
        Persist both the player's updated ability AND the skill's updated
        difficulty in a single write query.

        BUG FIX: skill difficulty was previously stuck at the seeded 0.5
        because update_difficulty() was computed but never saved.
        """
        with self.driver.session() as s:
            s.run(
                """
                MATCH (p:Player {id: $pid})-[r:PERFORMANCE]->(sk:Skill {name: $skill})
                SET r.irt_ability  = $ability,
                    sk.difficulty  = $difficulty
                """,
                pid=player_id, skill=skill_name,
                ability=new_ability, difficulty=new_difficulty
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_player_skill_profile(self, player_id: str) -> list[dict]:
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH (p:Player {id: $pid})-[r:PERFORMANCE]->(sk:Skill)
                RETURN sk.name AS skill,
                       r.attempts  AS attempts,
                       r.successes AS successes,
                       r.irt_ability AS irt_ability,
                       sk.difficulty AS difficulty
                ORDER BY r.attempts DESC
                """,
                pid=player_id
            )
            return [dict(row) for row in result]

    def get_all_skills(self) -> list[str]:
        with self.driver.session() as s:
            result = s.run("MATCH (sk:Skill) RETURN sk.name AS name")
            return [row["name"] for row in result]