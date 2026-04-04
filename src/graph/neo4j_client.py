import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neo4j import GraphDatabase
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

URI      = os.getenv("NEO4J_URI",      "neo4j://127.0.0.1:7687")
USER     = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "chess123")


class Neo4jClient:
    """
    Handles all Neo4j operations for the Chess Ecosystem.
    Graph schema:
      (Player)-[:PLAYED]->(Game)-[:HAS_MOVE]->(Move)-[:INVOLVES]->(Skill)
      (Player)-[:PERFORMANCE {attempts, successes, irt_ability}]->(Skill)

    Credentials are loaded from .env (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD).
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

    def record_move(
        self,
        game_id: str,
        move_number: int,
        uci: str,
        fen_before: str,
        skills_present: list[str],
        player_found_best: bool,
        cp_loss: int | None = None,
        move_class: str = "unknown",
    ):
        """
        Store a move node with engine evaluation metadata (cp_loss, move_class).
        """
        with self.driver.session() as s:
            s.run(
                """
                MATCH (g:Game {id: $game_id})
                CREATE (m:Move {
                    move_number: $move_num,
                    uci: $uci,
                    fen_before: $fen,
                    player_found_best: $best,
                    cp_loss: $cp_loss,
                    move_class: $move_class,
                    recorded_at: $ts
                })
                CREATE (g)-[:HAS_MOVE]->(m)
                """,
                game_id=game_id, move_num=move_number,
                uci=uci, fen=fen_before,
                best=player_found_best,
                cp_loss=cp_loss if cp_loss is not None else -1,
                move_class=move_class,
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

    def get_single_skill_profile(self, player_id: str,
                                  skill_name: str) -> dict | None:
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

    def update_irt_params(self, player_id: str, skill_name: str,
                           new_ability: float, new_difficulty: float):
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

    def get_player_move_history(self, player_id: str, limit: int = 50) -> list[dict]:
        """
        Returns recent moves with cp_loss and move_class for Elo validation
        and dashboard display.
        """
        with self.driver.session() as s:
            result = s.run(
                """
                MATCH (p:Player {id: $pid})-[:PLAYED]->(g:Game)-[:HAS_MOVE]->(m:Move)
                RETURN m.uci AS uci, m.move_number AS move_number,
                       m.cp_loss AS cp_loss, m.move_class AS move_class,
                       m.player_found_best AS found_best,
                       g.id AS game_id
                ORDER BY g.played_at DESC, m.move_number ASC
                LIMIT $limit
                """,
                pid=player_id, limit=limit
            )
            return [dict(row) for row in result]

    def get_all_skills(self) -> list[str]:
        with self.driver.session() as s:
            result = s.run("MATCH (sk:Skill) RETURN sk.name AS name")
            return [row["name"] for row in result]
