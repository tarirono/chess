import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chess
import uuid
from src.graph.neo4j_client import Neo4jClient
from src.graph.skill_tagger  import SkillTagger
from src.graph.irt_model     import IRTModel


class SkillTree:
    """
    Main Phase C orchestrator.
    Connects Neo4j, SkillTagger, and IRTModel together.
    Called by the game loop after each player move.
    """

    def __init__(self):
        self.db     = Neo4jClient()
        self.tagger = SkillTagger()
        self.irt    = IRTModel()
        print("SkillTree initialised.")

    def get_or_create_player(self, player_id: str,
                              elo: int = 1200) -> dict:
        return self.db.get_or_create_player(player_id, elo)

    def start_game(self, player_id: str, player_elo: int,
                   bot_bracket: str) -> str:
        """Create a new game record. Returns game_id."""
        game_id = str(uuid.uuid4())[:8]
        self.db.create_game(game_id, player_id, player_elo, bot_bracket)
        return game_id

    def record_player_move(self, game_id: str, player_id: str,
                            move_number: int, move: chess.Move,
                            board_before: chess.Board,
                            best_move: chess.Move = None):
        """
        After a player makes a move:
        1. Tag the position with skill concepts
        2. Check if player found the best move
        3. Store in Neo4j
        4. Update IRT ability estimates
        """
        # Tag skills present in this position
        skills = self.tagger.tag_position(board_before, move)

        # Did player find the best move?
        player_found_best = (best_move is not None and
                             move.uci() == best_move.uci())

        # Store move in graph
        self.db.record_move(
            game_id=game_id,
            move_number=move_number,
            uci=move.uci(),
            fen_before=board_before.fen(),
            skills_present=skills,
            player_found_best=player_found_best
        )

        # Update player skill performance + IRT ability
        for skill_name in skills:
            self.db.update_player_skill(
                player_id, skill_name, player_found_best
            )
            # Update IRT ability in the graph
            profile = self._get_skill_profile(player_id, skill_name)
            if profile:
                new_ability = self.irt.update_ability(
                    current_ability=profile.get("irt_ability", 0.0),
                    success=player_found_best,
                    difficulty=profile.get("difficulty", 0.5)
                )
                self._update_irt_ability(player_id, skill_name, new_ability)

        return skills

    def get_zpd_recommendations(self, player_id: str) -> list[dict]:
        """
        Return ZPD skill recommendations for a player.
        These are the skills they should practice next.
        """
        profiles = self.db.get_player_skill_profile(player_id)
        if not profiles:
            return []
        return self.irt.zone_of_proximal_development(profiles)

    def get_skill_summary(self, player_id: str) -> dict:
        """Full skill profile summary for dashboard display."""
        profiles  = self.db.get_player_skill_profile(player_id)
        zpd       = self.irt.zone_of_proximal_development(profiles)

        mastered  = [s for s in zpd if s["category"] == "mastered"]
        in_zpd    = [s for s in zpd if s["category"] == "zpd"]
        too_hard  = [s for s in zpd if s["category"] == "too_hard"]

        return {
            "player_id":   player_id,
            "total_skills_seen": len(profiles),
            "mastered":    mastered,
            "practice_now": in_zpd,
            "not_ready":   too_hard,
            "top_recommendation": in_zpd[0]["skill"] if in_zpd else None
        }

    def _get_skill_profile(self, player_id: str,
                            skill_name: str) -> dict | None:
        profiles = self.db.get_player_skill_profile(player_id)
        for p in profiles:
            if p["skill"] == skill_name:
                return p
        return None

    def _update_irt_ability(self, player_id: str,
                             skill_name: str, ability: float):
        with self.db.driver.session() as s:
            s.run(
                """
                MATCH (p:Player {id: $pid})-[r:PERFORMANCE]->(sk:Skill {name: $skill})
                SET r.irt_ability = $ability
                """,
                pid=player_id, skill=skill_name, ability=ability
            )

    def close(self):
        self.db.close()