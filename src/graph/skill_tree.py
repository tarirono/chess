import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import chess
import uuid
from dotenv import load_dotenv

from src.graph.skill_tagger    import SkillTagger
from src.graph.irt_model       import IRTModel
from src.graph.engine_analyzer import EngineAnalyzer

load_dotenv()


def _build_db_client():
    """
    Try to connect to Neo4j.  If it is unavailable (or NEO4J_MOCK=true in
    .env), fall back to the in-memory MockNeo4jClient so the dashboard and
    game flow still work without a database.
    """
    if os.getenv("NEO4J_MOCK", "false").lower() == "true":
        from src.graph.mock_neo4j_client import MockNeo4jClient
        return MockNeo4jClient()

    try:
        from src.graph.neo4j_client import Neo4jClient
        client = Neo4jClient()
        return client
    except Exception as e:
        print(
            f"[SkillTree] Neo4j unavailable ({e}).\n"
            "           Falling back to in-memory mock (data will NOT persist).\n"
            "           To silence this, set NEO4J_MOCK=true in your .env file."
        )
        from src.graph.mock_neo4j_client import MockNeo4jClient
        return MockNeo4jClient()


class SkillTree:
    """
    Main Phase C orchestrator.
    Connects Neo4j (or mock), SkillTagger, IRTModel, and EngineAnalyzer.
    """

    def __init__(self):
        self.db      = _build_db_client()
        self.tagger  = SkillTagger()
        self.irt     = IRTModel()
        self.engine  = EngineAnalyzer(depth=15)
        print("SkillTree initialised.")

    def get_or_create_player(self, player_id: str, elo: int = 1200) -> dict:
        return self.db.get_or_create_player(player_id, elo)

    def start_game(self, player_id: str, player_elo: int,
                   bot_bracket: str) -> str:
        game_id = str(uuid.uuid4())[:8]
        self.db.create_game(game_id, player_id, player_elo, bot_bracket)
        return game_id

    def record_player_move(self,
                            game_id: str,
                            player_id: str,
                            move_number: int,
                            move: chess.Move,
                            board_before: chess.Board,
                            best_move: chess.Move | None = None) -> dict:
        """
        After a player makes a move:
        1. Run Stockfish analysis.
        2. Tag with engine-backed skill concepts.
        3. Determine if player found the best move.
        4. Store move + skills in DB (real or mock).
        5. Update IRT ability and difficulty estimates.
        """
        # 1. Engine analysis
        analysis = self.engine.analyze_move(board_before, move)

        # 2. Skill tagging
        skills = self.tagger.tag_position(board_before, move, analysis)

        # 3. Best move check
        if analysis.best_move_uci:
            player_found_best = (move.uci() == analysis.best_move_uci)
        elif best_move is not None:
            player_found_best = (move.uci() == best_move.uci())
        else:
            player_found_best = analysis.classification in ("best", "good")

        # 4. Persist
        self.db.record_move(
            game_id=game_id,
            move_number=move_number,
            uci=move.uci(),
            fen_before=board_before.fen(),
            skills_present=skills,
            player_found_best=player_found_best,
            cp_loss=analysis.cp_loss if analysis.available else None,
            move_class=analysis.classification,
        )

        # 5. IRT update per skill
        for skill_name in skills:
            self.db.update_player_skill(player_id, skill_name, player_found_best)
            profile = self.db.get_single_skill_profile(player_id, skill_name)
            if not profile:
                continue
            new_ability    = self.irt.update_ability(
                profile.get("irt_ability", 0.0), player_found_best,
                profile.get("difficulty", 0.5)
            )
            new_difficulty = self.irt.update_difficulty(
                profile.get("difficulty", 0.5), player_found_best,
                profile.get("irt_ability", 0.0)
            )
            self.db.update_irt_params(player_id, skill_name, new_ability, new_difficulty)

        print(
            f"  [{move.uci()}] cp_loss={analysis.cp_loss:>4}  "
            f"class={analysis.classification:<12}  skills={skills}"
        )

        return {
            "skills":     skills,
            "cp_loss":    analysis.cp_loss if analysis.available else None,
            "move_class": analysis.classification,
        }

    def get_zpd_recommendations(self, player_id: str) -> list[dict]:
        profiles = self.db.get_player_skill_profile(player_id)
        if not profiles:
            return []
        return self.irt.zone_of_proximal_development(profiles)

    def recommend_bot_elo(self, player_id: str, baseline_elo: int) -> int:
        """
        Estimate a live opponent Elo from the player's tracked skill evolution.

        Starts from the initial baseline Elo, then nudges it based on:
        - per-skill IRT ability estimates
        - observed success rates across tracked skills
        - recent move-quality classifications

        The result is intentionally clamped to the trained bracket range so
        the behavioral bot stays within the available policy models.
        """
        profiles = self.db.get_player_skill_profile(player_id)
        if not profiles:
            return baseline_elo

        observed = [p for p in profiles if p.get("attempts", 0) > 0]
        if not observed:
            return baseline_elo

        total_weight = sum(min(p.get("attempts", 0), 5) for p in observed)
        if total_weight == 0:
            return baseline_elo

        weighted_ability = sum(
            p.get("irt_ability", 0.0) * min(p.get("attempts", 0), 5)
            for p in observed
        ) / total_weight
        weighted_success = sum(
            (p.get("successes", 0) / max(p.get("attempts", 1), 1))
            * min(p.get("attempts", 0), 5)
            for p in observed
        ) / total_weight

        recent_moves = self.db.get_player_move_history(player_id, limit=12)
        quality_delta = 0.0
        if recent_moves:
            quality_score = {
                "best": 1.0,
                "good": 0.4,
                "inaccuracy": -0.4,
                "mistake": -0.8,
                "blunder": -1.2,
            }
            quality_delta = (
                sum(quality_score.get(m.get("move_class", "unknown"), 0.0)
                    for m in recent_moves)
                / len(recent_moves)
            )

        ability_adjustment = max(-120, min(120, round(weighted_ability * 120)))
        success_adjustment = max(
            -80,
            min(80, round((weighted_success - 0.5) * 160))
        )
        quality_adjustment = max(-60, min(60, round(quality_delta * 50)))

        adaptive_elo = baseline_elo + ability_adjustment + success_adjustment + quality_adjustment
        return max(1200, min(1600, int(round(adaptive_elo / 50) * 50)))

    def recommend_bot_bracket(self, player_id: str, baseline_elo: int) -> tuple[int, str]:
        adaptive_elo = self.recommend_bot_elo(player_id, baseline_elo)
        bracket = min(
            ("1200", "1400", "1600"),
            key=lambda b: abs(int(b) - adaptive_elo)
        )
        return adaptive_elo, bracket

    def get_skill_summary(self, player_id: str) -> dict:
        profiles = self.db.get_player_skill_profile(player_id)
        zpd      = self.irt.zone_of_proximal_development(profiles)

        mastered = [s for s in zpd if s["category"] == "mastered"]
        in_zpd   = [s for s in zpd if s["category"] == "zpd"]
        too_hard = [s for s in zpd if s["category"] == "too_hard"]

        return {
            "player_id":          player_id,
            "total_skills_seen":  len(profiles),
            "mastered":           mastered,
            "practice_now":       in_zpd,
            "not_ready":          too_hard,
            "top_recommendation": in_zpd[0]["skill"] if in_zpd else None,
        }

    def close(self):
        self.engine.close()
        self.db.close()
