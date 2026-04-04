import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chess
import uuid
from src.graph.neo4j_client    import Neo4jClient
from src.graph.skill_tagger    import SkillTagger
from src.graph.irt_model       import IRTModel
from src.graph.engine_analyzer import EngineAnalyzer


class SkillTree:
    """
    Main Phase C orchestrator.
    Connects Neo4j, SkillTagger, IRTModel, and EngineAnalyzer.

    EngineAnalyzer provides per-move centipawn loss and move classification
    (best / good / inaccuracy / mistake / blunder) backed by Stockfish 16.
    The SkillTagger uses this data to accurately label blunders and confirm
    tactical patterns, replacing the previous pure-heuristic approach.
    """

    def __init__(self):
        self.db      = Neo4jClient()
        self.tagger  = SkillTagger()
        self.irt     = IRTModel()
        self.engine  = EngineAnalyzer(depth=15)
        print("SkillTree initialised (Stockfish engine active).")

    def get_or_create_player(self, player_id: str,
                              elo: int = 1200) -> dict:
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
        1.  Run Stockfish analysis on the position.
        2.  Tag the position with engine-backed skill concepts.
        3.  Determine if the player found the best move (engine best, not
            just the externally supplied hint).
        4.  Store move + skills in Neo4j.
        5.  Update IRT ability AND difficulty estimates.

        Returns the list of skill tags assigned to this move.
        """
        # ── 1. Engine analysis ────────────────────────────────────────
        analysis = self.engine.analyze_move(board_before, move)

        # ── 2. Skill tagging (engine-backed) ─────────────────────────
        skills = self.tagger.tag_position(board_before, move, analysis)

        # ── 3. Did the player find the best move? ─────────────────────
        if analysis.best_move_uci:
            player_found_best = (move.uci() == analysis.best_move_uci)
        elif best_move is not None:
            player_found_best = (move.uci() == best_move.uci())
        else:
            # Fall back: "good" or better counts as best
            player_found_best = analysis.classification in ("best", "good")

        # ── 4. Persist move in Neo4j ──────────────────────────────────
        self.db.record_move(
            game_id=game_id,
            move_number=move_number,
            uci=move.uci(),
            fen_before=board_before.fen(),
            skills_present=skills,
            player_found_best=player_found_best,
        )

        # ── 5. Update IRT per skill ───────────────────────────────────
        for skill_name in skills:
            self.db.update_player_skill(player_id, skill_name, player_found_best)
            profile = self.db.get_single_skill_profile(player_id, skill_name)
            if not profile:
                continue

            current_ability    = profile.get("irt_ability", 0.0)
            current_difficulty = profile.get("difficulty", 0.5)

            new_ability    = self.irt.update_ability(
                current_ability, player_found_best, current_difficulty
            )
            new_difficulty = self.irt.update_difficulty(
                current_difficulty, player_found_best, current_ability
            )
            self.db.update_irt_params(
                player_id, skill_name, new_ability, new_difficulty
            )

        # ── Log for debugging ─────────────────────────────────────────
        print(
            f"  [{move.uci()}] cp_loss={analysis.cp_loss:>4}  "
            f"class={analysis.classification:<12}  skills={skills}"
        )

        return {
        "skills": skills,
        "cp_loss": analysis.cp_loss if analysis.available else None,
        "move_class": analysis.classification,
    }

    def get_zpd_recommendations(self, player_id: str) -> list[dict]:
        profiles = self.db.get_player_skill_profile(player_id)
        if not profiles:
            return []
        return self.irt.zone_of_proximal_development(profiles)

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
