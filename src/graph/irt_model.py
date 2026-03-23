import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import math


class IRTModel:
    """
    Item Response Theory — 1-Parameter Logistic (1PL) model.

    Estimates player ability per skill and skill difficulty.
    Uses the Rasch model: P(correct) = 1 / (1 + exp(-(ability - difficulty)))
    """

    def probability_correct(self, ability: float,
                             difficulty: float) -> float:
        """Probability a player with given ability solves a skill
        of given difficulty."""
        return 1.0 / (1.0 + math.exp(-(ability - difficulty)))

    def update_ability(self, current_ability: float,
                       success: bool,
                       difficulty: float,
                       learning_rate: float = 0.3) -> float:
        """
        Update player ability estimate using gradient step.
        success: True if player found the best move in this skill context.
        """
        p = self.probability_correct(current_ability, difficulty)
        outcome = 1.0 if success else 0.0
        new_ability = current_ability + learning_rate * (outcome - p)
        return round(new_ability, 4)

    def update_difficulty(self, current_difficulty: float,
                          success: bool,
                          ability: float,
                          learning_rate: float = 0.1) -> float:
        """
        Update skill difficulty based on whether a player succeeded.
        If players keep succeeding → skill becomes easier (lower difficulty).
        """
        p = self.probability_correct(ability, current_difficulty)
        outcome = 1.0 if success else 0.0
        new_difficulty = current_difficulty - learning_rate * (outcome - p)
        return round(new_difficulty, 4)

    def zone_of_proximal_development(
            self,
            skill_profiles: list[dict],
            target_probability: float = 0.6
    ) -> list[dict]:
        """
        Identify skills in the Zone of Proximal Development.

        ZPD = skills where the player has ~60% success probability.
        Not too easy (>80%) and not too hard (<30%).

        Returns skills ranked by how close they are to the target.
        """
        zpd_skills = []

        for profile in skill_profiles:
            ability    = profile.get("irt_ability", 0.0)
            difficulty = profile.get("difficulty", 0.5)
            attempts   = profile.get("attempts", 0)

            if attempts < 3:
                continue  # not enough data

            p = self.probability_correct(ability, difficulty)

            # Skip mastered or too-hard skills
            if p > 0.85:
                category = "mastered"
            elif p < 0.25:
                category = "too_hard"
            else:
                category = "zpd"

            zpd_skills.append({
                "skill":       profile["skill"],
                "ability":     ability,
                "difficulty":  difficulty,
                "probability": round(p, 3),
                "attempts":    attempts,
                "successes":   profile.get("successes", 0),
                "category":    category,
                "zpd_score":   round(1.0 - abs(p - target_probability), 3)
            })

        # Sort: ZPD skills first, then by closeness to target probability
        zpd_skills.sort(
            key=lambda x: (x["category"] != "zpd", -x["zpd_score"])
        )
        return zpd_skills