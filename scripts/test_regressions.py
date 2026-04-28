import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import chess
import torch

from src.api.move_service import MoveService
from src.behavioral.encoder import NUM_MOVES, move_to_index
from src.graph.skill_tagger import SkillTagger
from src.graph.skill_tree import SkillTree


class DummyModel:
    def eval(self):
        return self

    def __call__(self, _tensor):
        logits = torch.full((1, NUM_MOVES), -10.0)
        logits[0, move_to_index(chess.Move.from_uci("e2e4"))] = 3.0
        logits[0, move_to_index(chess.Move.from_uci("d2d4"))] = 2.5
        return logits


def test_move_service_samples_distribution():
    service = MoveService.__new__(MoveService)
    service.models = {"1400": DummyModel()}
    service._closest_bracket = lambda elo: "1400"

    with patch(
        "src.api.move_service.torch.multinomial",
        return_value=torch.tensor([move_to_index(chess.Move.from_uci("d2d4"))]),
    ) as sampler:
        result = service.get_move(chess.STARTING_FEN, elo=1400, temperature=1.1)

    assert sampler.called, "expected stochastic sampling via torch.multinomial"
    assert result["uci"] == "d2d4", result


def test_skill_tagger_distinguishes_structure_and_activity():
    tagger = SkillTagger()

    doubled_board = chess.Board("4k3/8/8/8/8/2p5/1PP5/4K3 w - - 0 1")
    doubled_tags = tagger.tag_position(
        doubled_board,
        chess.Move.from_uci("b2c3"),
    )
    assert "Pawn_structure" in doubled_tags, doubled_tags
    assert "Doubled_pawns" in doubled_tags, doubled_tags

    isolated_board = chess.Board("4k3/8/8/8/8/8/1P6/4K3 w - - 0 1")
    isolated_tags = tagger.tag_position(
        isolated_board,
        chess.Move.from_uci("b2b4"),
    )
    assert "Isolated_pawn" in isolated_tags, isolated_tags

    outpost_board = chess.Board("4k3/2p3p1/8/8/3P4/5N2/8/4K3 w - - 0 1")
    outpost_tags = tagger.tag_position(
        outpost_board,
        chess.Move.from_uci("f3e5"),
    )
    assert "Piece_activity" in outpost_tags, outpost_tags
    assert "Outpost_control" in outpost_tags, outpost_tags


class DummySkillDb:
    def get_player_skill_profile(self, _player_id):
        return [
            {
                "skill": "Fork",
                "attempts": 6,
                "successes": 5,
                "irt_ability": 0.9,
                "difficulty": 0.4,
            },
            {
                "skill": "Endgame",
                "attempts": 4,
                "successes": 3,
                "irt_ability": 0.6,
                "difficulty": 0.5,
            },
        ]

    def get_player_move_history(self, _player_id, limit=12):
        return [
            {"move_class": "best"},
            {"move_class": "good"},
            {"move_class": "good"},
            {"move_class": "inaccuracy"},
        ][:limit]


def test_skill_tree_recommends_adaptive_bot_strength():
    tree = SkillTree.__new__(SkillTree)
    tree.db = DummySkillDb()

    adaptive_elo, adaptive_bracket = tree.recommend_bot_bracket("adam", 1400)

    assert adaptive_elo > 1400, (adaptive_elo, adaptive_bracket)
    assert adaptive_bracket in {"1400", "1600"}, adaptive_bracket


if __name__ == "__main__":
    test_move_service_samples_distribution()
    test_skill_tagger_distinguishes_structure_and_activity()
    test_skill_tree_recommends_adaptive_bot_strength()
    print("Regression checks passed.")
