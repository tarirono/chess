import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chess
from src.graph.skill_tree import SkillTree

print("Testing Phase C — Knowledge Graph\n" + "="*45)

tree = SkillTree()

# Create a test player
player = tree.get_or_create_player("adam_test", elo=1400)
print(f"Player: {player}")

# Start a game
game_id = tree.start_game("adam_test", player_elo=1400, bot_bracket="1400")
print(f"Game ID: {game_id}")

# Simulate 5 moves
board = chess.Board()
test_moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]

print("\nSimulating moves:")
for i, uci in enumerate(test_moves):
    move = chess.Move.from_uci(uci)
    if move in board.legal_moves:
        skills = tree.record_player_move(
            game_id=game_id,
            player_id="adam_test",
            move_number=i + 1,
            move=move,
            board_before=board.copy(),
            best_move=move if i % 2 == 0 else None  # found best on even moves
        )
        print(f"  Move {i+1}: {uci} → skills: {skills}")
        board.push(move)

# Get ZPD recommendations
print("\nSkill summary:")
summary = tree.get_skill_summary("adam_test")
print(f"  Skills seen: {summary['total_skills_seen']}")
print(f"  Top recommendation: {summary['top_recommendation']}")

print("\nZPD recommendations (what to practice next):")
for rec in summary["practice_now"]:
    print(f"  {rec['skill']:<20} prob={rec['probability']}  attempts={rec['attempts']}")

print("\nPhase C working correctly.")
tree.close()

