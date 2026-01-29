#!/usr/bin/env python3
"""Unit tests for the plan scoring helper."""

import sys
from pathlib import Path

SRC_PATH = Path(__file__).parent.parent / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from plan_scorer import PlanScorer


def test_maze3_example_plan_route_penalties(tmp_path):
    plan_actions = [
        "move_to_cell row=4 col=8",
        "move_to_cell row=2 col=8",
        "move_to_cell row=2 col=7",
        "start_automated_grasp object_type=maze item",
        "move_to_cell row=4 col=7",
        "move_to_cell row=4 col=4",
        "move_to_cell row=7 col=1",
        "move_to_cell row=8 col=1",
        "rotate_to direction=S",
        "start_drop_off",
    ]

    scorer = PlanScorer()
    log_file = tmp_path / "evaluation.log"
    result = scorer.score_plan(
        plan_actions,
        world_id="3",
        checkpoints=[(2, 7), (8, 1)],
        log_path=log_file,
    )

    assert result.total_score == 600
    assert result.achieved_checkpoints == [(2, 7), (8, 1)]
    assert result.missed_checkpoints == []
    assert result.final_pose == (8, 1)

    statuses = [detail.status for detail in result.breakdown]
    assert statuses[2] == "checkpoint"
    assert statuses[4] == "obstacle_collision"
    assert statuses[5] == "valid"
    assert statuses[6] == "diagonal_move+obstacle_collision"

    assert log_file.exists()
    log_text = log_file.read_text()
    assert "Rewarded actions" in log_text
    assert "Penalized actions" in log_text
    assert "-150" in log_text


def test_additional_rules_penalize_revisit():
    plan_actions = [
        "move_to_cell row=4 col=5",
        "move_to_cell row=4 col=4",
    ]
    scorer = PlanScorer(enable_additional_rules=True)
    result = scorer.score_plan(
        plan_actions,
        world_id="3",
        checkpoints=[],
        initial_pose=(4, 4),
    )

    assert result.total_score == 980
    statuses = [detail.status for detail in result.breakdown]
    assert statuses == ["valid", "valid+revisit_penalty"]


def test_additional_rules_enforce_checkpoint_order_and_penalty():
    plan_actions = [
        "move_to_cell row=5 col=4",
        "move_to_cell row=5 col=6",
        "move_to_cell row=4 col=6",
        "move_to_cell row=4 col=5",
        "move_to_cell row=4 col=6",
    ]
    checkpoints = [(4, 5), (4, 6)]

    scorer = PlanScorer(enable_additional_rules=True)
    result = scorer.score_plan(
        plan_actions,
        world_id="3",
        checkpoints=checkpoints,
        initial_pose=(4, 4),
    )

    assert result.total_score == 855
    statuses = [detail.status for detail in result.breakdown]
    assert statuses[2] == "checkpoint_out_of_order"
    assert statuses[3] == "checkpoint"
    assert statuses[4] == "checkpoint+revisit_penalty"
    assert result.breakdown[4].score_delta == -20
    assert result.achieved_checkpoints == checkpoints
    assert result.missed_checkpoints == []


def test_diagonal_move_still_marks_checkpoint():
    plan_actions = [
        "move_to_cell row=5 col=2",
    ]
    scorer = PlanScorer()
    result = scorer.score_plan(
        plan_actions,
        world_id="3",
        checkpoints=[(5, 2)],
        initial_pose=(4, 1),
    )

    assert result.total_score == 750  # baseline 1000 minus penalties for diagonal + wall hit
    assert result.achieved_checkpoints == [(5, 2)]
    assert result.missed_checkpoints == []
    assert result.breakdown[0].status == "diagonal_move+obstacle_collision+checkpoint"


def test_diagonal_move_out_of_order_checkpoint_no_reward():
    plan_actions = [
        "move_to_cell row=5 col=2",
        "move_to_cell row=4 col=1",
        "move_to_cell row=4 col=2",
    ]
    checkpoints = [(4, 2), (5, 2)]
    scorer = PlanScorer(enable_additional_rules=True)
    result = scorer.score_plan(
        plan_actions,
        world_id="3",
        checkpoints=checkpoints,
        initial_pose=(4, 1),
    )

    assert result.total_score == 80
    statuses = [detail.status for detail in result.breakdown]
    assert statuses[0] == "diagonal_move+obstacle_collision+checkpoint_out_of_order"
    assert result.breakdown[0].score_delta == -375
    assert statuses[1] == "diagonal_move+obstacle_collision+revisit_penalty"
    assert statuses[2] == "obstacle_collision+checkpoint"
    assert statuses[3] == "missed_checkpoint"
    assert result.achieved_checkpoints == [(4, 2)]
    assert result.missed_checkpoints == [(5, 2)]
