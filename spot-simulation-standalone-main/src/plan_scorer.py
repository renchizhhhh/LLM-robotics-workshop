#!/usr/bin/env python3
"""Score LLM-generated Spot plans against simple rule checks and checkpoints."""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple, Union

from world_manager import WorldManager

# Precompile for basic plan parsing
_ROW_REGEX = re.compile(r"row\s*=\s*(-?\d+)")
_COL_REGEX = re.compile(r"col\s*=\s*(-?\d+)")
_DIR_REGEX = re.compile(r"direction\s*=\s*([NSEW])", re.IGNORECASE)


@dataclass
class ActionScore:
    index: int
    action: str
    score_delta: int
    status: str
    resulting_pose: Tuple[int, int]


@dataclass
class ActionOutcome:
    action_type: str
    new_pose: Tuple[int, int]
    delta: int
    status_tags: List[str]
    path_cells: List[Tuple[int, int]]
    checkpoint_penalty: int = 0

    @property
    def base_status(self) -> str:
        return self.status_tags[0]


@dataclass
class _CheckpointResult:
    tag: Optional[str] = None
    penalty: int = 0
    replace_base: bool = False


class _CheckpointTracker:
    """Track checkpoint progress and enforce ordering when enabled."""

    def __init__(
        self,
        checkpoints: Sequence[Tuple[int, int]],
        penalty: int,
        enforce_order: bool,
    ) -> None:
        self._ordered: List[Tuple[int, int]] = list(checkpoints)
        self._set: Set[Tuple[int, int]] = set(self._ordered)
        self._penalty = penalty
        self._enforce_order = enforce_order
        self._next_index = 0
        self.achieved: List[Tuple[int, int]] = []
        self._achieved_set: Set[Tuple[int, int]] = set()

    def record(self, target: Tuple[int, int], base_status: str) -> _CheckpointResult:
        if target not in self._set:
            return _CheckpointResult()

        tag: Optional[str]
        penalty = 0

        if self._enforce_order:
            if self._next_index < len(self._ordered) and target == self._ordered[self._next_index]:
                tag = "checkpoint"
                if target not in self._achieved_set:
                    self.achieved.append(target)
                    self._achieved_set.add(target)
                self._next_index += 1
            else:
                tag = "checkpoint_out_of_order"
                penalty = self._penalty
        else:
            tag = "checkpoint"
            if target not in self._achieved_set:
                self.achieved.append(target)
                self._achieved_set.add(target)

        replace_base = bool(tag) and base_status == "valid"
        return _CheckpointResult(tag=tag, penalty=penalty, replace_base=replace_base)

    def missed_checkpoints(self) -> List[Tuple[int, int]]:
        if self._enforce_order:
            return self._ordered[self._next_index :]
        return [cp for cp in self._ordered if cp not in self._achieved_set]


@dataclass
class ScoreResult:
    total_score: int
    breakdown: List[ActionScore]
    achieved_checkpoints: List[Tuple[int, int]]
    missed_checkpoints: List[Tuple[int, int]]
    final_pose: Tuple[int, int]


class PlanScorer:
    """Assigns simple numeric scores to Spot plans using maze constraints."""

    def __init__(
        self,
        *,
        world_manager: Optional[WorldManager] = None,
        violation_penalty: int = 100,
        valid_action_score: int = 0,
        enable_additional_rules: bool = False,
        baseline_score: int = 1000,
        additional_wall_penalty: int = 50,
        revisit_penalty: int = 20,
        out_of_order_penalty: int = 125,
        missed_checkpoint_penalty: int = 125,
        out_of_bounds_penalty: int = 150,
    ) -> None:
        self.world_manager = world_manager or WorldManager()
        self.rule_violation_penalty = abs(violation_penalty)
        self.valid_action_score = valid_action_score
        self.enable_additional_rules = enable_additional_rules
        self.baseline_score = baseline_score
        self.additional_wall_penalty = abs(additional_wall_penalty)
        self._revisit_penalty_value = abs(revisit_penalty)
        self.out_of_order_penalty = abs(out_of_order_penalty)
        self.missed_checkpoint_penalty = abs(missed_checkpoint_penalty)
        self.out_of_bounds_penalty = abs(out_of_bounds_penalty)

    def score_plan(
        self,
        actions: Sequence[str],
        *,
        world_id: str,
        checkpoints: Sequence[Tuple[int, int]],
        initial_pose: Optional[Tuple[int, int]] = None,
        log_path: Optional[Union[str, Path]] = None,
    ) -> ScoreResult:
        if not actions:
            raise ValueError("Plan is empty; provide at least one action")

        config = self.world_manager.get_world_config(world_id)
        if not config:
            raise ValueError(f"Unknown world_id '{world_id}'")

        rows, cols = self.world_manager.get_world_dimensions(world_id)
        walls = {tuple(cell) for cell in config.get("obstacle_cells", [])}
        pose = initial_pose or self._resolve_initial_pose(world_id)
        pose = self._coerce_pose(pose)
        ordered_checkpoints = [self._coerce_pose(cp) for cp in checkpoints]
        checkpoint_tracker = _CheckpointTracker(
            ordered_checkpoints,
            penalty=self.out_of_order_penalty,
            enforce_order=self.enable_additional_rules,
        )
        breakdown: List[ActionScore] = []
        total = self.baseline_score
        visited_cells: Set[Tuple[int, int]] = {pose}
        obstacle_collision_count = 0

        # If all actions are FAIL/ignored placeholders, return zero without penalties
        def _has_valid_action(seq: Sequence[str]) -> bool:
            for a in seq:
                if not a:
                    continue
                if isinstance(a, str) and a.strip().upper().startswith("FAIL"):
                    continue
                return True
            return False

        def _contains_fail(seq: Sequence[str]) -> bool:
            for a in seq:
                if isinstance(a, str) and a.strip().upper().startswith("FAIL"):
                    return True
            return False

        if _contains_fail(actions):
            result = ScoreResult(0, [], [], ordered_checkpoints, pose)
            if log_path:
                self._write_log(result, log_path, world_id, checkpoints)
            return result

        if not _has_valid_action(actions):
            result = ScoreResult(0, [], [], ordered_checkpoints, pose)
            if log_path:
                self._write_log(result, log_path, world_id, checkpoints)
            return result

        for idx, raw_action in enumerate(actions):
            action = raw_action.strip()
            if not action:
                continue

            outcome = self._evaluate_action(
                action=action,
                pose=pose,
                rows=rows,
                cols=cols,
                walls=walls,
                checkpoint_tracker=checkpoint_tracker,
            )

            delta = 0
            status_tags = list(outcome.status_tags)

            if (
                self.enable_additional_rules
                and outcome.action_type == "move"
                and outcome.base_status not in {"invalid_move", "out_of_bounds", "invalid_rotation"}
            ):
                path_cells = outcome.path_cells
                revisit_detected = any(cell in visited_cells for cell in path_cells[1:])
                if revisit_detected:
                    delta -= self._revisit_penalty_value
                    status_tags.append("revisit_penalty")
                for cell in path_cells[1:]:
                    visited_cells.add(cell)

            violation_tags = {"obstacle_collision", "diagonal_move", "invalid_move", "out_of_bounds", "invalid_rotation"}
            current_tags = set(status_tags)
            if current_tags & violation_tags:
                multiplier = 2 if {"obstacle_collision", "diagonal_move"} <= current_tags else 1
                delta -= self.rule_violation_penalty * multiplier

            if "out_of_bounds" in current_tags:
                delta -= self.out_of_bounds_penalty

            if "obstacle_collision" in status_tags:
                obstacle_collision_count += 1
                if obstacle_collision_count >= 1:
                    delta -= self.additional_wall_penalty

            if outcome.checkpoint_penalty:
                delta -= outcome.checkpoint_penalty

            total += delta
            breakdown.append(ActionScore(idx, action, delta, "+".join(status_tags), outcome.new_pose))

            if outcome.base_status not in {"invalid_move", "out_of_bounds", "invalid_rotation"}:
                pose = outcome.new_pose

        missed = checkpoint_tracker.missed_checkpoints()
        for missed_cp in missed:
            penalty = self.missed_checkpoint_penalty
            total -= penalty
            breakdown.append(
                ActionScore(
                    len(breakdown),
                    f"missed_checkpoint row={missed_cp[0]} col={missed_cp[1]}",
                    -penalty,
                    "missed_checkpoint",
                    pose,
                )
            )
        result = ScoreResult(total, breakdown, checkpoint_tracker.achieved, missed, pose)

        if log_path:
            self._write_log(result, log_path, world_id, checkpoints)

        return result

    def score_plan_from_text(
        self,
        plan_text: str,
        *,
        world_id: str,
        checkpoints: Sequence[Tuple[int, int]],
        initial_pose: Optional[Tuple[int, int]] = None,
        log_path: Optional[Union[str, Path]] = None,
    ) -> ScoreResult:
        actions = [line.strip() for line in plan_text.splitlines() if line.strip()]
        if not actions:
            raise ValueError("No parsable actions found in plan text")
        return self.score_plan(
            actions,
            world_id=world_id,
            checkpoints=checkpoints,
            initial_pose=initial_pose,
            log_path=log_path,
        )

    def _evaluate_action(
        self,
        *,
        action: str,
        pose: Tuple[int, int],
        rows: int,
        cols: int,
        walls: Set[Tuple[int, int]],
        checkpoint_tracker: _CheckpointTracker,
    ) -> ActionOutcome:
        lower = action.lower()

        if lower.startswith("move_to_cell"):
            return self._score_move(
                action=action,
                pose=pose,
                rows=rows,
                cols=cols,
                walls=walls,
                checkpoint_tracker=checkpoint_tracker,
            )

        if lower.startswith("rotate_to"):
            return self._score_rotate(action=action, pose=pose)

        if lower.startswith("start_automated_grasp") or lower.startswith("start_drop_off"):
            return self._simple_outcome("task", pose, "task_action", self.valid_action_score)

        if lower.startswith("stand_up") or lower.startswith("sit_down"):
            return self._simple_outcome("posture", pose, "posture", self.valid_action_score)

        return self._simple_outcome("ignored", pose, "ignored", self.valid_action_score)

    def _score_move(
        self,
        *,
        action: str,
        pose: Tuple[int, int],
        rows: int,
        cols: int,
        walls: Set[Tuple[int, int]],
        checkpoint_tracker: _CheckpointTracker,
    ) -> ActionOutcome:
        target = self._extract_coordinate(action)
        if target is None:
            return self._simple_outcome("move", pose, "invalid_move", 0)

        target = self._coerce_pose(target)
        if not self._in_bounds(target, rows, cols):
            return self._simple_outcome("move", pose, "out_of_bounds", 0)

        new_pose = target
        path_cells = self._enumerate_path(pose, new_pose) if new_pose != pose else [pose]

        status_tags: List[str] = []

        if self._is_diagonal_move(pose, target):
            status_tags.append("diagonal_move")

        if self._line_hits_wall(pose, target, walls):
            status_tags.append("obstacle_collision")

        if not status_tags:
            status_tags = ["valid"]

        checkpoint_result = checkpoint_tracker.record(target, status_tags[0])
        if checkpoint_result.tag:
            if checkpoint_result.replace_base:
                status_tags = [checkpoint_result.tag]
            elif checkpoint_result.tag not in status_tags:
                status_tags.append(checkpoint_result.tag)

        return ActionOutcome(
            action_type="move",
            new_pose=new_pose,
            delta=0,
            status_tags=status_tags,
            path_cells=path_cells,
            checkpoint_penalty=checkpoint_result.penalty,
        )

    def _score_rotate(self, *, action: str, pose: Tuple[int, int]) -> ActionOutcome:
        direction = self._extract_direction(action)
        if direction is None:
            return self._simple_outcome("rotate", pose, "invalid_rotation", 0)
        return self._simple_outcome("rotate", pose, "valid", self.valid_action_score)

    def _simple_outcome(
        self,
        action_type: str,
        pose: Tuple[int, int],
        status: str,
        delta: int,
    ) -> ActionOutcome:
        return ActionOutcome(
            action_type=action_type,
            new_pose=pose,
            delta=delta,
            status_tags=[status],
            path_cells=[pose],
            checkpoint_penalty=0,
        )

    def _resolve_initial_pose(self, world_id: str) -> Tuple[int, int]:
        state = self.world_manager.get_initial_state(world_id)
        if not state:
            return (4, 4)
        return (int(state.get("robot_row", 4)), int(state.get("robot_col", 4)))

    @staticmethod
    def _coerce_pose(pose: Tuple[float, float]) -> Tuple[int, int]:
        row, col = pose
        return int(math.floor(row + 0.5)), int(math.floor(col + 0.5))

    @staticmethod
    def _extract_coordinate(action: str) -> Optional[Tuple[int, int]]:
        row_match = _ROW_REGEX.search(action)
        col_match = _COL_REGEX.search(action)
        if not row_match or not col_match:
            return None
        try:
            return int(row_match.group(1)), int(col_match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _extract_direction(action: str) -> Optional[str]:
        match = _DIR_REGEX.search(action)
        if not match:
            return None
        direction = match.group(1).upper()
        if direction not in {"N", "S", "E", "W"}:
            return None
        return direction

    @staticmethod
    def _in_bounds(cell: Tuple[int, int], rows: int, cols: int) -> bool:
        row, col = cell
        return 0 <= row < rows and 0 <= col < cols

    @staticmethod
    def _is_diagonal_move(start: Tuple[int, int], target: Tuple[int, int]) -> bool:
        """Check if a move is diagonal (not axis-aligned).
        
        A move is diagonal if it changes both row and col coordinates.
        Valid axis-aligned moves only change row OR col, not both.
        """
        start_row, start_col = PlanScorer._coerce_pose(start)
        target_row, target_col = PlanScorer._coerce_pose(target)
        
        row_change = start_row != target_row
        col_change = start_col != target_col
        
        # Diagonal if both row and col change
        return row_change and col_change

    @staticmethod
    def _line_hits_wall(
        start: Tuple[int, int],
        target: Tuple[int, int],
        walls: Set[Tuple[int, int]],
    ) -> bool:
        (sr, sc) = start
        (tr, tc) = target

        if start == target:
            return False

        # Axis-aligned movement: simple straight-line scan
        if sr == tr or sc == tc:
            primary = "row" if sc == tc else "col"
            path = PlanScorer._axis_path_cells(start, target, primary)
            return any(cell in walls for cell in path)

        # Diagonal or L-shaped movement: consider both row-first and column-first traversals.
        row_first_path = PlanScorer._axis_path_cells(start, target, "row")
        if any(cell in walls for cell in row_first_path):
            return True

        col_first_path = PlanScorer._axis_path_cells(start, target, "col")
        return any(cell in walls for cell in col_first_path)

    @staticmethod
    def _axis_path_cells(
        start: Tuple[int, int],
        target: Tuple[int, int],
        primary_axis: str,
    ) -> List[Tuple[int, int]]:
        """Generate inclusive Manhattan path, prioritising the chosen axis first."""
        sr, sc = start
        tr, tc = target
        path: List[Tuple[int, int]] = []
        r, c = sr, sc

        def _step_towards(current: int, goal: int) -> int:
            if current == goal:
                return current
            return current + (1 if goal > current else -1)

        if primary_axis == "row":
            while r != tr:
                r = _step_towards(r, tr)
                path.append((r, c))
            while c != tc:
                c = _step_towards(c, tc)
                path.append((r, c))
        else:
            while c != tc:
                c = _step_towards(c, tc)
                path.append((r, c))
            while r != tr:
                r = _step_towards(r, tr)
                path.append((r, c))

        return path

    @staticmethod
    def _enumerate_path(start: Tuple[int, int], target: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Enumerate inclusive path between two axis-aligned cells."""
        path = [start]
        sr, sc = start
        tr, tc = target
        if start == target:
            return path

        if sr == tr:
            step = 1 if tc > sc else -1
            for c in range(sc + step, tc + step, step):
                path.append((sr, c))
            return path

        if sc == tc:
            step = 1 if tr > sr else -1
            for r in range(sr + step, tr + step, step):
                path.append((r, sc))
            return path

        path.append(target)
        return path

    def _write_log(
        self,
        result: ScoreResult,
        log_path: Union[str, Path],
        world_id: str,
        checkpoints: Sequence[Tuple[int, int]],
    ) -> None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rewarded = [entry for entry in result.breakdown if entry.score_delta > 0]
        penalized = [entry for entry in result.breakdown if entry.score_delta < 0]
        neutral = [entry for entry in result.breakdown if entry.score_delta == 0]

        def _format_entry(entry: ActionScore) -> str:
            if entry.score_delta > 0:
                delta = f"+{entry.score_delta}"
            elif entry.score_delta < 0:
                delta = f"{entry.score_delta}"
            else:
                delta = "0"
            return (
                f"  [{entry.index}] {delta} | {entry.status} | {entry.action} "
                f"-> pose={entry.resulting_pose}"
            )

        lines: List[str] = []
        lines.append("Plan Evaluation Log")
        lines.append(f"World: {world_id}")
        lines.append(f"Checkpoints: {list(checkpoints)}")
        lines.append(f"Total score: {result.total_score}")
        lines.append(f"Achieved checkpoints: {result.achieved_checkpoints}")
        lines.append(f"Missed checkpoints: {result.missed_checkpoints}")
        lines.append("")

        lines.append("Rewarded actions:")
        if rewarded:
            lines.extend(_format_entry(entry) for entry in rewarded)
        else:
            lines.append("  (none)")
        lines.append("")

        lines.append("Penalized actions:")
        if penalized:
            lines.extend(_format_entry(entry) for entry in penalized)
        else:
            lines.append("  (none)")
        lines.append("")

        lines.append("Neutral actions:")
        if neutral:
            lines.extend(_format_entry(entry) for entry in neutral)
        else:
            lines.append("  (none)")

        path.write_text("\n".join(lines))


if __name__ == "__main__":
    # go to the pickup and then exit in maze 3
    demo_plan = [
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
    result = scorer.score_plan(
        demo_plan,
        world_id="3",
        checkpoints=[(2, 7), (8, 1)],
        log_path=Path("plan_scorer_demo.log"),
    )
    print(f"Total score: {result.total_score}")
    for detail in result.breakdown:
        print(f"[{detail.index}] {detail.action} -> {detail.score_delta} ({detail.status})")
