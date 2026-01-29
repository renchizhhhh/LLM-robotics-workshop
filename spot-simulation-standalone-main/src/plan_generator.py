"""LLM-backed plan generation utilities."""

from __future__ import annotations

import json
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from message_bus import rospy

from timing_utils import recorder
from prompt_manager import load_prompt


@dataclass(slots=True)
class PlanGenerationResult:
    actions: List[str]
    raw_response: str
    plan_text: str
    prompt: str
    enforcement_applied: bool


class PlanGenerationError(RuntimeError):
    """Raised when the LLM-based plan generation fails."""


class PlanGenerator:
    def __init__(self, llm_router) -> None:
        self.llm_router = llm_router

    def generate_plan(
        self,
        command: str,
        current_state: Dict[str, Any],
        world_model: Dict[str, Any],
        prompt_name: str,
        *,
        enforce_wall_constraints: bool = True,
    ) -> PlanGenerationResult:
        if not self.llm_router:
            raise PlanGenerationError("LLM router is unavailable")

        prompt_template = load_prompt(prompt_name)
        formatted_prompt = prompt_template.format(
            current_state_json=json.dumps(current_state, indent=2),
            world_model_json=json.dumps(world_model, indent=2),
            command=command,
        )

        recorder.publish_event("start_llm_processing")
        try:
            response = self.llm_router.generate(
                prompt=formatted_prompt,
                options={"temperature": 0.0, "num_predict": 8192},
            )
        finally:
            recorder.publish_event("stop_llm_processing")

        if not response or "response" not in response:
            raise PlanGenerationError("LLM returned no response")

        raw_text = response["response"].strip()
        plan_text, extracted_actions = self._extract_plan(raw_text)

        if not extracted_actions:
            raise PlanGenerationError("LLM response did not contain any actions")

        if enforce_wall_constraints:
            actions = self._enforce_wall_constraints(extracted_actions, current_state, world_model)
            enforcement_applied = actions != extracted_actions
        else:
            actions = list(extracted_actions)
            enforcement_applied = False

        plan_text = "\n".join(actions)

        rospy.loginfo(f"PlanGenerator: Generated {len(actions)} actions")
        return PlanGenerationResult(
            actions=actions,
            raw_response=raw_text,
            plan_text=plan_text,
            prompt=formatted_prompt,
            enforcement_applied=enforcement_applied,
        )

    @staticmethod
    def _extract_plan(raw_text: str) -> tuple[str, List[str]]:
        plan_match = re.search(r"\[Plan\](.*?)\[/Plan\]", raw_text, re.DOTALL)
        if plan_match:
            plan_content = plan_match.group(1).strip()
        else:
            plan_content = raw_text

        actions: List[str] = []
        for line in plan_content.splitlines():
            cleaned = PlanGenerator._clean_line(line)
            if cleaned:
                actions.append(cleaned)
        return plan_content, actions

    @staticmethod
    def _clean_line(line: str) -> Optional[str]:
        stripped = line.strip()
        if not stripped:
            return None

        if "#" in stripped:
            stripped = stripped.split("#", 1)[0].strip()
        if not stripped:
            return None

        skip_prefixes = [
            "#",
            "Technical Steps:",
            "Reasoning:",
            "Steps:",
            "Process:",
        ]
        if any(stripped.startswith(prefix) for prefix in skip_prefixes):
            return None

        enumerated_prefixes = tuple(f"{i}. #" for i in range(1, 11))
        if stripped.startswith(enumerated_prefixes):
            return None
        return stripped

    def _enforce_wall_constraints(
        self,
        actions: List[str],
        current_state: Dict[str, Any],
        world_model: Dict[str, Any],
    ) -> List[str]:
        walls = {tuple(cell) for cell in world_model.get("obstacle_cells", [])}

        rows: List[int] = [cell[0] for cell in walls]
        cols: List[int] = [cell[1] for cell in walls]

        for location in world_model.get("checkpoints", {}).values():
            rows.append(location.get("row", 0))
            cols.append(location.get("col", 0))

        for location in world_model.get("waypoints", {}).values():
            rows.append(location.get("row", 0))
            cols.append(location.get("col", 0))

        max_row = max(rows, default=9)
        max_col = max(cols, default=9)

        repaired: List[str] = []
        # Use rounded position, but prefer the exact target if it matches
        robot_cell = current_state.get("robot_cell", [4, 4])
        current_pos = tuple(int(round(x)) for x in robot_cell)
        rospy.loginfo(f"PlanGenerator: Starting from position {current_pos} (from robot_cell {robot_cell})")

        for action in actions:
            target = self._parse_move_to_cell(action)
            if target is None:
                repaired.append(action)
                continue

            # If we're already at the target (within rounding), skip path planning
            if current_pos == target:
                rospy.loginfo(f"PlanGenerator: Already at target {target}, skipping redundant move")
                continue

            rospy.loginfo(f"PlanGenerator: Planning path from {current_pos} to {target}")
            segments = self._plan_safe_segments(current_pos, target, walls, max_row, max_col)
            rospy.loginfo(f"PlanGenerator: Generated {len(segments)} segment(s): {segments}")
            if not segments:
                rospy.logwarn(f"PlanGenerator: Unable to find wall-safe path from {current_pos} to {target}; keeping original action")
                repaired.append(action)
                current_pos = target
            else:
                # Filter out redundant segments that match current position
                filtered_segments = []
                for seg in segments:
                    seg_target = self._parse_move_to_cell(seg)
                    if seg_target is None or seg_target != current_pos:
                        filtered_segments.append(seg)
                    elif seg_target == current_pos:
                        rospy.loginfo(f"PlanGenerator: Filtering out redundant segment to current position: {seg}")
                if filtered_segments:
                    repaired.extend(filtered_segments)
                    # Update current_pos to the last segment's target
                    last_seg_target = self._parse_move_to_cell(filtered_segments[-1])
                    if last_seg_target:
                        current_pos = last_seg_target
                else:
                    # All segments were redundant, just update position
                    current_pos = target

        return repaired

    @staticmethod
    def _parse_move_to_cell(action: str) -> Optional[Tuple[int, int]]:
        action = action.strip()
        if not action.startswith("move_to_cell"):
            return None

        parts = action.replace(",", " ").split()
        row_value: Optional[int] = None
        col_value: Optional[int] = None

        for part in parts:
            if part.startswith("row="):
                try:
                    row_value = int(float(part.split("=", 1)[1]))
                except ValueError:
                    return None
            elif part.startswith("col="):
                try:
                    col_value = int(float(part.split("=", 1)[1]))
                except ValueError:
                    return None

        if row_value is None or col_value is None:
            return None
        return (row_value, col_value)

    def _plan_safe_segments(
        self,
        start: Tuple[int, int],
        target: Tuple[int, int],
        walls: set[Tuple[int, int]],
        max_row: int,
        max_col: int,
    ) -> List[str]:
        if start == target:
            return []

        if self._path_is_clear(start, target, walls):
            rospy.loginfo(f"PlanGenerator: Path from {start} to {target} is clear, using direct movement")
            return [f"move_to_cell row={target[0]} col={target[1]}"]

        rospy.loginfo(f"PlanGenerator: Path from {start} to {target} is not clear, using BFS pathfinding")
        bounds_row = max(max_row, start[0], target[0])
        bounds_col = max(max_col, start[1], target[1])
        path = self._bfs_path(start, target, walls, bounds_row, bounds_col)
        if not path:
            rospy.logwarn(f"PlanGenerator: BFS could not find path from {start} to {target}")
            return []
        rospy.loginfo(f"PlanGenerator: BFS found path with {len(path)} cells: {path}")

        segments: List[str] = []
        seg_start = path[0]
        prev = path[0]
        direction: Optional[str] = None

        for cell in path[1:]:
            if cell[0] == prev[0]:
                step_dir = "h"
            elif cell[1] == prev[1]:
                step_dir = "v"
            else:
                rospy.logwarn(f"PlanGenerator: Non-axis step encountered between {prev} and {cell}")
                return []

            if direction is None:
                direction = step_dir
            elif step_dir != direction:
                # Direction changed - save segment for previous direction (but not if it's the start)
                if seg_start != prev and prev != start:
                    segments.append(f"move_to_cell row={prev[0]} col={prev[1]}")
                seg_start = prev
                direction = step_dir

            prev = cell

        # Add final segment if it's not the start position
        if prev != start and (not segments or segments[-1] != f"move_to_cell row={prev[0]} col={prev[1]}"):
            segments.append(f"move_to_cell row={prev[0]} col={prev[1]}")

        # Ensure target is included (but don't add if it matches start)
        if target != start and (not segments or segments[-1] != f"move_to_cell row={target[0]} col={target[1]}"):
            segments.append(f"move_to_cell row={target[0]} col={target[1]}")

        return segments

    @staticmethod
    def _path_is_clear(
        start: Tuple[int, int],
        target: Tuple[int, int],
        walls: set[Tuple[int, int]],
    ) -> bool:
        if start[0] == target[0]:
            row = start[0]
            c1, c2 = sorted((start[1], target[1]))
            for col in range(c1 + 1, c2):
                if (row, col) in walls:
                    return False
            return True

        if start[1] == target[1]:
            col = start[1]
            r1, r2 = sorted((start[0], target[0]))
            for row in range(r1 + 1, r2):
                if (row, col) in walls:
                    return False
            return True

        return False

    @staticmethod
    def _bfs_path(
        start: Tuple[int, int],
        target: Tuple[int, int],
        walls: set[Tuple[int, int]],
        max_row: int,
        max_col: int,
    ) -> List[Tuple[int, int]]:
        queue: deque[Tuple[int, int]] = deque([start])
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        while queue:
            current = queue.popleft()
            if current == target:
                break

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = current[0] + dr
                nc = current[1] + dc

                if nr < 0 or nc < 0 or nr > max_row or nc > max_col:
                    continue
                if (nr, nc) in walls:
                    continue

                neighbor = (nr, nc)
                if neighbor in came_from:
                    continue

                came_from[neighbor] = current
                queue.append(neighbor)

        if target not in came_from:
            return []

        path: List[Tuple[int, int]] = []
        node: Optional[Tuple[int, int]] = target
        while node is not None:
            path.append(node)
            node = came_from[node]
        path.reverse()
        return path
