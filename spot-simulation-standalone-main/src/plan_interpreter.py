"""Helpers to convert robot plans into natural language."""

from __future__ import annotations

import re
from typing import Iterable, List

from message_bus import rospy


PLAN_INTERPRETATION_PROMPT = """You are a robot plan interpreter. Convert a programmatic robot plan into natural language.

GRID COORDINATE SYSTEM:
- Grid is 10x10 cells (0-9 for both row and col)
- Row 0 = top, Row 9 = bottom
- Col 0 = left, Col 9 = right
- N = North (up), S = South (down), E = East (right), W = West (left)

ROBOT ACTIONS:
- stand_up: Robot stands up
- sit_down: Robot sits down  
- move_to_cell row=R col=C: Robot moves to cell (R, C)
- rotate_to direction=D: Robot rotates to face direction D (N/S/E/W)
- start_automated_grasp object_type="OBJECT": Robot grasps the OBJECT
- start_drop_off: Robot performs full drop-off sequence (move arm to position, open gripper, close gripper, stow arm)

PLAN: {plan}

INSTRUCTIONS:
1. Write in natural, conversational language as if explaining to a friend
2. Use flowing sentences, not bullet points or numbered lists
3. Include ALL steps from the plan - do not skip any actions
4. Use correct directional language: "north" for N, "south" for S, "east" for E, "west" for W
5. For movements: describe the path naturally, mentioning key locations using "row X, column Y" format
6. For rotations: mention the direction the robot will face
7. For grasping: mention the specific object being picked up
8. For drop-off: mention where the robot will deliver the item
9. Group similar movements together when they form a logical sequence
10. Be specific about locations and objects
11. Write as a single flowing paragraph or a few connected sentences
12. Avoid formatting like **bold** or bullet points - just natural speech

NATURAL LANGUAGE DESCRIPTION:"""


def generate_fallback_interpretation(plan: Iterable[str]) -> str:
    steps: List[str] = []
    for raw_action in plan:
        action = raw_action.lower().strip()
        if action == "stand_up":
            steps.append("stand up")
            continue
        if action == "sit_down":
            steps.append("sit down")
            continue
        if action.startswith("move_to_cell"):
            row, col = _extract_pair(action, "row", "col")
            if row is not None and col is not None:
                steps.append(f"move to row {row}, column {col}")
            else:
                steps.append("move to a new location")
            continue
        if action.startswith("rotate_to"):
            direction = _extract_single(action, "direction")
            if direction:
                steps.append(f"turn to face { _direction_name(direction) }")
            else:
                steps.append("turn to a new direction")
            continue
        if action.startswith("start_automated_grasp"):
            object_name = _extract_string(action, "object_type")
            if object_name:
                steps.append(f"pick up the {object_name}")
            else:
                steps.append("pick up an object")
            continue
        if action == "start_drop_off":
            steps.append("deliver the item to the destination")
            continue
        steps.append(f"perform action: {raw_action}")

    if not steps:
        return "No plan generated"
    if len(steps) == 1:
        return f"The robot will {steps[0]}."
    if len(steps) == 2:
        return f"The robot will {steps[0]} and then {steps[1]}."
    if len(steps) == 3:
        return f"The robot will {steps[0]}, then {steps[1]}, and finally {steps[2]}."
    return f"The robot will {', '.join(steps[:-1])}, and finally {steps[-1]}."


class PlanInterpreter:
    def __init__(self, llm_router) -> None:
        self.llm_router = llm_router

    def interpret(self, plan: List[str]) -> str:
        if not plan:
            return "No plan generated"

        if not self.llm_router:
            return generate_fallback_interpretation(plan)

        plan_text = "\n".join(f"{idx + 1}. {action}" for idx, action in enumerate(plan))
        try:
            response = self.llm_router.generate(
                prompt=PLAN_INTERPRETATION_PROMPT.format(plan=plan_text),
                options={"temperature": 0.3, "num_predict": 150},
                use_20b=True,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            rospy.logwarn(f"PlanInterpreter: interpretation failed: {exc}")
            return generate_fallback_interpretation(plan)

        if response and "response" in response:
            interpretation = response["response"].strip()
            if interpretation:
                rospy.loginfo("PlanInterpreter: Generated LLM interpretation")
                return interpretation

        rospy.logwarn("PlanInterpreter: Falling back to heuristic interpretation")
        return generate_fallback_interpretation(plan)


def _extract_pair(action: str, first_key: str, second_key: str) -> tuple[str | None, str | None]:
    first = _extract_single(action, first_key)
    second = _extract_single(action, second_key)
    return first, second


def _extract_single(action: str, key: str) -> str | None:
    pattern = rf"{key}=([0-9A-Za-z]+)"
    match = re.search(pattern, action)
    if match:
        return match.group(1)
    return None


def _extract_string(action: str, key: str) -> str | None:
    pattern = rf'{key}="([^"\\]+)"'
    match = re.search(pattern, action)
    if match:
        return match.group(1)
    return None


def _direction_name(direction: str) -> str:
    mapping = {"N": "north", "S": "south", "E": "east", "W": "west"}
    return mapping.get(direction.upper(), direction)


__all__ = [
    "PlanInterpreter",
    "PLAN_INTERPRETATION_PROMPT",
    "generate_fallback_interpretation",
]
