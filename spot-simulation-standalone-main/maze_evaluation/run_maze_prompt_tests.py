#!/usr/bin/env python3
"""Run maze prompt evaluations and log robot positions to CSV."""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import re

# =============================================================================
# CONFIGURATION - Edit these settings to customize the test run
# =============================================================================

# Test command to send to the LLM
TEST_COMMAND = "Move to the pick up position and then go to the maze exit"

# Initial robot position(s)
# Can be a list of (row, col) tuples to test multiple start poses
INITIAL_ROBOT_POSITIONS = [(3,1),(3,2),(3,3), (4, 5), (5, 4), (6, 3),(6,6),(7,6),(8,6)]
# INITIAL_ROBOT_POSITIONS = [(4, 5)]
# Prompt(s) to test - Options: "base", "reasoning", "stupid_maze"
# Leave as None to test all available prompts
# Examples:
#   PROMPTS_TO_TEST = ["base"]              # Test only base prompt
#   PROMPTS_TO_TEST = ["base", "reasoning"] # Test multiple prompts
#   PROMPTS_TO_TEST = None                  # Test all available prompts
PROMPTS_TO_TEST = ["stupid_maze"]  # Test all available prompts by default

# LLM Model to use - Options: "base-120b-low", "gemini-robotics-er-1.5-preview", "gemini-2.5-pro"
# Note: Gemini models require GEMINI_API_KEY environment variable
LLM_MODEL = "base-120b-high"

# Maze world(s) to test - Options: "11", "12", "13", "14" (or any maze world ID)
# Leave as None to test all maze worlds
# Examples:
#   MAZE_WORLDS_TO_TEST = ["11"]        # Test only world 11
#   MAZE_WORLDS_TO_TEST = ["11", "12"]  # Test multiple worlds
#   MAZE_WORLDS_TO_TEST = None          # Test all maze worlds
MAZE_WORLDS_TO_TEST = ["14"]  # Test only world 11 by default

# Output directory for test results
OUTPUT_DIRECTORY = "maze_prompt_logs"

# LLM generation options
LLM_TEMPERATURE = 0.00000001
LLM_MAX_TOKENS = 16192

# Number of times to repeat each test
REPEAT_COUNT = 20

# =============================================================================
# END CONFIGURATION
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
# Also ensure the repository root is on sys.path so modules at project root (e.g. ollama_client.py)
# can be imported when this script is executed from the maze_evaluation/ folder.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from world_manager import WorldManager  # type: ignore
from nl_control import load_prompt, AVAILABLE_PROMPTS, LLMRouter  # type: ignore

PLAN_REGEX = re.compile(r"\[Plan\](.*?)\[/Plan\]", re.DOTALL)
ROW_REGEX = re.compile(r"row\s*=\s*(-?\d+)")
COL_REGEX = re.compile(r"col\s*=\s*(-?\d+)")


def format_pose(pose: Tuple[int, int]) -> str:
    return f"({pose[0]},{pose[1]})"


def extract_response_text(response) -> Optional[str]:
    if isinstance(response, dict):
        for key in ("response", "output_text", "text", "message"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    elif isinstance(response, str):
        value = response.strip()
        if value:
            return value
    return None


def extract_plan_section(response_text: str) -> Tuple[str, Optional[str]]:
    match = PLAN_REGEX.search(response_text)
    if match:
        return match.group(1).strip(), response_text
    return response_text.strip(), None


def extract_actions(plan_section: str) -> List[str]:
    actions: List[str] = []
    skip_prefixes = (
        "#",
        "Reasoning:",
        "Technical Steps:",
        "Steps:",
        "Process:",
        "Thought:",
        "Reflection:",
        "Summary:"
    )
    for raw_line in plan_section.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(line.startswith(prefix) for prefix in skip_prefixes):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if line:
            actions.append(line)
    return actions


def parse_int_param(text: str, regex: re.Pattern) -> Optional[int]:
    match = regex.search(text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def accumulate_positions(actions: Sequence[str], initial_pose: Tuple[int, int]) -> List[Tuple[int, int]]:
    positions: List[Tuple[int, int]] = [initial_pose]
    current_row, current_col = initial_pose
    for action in actions:
        row = parse_int_param(action, ROW_REGEX)
        col = parse_int_param(action, COL_REGEX)
        if action.lower().startswith("move_to_cell") and row is not None and col is not None:
            current_row, current_col = row, col
        positions.append((current_row, current_col))
    return positions


def build_prompt(prompt_name: str, world_config: Dict, command: str, initial_pose: Tuple[int, int]) -> Tuple[str, Dict[str, object]]:
    prompt_template = load_prompt(prompt_name)
    current_state = {
        "robot_state": "Stand",
        "standing": True,
        "robot_cell": list(initial_pose),
        "facing": "N",
        "arm": "stowed",
        "held_object": None
    }
    prompt_text = prompt_template.format(
        current_state_json=json.dumps(current_state, indent=2),
        world_model_json=json.dumps(world_config, indent=2),
        command=command
    )
    return prompt_text, current_state


def run_single_test(router: LLMRouter, prompt_name: str, world_id: str, world_name: str,
                    world_config: Dict, command: str, initial_pose: Tuple[int, int],
                    output_detail_dir: Path, temperature: float = 0.1, 
                    max_tokens: int = 8192, repeat_idx: Optional[int] = None) -> Dict[str, object]:
    prompt_text, current_state = build_prompt(prompt_name, world_config, command, initial_pose)
    response = router.generate(
        prompt=prompt_text,
        options={
            "temperature": temperature,
            "num_predict": max_tokens
        }
    )
    response_text = extract_response_text(response)
    if not response_text:
        error_msg = "No response from LLM"
        return {
            "prompt": prompt_name,
            "world_id": world_id,
            "world_name": world_name,
            "error": error_msg,
            "prompt_text": prompt_text,
            "response_raw": response
        }

    plan_section, full_text = extract_plan_section(response_text)
    actions = extract_actions(plan_section)
    positions = accumulate_positions(actions, initial_pose)

    # Add repeat index to filename if provided
    repeat_suffix = f"_run{repeat_idx}" if repeat_idx is not None else ""
    detail_path = output_detail_dir / f"prompt_{prompt_name}_world_{world_id}{repeat_suffix}.txt"
    detail_payload = {
        "prompt_name": prompt_name,
        "world_id": world_id,
        "world_name": world_name,
        "command": command,
        "initial_pose": initial_pose,
        "current_state": current_state,
        "actions": actions,
        "positions": positions,
        "prompt_text": prompt_text,
        "response_text": full_text or response_text,
        "raw_response": response
    }
    detail_path.write_text(json.dumps(detail_payload, indent=2))

    return {
        "prompt": prompt_name,
        "world_id": world_id,
        "world_name": world_name,
        "actions": actions,
        "positions": positions,
        "detail_file": detail_path.name
    }


def collect_maze_worlds(world_manager: WorldManager, explicit_worlds: Optional[List[str]]) -> List[Tuple[str, str, Dict]]:
    maze_worlds: List[Tuple[str, str, Dict]] = []
    for world_id, name, _ in world_manager.get_world_list():
        if explicit_worlds and world_id not in explicit_worlds:
            continue
        if not explicit_worlds and "maze" not in name.lower():
            continue
        config = world_manager.get_world_config(world_id)
        if config:
            maze_worlds.append((world_id, name, config))
    if explicit_worlds:
        missing = sorted(set(explicit_worlds) - {w[0] for w in maze_worlds})
        if missing:
            raise ValueError(f"Unknown world ids: {', '.join(missing)}")
    if not maze_worlds:
        raise ValueError("No maze worlds found")
    return maze_worlds


def resolve_prompts(requested_prompts: Optional[List[str]]) -> List[str]:
    available = set(AVAILABLE_PROMPTS.keys())
    available.add("base")
    if not available:
        raise ValueError("No prompts available")
    if not requested_prompts:
        return sorted(available)
    missing = sorted([p for p in requested_prompts if p not in available])
    if missing:
        raise ValueError(f"Unknown prompts: {', '.join(missing)}")
    return requested_prompts


def write_csv(output_path: Path, rows: List[List[str]]):
    if not rows:
        output_path.write_text("")
        return
    max_len = max(len(row) for row in rows)
    header = ["initial_pose"] + [f"step_{i}" for i in range(1, max_len)]
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            padded = row + [""] * (max_len - len(row))
            writer.writerow(padded)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run maze prompt tests and log outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Configuration can be set either via:
1. Command-line arguments (override config variables)
2. Edit configuration variables at the top of this script

Examples:
  # Use default config from script
  python run_maze_prompt_tests.py
  
  # Override specific settings
  python run_maze_prompt_tests.py --prompt base --world 11
  
  # Test with different model
  python run_maze_prompt_tests.py --model gemini-2.5-pro
        """
    )
    parser.add_argument("--command", default=TEST_COMMAND, 
                        help=f"Command to send to the prompts (default: '{TEST_COMMAND}')")
    parser.add_argument("--output-dir", default=OUTPUT_DIRECTORY, 
                        help=f"Directory for log output (default: {OUTPUT_DIRECTORY})")
    # Legacy single-row/col flags (optional). Use --initial-pos for multiple starts.
    parser.add_argument("--initial-row", type=int, default=None,
                        help=f"Initial robot row (overrides config when used together with --initial-col)")
    parser.add_argument("--initial-col", type=int, default=None,
                        help=f"Initial robot col (overrides config when used together with --initial-row)")
    parser.add_argument("--initial-pos", action="append", dest="initial_positions",
                        help="Initial robot position as 'row,col'. Can be repeated to test multiple starts. Overrides config.")
    parser.add_argument("--prompt", action="append", dest="prompts",
                        help="Prompt name to test (repeat for multiple). Defaults to config or all available.")
    parser.add_argument("--world", action="append", dest="worlds",
                        help="Maze world ID to test (repeat for multiple). Defaults to config or all maze worlds.")
    parser.add_argument("--model", default=LLM_MODEL,
                        help=f"LLM model to use (default: {LLM_MODEL})")
    parser.add_argument("--temperature", type=float, default=LLM_TEMPERATURE,
                        help=f"LLM temperature (default: {LLM_TEMPERATURE})")
    parser.add_argument("--max-tokens", type=int, default=LLM_MAX_TOKENS,
                        help=f"LLM max tokens (default: {LLM_MAX_TOKENS})")
    parser.add_argument("--repeat", type=int, default=REPEAT_COUNT,
                        help=f"Number of times to repeat each test (default: {REPEAT_COUNT})")
    args = parser.parse_args()

    # Resolve initial positions: priority -> --initial-pos (repeated) -> explicit --initial-row/--initial-col -> config list
    if args.initial_positions:
        initial_positions: List[Tuple[int, int]] = []
        for text in args.initial_positions:
            try:
                parts = [p.strip() for p in text.split(",")]
                if len(parts) != 2:
                    raise ValueError()
                r = int(parts[0]); c = int(parts[1])
                initial_positions.append((r, c))
            except Exception:
                raise ValueError(f"Invalid --initial-pos value: '{text}'. Expected format 'row,col'")
    elif args.initial_row is not None and args.initial_col is not None:
        initial_positions = [(args.initial_row, args.initial_col)]
    else:
        # Use config list (already a list of tuples)
        initial_positions = list(INITIAL_ROBOT_POSITIONS)

    output_root = Path(args.output_dir).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    details_dir = run_dir / "details"
    run_dir.mkdir(parents=True, exist_ok=True)
    details_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("MAZE PROMPT TEST RUNNER")
    print("="*80)
    print(f"Command: {args.command}")
    print(f"Initial positions: {initial_positions}")
    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Repeat count: {args.repeat}")
    print(f"Output directory: {run_dir}")
    print("="*80)
    print()

    world_manager = WorldManager()
    
    # Use config variables if no command-line override
    worlds_to_test = args.worlds if args.worlds else MAZE_WORLDS_TO_TEST
    prompts_to_test = args.prompts if args.prompts else PROMPTS_TO_TEST
    
    maze_worlds = collect_maze_worlds(world_manager, worlds_to_test)
    prompt_names = resolve_prompts(prompts_to_test)
    
    print(f"Testing {len(prompt_names)} prompt(s): {', '.join(prompt_names)}")
    print(f"Testing {len(maze_worlds)} world(s): {', '.join(f'{wid} ({wname})' for wid, wname, _ in maze_worlds)}")
    print()

    router = LLMRouter()
    router.set_model(args.model)
    print(f"LLM Router initialized with model: {args.model}")
    print()

    run_metadata: List[Dict[str, object]] = []
    total_tests = len(prompt_names) * len(maze_worlds) * len(initial_positions) * args.repeat
    current_test = 0

    for prompt_name in prompt_names:
        for initial_pose in initial_positions:
            csv_rows: List[List[str]] = []
            for world_id, world_name, world_config in maze_worlds:
                # Repeat each test multiple times if requested
                for repeat_idx in range(args.repeat):
                    current_test += 1
                    repeat_suffix = f" (run {repeat_idx + 1}/{args.repeat})" if args.repeat > 1 else ""
                    print(f"[{current_test}/{total_tests}] Testing prompt '{prompt_name}' on world {world_id} ({world_name}), initial {initial_pose}{repeat_suffix}...")
                    
                    result = run_single_test(
                        router=router,
                        prompt_name=prompt_name,
                        world_id=world_id,
                        world_name=world_name,
                        world_config=world_config,
                        command=args.command,
                        initial_pose=initial_pose,
                        output_detail_dir=details_dir,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        repeat_idx=repeat_idx if args.repeat > 1 else None
                    )

                    meta_entry: Dict[str, object] = {
                        "prompt": prompt_name,
                        "world_id": world_id,
                        "world_name": world_name,
                        "initial_pose": initial_pose,
                    }
                    if args.repeat > 1:
                        meta_entry["repeat_idx"] = repeat_idx
                    for key, value in result.items():
                        if key in {"prompt", "world_id", "world_name"}:
                            continue
                        meta_entry[key] = value
                    run_metadata.append(meta_entry)

                    if "error" in result:
                        print(f"  ERROR: {result.get('error')}")
                        continue
                    
                    # Display summary of result
                    actions = result.get("actions", [])
                    positions = result.get("positions", [])
                    if isinstance(actions, list) and isinstance(positions, list):
                        print(f"  Generated {len(actions)} actions, {len(positions)} positions")
                    else:
                        print(f"  Generated result (type mismatch)")

                    if isinstance(positions, list):
                        row = [format_pose(pose) for pose in positions if isinstance(pose, tuple)]
                        if row:
                            csv_rows.append(row)

            csv_path = run_dir / f"positions_{prompt_name}_init_{initial_pose[0]}_{initial_pose[1]}.csv"
            write_csv(csv_path, csv_rows)
            print(f"  Saved positions to: {csv_path.name}")
            print()

    meta_path = run_dir / "run_metadata.json"
    meta_path.write_text(json.dumps({
        "command": args.command,
        "initial_positions": initial_positions,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "repeat_count": args.repeat,
        "prompts": prompt_names,
        "worlds": [
            {"id": wid, "name": wname} for wid, wname, _ in maze_worlds
        ],
        "results": run_metadata
    }, indent=2))

    print("="*80)
    print("TEST RUN COMPLETE")
    print("="*80)
    print(f"Results saved to: {run_dir}")
    print(f"  - Metadata: {meta_path.name}")
    print(f"  - CSV files: {len(prompt_names)} file(s)")
    print(f"  - Detail files: {len(run_metadata)} file(s) in details/")
    print("="*80)
    
    # TODO: add clean up for the rospy node 


if __name__ == "__main__":
    main()
