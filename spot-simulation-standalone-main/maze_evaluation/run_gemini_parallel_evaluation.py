#!/usr/bin/env python3
"""Parallel Gemini-based maze evaluation with automated scoring."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import re
import os
import random
from dotenv import load_dotenv
load_dotenv()

# Ensure project modules are importable when running from CLI
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prompt_manager import load_prompt  # type: ignore  # noqa: E402
from llm_router import LLMRouter  # type: ignore  # noqa: E402
from plan_scorer import PlanScorer, ScoreResult  # type: ignore  # noqa: E402
from world_manager import WorldManager  # type: ignore  # noqa: E402

###############################################################################
# Global configuration defaults
###############################################################################

# Gemini model to use. Override with --model if needed.
DEFAULT_MODEL_ID = "gemini-2.5-pro"

# Command to evaluate. Override with --command.
DEFAULT_COMMAND = "Visit all checkpoints"

# Prompt(s) to evaluate. Override with --prompt repeatedly.
DEFAULT_PROMPTS = ["sim"]

# Default world definition. This can be a world id or a path to a JSON file.
DEFAULT_WORLD_SPEC = str(REPO_ROOT / "worlds" / "custom_sim.json")

# Checkpoints expected for scoring (names must exist in the world config).
DEFAULT_CHECKPOINT_SEQUENCE = ["CHECKPOINT_A", "CHECKPOINT_B", "CHECKPOINT_C"]

# Optional list of explicit initial poses. When empty, the script will use the
# robot_start_position from the selected world (or fall back to (4, 4)).
DEFAULT_INITIAL_POSES: Sequence[Tuple[int, int]] = ()

# Sampling / batching controls
DEFAULT_REPEAT_COUNT = 5
DEFAULT_MAX_WORKERS = 5

# Generation options
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 8192 / 2
DEFAULT_THINKING_BUDGET = 3000

# Scoring options
DEFAULT_BASELINE_SCORE = 1000
DEFAULT_VIOLATION_PENALTY = 100
DEFAULT_VALID_ACTION_SCORE = 0
DEFAULT_ADDITIONAL_WALL_PENALTY = 50
DEFAULT_REVISIT_PENALTY = 20
DEFAULT_OUT_OF_ORDER_PENALTY = 125
DEFAULT_MISSED_CHECKPOINT_PENALTY = 125
DEFAULT_ENABLE_ADDITIONAL_RULES = False
DEFAULT_REPEAT_JITTER_SECONDS = 0.0
DEFAULT_RESPONSE_TIMEOUT_SECONDS = 180.0

# Output settings
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "gemini_eval_runs"
SAVE_SCORE_LOGS = True

# Regex helpers for plan parsing
PLAN_REGEX = re.compile(r"\[Plan\](.*?)\[/Plan\]", re.DOTALL)
ROW_REGEX = re.compile(r"row\s*=\s*(-?\d+)")
COL_REGEX = re.compile(r"col\s*=\s*(-?\d+)")

###############################################################################
# Data structures
###############################################################################


@dataclass(frozen=True)
class EvaluationJob:
    index: int
    prompt_name: str
    repeat_index: int
    initial_pose: Tuple[int, int]
    job_label: str


@dataclass(frozen=True)
class RuntimeContext:
    command: str
    model_id: str
    temperature: float
    max_tokens: int
    thinking_budget: Optional[int]
    world_id: str
    world_config: Dict[str, Any]
    world_data: Dict[str, Any]
    checkpoints: Sequence[Tuple[int, int]]
    detail_dir: Path
    score_log_dir: Optional[Path]
    plan_scorer: PlanScorer
    prompt_cache: Dict[str, str]
    repeat_jitter_seconds: float
    response_timeout_seconds: float


@dataclass
class JobResult:
    job: EvaluationJob
    score_result: Optional[ScoreResult]
    actions: List[str]
    response_text: Optional[str]
    raw_response: Optional[Dict[str, Any]]
    usage_metadata: Optional[Dict[str, Any]]
    error: Optional[str]
    detail_file: Optional[str]
    score_log_file: Optional[str]
    runtime_seconds: float

    @property
    def success(self) -> bool:
        if not self.score_result:
            return False
        return len(self.score_result.missed_checkpoints) == 0

    @property
    def total_score(self) -> Optional[int]:
        return self.score_result.total_score if self.score_result else None


###############################################################################
# Utility functions
###############################################################################


def locate_world_path(world_spec: str) -> Optional[Path]:
    """Return a filesystem path for the given world spec if it exists."""
    candidate = Path(world_spec).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    relative_candidate = REPO_ROOT / "worlds" / f"{world_spec}.json"
    if relative_candidate.is_file():
        return relative_candidate.resolve()
    return None


def load_world_definition(
    manager: WorldManager,
    world_spec: str,
) -> Tuple[str, Dict[str, Any], Dict[str, Any], Optional[Path]]:
    """Load a world configuration either by id or from a JSON path."""
    world_path = locate_world_path(world_spec)
    if world_path:
        with world_path.open("r", encoding="utf-8") as handle:
            world_data = json.load(handle)
        world_id = world_path.stem
        manager.custom_worlds[world_id] = world_data
        world_config = world_data.get("config")
        if not world_config:
            raise ValueError(f"World file '{world_path}' missing 'config' section")
        return world_id, world_config, world_data, world_path

    world_id = world_spec
    world_config = manager.get_world_config(world_id)
    if not world_config:
        raise ValueError(f"Unknown world spec '{world_spec}'")

    world_data = manager.custom_worlds.get(world_id)
    if not world_data:
        world_data = {"config": world_config}
    return world_id, world_config, world_data, None


def resolve_initial_positions(
    world_data: Dict[str, Any],
    explicit: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    if explicit:
        return [tuple(map(int, pose)) for pose in explicit]
    start = world_data.get("robot_start_position")
    if isinstance(start, dict):
        row = int(start.get("row", 4))
        col = int(start.get("col", 4))
        return [(row, col)]
    return [(4, 4)]


def resolve_checkpoints(
    world_data: Dict[str, Any],
    names: Sequence[str],
) -> List[Tuple[int, int]]:
    config = world_data.get("config", {})
    checkpoint_map = config.get("checkpoints", {})
    if not checkpoint_map:
        if names:
            raise ValueError("Checkpoint names provided but world has no checkpoints")
        return []

    resolved: List[Tuple[int, int]] = []
    if names:
        for name in names:
            entry = checkpoint_map.get(name)
            if not entry:
                raise ValueError(f"Checkpoint '{name}' not found in world config")
            resolved.append((int(entry["row"]), int(entry["col"])))
        return resolved

    # Default: alphabetical order for deterministic scoring
    for name, entry in sorted(checkpoint_map.items()):
        resolved.append((int(entry["row"]), int(entry["col"])))
    return resolved


def extract_response_text(response: Any) -> Optional[str]:
    if isinstance(response, dict):
        for key in ("response", "output_text", "text", "message"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    elif isinstance(response, str):
        text = response.strip()
        if text:
            return text
    return None


def extract_plan_section(response_text: str) -> Tuple[str, Optional[str]]:
    match = PLAN_REGEX.search(response_text)
    if match:
        return match.group(1).strip(), response_text
    return response_text.strip(), None


def extract_actions(plan_text: str) -> List[str]:
    actions: List[str] = []
    skip_prefixes = (
        "#",
        "Reasoning:",
        "Thought:",
        "Steps:",
        "Process:",
        "Summary:",
        "Reflection:",
    )
    for raw_line in plan_text.splitlines():
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


def format_usage_summary(usage: Optional[Dict[str, Any]]) -> str:
    if not usage:
        return ""
    total = usage.get("total_token_count")
    prompt = usage.get("prompt_token_count")
    response = usage.get("candidates_token_count")
    thoughts = (
        usage.get("thinking_token_count")
        or usage.get("thoughts_token_count")
        or usage.get("cached_content_token_count")
    )
    total_s = str(total) if total is not None else "?"
    prompt_s = str(prompt) if prompt is not None else "?"
    response_s = str(response) if response is not None else "?"
    thought_s = str(thoughts) if thoughts is not None else "?"
    return f" tokens total/prompt/thinking/response={total_s}/{prompt_s}/{thought_s}/{response_s}"


def build_prompt(
    prompt_template: str,
    world_config: Dict[str, Any],
    command: str,
    initial_pose: Tuple[int, int],
    world_data: Dict[str, Any],
) -> str:
    start_facing = "N"
    start_info = world_data.get("robot_start_position")
    if isinstance(start_info, dict):
        start_facing = start_info.get("facing", start_facing)

    current_state = {
        "robot_state": "Stand",
        "standing": True,
        "robot_cell": [int(initial_pose[0]), int(initial_pose[1])],
        "facing": start_facing,
        "arm": "stowed",
        "held_object": None,
    }
    return prompt_template.format(
        current_state_json=json.dumps(current_state, indent=2),
        world_model_json=json.dumps(world_config, indent=2),
        command=command,
    )


def format_pose(pose: Tuple[int, int]) -> str:
    return f"({pose[0]},{pose[1]})"


def score_breakdown_to_dict(result: ScoreResult) -> List[Dict[str, Any]]:
    breakdown: List[Dict[str, Any]] = []
    for entry in result.breakdown:
        breakdown.append(
            {
                "index": entry.index,
                "action": entry.action,
                "score_delta": entry.score_delta,
                "status": entry.status,
                "resulting_pose": list(entry.resulting_pose),
            }
        )
    return breakdown


def allocate_run_dir(output_root: Path) -> Path:
    """Create a unique run directory under the given root."""
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    if not run_dir.exists():
        run_dir.mkdir()
        return run_dir

    # Concurrent runs can collide on the same second-level timestamp; add a suffix to keep outputs isolated.
    suffix = 1
    while True:
        candidate = output_root / f"run_{timestamp}_{suffix}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            suffix += 1


###############################################################################
# LLM routing helpers
###############################################################################

_THREAD_LOCAL = threading.local()


def get_thread_local_router(model_id: str) -> LLMRouter:
    router = getattr(_THREAD_LOCAL, "router", None)
    current_model = getattr(_THREAD_LOCAL, "model_id", None)
    if router is None:
        router = LLMRouter()
        router.set_model(model_id)
        _THREAD_LOCAL.router = router
        _THREAD_LOCAL.model_id = model_id
        return router

    if current_model != model_id:
        router.set_model(model_id)
        _THREAD_LOCAL.model_id = model_id
    return router


###############################################################################
# Evaluation logic
###############################################################################


def evaluate_job(job: EvaluationJob, context: RuntimeContext) -> JobResult:
    start_time = time.time()
    prompt_template = context.prompt_cache[job.prompt_name]
    prompt_text = build_prompt(
        prompt_template,
        context.world_config,
        context.command,
        job.initial_pose,
        context.world_data,
    )

    # Optional small delay to stagger repeat submissions across threads.
    if context.repeat_jitter_seconds > 0:
        time.sleep(random.uniform(0, context.repeat_jitter_seconds))

    router = get_thread_local_router(context.model_id)
    generate_options = {
        "temperature": context.temperature,
        "num_predict": context.max_tokens,
    }
    if context.thinking_budget is not None:
        generate_options["thinking_budget"] = context.thinking_budget

    # Run generation with a timeout to avoid hanging repeats.
    raw_response: Optional[Dict[str, Any]] = None
    with ThreadPoolExecutor(max_workers=1) as local_executor:
        future = local_executor.submit(router.generate, prompt_text, generate_options)
        try:
            raw_response = future.result(timeout=context.response_timeout_seconds)
        except TimeoutError:
            future.cancel()
            return JobResult(
                job=job,
                score_result=None,
                actions=[],
                response_text=None,
                raw_response=None,
                usage_metadata=None,
                error=f"LLM response timed out after {context.response_timeout_seconds:.1f} seconds.",
                detail_file=None,
                score_log_file=None,
                runtime_seconds=time.time() - start_time,
            )
        except Exception as exc:
            future.cancel()
            return JobResult(
                job=job,
                score_result=None,
                actions=[],
                response_text=None,
                raw_response=None,
                usage_metadata=None,
                error=f"LLM generation failed: {exc}",
                detail_file=None,
                score_log_file=None,
                runtime_seconds=time.time() - start_time,
            )
    usage_metadata = raw_response.get("usage_metadata") if isinstance(raw_response, dict) else None
    response_text = extract_response_text(raw_response)
    if not response_text:
        return JobResult(
            job=job,
            score_result=None,
            actions=[],
            response_text=None,
            raw_response=raw_response if isinstance(raw_response, dict) else None,
            usage_metadata=usage_metadata if isinstance(usage_metadata, dict) else None,
            error="No response text received from LLM.",
            detail_file=None,
            score_log_file=None,
            runtime_seconds=time.time() - start_time,
        )

    plan_text, _ = extract_plan_section(response_text)
    actions = extract_actions(plan_text)
    if not actions:
        return JobResult(
            job=job,
            score_result=None,
            actions=[],
            response_text=response_text,
            raw_response=raw_response if isinstance(raw_response, dict) else None,
            usage_metadata=usage_metadata if isinstance(usage_metadata, dict) else None,
            error="Failed to parse any actions from model response.",
            detail_file=None,
            score_log_file=None,
            runtime_seconds=time.time() - start_time,
        )

    score_log_path: Optional[Path] = None
    if context.score_log_dir:
        score_log_dir = context.score_log_dir
        score_log_dir.mkdir(parents=True, exist_ok=True)
        score_log_path = score_log_dir / f"{job.job_label}_score.log"

    try:
        score_result = context.plan_scorer.score_plan(
            actions,
            world_id=context.world_id,
            checkpoints=context.checkpoints,
            initial_pose=job.initial_pose,
            log_path=score_log_path,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        return JobResult(
            job=job,
            score_result=None,
            actions=actions,
            response_text=response_text,
            raw_response=raw_response if isinstance(raw_response, dict) else None,
            usage_metadata=usage_metadata if isinstance(usage_metadata, dict) else None,
            error=f"Scoring failed: {exc}",
            detail_file=None,
            score_log_file=str(score_log_path) if score_log_path else None,
            runtime_seconds=time.time() - start_time,
        )

    detail_payload = {
        "job": {
            "index": job.index,
            "prompt": job.prompt_name,
            "repeat_index": job.repeat_index,
            "initial_pose": list(job.initial_pose),
            "label": job.job_label,
        },
        "command": context.command,
        "model": context.model_id,
        "world_id": context.world_id,
        "checkpoints": [list(cp) for cp in context.checkpoints],
        "prompt_text": prompt_text,
        "response_text": response_text,
        "actions": actions,
        "thinking_budget": context.thinking_budget,
        "score": {
            "total": score_result.total_score,
            "achieved": [list(cp) for cp in score_result.achieved_checkpoints],
            "missed": [list(cp) for cp in score_result.missed_checkpoints],
            "final_pose": list(score_result.final_pose),
            "breakdown": score_breakdown_to_dict(score_result),
        },
        "raw_response": raw_response if isinstance(raw_response, dict) else {"text": response_text},
        "token_usage": usage_metadata if isinstance(usage_metadata, dict) else None,
        "timestamps": {
            "evaluated_at": datetime.utcnow().isoformat() + "Z",
            "runtime_seconds": time.time() - start_time,
        },
        "paths": {
            "score_log": str(score_log_path) if score_log_path else None,
        },
    }

    context.detail_dir.mkdir(parents=True, exist_ok=True)
    detail_path = context.detail_dir / f"{job.job_label}.json"
    detail_path.write_text(json.dumps(detail_payload, indent=2), encoding="utf-8")

    return JobResult(
        job=job,
        score_result=score_result,
        actions=actions,
        response_text=response_text,
        raw_response=raw_response if isinstance(raw_response, dict) else None,
        usage_metadata=usage_metadata if isinstance(usage_metadata, dict) else None,
        error=None,
        detail_file=str(detail_path),
        score_log_file=str(score_log_path) if score_log_path else None,
        runtime_seconds=time.time() - start_time,
    )


def create_jobs(
    prompts: Sequence[str],
    initial_poses: Sequence[Tuple[int, int]],
    repeat_count: int,
) -> List[EvaluationJob]:
    jobs: List[EvaluationJob] = []
    counter = 1
    for prompt_name in prompts:
        for pose in initial_poses:
            for repeat_idx in range(repeat_count):
                label = f"{prompt_name}_r{repeat_idx+1:02d}_row{pose[0]}_col{pose[1]}"
                jobs.append(
                    EvaluationJob(
                        index=counter,
                        prompt_name=prompt_name,
                        repeat_index=repeat_idx,
                        initial_pose=(int(pose[0]), int(pose[1])),
                        job_label=label,
                    )
                )
                counter += 1
    return jobs


def summarize_results(results: Iterable[JobResult]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "total_jobs": 0,
        "completed": 0,
        "failed": 0,
        "successes": 0,
        "avg_score": None,
        "min_score": None,
        "max_score": None,
    }
    scores: List[int] = []
    successes = 0
    completed = 0
    failed = 0
    for result in results:
        summary["total_jobs"] += 1
        if result.error:
            failed += 1
            continue
        completed += 1
        if result.success:
            successes += 1
        if result.total_score is not None:
            scores.append(result.total_score)

    summary["completed"] = completed
    summary["failed"] = failed
    summary["successes"] = successes
    if scores:
        summary["avg_score"] = sum(scores) / len(scores)
        summary["min_score"] = min(scores)
        summary["max_score"] = max(scores)
    return summary


###############################################################################
# CLI
###############################################################################


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run parallel Gemini maze evaluations with scoring.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--command", default=DEFAULT_COMMAND, help="Task command to embed in the prompt.")
    parser.add_argument(
        "--prompt",
        dest="prompts",
        action="append",
        help="Prompt template name to evaluate (can be repeated). Defaults to configured list.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Gemini model identifier to use.")
    parser.add_argument(
        "--world",
        default=DEFAULT_WORLD_SPEC,
        help="World spec (id or path to JSON).",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        help="Checkpoint name in desired order (repeat for multiple). Defaults to configured list.",
    )
    parser.add_argument(
        "--initial-pos",
        dest="initial_positions",
        action="append",
        help="Initial robot position as 'row,col'. Repeat to test multiple starts.",
    )
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT_COUNT, help="Number of repetitions per prompt/pose.")
    parser.add_argument("--parallelism", type=int, default=DEFAULT_MAX_WORKERS, help="Number of concurrent requests.")
    parser.add_argument(
        "--repeat-jitter-seconds",
        type=float,
        default=DEFAULT_REPEAT_JITTER_SECONDS,
        help="Random sleep of up to this many seconds before each repeat to stagger requests.",
    )
    parser.add_argument(
        "--response-timeout-seconds",
        type=float,
        default=DEFAULT_RESPONSE_TIMEOUT_SECONDS,
        help="Fail a repeat if the LLM response takes longer than this many seconds.",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Generation temperature.")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Max tokens / num_predict for generation.")
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=DEFAULT_THINKING_BUDGET,
        help="Token budget for Gemini thinking traces (use 0 to disable passing thinking_config).",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory to store run artifacts.")
    parser.add_argument(
        "--baseline-score",
        type=int,
        default=DEFAULT_BASELINE_SCORE,
        help="Starting score before penalties are applied.",
    )
    parser.add_argument(
        "--violation-penalty",
        type=int,
        default=DEFAULT_VIOLATION_PENALTY,
        help="Penalty applied for invalid moves.",
    )
    parser.add_argument(
        "--valid-action-score",
        type=int,
        default=DEFAULT_VALID_ACTION_SCORE,
        help="Score applied to valid non-rewarding actions.",
    )
    parser.add_argument(
        "--additional-wall-penalty",
        type=int,
        default=DEFAULT_ADDITIONAL_WALL_PENALTY,
        help="Extra penalty after the first wall collision.",
    )
    parser.add_argument(
        "--revisit-penalty",
        type=int,
        default=DEFAULT_REVISIT_PENALTY,
        help="Penalty for revisiting a previously visited cell (when additional rules are enabled).",
    )
    parser.add_argument(
        "--out-of-order-penalty",
        type=int,
        default=DEFAULT_OUT_OF_ORDER_PENALTY,
        help="Penalty for reaching checkpoints out of the provided order.",
    )
    parser.add_argument(
        "--missed-checkpoint-penalty",
        type=int,
        default=DEFAULT_MISSED_CHECKPOINT_PENALTY,
        help="Penalty applied for each missed checkpoint.",
    )
    parser.add_argument(
        "--enable-additional-rules",
        dest="enable_additional_rules",
        action="store_true",
        help="Enable additional scoring rules (revisit penalties, ordered checkpoints).",
    )
    parser.add_argument(
        "--no-additional-rules",
        dest="enable_additional_rules",
        action="store_false",
        help="Disable additional scoring rules (revisit penalties, ordered checkpoints).",
    )
    parser.set_defaults(enable_additional_rules=DEFAULT_ENABLE_ADDITIONAL_RULES)
    return parser.parse_args()


def parse_initial_positions(values: Optional[Sequence[str]]) -> List[Tuple[int, int]]:
    if not values:
        return []
    positions: List[Tuple[int, int]] = []
    for text in values:
        try:
            row_str, col_str = [segment.strip() for segment in text.split(",", 1)]
            positions.append((int(row_str), int(col_str)))
        except Exception as exc:
            raise ValueError(f"Invalid --initial-pos value '{text}': {exc}") from exc
    return positions


def main() -> None:
    args = parse_args()

    # Resolve prompts with fallback defaults.
    prompts = args.prompts if args.prompts else list(DEFAULT_PROMPTS)
    if not prompts:
        raise SystemExit("No prompts specified and default prompt list is empty.")

    prompt_cache = {name: load_prompt(name) for name in prompts}

    # Setup world manager and scoring
    world_manager = WorldManager()
    world_id, world_config, world_data, world_path = load_world_definition(world_manager, args.world)

    # Determine checkpoints and initial positions.
    checkpoint_names = args.checkpoints if args.checkpoints else list(DEFAULT_CHECKPOINT_SEQUENCE)
    checkpoints = resolve_checkpoints(world_data, checkpoint_names)
    explicit_initials = parse_initial_positions(args.initial_positions)
    initial_poses = resolve_initial_positions(world_data, explicit_initials or DEFAULT_INITIAL_POSES)

    jobs = create_jobs(prompts, initial_poses, max(1, args.repeat))
    if not jobs:
        raise SystemExit("No evaluation jobs generated.")

    output_root = Path(args.output_dir).expanduser().resolve()
    run_dir = allocate_run_dir(output_root)
    detail_dir = run_dir / "details"
    score_log_dir = run_dir / "score_logs" if SAVE_SCORE_LOGS else None

    plan_scorer = PlanScorer(
        world_manager=world_manager,
        violation_penalty=args.violation_penalty,
        valid_action_score=args.valid_action_score,
        enable_additional_rules=args.enable_additional_rules,
        baseline_score=args.baseline_score,
        additional_wall_penalty=args.additional_wall_penalty,
        revisit_penalty=args.revisit_penalty,
        out_of_order_penalty=args.out_of_order_penalty,
        missed_checkpoint_penalty=args.missed_checkpoint_penalty,
    )

    thinking_budget = max(0, args.thinking_budget)

    context = RuntimeContext(
        command=args.command,
        model_id=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        thinking_budget=thinking_budget,
        world_id=world_id,
        world_config=world_config,
        world_data=world_data,
        checkpoints=checkpoints,
        detail_dir=detail_dir,
        score_log_dir=score_log_dir,
        plan_scorer=plan_scorer,
        prompt_cache=prompt_cache,
        repeat_jitter_seconds=max(0.0, float(args.repeat_jitter_seconds)),
        response_timeout_seconds=max(1.0, float(args.response_timeout_seconds)),
    )

    print("=" * 80)
    print("PARALLEL GEMINI MAZE EVALUATION")
    print("=" * 80)
    print(f"Run directory        : {run_dir}")
    if world_path:
        print(f"World file           : {world_path}")
    else:
        print(f"World id             : {world_id}")
    print(f"Model                : {args.model}")
    print(f"Command              : {args.command}")
    print(f"Prompts              : {', '.join(prompts)}")
    print(f"Initial poses        : {', '.join(format_pose(p) for p in initial_poses)}")
    print(f"Checkpoints          : {checkpoints}")
    print(f"Repeat count         : {args.repeat}")
    print(f"Repeat jitter (s)    : {max(0.0, float(args.repeat_jitter_seconds))}")
    print(f"Response timeout (s) : {max(1.0, float(args.response_timeout_seconds))}")
    print(f"Parallel workers     : {max(1, args.parallelism)}")
    print(f"Thinking budget      : {thinking_budget}")
    print(f"Baseline score       : {args.baseline_score}")
    print(f"Violation penalty    : {args.violation_penalty}")
    print(f"Additional wall pen. : {args.additional_wall_penalty}")
    print(f"Revisit penalty      : {args.revisit_penalty}")
    print(f"Out-of-order pen.    : {args.out_of_order_penalty}")
    print(f"Missed checkpoint    : {args.missed_checkpoint_penalty}")
    print(f"Additional rules     : {args.enable_additional_rules}")
    print("-" * 80)

    results: List[JobResult] = []
    total_jobs = len(jobs)
    max_workers = max(1, args.parallelism)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(evaluate_job, job, context): job for job in jobs}
        for idx, future in enumerate(as_completed(future_map), start=1):
            job = future_map[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive logging
                print(f"[{idx}/{total_jobs}] {job.job_label}: failed with exception {exc}")
                results.append(
                    JobResult(
                        job=job,
                        score_result=None,
                        actions=[],
                        response_text=None,
                        raw_response=None,
                        usage_metadata=None,
                        error=f"Unhandled exception: {exc}",
                        detail_file=None,
                        score_log_file=None,
                        runtime_seconds=0.0,
                    )
                )
                continue

            results.append(result)
            if result.error:
                print(f"[{idx}/{total_jobs}] {job.job_label}: ERROR - {result.error}")
            else:
                score = result.total_score if result.total_score is not None else "n/a"
                status = "success" if result.success else "incomplete"
                token_info = format_usage_summary(result.usage_metadata)
                print(
                    f"[{idx}/{total_jobs}] {job.job_label}: {status}, "
                    f"score={score}, actions={len(result.actions)}{token_info}"
                )

    summary = summarize_results(results)
    metadata = {
        "run_dir": str(run_dir),
        "world": {
            "id": world_id,
            "path": str(world_path) if world_path else None,
            "checkpoints": [list(cp) for cp in checkpoints],
        },
        "model": args.model,
        "command": args.command,
        "prompts": prompts,
        "initial_poses": [list(p) for p in initial_poses],
        "parallelism": max_workers,
        "repeat_count": args.repeat,
        "thinking_budget": thinking_budget,
        "scoring": {
            "baseline_score": args.baseline_score,
            "violation_penalty": args.violation_penalty,
            "valid_action_score": args.valid_action_score,
            "additional_wall_penalty": args.additional_wall_penalty,
            "revisit_penalty": args.revisit_penalty,
            "out_of_order_penalty": args.out_of_order_penalty,
            "missed_checkpoint_penalty": args.missed_checkpoint_penalty,
            "enable_additional_rules": args.enable_additional_rules,
        },
        "results": [
            {
                "label": r.job.job_label,
                "prompt": r.job.prompt_name,
                "repeat_index": r.job.repeat_index,
                "initial_pose": list(r.job.initial_pose),
                "score": r.total_score,
                "success": r.success,
                "actions": len(r.actions),
                "detail_file": r.detail_file,
                "score_log_file": r.score_log_file,
                "error": r.error,
                "runtime_seconds": r.runtime_seconds,
                "token_usage": r.usage_metadata,
                "thinking_budget": thinking_budget,
                "achieved_checkpoints": (
                    [list(cp) for cp in r.score_result.achieved_checkpoints]
                    if r.score_result
                    else None
                ),
                "missed_checkpoints": (
                    [list(cp) for cp in r.score_result.missed_checkpoints]
                    if r.score_result
                    else None
                ),
                "response_preview": (r.response_text[:200] + "…") if r.response_text and len(r.response_text) > 200 else r.response_text,
            }
            for r in results
        ],
        "summary": summary,
    }

    summary_path = run_dir / "run_metadata.json"
    summary_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("-" * 80)
    print("RUN COMPLETE")
    print(f"Results stored in: {run_dir}")
    print(f"  Detail files : {len([r for r in results if r.detail_file])}")
    print(f"  Failures     : {summary['failed']}")
    if summary["avg_score"] is not None:
        print(
            f"  Score stats  : avg={summary['avg_score']:.2f}, "
            f"min={summary['min_score']}, max={summary['max_score']}"
        )
    print("-" * 80)


if __name__ == "__main__":
    main()
