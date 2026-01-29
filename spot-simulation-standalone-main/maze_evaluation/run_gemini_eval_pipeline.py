#!/usr/bin/env python3
"""Batch runner for Gemini maze evaluations driven by a JSON config."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = REPO_ROOT / "maze_evaluation" / "run_gemini_parallel_evaluation.py"
DEFAULT_CONFIG = REPO_ROOT / "maze_evaluation" / "configs" / "gemini_eval_pipeline.json"


@dataclass(frozen=True)
class PipelineJob:
    task_name: str
    command_text: str
    checkpoint_order: str
    args: List[str]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_pipeline_workers(arg_workers: Optional[int], config: Dict[str, Any]) -> int:
    """Resolve pipeline-level worker count from CLI -> config -> CPU count."""
    if arg_workers is not None:
        return max(1, arg_workers)

    config_workers = config.get("pipeline_workers", config.get("workers"))
    if config_workers is not None:
        try:
            return max(1, int(config_workers))
        except Exception as exc:
            raise SystemExit(f"Invalid pipeline worker value in config: {config_workers}") from exc

    return max(1, os.cpu_count() or 1)


def resolve_paths(paths: Sequence[str]) -> List[Path]:
    resolved: List[Path] = []
    for entry in paths:
        resolved.append(Path(entry).expanduser().resolve())
    return resolved


def load_commands(paths: Iterable[Path]) -> List[str]:
    commands: List[str] = []
    for input_path in paths:
        data = load_json(input_path)
        if not isinstance(data, dict):
            continue
        for value in data.values():
            if isinstance(value, str):
                text = value.strip()
                if text:
                    commands.append(text)
    return commands


def checkpoint_names(world_path: Path, order: str) -> List[str]:
    data = load_json(world_path)
    checkpoints = list(data.get("config", {}).get("checkpoints", {}).keys())
    if not checkpoints:
        return []
    if order == "alphabetical":
        return sorted(checkpoints)
    return checkpoints


def build_checkpoint_args(names: Iterable[str]) -> List[str]:
    args: List[str] = []
    for name in names:
        args.extend(["--checkpoint", name])
    return args


def merge_settings(defaults: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(defaults)
    for key in (
        "prompts",
        "temperature",
        "max_tokens",
        "thinking_budget",
        "repeat",
        "parallelism",
        "repeat_jitter_seconds",
        "response_timeout_seconds",
        "baseline_score",
        "violation_penalty",
        "valid_action_score",
        "additional_wall_penalty",
        "revisit_penalty",
        "out_of_order_penalty",
        "missed_checkpoint_penalty",
    ):
        if key in task:
            merged[key] = task[key]
    return merged


def build_jobs_for_task(
    *,
    task: Dict[str, Any],
    settings: Dict[str, Any],
    commands: List[str],
    world_path: Path,
    task_output_root: Path,
) -> List[PipelineJob]:
    checkpoint_mode = task.get("checkpoint_order", "unordered")
    checkpoints = checkpoint_names(world_path, checkpoint_mode)
    checkpoint_args = build_checkpoint_args(checkpoints)
    task_output_root.mkdir(parents=True, exist_ok=True)

    jobs: List[PipelineJob] = []
    for command_text in commands:
        cmd: List[str] = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--command",
            command_text,
            "--model",
            task["model"],
            "--world",
            str(world_path),
            "--repeat",
            str(settings["repeat"]),
            "--parallelism",
            str(settings["parallelism"]),
            "--repeat-jitter-seconds",
            str(settings.get("repeat_jitter_seconds", 0)),
            "--response-timeout-seconds",
            str(settings.get("response_timeout_seconds", 180)),
            "--temperature",
            str(settings["temperature"]),
            "--max-tokens",
            str(settings["max_tokens"]),
            "--output-dir",
            str(task_output_root),
        ]

        if settings.get("thinking_budget") is not None:
            cmd.extend(["--thinking-budget", str(settings["thinking_budget"])])

        for prompt in settings.get("prompts", []):
            cmd.extend(["--prompt", str(prompt)])

        cmd.extend(checkpoint_args)

        enable_rules = bool(task.get("enable_additional_rules"))
        cmd.append("--enable-additional-rules" if enable_rules else "--no-additional-rules")

        jobs.append(
            PipelineJob(
                task_name=task["name"],
                command_text=command_text,
                checkpoint_order=checkpoint_mode,
                args=cmd,
            )
        )

    return jobs


def run_job(job: PipelineJob, dry_run: bool = False) -> None:
    print("\n=== Running task:", job.task_name)
    print("Command:", job.command_text)
    print("Checkpoint order:", job.checkpoint_order)
    print("Args:", job.args)
    if dry_run:
        return
    subprocess.run(job.args, check=True)


def run_jobs(jobs: Sequence[PipelineJob], max_workers: int, dry_run: bool) -> None:
    if not jobs:
        return

    worker_count = max(1, min(max_workers, len(jobs)))
    if dry_run or worker_count == 1:
        for job in jobs:
            run_job(job, dry_run=dry_run)
        return

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(run_job, job, False) for job in jobs]
        for future in as_completed(futures):
            future.result()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of Gemini maze evaluations from a JSON config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to JSON config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print the planned commands without executing.")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel process count for dispatching evaluation runs (overrides config; defaults to CPU count).",
    )
    args = parser.parse_args()

    config = load_json(args.config)
    pipeline_workers = resolve_pipeline_workers(args.workers, config)
    default_world = config.get("world")
    output_root = Path(config["output_root"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    defaults = config.get("defaults", {})
    tasks = config.get("tasks", [])
    if not tasks:
        raise SystemExit("No tasks defined in config.")

    all_jobs: List[PipelineJob] = []
    for task in tasks:
        if "model" not in task or "name" not in task:
            raise SystemExit(f"Task entry missing 'name' or 'model': {task}")

        task_input_paths = resolve_paths(task.get("input_files", []))
        if not task_input_paths:
            raise SystemExit(f"No input files provided for task '{task.get('name')}'.")

        commands = load_commands(task_input_paths)
        if not commands:
            raise SystemExit(f"No commands found in input files for task '{task.get('name')}'.")

        world_value = task.get("world", default_world)
        if not world_value:
            raise SystemExit(f"No world provided for task '{task.get('name')}' and no default 'world' in config.")
        world_path = Path(world_value).expanduser().resolve()

        settings = merge_settings(defaults, task)
        settings.setdefault("repeat_jitter_seconds", 0)
        settings.setdefault("response_timeout_seconds", 180)
        task_output_root = output_root / task["name"]
        all_jobs.extend(
            build_jobs_for_task(
                task=task,
                settings=settings,
                commands=commands,
                world_path=world_path,
                task_output_root=task_output_root,
            )
        )

    if not all_jobs:
        raise SystemExit("No evaluation jobs generated.")

    run_jobs(all_jobs, pipeline_workers, args.dry_run)


if __name__ == "__main__":
    main()
