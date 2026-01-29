#!/usr/bin/env python3
"""Summarize Gemini pipeline evaluation runs into a CSV report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "maze_evaluation" / "configs" / "gemini_eval_pipeline.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "maze_evaluation" / "gemini_eval_runs" / "pipeline_dec22_1618"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_ROOT / "pipeline_summary.csv"
DEFAULT_PENALTY_OUTPUT = DEFAULT_OUTPUT_ROOT / "penalty_distribution.csv"
PENALTY_COMPONENTS = {
    "obstacle_collision",
    "revisit_penalty",
    "checkpoint_out_of_order",
    "diagonal_move",
    "missed_checkpoint",
    "ignored",
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_output_root(config: Dict[str, Any], override: Optional[Path]) -> Path:
    root = Path(override or config["output_root"]).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def iter_run_metadata(task_dir: Path) -> Iterable[Path]:
    yield from sorted(task_dir.glob("run_*/run_metadata.json"))


def select_runs(meta_paths: Iterable[Path], latest_only: bool) -> List[Path]:
    paths = sorted(meta_paths)
    if latest_only and paths:
        return [paths[-1]]
    return paths


def extract_input_from_detail(
    detail_file: str,
    run_dirs: Iterable[Path] | Path | None,
) -> Optional[str]:
    """Extract the input/command from a detail file."""
    try:
        detail_path = Path(detail_file)
        if detail_path.exists():
            detail_data = load_json(detail_path)
            return detail_data.get("command")

        candidates: List[Path] = []
        if isinstance(run_dirs, Path):
            candidates.append(run_dirs / "details" / detail_path.name)
        elif run_dirs is not None:
            candidates.extend(base / "details" / detail_path.name for base in run_dirs)

        for candidate in candidates:
            if candidate.exists():
                detail_data = load_json(candidate)
                return detail_data.get("command")
    except Exception:
        pass
    return None


def extract_prompt_stats(
    results: List[Dict[str, Any]], 
    prompt_name: str,
    run_dirs: Iterable[Path] | Path | None,
    default_thinking_budget: Optional[int],
    default_input: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract statistics for a specific prompt from results."""
    prompt_results = [r for r in results if r.get("prompt") == prompt_name]
    
    if not prompt_results:
        return {
            "count": 0,
            "scores": [],
            "max_score": None,
            "min_score": None,
            "mean_score": None,
            "max_response_time": None,
            "min_response_time": None,
            "mean_response_time": None,
            "input": None,
            "mean_total_tokens": None,
            "thinking_budget": default_thinking_budget,
        }
    
    scores = [r.get("score") for r in prompt_results if r.get("score") is not None]
    runtimes = [r.get("runtime_seconds") for r in prompt_results if r.get("runtime_seconds") is not None]
    
    # Extract token usage
    total_tokens = []
    thinking_budgets = []
    
    for r in prompt_results:
        token_usage = r.get("token_usage", {})
        if token_usage:
            # Try both possible key names for compatibility
            total = token_usage.get("total_token_count") or token_usage.get("total_tokens")
            if total is not None:
                total_tokens.append(total)
        budget_value = r.get("thinking_budget")
        if budget_value is not None:
            try:
                thinking_budgets.append(int(budget_value))
            except (TypeError, ValueError):
                continue
    
    # Extract input from the first result's detail file
    input_value = None
    if prompt_results and prompt_results[0].get("detail_file"):
        input_value = extract_input_from_detail(prompt_results[0]["detail_file"], run_dirs)
    if input_value is None:
        input_value = default_input

    thinking_budget_value: Optional[int | List[int]] = None
    try:
        default_budget_int = int(default_thinking_budget) if default_thinking_budget is not None else None
    except (TypeError, ValueError):
        default_budget_int = None

    unique_budgets = sorted(set(thinking_budgets))
    if unique_budgets:
        thinking_budget_value = unique_budgets[0] if len(unique_budgets) == 1 else unique_budgets
    elif default_budget_int is not None:
        thinking_budget_value = default_budget_int

    error_free_count = sum(1 for s in scores if s == 1000)
    no_plan_count = sum(1 for s in scores if s == 0)
    
    return {
        "count": len(prompt_results),
        "scores": scores,
        "max_score": max(scores) if scores else None,
        "min_score": min(scores) if scores else None,
        "mean_score": sum(scores) / len(scores) if scores else None,
        "max_response_time": max(runtimes) if runtimes else None,
        "min_response_time": min(runtimes) if runtimes else None,
        "mean_response_time": sum(runtimes) / len(runtimes) if runtimes else None,
        "input": input_value,
        "mean_total_tokens": round(sum(total_tokens) / len(total_tokens), 2) if total_tokens else None,
        "thinking_budget": thinking_budget_value,
        "error_free_count": error_free_count,
        "no_plan_count": no_plan_count,
    }


def extract_csv_rows(
    *,
    meta_path: Path,
    task_name: str,
) -> List[Dict[str, Any]]:
    """Extract CSV rows for each prompt in a run."""
    data = load_json(meta_path)
    model = data.get("model", "unknown")
    repeat_count = data.get("repeat_count", 0)
    results = data.get("results", [])
    default_thinking_budget = data.get("thinking_budget")
    run_dir = meta_path.parent
    
    # Get unique prompts from results
    prompts = sorted(set(r.get("prompt") for r in results if r.get("prompt")))
    
    rows = []
    for prompt_name in prompts:
        stats = extract_prompt_stats(
            results,
            prompt_name,
            run_dir,
            default_thinking_budget,
            data.get("command"),
        )
        thinking_budget_value = stats["thinking_budget"]
        if isinstance(thinking_budget_value, list):
            thinking_budget_value = json.dumps(thinking_budget_value)
        rows.append({
            "task": task_name,
            "model": model,
            "input": stats["input"],
            "system": prompt_name,
            "thinking_budget": thinking_budget_value,
            "repeat_count": stats["count"],
            "max_score": stats["max_score"],
            "min_score": stats["min_score"],
            "mean_score": round(stats["mean_score"], 2) if stats["mean_score"] is not None else None,
            "score_list": json.dumps(stats["scores"]),
            "max_response_time": round(stats["max_response_time"], 4) if stats["max_response_time"] is not None else None,
            "min_response_time": round(stats["min_response_time"], 4) if stats["min_response_time"] is not None else None,
            "mean_response_time": round(stats["mean_response_time"], 4) if stats["mean_response_time"] is not None else None,
            "mean_total_tokens": stats["mean_total_tokens"],
            "error_free_count": stats["error_free_count"],
            "no_plan_count": stats["no_plan_count"],
        })
    
    return rows


def summarize_runs_to_csv(
    *,
    output_root: Path,
    latest_only: bool,
    aggregate_prompts: bool,
) -> List[Dict[str, Any]]:
    """Collect all CSV rows from all tasks and runs."""
    all_rows: List[Dict[str, Any]] = []
    
    for task_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        meta_paths = select_runs(iter_run_metadata(task_dir), latest_only)

        if aggregate_prompts:
            grouped_results: Dict[tuple[str, Optional[str]], List[Dict[str, Any]]] = {}
            grouped_run_dirs: Dict[tuple[str, Optional[str]], List[Path]] = {}
            grouped_default_budget: Dict[tuple[str, Optional[str]], Optional[int]] = {}
            grouped_models: Dict[tuple[str, Optional[str]], List[str]] = {}

            for meta_path in meta_paths:
                data = load_json(meta_path)
                model = data.get("model")
                run_dir = meta_path.parent
                default_budget = data.get("thinking_budget")
                command_text: Optional[str] = data.get("command")

                for result in data.get("results", []):
                    prompt_name = result.get("prompt")
                    if not prompt_name:
                        continue
                    key = (prompt_name, command_text)
                    grouped_results.setdefault(key, []).append(result)
                    grouped_run_dirs.setdefault(key, []).append(run_dir)
                    grouped_models.setdefault(key, [])
                    if model:
                        grouped_models[key].append(str(model))
                    if default_budget is not None and key not in grouped_default_budget:
                        grouped_default_budget[key] = default_budget

            def sort_key(item: tuple[str, Optional[str]]) -> tuple[str, str]:
                prompt_name, cmd = item
                return (prompt_name, cmd or "")

            for key in sorted(grouped_results, key=sort_key):
                prompt_name, command_text = key
                stats = extract_prompt_stats(
                    grouped_results[key],
                    prompt_name,
                    grouped_run_dirs.get(key),
                    grouped_default_budget.get(key),
                    command_text,
                )
                thinking_budget_value = stats["thinking_budget"]
                if isinstance(thinking_budget_value, list):
                    thinking_budget_value = json.dumps(thinking_budget_value)

                models = grouped_models.get(key, [])
                model_value = "unknown"
                unique_models = sorted(set(models))
                if unique_models:
                    model_value = unique_models[0] if len(unique_models) == 1 else ",".join(unique_models)

                all_rows.append(
                    {
                        "task": task_dir.name,
                        "model": model_value,
                        "input": stats["input"],
                        "system": prompt_name,
                        "thinking_budget": thinking_budget_value,
                        "repeat_count": stats["count"],
                        "max_score": stats["max_score"],
                        "min_score": stats["min_score"],
                        "mean_score": round(stats["mean_score"], 2) if stats["mean_score"] is not None else None,
                        "score_list": json.dumps(stats["scores"]),
                        "max_response_time": round(stats["max_response_time"], 4) if stats["max_response_time"] is not None else None,
                        "min_response_time": round(stats["min_response_time"], 4) if stats["min_response_time"] is not None else None,
                        "mean_response_time": round(stats["mean_response_time"], 4) if stats["mean_response_time"] is not None else None,
                        "mean_total_tokens": stats["mean_total_tokens"],
                        "error_free_count": stats["error_free_count"],
                        "no_plan_count": stats["no_plan_count"],
                    }
                )
        else:
            for meta_path in meta_paths:
                rows = extract_csv_rows(meta_path=meta_path, task_name=task_dir.name)
                all_rows.extend(rows)
    
    return all_rows


def collect_penalties_from_detail(detail_file: str) -> Counter:
    """Count penalty components in a single detail file."""
    penalties = Counter()
    detail_path = Path(detail_file)
    if not detail_path.exists():
        return penalties

    try:
        detail_data = load_json(detail_path)
    except Exception:
        return penalties

    for breakdown in detail_data.get("score", {}).get("breakdown", []):
        status = breakdown.get("status", "") or ""
        for comp in status.split("+"):
            if comp in PENALTY_COMPONENTS:
                penalties[comp] += 1
    return penalties


def summarize_penalties(
    *,
    output_root: Path,
    latest_only: bool,
) -> List[Dict[str, Any]]:
    """Aggregate penalty distributions per task."""
    summary_rows: List[Dict[str, Any]] = []

    for task_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        penalty_counter = Counter()
        meta_paths = select_runs(iter_run_metadata(task_dir), latest_only)

        for meta_path in meta_paths:
            data = load_json(meta_path)
            for result in data.get("results", []):
                detail_file = result.get("detail_file")
                if not detail_file:
                    continue
                penalty_counter.update(collect_penalties_from_detail(detail_file))

        total = sum(penalty_counter.values())
        if total == 0:
            summary_rows.append(
                {
                    "task": task_dir.name,
                    "penalty_type": "(none)",
                    "count": 0,
                    "percentage": 0.0,
                    "total_penalties": 0,
                }
            )
            continue

        for penalty_type, count in sorted(
            penalty_counter.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            summary_rows.append(
                {
                    "task": task_dir.name,
                    "penalty_type": penalty_type,
                    "count": count,
                    "percentage": round((count / total) * 100, 1),
                    "total_penalties": total,
                }
            )

    return summary_rows


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write rows to CSV file."""
    if not rows:
        print("No data to write.")
        return
    
    fieldnames = [
        "task",
        "model",
        "input",
        "system",
        "thinking_budget",
        "repeat_count",
        "max_score",
        "min_score",
        "mean_score",
        "score_list",
        "max_response_time",
        "min_response_time",
        "mean_response_time",
        "mean_total_tokens",
        "error_free_count",
        "no_plan_count",
        ]
    
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_penalty_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    """Write penalty distribution rows to CSV file."""
    if not rows:
        print("No penalty data to write.")
        return

    fieldnames = ["task", "penalty_type", "count", "percentage", "total_penalties"]
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Gemini pipeline evaluation runs to CSV.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to the pipeline config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination for the summary CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Override for the output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Only include the most recent run for each task.",
    )
    parser.add_argument(
        "--aggregate-prompts",
        action="store_true",
        help="Aggregate results across runs within each task for the same prompt.",
    )
    parser.add_argument(
        "--penalty-output",
        action="store_true",
        help=f"Also write penalty distribution CSV to {DEFAULT_PENALTY_OUTPUT}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    rows = summarize_runs_to_csv(
        output_root=output_root,
        latest_only=args.latest_only,
        aggregate_prompts=args.aggregate_prompts,
    )
    
    csv_path = Path(args.output).expanduser().resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, csv_path)
    
    print(f"Wrote summary with {len(rows)} prompt results to {csv_path}")

    if args.penalty_output:
        penalty_rows = summarize_penalties(
            output_root=output_root,
            latest_only=args.latest_only,
        )
        penalty_csv = DEFAULT_PENALTY_OUTPUT
        penalty_csv.parent.mkdir(parents=True, exist_ok=True)
        write_penalty_csv(penalty_rows, penalty_csv)
        print(f"Wrote penalty distribution with {len(penalty_rows)} rows to {penalty_csv}")


if __name__ == "__main__":
    main()
