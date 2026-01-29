from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import pstdev
from typing import Dict, Iterable, List, Tuple


def iter_run_metadata(base_dir: Path) -> Iterable[Tuple[str, str, List[dict]]]:
    """
    Yield (task, command, results) tuples from every run_metadata.json found under base_dir,
    skipping any pipeline folder named "archive".
    """
    for pipeline in sorted(p for p in base_dir.iterdir() if p.is_dir() and p.name != "archive"):
        for task_dir in sorted(p for p in pipeline.iterdir() if p.is_dir() and p.name.startswith("prompt")):
            for run_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                meta_path = run_dir / "run_metadata.json"
                if not meta_path.exists():
                    continue
                with meta_path.open() as f:
                    data = json.load(f)
                command = data.get("command", "")
                results = data.get("results") or []
                yield task_dir.name, command, results


def aggregate(base_dir: Path) -> List[Dict[str, object]]:
    """Aggregate scores, runtimes, and token counts grouped by (task, command)."""
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    for task, command, results in iter_run_metadata(base_dir):
        key = (task, command)
        entry = grouped.setdefault(key, {"scores": [], "runtimes": [], "tokens": []})

        for res in results:
            score = res.get("score")
            if score is not None:
                entry["scores"].append(score)

            rt = res.get("runtime_seconds")
            if rt is not None:
                entry["runtimes"].append(rt)

            tok = (res.get("token_usage") or {}).get("total_token_count")
            if tok is not None:
                entry["tokens"].append(tok)

    rows: List[Dict[str, object]] = []
    for (task, command) in sorted(grouped.keys()):
        data = grouped[(task, command)]
        scores = data["scores"]
        runtimes = data["runtimes"]
        tokens = data["tokens"]

        if not scores:
            continue

        repeat_count = len(scores)
        success_count = sum(1 for s in scores if s == 1000)
        rejection_count = sum(1 for s in scores if s == 0)
        rows.append(
            {
                "task": task,
                "input": command,
                "repeat_count": repeat_count,
                "max_score": max(scores),
                "min_score": min(scores),
                "mean_score": round(sum(scores) / repeat_count, 4),
                "success_count": success_count,
                "rejection_count": rejection_count,
                "score_list": f"[{', '.join(str(s) for s in scores)}]",
                "max_response_time": round(max(runtimes), 4) if runtimes else "",
                "min_response_time": round(min(runtimes), 4) if runtimes else "",
                "mean_response_time": round(sum(runtimes) / len(runtimes), 4) if runtimes else "",
                "mean_total_tokens": round(sum(tokens) / len(tokens), 4) if tokens else "",
                "std_score": round(pstdev(scores), 4) if scores else "",
                "std_response_time": round(pstdev(runtimes), 4) if runtimes else "",
                "std_total_tokens": round(pstdev(tokens), 4) if tokens else "",
            }
        )

    return rows


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    header = [
        "task",
        "input",
        "repeat_count",
        "max_score",
        "min_score",
        "mean_score",
        "success_count",
        "rejection_count",
        "score_list",
        "max_response_time",
        "min_response_time",
        "mean_response_time",
        "mean_total_tokens",
        "std_score",
        "std_response_time",
        "std_total_tokens",
    ]

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    base_dir = Path("maze_evaluation/gemini_eval_runs")
    output_path = base_dir / "combined_pipeline_summary.csv"

    rows = aggregate(base_dir)
    write_csv(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
