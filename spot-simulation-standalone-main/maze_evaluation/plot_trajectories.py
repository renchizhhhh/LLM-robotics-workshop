#!/usr/bin/env python3
"""Plot robot trajectories from CSV files so lines pass through cell centers.

Usage:
  python scripts/plot_trajectories.py --input <dir_or_file> [--grid-size 10]

This script reads CSV files produced by the maze prompt runner (positions_*.csv).
Each CSV row is a trajectory: cells are strings like "(3,1)". The script maps
each cell to the center of that grid cell and draws the trajectory as a line
running through centers (not along cell borders).

Saves PNG(s) next to each CSV with the same basename.
"""

import argparse
import sys
import csv
import re
from pathlib import Path
from typing import List, Optional, Tuple
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

COORD_RE = re.compile(r"\(?\s*(-?\d+)\s*,\s*(-?\d+)\s*\)?")

# ----------------------
# Top-level default config
# ----------------------
DEFAULT_INPUT = Path(__file__).parent.parent / 'maze_prompt_logs'
DEFAULT_PATTERN = '*.csv'
DEFAULT_RECURSIVE = True
DEFAULT_GRID_SIZE = 10
DEFAULT_OUT_DIR = None
WORLD_ID = '14'

SEGMENT_FULL = 'full'
SEGMENT_START_TO_OBJECT = 'start-to-object'
SEGMENT_OBJECT_TO_EXIT = 'object-to-exit'
SEGMENT_CHOICES = (SEGMENT_FULL, SEGMENT_START_TO_OBJECT, SEGMENT_OBJECT_TO_EXIT)


def fetch_obstacle_cells(world_id: str) -> List[Tuple[int, int]]:
    """Attempt to load wall cells for a given world id from the project's WorldManager.

    Returns a list of (row, col) tuples. If the WorldManager can't be imported or no
    walls are defined for the world, returns an empty list.
    """
    try:
        # Prefer package import if project is installed as a package
        from src.world_manager import WorldManager  # type: ignore
    except Exception:
        # Try adding the src folder to sys.path and import directly
        try:
            src_dir = Path(__file__).resolve().parents[1] / 'src'
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            from world_manager import WorldManager  # type: ignore
        except Exception:
            return []

    try:
        wm = WorldManager()
        cfg = wm.get_world_config(str(world_id))
        # The world config might either be the inner 'config' dict or already the config
        if not cfg:
            return []
        # Try a few places where obstacle_cells might live
        walls = []
        if isinstance(cfg, dict):
            # If get_world_config returned the wrapper that contains 'config'
            if 'obstacle_cells' in cfg and isinstance(cfg['obstacle_cells'], list):
                walls = cfg['obstacle_cells']
            elif 'config' in cfg and isinstance(cfg['config'], dict):
                walls = cfg['config'].get('obstacle_cells', []) or []
            else:
                # Maybe get_world_config already returned the inner config
                walls = cfg.get('obstacle_cells', []) if hasattr(cfg, 'get') else []
        # Normalize entries to tuples
        parsed: List[Tuple[int, int]] = []
        for item in walls:
            try:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    parsed.append((int(item[0]), int(item[1])))
            except Exception:
                continue
        return parsed
    except Exception:
        return []

def parse_position(token: str) -> Optional[Tuple[int, int]]:
    if not token:
        return None
    token = token.strip().strip('"').strip("'")
    m = COORD_RE.search(token)
    if not m:
        return None
    try:
        r = int(m.group(1))
        c = int(m.group(2))
        return (r, c)
    except Exception:
        return None


def cell_to_center(rc: Tuple[int, int], grid_size: int = 10) -> Tuple[float, float]:
    r, c = rc
    x = c + 0.5
    y = r + 0.5
    return (x, y)


def read_positions_from_csv(path: Path) -> List[List[Tuple[int, int]]]:
    trajectories: List[List[Tuple[int, int]]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return trajectories
        for row in reader:
            traj: List[Tuple[int, int]] = []
            for token in row:
                pos = parse_position(token)
                if pos is not None:
                    traj.append(pos)
            if traj:
                trajectories.append(traj)
    return trajectories


def _in_bounds(rc: Tuple[int, int], grid_size: int) -> bool:
    r, c = rc
    return 0 <= r < grid_size and 0 <= c < grid_size


def _iter_cells_between(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Return the inclusive Manhattan path between two cells, excluding the starting cell."""
    path: List[Tuple[int, int]] = []
    cur_r, cur_c = start
    target_r, target_c = end
    while (cur_r, cur_c) != (target_r, target_c):
        if cur_r != target_r:
            cur_r += 1 if target_r > cur_r else -1
        elif cur_c != target_c:
            cur_c += 1 if target_c > cur_c else -1
        path.append((cur_r, cur_c))
    return path


def slice_trajectory(traj: List[Tuple[int, int]], segment: str,
                     object_pos: Tuple[int, int], exit_pos: Optional[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not traj:
        return []
    if segment == SEGMENT_FULL:
        return traj

    try:
        obj_idx = traj.index(object_pos)
    except ValueError:
        return []

    if segment == SEGMENT_START_TO_OBJECT:
        return traj[:obj_idx + 1]

    if segment == SEGMENT_OBJECT_TO_EXIT:
        if exit_pos is None:
            return []
        try:
            exit_idx = traj.index(exit_pos, obj_idx)
        except ValueError:
            try:
                exit_idx = traj.index(exit_pos)
            except ValueError:
                return []
        if exit_idx < obj_idx:
            return []
        return traj[obj_idx:exit_idx + 1]

    return []


def build_heatmap(trajectories: List[List[Tuple[int, int]]], grid_size: int) -> np.ndarray:
    heatmap = np.zeros((grid_size, grid_size), dtype=float)
    for traj in trajectories:
        if not traj:
            continue
        previous = traj[0]
        if _in_bounds(previous, grid_size):
            heatmap[previous[0], previous[1]] += 1
        for current in traj[1:]:
            if current == previous:
                if _in_bounds(current, grid_size):
                    heatmap[current[0], current[1]] += 1
            else:
                for cell in _iter_cells_between(previous, current):
                    if _in_bounds(cell, grid_size):
                        heatmap[cell[0], cell[1]] += 1
            previous = current
    return heatmap


def plot_trajectories(trajectories: List[List[Tuple[int, int]]], out_png: Path,
                      grid_size: int = 10, title: Optional[str] = None,
                      segment: str = SEGMENT_FULL):
    fig, ax = plt.subplots(figsize=(6, 6))
    legend_extra_handles: List[Patch] = []
    divider = make_axes_locatable(ax)
    # Make the colorbar thinner and move it further right (larger pad) to avoid overlapping the legend
    cax = divider.append_axes("right", size="2%", pad=1.4)

    object_pos = (2, 7)
    final_positions = [traj[-1] for traj in trajectories if traj]
    exit_pos: Optional[Tuple[int, int]] = None
    if final_positions:
        most_common, _ = Counter(final_positions).most_common(1)[0]
        exit_pos = most_common

    filtered_trajectories: List[List[Tuple[int, int]]] = []
    skipped = 0
    for traj in trajectories:
        sliced = slice_trajectory(traj, segment, object_pos, exit_pos)
        if sliced:
            filtered_trajectories.append(sliced)
        else:
            skipped += 1
    if not filtered_trajectories:
        filtered_trajectories = []
    if segment != SEGMENT_FULL and skipped:
        label = title if title else out_png.name
        print(f"[{label}] segment filter '{segment}' removed {skipped} trajectory(s) that lacked the required waypoints.")

    heatmap = build_heatmap(filtered_trajectories, grid_size)

    # If a WORLD_ID is set, fetch wall cells for that world and draw them as gray squares
    try:
        obstacle_cells = fetch_obstacle_cells(WORLD_ID) if WORLD_ID else []
    except Exception:
        obstacle_cells = []
    for (wr, wc) in obstacle_cells:
        if _in_bounds((wr, wc), grid_size):
            heatmap[wr, wc] = np.nan

    vmax = np.nanmax(heatmap) if np.any(np.isfinite(heatmap)) else 0.0
    if vmax <= 0:
        vmax = 1.0

    im = ax.imshow(
        heatmap,
        cmap='YlGnBu_r',
        origin='upper',
        extent=[0, grid_size, grid_size, 0],
        vmin=0,
        vmax=vmax,
        interpolation='nearest'
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Traversal frequency')
    # Remove tick marks and numeric labels from the colorbar
    try:
        cbar.set_ticks([])
        cbar.ax.tick_params(length=0)
        cbar.ax.set_yticklabels([])
    except Exception:
        pass

    # Draw grid lines on top of the heatmap
    for i in range(grid_size + 1):
        ax.plot([0, grid_size], [i, i], color="#cccccc", linewidth=0.5)
        ax.plot([i, i], [0, grid_size], color="#cccccc", linewidth=0.5)

    if obstacle_cells:
        for (wr, wc) in obstacle_cells:
            if not _in_bounds((wr, wc), grid_size):
                continue
            rect = Rectangle((wc, wr), 1.0, 1.0, facecolor='lightgray', edgecolor='none', alpha=0.8)
            ax.add_patch(rect)
        legend_extra_handles.append(Patch(facecolor='lightgray', edgecolor='none', label='wall'))
        ax.add_patch(Rectangle((-10, -10), 0, 0, alpha=0))

    color_values = list(mcolors.TABLEAU_COLORS.values())

    for idx, traj in enumerate(filtered_trajectories):
        if not traj:
            continue
        centers = [cell_to_center(p, grid_size) for p in traj]
        xs = [p[0] for p in centers]
        ys = [p[1] for p in centers]
        color = color_values[idx % len(color_values)]
        ax.plot(xs, ys, marker='o', linewidth=2.0, alpha=0.5, color=color, label=f'traj {idx+1}')
        ax.plot(xs[0], ys[0], marker='*', color='gold', markersize=10, markeredgecolor='k')
        ax.plot(xs[-1], ys[-1], marker='o', color=color, markersize=6)

    try:
        obj_x, obj_y = cell_to_center(object_pos, grid_size)
        ax.plot(obj_x, obj_y, marker='^', color='tab:orange', markersize=10, label='object (2,7)')
    except Exception:
        pass

    if exit_pos is not None:
        try:
            ex_x, ex_y = cell_to_center(exit_pos, grid_size)
            ax.plot(ex_x, ex_y, marker='X', color='red', markersize=10, label=f'exit {exit_pos}')
        except Exception:
            pass

    ticks = [i + 0.5 for i in range(grid_size)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(i) for i in range(grid_size)])
    ax.set_yticklabels([str(i) for i in range(grid_size)])

    ax.grid(False)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    if title:
        ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    if legend_extra_handles:
        handles = handles + legend_extra_handles
        labels = labels + [h.get_label() for h in legend_extra_handles]
    if handles:
        ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

    fig.tight_layout()
    fig.savefig(str(out_png), dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot trajectories through cell centers")
    parser.add_argument('--input', '-i', required=True,
                        help='Input CSV file or directory containing position CSVs')
    parser.add_argument('--pattern', default='*.csv',
                        help='Glob pattern to match CSV files when --input is a directory (default: "*.csv")')
    parser.add_argument('--recursive', action='store_true',
                        help='Search recursively in subdirectories for matching files')
    parser.add_argument('--grid-size', type=int, default=10,
                        help='Grid size (number of rows/cols). Default: 10')
    parser.add_argument('--out-dir', help='Optional output directory. If omitted, PNGs are saved next to CSVs')
    parser.add_argument('--segment', choices=SEGMENT_CHOICES, default=SEGMENT_FULL,
                        help='Which portion of each trajectory to visualize: "full", "start-to-object", or "object-to-exit"')
    args = parser.parse_args()
    run_plotter(args.input, pattern=args.pattern, recursive=args.recursive,
                grid_size=args.grid_size, out_dir=args.out_dir, segment=args.segment)


def run_plotter(input_path: str, pattern: str = '*.csv', recursive: bool = False,
                grid_size: int = 10, out_dir: Optional[str] = None,
                segment: str = SEGMENT_FULL):
    """Run the plotting using explicit parameters (callable from other code or a top-level config).

    This function mirrors the behavior of the original script body but accepts parameters
    so the script can be executed without using command-line arguments.
    """
    inp = Path(input_path)
    files: List[Path] = []
    if inp.is_dir():
        if recursive:
            files = sorted(inp.rglob(pattern))
        else:
            files = sorted(inp.glob(pattern))
    elif inp.is_file():
        files = [inp]
    else:
        print(f"No such file or directory: {inp}")
        return

    out_dir_path = Path(out_dir) if out_dir else None
    if out_dir_path and not out_dir_path.exists():
        out_dir_path.mkdir(parents=True, exist_ok=True)

    if not files:
        print(f"No files matched in {inp} using pattern '{pattern}' (recursive={recursive})")
        return
    for f in files:
        try:
            trajectories = read_positions_from_csv(f)
            if not trajectories:
                print(f"No trajectories found in {f}, skipping")
                continue
            out_png = (out_dir_path / (f.stem + '.png')) if out_dir_path else (f.with_suffix('.png'))
            title = f.name
            plot_trajectories(trajectories, out_png, grid_size=grid_size,
                              title=title, segment=segment)
            print(f"Saved: {out_png}")
        except Exception as e:
            print(f"Error processing {f}: {e}")


if __name__ == '__main__':

    if len(sys.argv) <= 1:
        run_plotter(str(DEFAULT_INPUT), pattern=DEFAULT_PATTERN,
                    recursive=DEFAULT_RECURSIVE, grid_size=DEFAULT_GRID_SIZE,
                    out_dir=DEFAULT_OUT_DIR, segment=SEGMENT_START_TO_OBJECT)
    else:
        main()
