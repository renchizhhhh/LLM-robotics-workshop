#!/usr/bin/env python3
"""Utilities for tracking Spot's pose and converting between grid and robot frames."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from tf.transformations import euler_from_quaternion  # type: ignore
except ImportError:  # pragma: no cover - fallback for environments without tf
    def euler_from_quaternion(quat: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
        """Compute roll, pitch, yaw from a quaternion without tf."""
        x, y, z, w = quat
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
        roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
        return roll, pitch, yaw


def _normalize_angle(angle: float) -> float:
    """Wrap an angle to the [-pi, pi] interval."""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class GridFrameMapper:
    """Bidirectional transforms between grid cells and robot frames."""

    cell_size: float = 0.57
    origin_cell: Tuple[float, float] = (4.0, 4.0)
    grid_offset_row: Optional[float] = None
    grid_offset_col: Optional[float] = None
    grid_yaw_offset: Optional[float] = None
    facing_baseline: str = "N"

    def has_offsets(self) -> bool:
        return self.grid_offset_row is not None and self.grid_offset_col is not None

    def reset_offsets(self) -> None:
        self.grid_offset_row = None
        self.grid_offset_col = None
        self.grid_yaw_offset = None

    def set_origin_cell(self, row: float, col: float) -> None:
        """Set the grid origin cell and clear cached offsets so the next pose reinitialises them."""
        origin = (float(row), float(col))
        if origin != self.origin_cell:
            self.origin_cell = origin
            self.reset_offsets()

    def set_offsets(self, row_offset: float, col_offset: float) -> None:
        self.grid_offset_row = row_offset
        self.grid_offset_col = col_offset

    @staticmethod
    def _yaw_from_pose(pose: Any) -> float:
        quat = (
            getattr(pose.orientation, "x", 0.0),
            getattr(pose.orientation, "y", 0.0),
            getattr(pose.orientation, "z", 0.0),
            getattr(pose.orientation, "w", 1.0),
        )
        _, _, yaw = euler_from_quaternion(quat)
        return float(yaw)

    def _rotate_to_grid_axes(self, x: float, y: float) -> Tuple[float, float]:
        """Rotate vision/world (x,y) into the grid-aligned axes using the initial yaw."""
        theta = self.grid_yaw_offset or 0.0
        c = math.cos(-theta)
        s = math.sin(-theta)
        return c * x - s * y, s * x + c * y

    def _rotate_to_vision_axes(self, xg: float, yg: float) -> Tuple[float, float]:
        """Rotate grid-aligned coordinates back into the vision/world frame."""
        theta = self.grid_yaw_offset or 0.0
        c = math.cos(theta)
        s = math.sin(theta)
        return c * xg - s * yg, s * xg + c * yg

    def ensure_offsets_from_pose(self, pose: Any) -> bool:
        """Initialise/refresh grid orientation and offsets using the provided pose.

        Behavior:
        - If grid_yaw_offset is not set, initialize it from the pose yaw so that the
          initial robot facing aligns with grid North.
        - If offsets are missing, compute them from the (possibly rotated) pose.
        - If offsets already exist but we just initialized grid_yaw_offset, recompute
          offsets so the origin stays consistent under the new rotation.
        Returns True if either the yaw offset or the grid offsets were set/updated.
        """
        updated = False

        # Establish the grid's orientation the first time we see a pose
        yaw_was_missing = self.grid_yaw_offset is None
        if yaw_was_missing:
            try:
                yaw0 = self._yaw_from_pose(pose)
                # Rotate grid axes so that the world/grid baseline facing aligns
                # with the intended grid axes:
                #  - N baseline: forward maps to grid North     (0°)
                #  - E baseline: forward maps to grid East      (+90°)
                #  - S baseline: forward maps to grid South     (+180°)
                #  - W baseline: forward maps to grid West      (-90°)
                baseline_rot = {"N": 0.0, "E": math.pi / 2.0, "S": math.pi, "W": -math.pi / 2.0}.get(
                    self.facing_baseline, 0.0
                )
                self.grid_yaw_offset = _normalize_angle(yaw0 + baseline_rot)
            except Exception:
                self.grid_yaw_offset = 0.0
            updated = True

        # Compute or refresh offsets if missing, or if yaw was just established
        if yaw_was_missing or not self.has_offsets():
            x = getattr(pose.position, "x", 0.0)
            y = getattr(pose.position, "y", 0.0)
            xg, yg = self._rotate_to_grid_axes(x, y)
            row_cont = -xg / self.cell_size
            col_cont = -yg / self.cell_size
            self.grid_offset_row = self.origin_cell[0] - row_cont
            self.grid_offset_col = self.origin_cell[1] - col_cont
            updated = True

        return updated

    def vision_xy_to_grid(self, x: float, y: float) -> Tuple[float, float]:
        if not self.has_offsets():
            raise ValueError("Grid offsets are not initialised")
        xg, yg = self._rotate_to_grid_axes(x, y)
        row_cont = -xg / self.cell_size
        col_cont = -yg / self.cell_size
        grid_row = row_cont + float(self.grid_offset_row)
        grid_col = col_cont + float(self.grid_offset_col)
        return grid_row, grid_col

    def grid_to_vision(self, row: float, col: float) -> Tuple[float, float]:
        if not self.has_offsets():
            raise ValueError("Grid offsets are not initialised")
        row_cont = row - float(self.grid_offset_row)
        col_cont = col - float(self.grid_offset_col)
        xg = -row_cont * self.cell_size
        yg = -col_cont * self.cell_size
        xv, yv = self._rotate_to_vision_axes(xg, yg)
        return xv, yv

    def grid_to_body_delta(self, target_row: float, target_col: float, pose: Any) -> Tuple[float, float]:
        """Convert a grid target to body-frame deltas using the current pose."""
        self.ensure_offsets_from_pose(pose)
        x_current = getattr(pose.position, "x", 0.0)
        y_current = getattr(pose.position, "y", 0.0)
        x_target, y_target = self.grid_to_vision(target_row, target_col)
        dx_v = x_target - x_current
        dy_v = y_target - y_current
        quat = (
            getattr(pose.orientation, "x", 0.0),
            getattr(pose.orientation, "y", 0.0),
            getattr(pose.orientation, "z", 0.0),
            getattr(pose.orientation, "w", 1.0),
        )
        _, _, yaw = euler_from_quaternion(quat)
        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)
        body_x = cos_yaw * dx_v - sin_yaw * dy_v
        body_y = sin_yaw * dx_v + cos_yaw * dy_v
        return body_x, body_y


@dataclass
class PoseTracker:
    """Track Spot's grid-aligned pose based on real-robot feedback."""

    grid_mapper: GridFrameMapper
    yaw_origin: Optional[float] = None
    current_cell: List[float] = field(default_factory=lambda: [4.0, 4.0])
    current_facing: str = "N"
    current_yaw_rel: float = 0.0
    # Baseline facing taken from the world's start pose. All subsequent
    # yaw-to-cardinal mapping is done relative to this baseline so the
    # GUI preview aligns with the world start orientation.
    facing_baseline: str = "N"

    def apply_start_pose(self, start_pose: Dict[str, Any]) -> None:
        """Align internal state and grid origin with a provided grid-frame start pose."""
        import math
        row = float(start_pose.get("row", self.current_cell[0]))
        col = float(start_pose.get("col", self.current_cell[1]))
        facing = start_pose.get("facing", self.current_facing)

        self.grid_mapper.set_origin_cell(row, col)
        self.current_cell = [row, col]
        if isinstance(facing, str):
            self.current_facing = facing
            # Capture the world-provided baseline so 0° yaw aligns with it
            self.facing_baseline = facing
            # Propagate baseline to the mapper so its yaw offset uses the same reference
            self.grid_mapper.facing_baseline = facing
            
            # Set current_yaw_rel based on the starting facing direction relative to North
            # This ensures rotation calculations work correctly before a real pose arrives
            facing_to_yaw = {
                'N': 0.0,
                'E': -math.pi / 2,
                'S': math.pi,
                'W': math.pi / 2,
            }
            self.current_yaw_rel = facing_to_yaw.get(facing, 0.0)

        # Clear yaw_origin so the next live pose establishes yaw origin relative to the grid start.
        self.yaw_origin = None

    def reset_yaw_origin(self) -> None:
        self.yaw_origin = None # only for real-robot now
        self.current_yaw_rel = 0.0
        # Preserve current facing and baseline; only reset mapper offsets
        self.grid_mapper.reset_offsets()


    def update_from_pose(self, pose: Any) -> Dict[str, Any]:
        offsets_initialised = self.grid_mapper.ensure_offsets_from_pose(pose)

        x = getattr(pose.position, "x", 0.0)
        y = getattr(pose.position, "y", 0.0)
        row, col = self.grid_mapper.vision_xy_to_grid(x, y)

        quat = (
            getattr(pose.orientation, "x", 0.0),
            getattr(pose.orientation, "y", 0.0),
            getattr(pose.orientation, "z", 0.0),
            getattr(pose.orientation, "w", 1.0),
        )
        _, _, yaw = euler_from_quaternion(quat)

        if self.yaw_origin is None:
            # Set yaw_origin such that current_yaw_rel matches the baseline facing
            # If baseline is East, we want current_yaw_rel = -π/2
            # So: yaw_origin = yaw - (-π/2) = yaw + π/2
            facing_to_yaw = {
                'N': 0.0,
                'E': -math.pi / 2,
                'S': math.pi,
                'W': math.pi / 2,
            }
            baseline_yaw_rel = facing_to_yaw.get(self.facing_baseline, 0.0)
            # Adjust yaw_origin so that the first pose gives us the correct baseline yaw
            self.yaw_origin = yaw - baseline_yaw_rel

        self.current_yaw_rel = _normalize_angle(yaw - float(self.yaw_origin))
        self.current_facing = self._yaw_to_cardinal(self.current_yaw_rel)
        self.current_cell = [row, col]

        return {
            "cell": [row, col],
            "facing": self.current_facing,
            "yaw": yaw,
            "yaw_rel": self.current_yaw_rel,
            "offsets_initialized": offsets_initialised,
        }

    def _yaw_to_cardinal(self, yaw_rel: float) -> str:
        """Map relative yaw to a cardinal direction.

        The yaw_rel is already relative to North (0° = North, -90° = East, etc.),
        so we map directly to cardinal directions without applying baseline shift.
        """
        yaw_deg = math.degrees(yaw_rel)
        # Map yaw relative to North directly to cardinal directions
        if -45.0 <= yaw_deg <= 45.0:
            return "N"
        elif 45.0 < yaw_deg <= 135.0:
            return "W"
        elif -135.0 <= yaw_deg < -45.0:
            return "E"
        else:
            return "S"
