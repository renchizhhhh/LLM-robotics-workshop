#!/usr/bin/env python3
"""
GUI Canvas Module - Canvas Rendering
Handles world drawing, robot rendering, and coordinate transformations.
"""

import logging
import math
import os
import tkinter as tk
import tkinter.font as tkfont
from message_bus import rospy

logger = logging.getLogger(__name__)

# Preferred TTF families; Tk will pick the first available with Xft-enabled builds.
CANVAS_FONT_CANDIDATES = (
    "DejaVu Sans",
    "Noto Sans",
    "Liberation Sans",
    "Bitstream Vera Sans",
    "Arial",
    "Helvetica",
)


class GUICanvasRenderer:
    """Handles all canvas rendering operations."""
    
    def __init__(self, canvas, grid_size=10, cell_size=45, scale_factor=0.90):
        self.canvas = canvas
        # Backward-compatible single-size; also support rows/cols distinct
        self.grid_size = grid_size
        self.grid_rows = grid_size
        self.grid_cols = grid_size
        self.cell_size = cell_size
        self.scale_factor = scale_factor
        self._grid_padding = None
        self.canvas_font_family = self._resolve_canvas_font_family()
        try:
            rospy.loginfo(f"Canvas font family resolved to: {self.canvas_font_family}")
        except Exception:
            pass
        
        # Track canvas resize events so grid stays centered after window adjustments
        self._last_canvas_size = None
        self._canvas_resize_after = None
        # Note: Canvas resize binding will be handled by the main GUI core
    
    def get_canvas_dimensions(self):
        """Return current canvas dimensions with fallbacks for early layout."""
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            width = self.canvas.winfo_reqwidth()
            height = self.canvas.winfo_reqheight()
        if width <= 1 or height <= 1:
            try:
                width = int(self.canvas['width'])
                height = int(self.canvas['height'])
            except Exception:
                width, height = 1200, 800
        return width, height
    
    def get_cell_size(self):
        """Return cell size scaled to fill the available canvas area."""
        canvas_width, canvas_height = self.get_canvas_dimensions()
        cols = max(1, self.grid_cols)
        rows = max(1, self.grid_rows)
        
        # Reserve padding for axis labels and breathing room so labels stay visible
        base_ref = self.cell_size or 45
        horizontal_padding = max(60, int(base_ref * 1.1))
        vertical_padding_top = max(70, int(base_ref * 1.35))
        vertical_padding_bottom = max(25, int(base_ref * 0.6))
        
        available_width = canvas_width - 2 * horizontal_padding
        available_height = canvas_height - (vertical_padding_top + vertical_padding_bottom)
        
        # Determine the largest square that fits inside the padded canvas
        max_cell_width = available_width / cols if available_width > 0 else 0
        max_cell_height = available_height / rows if available_height > 0 else 0
        if max_cell_width <= 0 or max_cell_height <= 0:
            base_size = self.cell_size if self.cell_size > 0 else 45
        else:
            base_size = min(max_cell_width, max_cell_height)
        
        scaled_size = base_size * (self.scale_factor or 1.0)
        
        # Persist padding info for origin calculations
        self._grid_padding = {
            "left": horizontal_padding,
            "right": horizontal_padding,
            "top": vertical_padding_top,
            "bottom": vertical_padding_bottom,
        }
        
        return max(4, int(scaled_size))
    
    def scale(self, value):
        """Scale a design-space value based on the current cell size.

        Maintains proportional sizing as the canvas grows/shrinks while
        preserving negative values for offsets.
        """
        if not isinstance(value, (int, float)):
            return value
        base_cell = self.cell_size if self.cell_size > 0 else 45
        ratio = 1.0
        try:
            current_cell = self.get_cell_size()
            if base_cell > 0:
                ratio = current_cell / base_cell
        except Exception:
            ratio = self.scale_factor or 1.0
        scaled = value * ratio
        return int(round(scaled)) if isinstance(value, (int, float)) else scaled
    
    def _compute_grid_origin(self, cell_size):
        """Compute the top-left grid origin using reserved padding."""
        canvas_width, canvas_height = self.get_canvas_dimensions()
        grid_width = self.grid_cols * cell_size
        grid_height = self.grid_rows * cell_size
        padding = self._grid_padding or {}
        
        left_pad = padding.get("left", 0)
        right_pad = padding.get("right", 0)
        top_pad = padding.get("top", 0)
        bottom_pad = padding.get("bottom", 0)
        
        avail_width = max(0, canvas_width - left_pad - right_pad)
        avail_height = max(0, canvas_height - top_pad - bottom_pad)
        
        extra_w = max(0, avail_width - grid_width)
        extra_h = max(0, avail_height - grid_height)
        
        grid_start_x = left_pad + extra_w // 2
        grid_start_y = top_pad + extra_h // 2
        
        # Fallback to centered placement if padding collapses
        if avail_width <= 0 or avail_height <= 0:
            grid_start_x = (canvas_width - grid_width) // 2
            grid_start_y = (canvas_height - grid_height) // 2
        
        return grid_start_x, grid_start_y
    
    def _convert_path_points(self, points, grid_start_x, grid_start_y, cell_size):
        """Convert grid-based points into canvas coordinates (with fallback)."""
        if not points:
            return []
        
        converted = []
        grid_limit = max(self.grid_rows, self.grid_cols) + 5
        
        for point in points:
            row = col = None
            if isinstance(point, dict):
                row = point.get("row")
                col = point.get("col")
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                row, col = point[0], point[1]
            else:
                continue
            
            if row is None or col is None:
                continue
            
            try:
                row_val = float(row)
                col_val = float(col)
            except (TypeError, ValueError):
                continue
            
            if abs(row_val) > grid_limit or abs(col_val) > grid_limit:
                converted.append((row_val, col_val))
            else:
                # Use cell_to_canvas for consistent coordinate transformation
                x, y = self.cell_to_canvas(row_val, col_val)
                converted.append((int(round(x)), int(round(y))))
        
        return converted

    def _draw_dotted_path(self, coords, color, radius=None, step=None):
        """Draw a path as evenly spaced dots instead of dashed lines."""
        if not coords or len(coords) < 2:
            return
        r = radius if radius is not None else max(1, self.scale(2))  # smaller dots
        spacing = step if step is not None else self.scale(14)       # slightly wider spacing
        for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
            dx = x2 - x1
            dy = y2 - y1
            dist = math.hypot(dx, dy)
            if dist <= 0:
                continue
            steps = max(1, int(dist // max(1, spacing)))
            for i in range(steps + 1):
                t = i / steps
                x = x1 + dx * t
                y = y1 + dy * t
                self.canvas.create_oval(
                    x - r, y - r, x + r, y + r,
                    fill=color, outline=color, width=0
                )
    
    def _on_canvas_resize(self, event):
        """Handle canvas resize events by redrawing the grid in the new bounds."""
        # Debounce resize events to avoid excessive redraws
        if self._canvas_resize_after:
            self.canvas.after_cancel(self._canvas_resize_after)

        self._canvas_resize_after = self.canvas.after(80, self._redraw_after_resize)
    
    def _redraw_after_resize(self):
        """Redraw world/robot after a resize debounce window."""
        self._canvas_resize_after = None
        # Trigger a redraw by calling the main GUI's update_display method
        # This will be handled by the main GUI core

    def _resolve_canvas_font_family(self):
        """Pick the first preferred family that Tk can instantiate; fall back to default."""
        for candidate in CANVAS_FONT_CANDIDATES:
            try:
                tkfont.Font(family=candidate, size=12)
                return candidate
            except Exception:
                continue
        return "TkDefaultFont"

    def _font(self, size, weight="bold"):
        """Return a canvas font tuple with the resolved family."""
        return (self.canvas_font_family, size, weight)
    
    def set_dimensions(self, rows, cols):
        """Update grid dimensions and request a redraw by caller."""
        try:
            self.grid_rows = max(1, int(rows))
            self.grid_cols = max(1, int(cols))
            # Maintain old attribute for any legacy callers using square size
            self.grid_size = max(self.grid_rows, self.grid_cols)
        except Exception:
            pass

    def cell_to_canvas(self, row, col):
        """Convert grid cell coordinates to canvas coordinates"""
        # Grid cell size (pixels) - configurable and scaled
        cell_size = self.get_cell_size()
        grid_start_x, grid_start_y = self._compute_grid_origin(cell_size)
        
        # Convert grid coordinates to canvas coordinates
        canvas_x = grid_start_x + col * cell_size + cell_size // 2
        canvas_y = grid_start_y + row * cell_size + cell_size // 2
        
        return canvas_x, canvas_y
    
    def body_to_vision(self, body_x, body_y, body_yaw):
        """Convert body frame movement to vision frame position"""
        # Use the provided body_yaw for coordinate transformation
        vision_yaw = body_yaw
        
        cos_yaw = math.cos(vision_yaw)
        sin_yaw = math.sin(vision_yaw)
        
        vision_dx = cos_yaw * body_x - sin_yaw * body_y
        vision_dy = sin_yaw * body_x + cos_yaw * body_y
        
        return vision_dx, vision_dy
    
    def _heatmap_color(self, count, max_count):
        """Generate heatmap color based on count and max count"""
        if max_count <= 1:
            return "#ff9500"
        
        intensity = count / max_count
        
        if intensity <= 0.5:
            r = int(255 * intensity * 2)
            g = 255
            b = 0
        else:
            r = 255
            g = int(255 * (2 - intensity * 2))
            b = 0
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def draw_world(self, world_config, objects, waypoints, zones, multi_preview_paths=None,
                   multi_preview_counts=None, heatmap_visible=False, all_preview_visible=False,
                   preview_path_coords=None, preview_waypoints=None):
        """Draw the static world elements"""
        logger.debug(
            "draw_world called - canvas size: %dx%d",
            self.canvas.winfo_width(),
            self.canvas.winfo_height()
        )
        self.canvas.delete("all")
        # Draw grid using configured size and cell pixels (scaled)
        cell_size = self.get_cell_size()
        # Use non-square dimensions: cols define width, rows define height
        grid_width = self.grid_cols * cell_size
        grid_height = self.grid_rows * cell_size

        # Position grid with reserved padding so labels stay visible
        grid_start_x, grid_start_y = self._compute_grid_origin(cell_size)
        checkpoints = world_config.get("checkpoints", {})
        
        # Draw uniform cell backgrounds for easier readability
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x1 = grid_start_x + col * cell_size
                y1 = grid_start_y + row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill="#4a4a4a", outline="")
        
        # Draw wall cells (background layer)
        obstacle_cells = world_config.get("obstacle_cells", [])
        logger.debug("Drawing %d wall cells", len(obstacle_cells))
        
        for wall_cell in obstacle_cells:
            row, col = wall_cell[0], wall_cell[1]
            if not (0 <= row < self.grid_rows and 0 <= col < self.grid_cols):
                logger.warning("Skipping obstacle cell out of bounds: (%s, %s)", row, col)
                continue

            # Get cell position
            cell_x, cell_y = self.cell_to_canvas(row, col)
            wall_size = cell_size

            # Draw wall cell as a filled rectangle
            x1 = cell_x - wall_size // 2
            y1 = cell_y - wall_size // 2
            x2 = cell_x + wall_size // 2
            y2 = cell_y + wall_size // 2

            # Slightly thicken the outline for perimeter walls so they still read as obstacles.
            is_outer_wall = (row == 0 or row == self.grid_rows - 1 or 
                            col == 0 or col == self.grid_cols - 1)
            outline_width = max(1, self.scale(3 if is_outer_wall else 2))
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill="#c79a63",
                outline="#a87436",
                width=outline_width
            )
        
        # Draw grid lines with thin light lines (after wall cells) to emphasize cell boundaries
        for i in range(self.grid_cols + 1):
            # Vertical lines - thin lines
            x = grid_start_x + i * cell_size
            self.canvas.create_line(x, grid_start_y, x, grid_start_y + grid_height, fill="#ffffff", width=1)
            
        for j in range(self.grid_rows + 1):
            # Horizontal lines - thin lines
            y = grid_start_y + j * cell_size
            self.canvas.create_line(grid_start_x, y, grid_start_x + grid_width, y, fill="#ffffff", width=1)
        
        # Draw coordinate labels (on top of walls)
        for i in range(self.grid_rows):
            # Row labels (left side)
            x = grid_start_x - self.scale(20)
            y = grid_start_y + i * cell_size + cell_size // 2
            self.canvas.create_text(x, y, text=str(i), fill="#58a6ff", font=self._font(max(8, self.scale(12))))
        # Column labels (top side)
        for j in range(self.grid_cols):
            x = grid_start_x + j * cell_size + cell_size // 2
            y = grid_start_y - self.scale(20)
            self.canvas.create_text(x, y, text=str(j), fill="#58a6ff", font=self._font(max(8, self.scale(12))))

        # Draw checkpoints before waypoints so waypoint markers sit on top
        for checkpoint_name, checkpoint_data in checkpoints.items():
            row, col = checkpoint_data["row"], checkpoint_data["col"]
            canvas_x, canvas_y = self.cell_to_canvas(row, col)
            
            letter = checkpoint_name.split('_')[-1] if '_' in checkpoint_name else checkpoint_name
            half = cell_size // 2
            x1 = canvas_x - half
            y1 = canvas_y - half
            x2 = canvas_x + half
            y2 = canvas_y + half
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#ffffff", outline="#d0d0d0", width=max(1, self.scale(2)))
            self.canvas.create_text(canvas_x, canvas_y, text=letter,
                                  fill="#000000", font=self._font(max(12, self.scale(16))))
        
        # Draw waypoints dynamically (on top of walls)
        for waypoint_name, waypoint_data in waypoints.items():
            row, col = waypoint_data["row"], waypoint_data["col"]
            canvas_x, canvas_y = self.cell_to_canvas(row, col)
            
            # Determine waypoint type and styling
            if waypoint_data.get("pick_direction") is not None:
                # Pickup waypoint - show category name
                color = "#ffa657"
                outline = "#ff7b72"
                label = waypoint_name.upper()
            else:
                # General waypoint (should not exist anymore since we removed dropoff waypoints)
                color = "#58a6ff"
                outline = "#39d353"
                label = waypoint_name.upper()
            
            # Draw waypoint rectangle (scaled)
            sz = max(6, self.scale(25))
            self.canvas.create_rectangle(canvas_x-sz, canvas_y-sz, canvas_x+sz, canvas_y+sz, 
                                       fill=color, outline=outline, width=max(1, self.scale(3)))
            self.canvas.create_text(canvas_x, canvas_y-self.scale(40), text=label, font=self._font(max(8, self.scale(12))), fill=color)
            
            # Draw orientation indicator if specified
            if waypoint_data.get("pick_direction") is not None:
                direction = waypoint_data["pick_direction"]
                arrow_length = self.scale(30)
                # Calculate arrow direction based on cardinal direction
                if direction == "N":
                    arrow_x, arrow_y = canvas_x, canvas_y - arrow_length
                elif direction == "S":
                    arrow_x, arrow_y = canvas_x, canvas_y + arrow_length
                elif direction == "E":
                    arrow_x, arrow_y = canvas_x + arrow_length, canvas_y
                elif direction == "W":
                    arrow_x, arrow_y = canvas_x - arrow_length, canvas_y
                else:
                    arrow_x, arrow_y = canvas_x, canvas_y - arrow_length  # Default to North
                
                self.canvas.create_line(canvas_x, canvas_y, arrow_x, arrow_y, fill="#ffffff", width=max(1, self.scale(3)), arrow=tk.LAST)
        
        # Draw zones dynamically (these are dropoff points) - on top of walls
        for zone_name, zone_data in zones.items():
            row, col = zone_data["row"], zone_data["col"]
            canvas_x, canvas_y = self.cell_to_canvas(row, col)
            
            # Draw zone as a larger circle with dropoff styling
            zone_r = self.scale(40)
            self.canvas.create_oval(canvas_x-zone_r, canvas_y-zone_r, canvas_x+zone_r, canvas_y+zone_r,
                                   fill="#2d2d2d", outline="#00d4aa", width=max(1, self.scale(3)), stipple="gray25")
            self.canvas.create_text(canvas_x, canvas_y, text=zone_name.upper(), font=self._font(max(8, self.scale(10))), fill="#00d4aa")
            
            # Draw orientation hint if available
            if zone_data.get("direction") is not None:
                direction = zone_data["direction"]
                arrow_length = self.scale(30)
                # Calculate arrow direction based on cardinal direction
                if direction == "N":
                    arrow_x, arrow_y = canvas_x, canvas_y - arrow_length
                elif direction == "S":
                    arrow_x, arrow_y = canvas_x, canvas_y + arrow_length
                elif direction == "E":
                    arrow_x, arrow_y = canvas_x + arrow_length, canvas_y
                elif direction == "W":
                    arrow_x, arrow_y = canvas_x - arrow_length, canvas_y
                else:
                    arrow_x, arrow_y = canvas_x, canvas_y - arrow_length  # Default to North
                
                self.canvas.create_line(canvas_x, canvas_y, arrow_x, arrow_y, fill="#00d4aa", width=max(1, self.scale(3)), arrow=tk.LAST)
        
        # Draw heatmap overlay before objects so objects remain visible
        if heatmap_visible and multi_preview_counts:
            max_count = max(multi_preview_counts.values()) if multi_preview_counts else 0
            if max_count > 0:
                for (row, col), count in multi_preview_counts.items():
                    if not (0 <= row < self.grid_rows and 0 <= col < self.grid_cols):
                        continue
                    # Use cell_to_canvas for consistent coordinate transformation
                    center_x, center_y = self.cell_to_canvas(row, col)
                    half_size = cell_size // 2
                    x1 = center_x - half_size
                    y1 = center_y - half_size
                    x2 = center_x + half_size
                    y2 = center_y + half_size
                    color = self._heatmap_color(count, max_count)
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="", stipple="gray25")

        # Draw objects with better visibility (on top of everything)
        for obj_name, obj_data in objects.items():
            if obj_data["present"]:
                obj_x, obj_y = self.cell_to_canvas(obj_data["row"], obj_data["col"])
                r = max(2, self.scale(12))
                text_off = self.scale(25)
                self.canvas.create_oval(obj_x-r, obj_y-r, obj_x+r, obj_y+r,
                                      fill=obj_data["color"], outline="#f0f6fc", width=max(1, self.scale(2)))
                self.canvas.create_text(obj_x, obj_y+text_off, text=obj_name, font=self._font(max(8, self.scale(10))), fill="#f0f6fc")
        
        if all_preview_visible and multi_preview_paths:
            for entry in multi_preview_paths:
                raw_points = entry.get("cells")
                if raw_points is None:
                    raw_points = entry.get("coords") or []
                coords = self._convert_path_points(raw_points, grid_start_x, grid_start_y, cell_size)
                if not coords:
                    continue
                color = entry.get("color", "#FF9500")
                label = entry.get("label")
                if len(coords) >= 2:
                    # Render multi-preview as dotted path for clearer stacking
                    self._draw_dotted_path(
                        coords,
                        color=color,
                        radius=max(1, self.scale(2)),
                        step=self.scale(16)
                    )
                for x, y in coords:
                    r = max(2, self.scale(5))
                    self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                            outline=color, width=max(1, self.scale(2)), fill="")
                if label:
                    label_x, label_y = coords[0]
                    self.canvas.create_text(label_x, label_y - self.scale(18),
                                            text=label, fill=color,
                                            font=self._font(max(8, self.scale(9))))

        # Draw pending plan preview path if available
        if preview_path_coords and len(preview_path_coords) > 1:
            resolved_preview = self._convert_path_points(preview_path_coords, grid_start_x, grid_start_y, cell_size)
            if len(resolved_preview) > 1:
                # Draw primary preview as dotted path with smaller dots and wider spacing
                self._draw_dotted_path(
                    resolved_preview,
                    color="#fde047",
                    radius=max(1, self.scale(2)),
                    step=self.scale(10)
                )

                # Overlay numbered waypoint markers derived from the planned actions
                waypoint_groups = {}
                if preview_waypoints:
                    fallback_counter = 1
                    for wp in preview_waypoints:
                        row = col = None
                        label = None
                        if isinstance(wp, dict):
                            row = wp.get("row")
                            col = wp.get("col")
                            label = wp.get("number") or wp.get("index") or wp.get("label")
                        elif isinstance(wp, (list, tuple)) and len(wp) >= 2:
                            row, col = wp[0], wp[1]
                            label = wp[2] if len(wp) >= 3 else None

                        if row is None or col is None:
                            continue

                        try:
                            row_val = int(round(float(row)))
                            col_val = int(round(float(col)))
                        except (TypeError, ValueError):
                            continue

                        if label is None:
                            label = fallback_counter
                            fallback_counter += 1

                        key = f"{row_val},{col_val}"
                        if key not in waypoint_groups:
                            waypoint_groups[key] = {"row": row_val, "col": col_val, "labels": []}
                        waypoint_groups[key]["labels"].append(str(label))

                if waypoint_groups:
                    marker_r = max(8, self.scale(14))
                    font_size = max(8, self.scale(12))
                    for group in waypoint_groups.values():
                        wx, wy = self.cell_to_canvas(group["row"], group["col"])
                        label_text = "/".join(group["labels"])
                        self.canvas.create_oval(
                            wx - marker_r, wy - marker_r, wx + marker_r, wy + marker_r,
                            fill="#fde047", outline="#facc15", width=max(1, self.scale(2))
                        )
                        self.canvas.create_text(
                            wx, wy, text=label_text,
                            fill="#0f172a", font=self._font(font_size)
                        )

        # Draw checkpoints last so labels stay above any trajectory overlays

        # Draw legend at bottom of canvas
        self._draw_legend(grid_start_x, grid_start_y, grid_width, grid_height, cell_size)
    
    def draw_robot(self, robot_row, robot_col, robot_facing, robot_state, has_object, carried_object_name, arm_status, gripper_status):
        """Draw the robot at current position as a simple yellow marker."""
        logger.debug(
            "draw_robot called - position: (%s, %s), facing: %s",
            robot_col,
            robot_row,
            robot_facing
        )
        # Unused parameters kept for compatibility with callers
        _ = (robot_state, carried_object_name, arm_status, gripper_status)
        # Convert robot position to canvas coordinates
        robot_canvas_x, robot_canvas_y = self.cell_to_canvas(robot_row, robot_col)

        # Calculate facing for forward offsets (object placement)
        direction_angles = {"N": 0, "E": 3*math.pi/2, "S": math.pi, "W": math.pi/2}
        display_yaw = direction_angles.get(robot_facing, 0)
        cos_yaw = math.cos(display_yaw)
        sin_yaw = math.sin(display_yaw)

        # Draw the robot as a yellow dot
        marker_radius = max(self.scale(12), 6)
        self.canvas.create_oval(
            robot_canvas_x - marker_radius,
            robot_canvas_y - marker_radius,
            robot_canvas_x + marker_radius,
            robot_canvas_y + marker_radius,
            fill="#fde047",
            outline="",
            width=0,
            tags="robot"
        )

        # Draw carried object ahead of the robot marker if present
        if has_object:
            obj_offset = marker_radius + self.scale(28)
            obj_r = max(3, self.scale(8))
            carried_x = robot_canvas_x - sin_yaw * obj_offset
            carried_y = robot_canvas_y - cos_yaw * obj_offset
            self.canvas.create_oval(
                carried_x - obj_r,
                carried_y - obj_r,
                carried_x + obj_r,
                carried_y + obj_r,
                fill="#d2a8ff",
                outline="#f0f6fc",
                width=max(1, self.scale(2)),
                tags="robot"
            )
            self.canvas.create_text(
                carried_x,
                carried_y + self.scale(20),
                text="OBJ",
                font=self._font(max(8, self.scale(10))),
                fill="#f0f6fc",
                tags="robot"
            )
    
    def draw_gripper(self, robot_x, robot_y, cos_yaw, sin_yaw, robot_length, arm_status, gripper_status):
        """Draw realistic gripper based on arm status"""
        # Scale gripper sizes/offsets
        if arm_status == "stowed":
            # Gripper at robot center when stowed
            gripper_x = robot_x
            gripper_y = robot_y
            gripper_size = self.scale(8)
            arm_length = 0
        elif arm_status == "carry":
            # Gripper at front edge of robot when in carry position
            offset = self.scale(10)
            gripper_x = robot_x + cos_yaw * 0 - sin_yaw * (robot_length/2 - offset)
            gripper_y = robot_y - (sin_yaw * 0 + cos_yaw * (robot_length/2 - offset))
            gripper_size = self.scale(10)
            arm_length = max(0, robot_length/2 - offset)
        else:  # extended, grasping, etc.
            # Gripper extended further from robot body
            offset = self.scale(30)
            gripper_x = robot_x + cos_yaw * 0 - sin_yaw * (robot_length/2 + offset)
            gripper_y = robot_y - (sin_yaw * 0 + cos_yaw * (robot_length/2 + offset))
            gripper_size = self.scale(12)
            arm_length = robot_length/2 + offset
        
        # Draw arm if extended
        if arm_length > 0:
            # Arm extends from robot front edge
            arm_start_x = robot_x + cos_yaw * 0 - sin_yaw * (robot_length/2)
            arm_start_y = robot_y - (sin_yaw * 0 + cos_yaw * (robot_length/2))
            self.canvas.create_line(arm_start_x, arm_start_y, gripper_x, gripper_y, 
                                  fill="#f0f6fc", width=max(1, self.scale(3)), tags="robot")
        
        # Draw gripper based on status
        if gripper_status == "closed":
            # Closed gripper - two parallel lines
            gripper_width = self.scale(8)
            gripper_x1 = gripper_x - gripper_width/2
            gripper_x2 = gripper_x + gripper_width/2
            gripper_y1 = gripper_y - self.scale(2)
            gripper_y2 = gripper_y + self.scale(2)
            self.canvas.create_line(gripper_x1, gripper_y1, gripper_x2, gripper_y1, 
                                  fill="#f85149", width=max(1, self.scale(3)), tags="robot")
            self.canvas.create_line(gripper_x1, gripper_y2, gripper_x2, gripper_y2, 
                                  fill="#f85149", width=max(1, self.scale(3)), tags="robot")
        else:  # open
            # Open gripper - two angled lines
            gripper_width = self.scale(12)
            gripper_x1 = gripper_x - gripper_width/2
            gripper_x2 = gripper_x + gripper_width/2
            gripper_y1 = gripper_y - self.scale(2)
            gripper_y2 = gripper_y + self.scale(2)
            self.canvas.create_line(gripper_x1, gripper_y1, gripper_x2, gripper_y2, 
                                  fill="#f85149", width=max(1, self.scale(3)), tags="robot")
            self.canvas.create_line(gripper_x1, gripper_y2, gripper_x2, gripper_y1, 
                                  fill="#f85149", width=max(1, self.scale(3)), tags="robot")
    
    def _draw_legend(self, grid_start_x, grid_start_y, grid_width, grid_height, cell_size):
        """Draw legend under the grid showing Obstacles and Checkpoints."""
        canvas_width, canvas_height = self.get_canvas_dimensions()
        
        # Position legend at bottom of canvas, below grid
        legend_y = grid_start_y + grid_height + 75
        legend_x = grid_start_x
        item_y = legend_y + 12
        
        # Legend title
        self.canvas.create_text(legend_x, legend_y - 30, text="Legend", 
                              fill="#8b949e", font=self._font(18), anchor='w')
        
        # Obstacles item
        obstacle_size = 30
        obstacle_x = legend_x
        obstacle_y = item_y
        self.canvas.create_rectangle(obstacle_x, obstacle_y - obstacle_size//2, obstacle_x + obstacle_size, obstacle_y + obstacle_size//2,
                                     fill="#c79a63", outline="#a87436", width=1)
        self.canvas.create_text(obstacle_x + obstacle_size + 15, obstacle_y, text="Obstacle", 
                              fill="#f0f6fc", font=self._font(17), anchor='w')
        
        # Checkpoints item
        checkpoint_x = legend_x + 210
        checkpoint_y = item_y
        checkpoint_sz = obstacle_size // 2
        self.canvas.create_rectangle(
            checkpoint_x - checkpoint_sz, checkpoint_y - checkpoint_sz,
            checkpoint_x + checkpoint_sz, checkpoint_y + checkpoint_sz,
            fill="#ffffff", outline="#cccccc", width=1
        )
        self.canvas.create_text(checkpoint_x + checkpoint_sz + 18, checkpoint_y, text="Checkpoint", 
                              fill="#f0f6fc", font=self._font(17), anchor='w')

        # Spot marker
        spot_x = legend_x + 420
        spot_y = item_y
        spot_r = 18
        self.canvas.create_oval(
            spot_x - spot_r, spot_y - spot_r,
            spot_x + spot_r, spot_y + spot_r,
            fill="#fde047", outline="", width=0
        )
        self.canvas.create_text(spot_x + spot_r + 12, spot_y, text="Spot", 
                              fill="#f0f6fc", font=self._font(17), anchor='w')
