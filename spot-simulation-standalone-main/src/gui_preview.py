#!/usr/bin/env python3
"""
GUI Preview Module - Plan Preview & Animation
Handles trajectory preview, animation, and multi-trajectory overlays.
"""

import json
import re
import math
from message_bus import rospy


class GUIPreviewManager:
    """Manages plan preview and animation functionality."""
    
    def __init__(self, canvas_renderer, gui_core=None):
        self.canvas_renderer = canvas_renderer
        self.canvas = canvas_renderer.canvas
        self.gui_core = gui_core
        
        # Plan preview state for animation and path overlay
        self.pending_plan_actions = []
        self.pending_plan_start = None
        self.preview_path_coords = None
        self.preview_waypoints = []
        self._preview_frames = []
        self._preview_animation_running = False
        self._preview_after_id = None
        self._preview_original_state = None
        self._preview_start_state = None
        self._preview_frame_index = 0
        self._preview_frame_delay_ms = 350
        self._preview_cancelled = False
        self._preview_paused = False
        
        # Path visibility flag (for toggle button)
        self._path_hidden = False

        # Aggregated preview overlay state
        self.multi_preview_paths = []
        self.multi_preview_counts = {}
        self.all_preview_visible = False
        self.heatmap_visible = False
    
    def compute_preview_frames(self, actions, start_state):
        """Convert action strings into per-step frames for animation."""
        rospy.loginfo(f"DEBUG: compute_preview_frames called with {len(actions) if actions else 0} actions, start_state: {start_state}")
        frames = []

        current_row = int(round(start_state.get("row", 4)))
        current_col = int(round(start_state.get("col", 4)))
        current_facing = start_state.get("facing", "N")

        frames.append({
            "row": current_row,
            "col": current_col,
            "facing": current_facing,
            "label": "start"
        })

        for action in actions:
            action_text = action.strip()
            lower = action_text.lower()

            if lower.startswith("move_to_cell"):
                match_row = re.search(r"row\s*=\s*(-?\d+)", action_text, re.IGNORECASE)
                match_col = re.search(r"col\s*=\s*(-?\d+)", action_text, re.IGNORECASE)
                
                if match_row and match_col:
                    target_row = int(match_row.group(1))
                    target_col = int(match_col.group(1))
                    
                    # Move in steps; handle float deltas from non-integer current pose
                    start_row = current_row
                    start_col = current_col
                    d_row = abs(target_row - start_row)
                    d_col = abs(target_col - start_col)
                    steps = max(d_row, d_col)
                    if steps > 0:
                        row_step = 0 if d_row == 0 else (1 if target_row > start_row else -1)
                        col_step = 0 if d_col == 0 else (1 if target_col > start_col else -1)
                        next_row = start_row
                        next_col = start_col
                        for step_index in range(1, steps + 1):
                            if step_index <= d_row:
                                next_row += row_step
                            if step_index <= d_col:
                                next_col += col_step
                            
                            frames.append({
                                "row": next_row,
                                "col": next_col,
                                "facing": current_facing,
                                "label": f"move_{step_index}"
                            })
                    
                    current_row, current_col = target_row, target_col

            elif lower.startswith("turn_to_face") or lower.startswith("turn to face"):
                direction_match = re.search(r"face\s+([NESW])", action_text, re.IGNORECASE)
                if direction_match:
                    new_facing = direction_match.group(1).upper()
                    frames.append({
                        "row": current_row,
                        "col": current_col,
                        "facing": new_facing,
                        "label": "turn"
                    })
                    current_facing = new_facing

            elif lower.startswith("rotate_to"):
                direction_match = re.search(r"direction\s*=\s*([a-z]+)", action_text, re.IGNORECASE)
                if not direction_match:
                    direction_match = re.search(r"rotate_to\s+([a-z]+)", action_text, re.IGNORECASE)

                if direction_match:
                    token = direction_match.group(1).strip().upper()
                    token_map = {
                        "NORTH": "N",
                        "EAST": "E",
                        "SOUTH": "S",
                        "WEST": "W",
                    }
                    if token not in token_map and len(token) > 1:
                        token = token[0]
                    new_facing = token_map.get(token, token)
                    if new_facing in {"N", "E", "S", "W"}:
                        frames.append({
                            "row": current_row,
                            "col": current_col,
                            "facing": new_facing,
                            "label": "rotate"
                        })
                        current_facing = new_facing

            elif lower.startswith("grasp_object"):
                frames.append({
                    "row": current_row,
                    "col": current_col,
                    "facing": current_facing,
                    "label": "grasp"
                })

            elif lower.startswith("place_object"):
                frames.append({
                    "row": current_row,
                    "col": current_col,
                    "facing": current_facing,
                    "label": "place"
                })

        rospy.loginfo(f"DEBUG: compute_preview_frames returning {len(frames)} frames")
        return frames
    
    def build_preview_path(self, frames):
        """Build grid-based path coordinates from frames."""
        if not frames:
            return []
        
        path_coords = []
        for frame in frames:
            row = float(frame.get("row", 0))
            col = float(frame.get("col", 0))
            path_coords.append((row, col))
        
        return path_coords

    def extract_waypoints(self, actions):
        """Extract ordered waypoint targets from move_to_cell actions."""
        if not actions:
            return []

        waypoints = []
        next_number = 1
        for action in actions:
            if not isinstance(action, str):
                continue

            match = re.search(
                r"move(?:_|\s*)to(?:_|\s*)cell\s*row\s*=?\s*(-?\d+)\s*col\s*=?\s*(-?\d+)",
                action,
                re.IGNORECASE,
            )
            if not match:
                continue

            try:
                row = int(match.group(1))
                col = int(match.group(2))
            except (TypeError, ValueError):
                continue

            waypoints.append({"row": row, "col": col, "number": next_number})
            next_number += 1

        return waypoints
    
    def heatmap_color(self, count, max_count):
        """Generate heatmap color based on count."""
        if max_count == 0:
            return "#30363d"
        
        intensity = min(count / max_count, 1.0)
        
        # Color gradient from blue (low) to red (high)
        if intensity < 0.5:
            # Blue to yellow
            r = int(255 * intensity * 2)
            g = int(255 * intensity * 2)
            b = 255
        else:
            # Yellow to red
            r = 255
            g = int(255 * (2 - intensity * 2))
            b = 0
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def start_preview_animation(self):
        """Start the preview animation."""
        rospy.loginfo(f"DEBUG: start_preview_animation called - frames: {len(self._preview_frames) if self._preview_frames else 0}, running: {self._preview_animation_running}")
        if not self._preview_frames or self._preview_animation_running:
            rospy.loginfo(f"DEBUG: Preview animation not started - no frames or already running")
            return

        self._preview_animation_running = True
        self._preview_cancelled = False
        self._preview_frame_index = 0

        # Store original robot state
        if self.gui_core:
            self._preview_original_state = {
                "row": float(self.gui_core.robot_row),
                "col": float(self.gui_core.robot_col),
                "facing": self.gui_core.robot_facing,
                "action": self.gui_core.current_action
            }
        else:
            raise Exception("GUI Core is not available for preview animation.")

        # Capture the intended preview start state for restoration fallback
        if self._preview_frames:
            first_frame = self._preview_frames[0]
            self._preview_start_state = {
                "row": float(first_frame.get("row", self._preview_original_state.get("row", 4))),
                "col": float(first_frame.get("col", self._preview_original_state.get("col", 4))),
                "facing": first_frame.get("facing", self._preview_original_state.get("facing", "N"))
            }
        else:
            self._preview_start_state = None

        rospy.loginfo("Starting preview animation")
        self._run_preview_step()

    def _run_preview_step(self):
        """Run a single step of the preview animation."""
        if self._preview_cancelled or not self._preview_animation_running:
            return
        
        if self._preview_frame_index >= len(self._preview_frames):
            self.end_preview_animation()
            return
        
        # Get current frame
        frame = self._preview_frames[self._preview_frame_index]
        
        # Update robot position for preview
        if self.gui_core:
            with self.gui_core.update_lock:
                self.gui_core.robot_row = frame["row"]
                self.gui_core.robot_col = frame["col"]
                self.gui_core.robot_facing = frame["facing"]
                label = frame.get("label")
                if label:
                    self.gui_core.current_action = f"preview: {label}"
                else:
                    self.gui_core.current_action = "preview"
            
            # Update display to show new robot position
            self.gui_core.update_display()
        
        # Draw preview path up to current position
        if self._preview_frame_index > 0:
            self._draw_preview_path_up_to_frame(self._preview_frame_index)
        
        self._preview_frame_index += 1

        # Schedule next frame (unless paused)
        if not self._preview_paused:
            self._preview_after_id = self.canvas.after(self._preview_frame_delay_ms, self._run_preview_step)

    def _draw_preview_path_up_to_frame(self, frame_index):
        """Draw preview path up to the specified frame."""
        if frame_index <= 0:
            return
        
        # Clear previous preview path
        self.canvas.delete("preview_path")
        
        # Draw path up to current frame
        for i in range(min(frame_index, len(self._preview_frames))):
            frame = self._preview_frames[i]
            x, y = self.canvas_renderer.cell_to_canvas(frame["row"], frame["col"])
            
            if i == 0:
                # Start point
                self.canvas.create_oval(x - 5, y - 5, x + 5, y + 5, 
                                      fill="#39d353", outline="#f0f6fc", width=2, tags="preview_path")
            else:
                # Path line
                prev_frame = self._preview_frames[i - 1]
                prev_x, prev_y = self.canvas_renderer.cell_to_canvas(prev_frame["row"], prev_frame["col"])
                self.canvas.create_line(prev_x, prev_y, x, y, 
                                      fill="#58a6ff", width=3, tags="preview_path")

                # Current position
                self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, 
                                      fill="#58a6ff", outline="#f0f6fc", width=1, tags="preview_path")

    def _restore_robot_state(self):
        """Restore the robot to its state before preview playback."""
        if not self.gui_core:
            return

        target_state = None
        if self._preview_original_state:
            target_state = self._preview_original_state
        elif self._preview_start_state:
            target_state = self._preview_start_state

        if not target_state:
            return

        with self.gui_core.update_lock:
            if "row" in target_state:
                self.gui_core.robot_row = float(target_state["row"])
            if "col" in target_state:
                self.gui_core.robot_col = float(target_state["col"])
            if "facing" in target_state:
                self.gui_core.robot_facing = target_state["facing"]
            # Reset to idle (or previously stored action if available)
            restored_action = target_state.get("action") if isinstance(target_state, dict) else None
            self.gui_core.current_action = restored_action or "idle"

        self.gui_core.update_display()

    def end_preview_animation(self):
        """End the preview animation."""
        self._preview_animation_running = False
        self._preview_cancelled = False
        
        if self._preview_after_id:
            self.canvas.after_cancel(self._preview_after_id)
            self._preview_after_id = None
        
        # Clear preview path
        self.canvas.delete("preview_path")

        # Restore original robot state
        self._restore_robot_state()

        self._preview_start_state = None
        self._preview_original_state = None
        rospy.loginfo("Preview animation ended")

    def cancel_preview_animation(self):
        """Cancel the preview animation."""
        self._preview_cancelled = True
        self._preview_animation_running = False

        if self._preview_after_id:
            self.canvas.after_cancel(self._preview_after_id)
            self._preview_after_id = None

        # Clear preview path
        self.canvas.delete("preview_path")

        # Restore robot to the beginning of the preview if it has already moved
        self._restore_robot_state()
        self._preview_start_state = None
        self._preview_original_state = None

        rospy.loginfo("Preview animation cancelled")
    
    def pause_preview_animation(self):
        """Pause the preview animation."""
        if self._preview_animation_running and not self._preview_paused:
            self._preview_paused = True
            if self._preview_after_id:
                self.canvas.after_cancel(self._preview_after_id)
                self._preview_after_id = None
            rospy.loginfo("Preview animation paused")
    
    def resume_preview_animation(self):
        """Resume the preview animation."""
        if self._preview_animation_running and self._preview_paused:
            self._preview_paused = False
            self._run_preview_step()
            rospy.loginfo("Preview animation resumed")
    
    def step_preview_animation(self):
        """Step forward one frame in the preview animation."""
        if self._preview_frames and self._preview_frame_index < len(self._preview_frames):
            # If not running, start in paused mode
            if not self._preview_animation_running:
                self._preview_animation_running = True
                self._preview_paused = True
                self._preview_cancelled = False
                if self._preview_frame_index == 0:
                    # Save original state for restore
                    if self.gui_core and hasattr(self.gui_core, 'robot_row'):
                        self._preview_original_state = {
                            "row": self.gui_core.robot_row,
                            "col": self.gui_core.robot_col,
                            "facing": self.gui_core.robot_facing
                        }
            
            # Execute one step
            if self._preview_paused or not self._preview_animation_running:
                self._run_preview_step()
                rospy.loginfo(f"Stepped to frame {self._preview_frame_index}/{len(self._preview_frames)}")
    
    def clear_plan_preview_state(self, clear_path=False):
        """Clear plan preview state."""
        self.pending_plan_actions = []
        self.pending_plan_start = None
        self._preview_frames = []
        self.preview_waypoints = []
        
        if clear_path:
            self.preview_path_coords = None
            self.canvas.delete("preview_path")
            if self.gui_core and hasattr(self.gui_core, "preview_path_coords"):
                try:
                    self.gui_core.preview_path_coords = None
                except Exception:
                    pass
        if self.gui_core and hasattr(self.gui_core, "preview_waypoints"):
            try:
                self.gui_core.preview_waypoints = []
            except Exception:
                pass
        
        self.cancel_preview_animation()
    
    def setup_multi_preview_overlay(self, combined_paths):
        """Setup multi-trajectory overlay display."""
        self.multi_preview_paths = combined_paths
        self.multi_preview_counts = {}
        self.all_preview_visible = True
        
        # Count occurrences of each coordinate
        for path in combined_paths:
            for coord in path:
                self.multi_preview_counts[coord] = self.multi_preview_counts.get(coord, 0) + 1
        
        # Draw overlay
        self._draw_multi_preview_overlay()
    
    def _draw_multi_preview_overlay(self):
        """Draw the multi-trajectory overlay."""
        self.canvas.delete("multi_preview")
        
        if not self.multi_preview_paths:
            return
        
        max_count = max(self.multi_preview_counts.values()) if self.multi_preview_counts else 1
        
        for path in self.multi_preview_paths:
            for i, coord in enumerate(path):
                x, y = coord
                count = self.multi_preview_counts.get(coord, 1)
                color = self.heatmap_color(count, max_count)
                
                if i == 0:
                    # Start point
                    self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, 
                                          fill=color, outline="#f0f6fc", width=2, tags="multi_preview")
                else:
                    # Path line
                    prev_coord = path[i - 1]
                    prev_x, prev_y = prev_coord
                    self.canvas.create_line(prev_x, prev_y, x, y, 
                                          fill=color, width=2, tags="multi_preview")
                    
                    # Current position
                    self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, 
                                          fill=color, outline="#f0f6fc", width=1, tags="multi_preview")
    
    def clear_all_preview_overlay(self):
        """Remove multi-trajectory overlay from the canvas."""
        if not self.all_preview_visible:
            self.multi_preview_counts = {}
            self.heatmap_visible = False
            return

        self.multi_preview_paths = []
        self.multi_preview_counts = {}
        self.all_preview_visible = False
        self.heatmap_visible = False

        self.canvas.delete("multi_preview")
