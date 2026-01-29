#!/usr/bin/env python3
"""
GUI Handlers Module - Event Handlers & ROS
Handles all event callbacks, ROS message processing, and user interactions.
"""

import json
import re
import time
import threading
import tkinter as tk
from pathlib import Path
from message_bus import rospy, String, Empty, Publisher
from plan_scorer import PlanScorer


class GUIHandlers:
    """Manages all event handlers and ROS message processing."""
    
    def __init__(self, gui_core, widgets, world_manager_component, preview_manager):
        self.gui_core = gui_core
        self.widgets = widgets
        self.world_manager_component = world_manager_component
        self.preview_manager = preview_manager
        self.model_id_map = {
            'Gemini Flash 2.0 (lite)': 'gemini-2.0-flash-lite',
            'Gemini Pro 2.5 (reasoning)': 'gemini-2.5-pro',
        }
        
        # Batch input planning state
        self.loaded_inputs = []
        self.input_plan_cache = {}
        self.batch_preview_queue = []
        self.batch_preview_inflight = None
        self.batch_preview_limit = 30
        self.active_batch_requests: set[str] = set()
        self._score_log_dir = Path(__file__).resolve().parents[1] / 'user_data'
        self._score_log_lock = threading.Lock()
        
        # Initialize tree state flag
        self.inputs_tree_disabled = False
        
        # Initialize plan scorer
        self.plan_scorer = PlanScorer(
            world_manager=world_manager_component.world_manager
        )
        if getattr(self.widgets, "strict_scoring_var", None) is not None:
            self.plan_scorer.enable_additional_rules = bool(self.widgets.strict_scoring_var.get())
        
        self.pub_wall_enforcement = Publisher('/nl_control/wall_enforcement')
        self.wall_enforcement_enabled = True
        if getattr(self.widgets, "wall_enforcement_var", None) is not None:
            initial_enabled = bool(self.widgets.wall_enforcement_var.get())
            self.wall_enforcement_enabled = initial_enabled
            try:
                self.pub_wall_enforcement.publish(String(data="true" if initial_enabled else "false"))
            except Exception as exc:
                rospy.logwarn(f"GUI: Failed to publish initial wall enforcement state: {exc}")
        
        # Detect models that allow parallel batch processing
        self._parallel_model_prefixes = ("gemini-",)
        
        # Baseline mode state
        self.current_baseline_data = None
    
    def _get_checkpoints_for_world(self, world_id):
        """Extract checkpoint coordinates from world configuration.
        
        Returns a list of (row, col) tuples representing checkpoints.
        Checkpoints are returned in alphabetical order by name for consistent scoring.
        """
        config = self.world_manager_component.world_manager.get_world_config(world_id)
        if not config:
            return []
        
        checkpoints = []
        checkpoint_data = config.get("checkpoints", {})
        
        # Sort checkpoints by name for consistent ordering (important for strict scoring)
        for checkpoint_name in sorted(checkpoint_data.keys()):
            checkpoint_info = checkpoint_data[checkpoint_name]
            row = checkpoint_info.get("row")
            col = checkpoint_info.get("col")
            if row is not None and col is not None:
                checkpoints.append((int(row), int(col)))
        
        # If no checkpoints defined, use waypoints and zones as implicit checkpoints
        if not checkpoints:
            waypoints = config.get("waypoints", {})
            for wp_name, wp_data in waypoints.items():
                row = wp_data.get("row")
                col = wp_data.get("col")
                if row is not None and col is not None:
                    checkpoints.append((int(row), int(col)))
            
            zones = config.get("zones", {})
            for zone_name, zone_data in zones.items():
                row = zone_data.get("row")
                col = zone_data.get("col")
                if row is not None and col is not None:
                    checkpoints.append((int(row), int(col)))
        
        return checkpoints
    
    def _current_model_id(self):
        """Return the internal model identifier for the selected LLM."""
        selection = self.widgets.model_combo.get()
        return self.model_id_map.get(selection, 'base-120b-low')
    
    def _is_parallel_batch_supported(self):
        """Check if current model allows parallel batch processing."""
        model_id = self._current_model_id()
        return any(model_id.startswith(prefix) for prefix in self._parallel_model_prefixes)
    
    def _run_on_ui_thread(self, callback, *args, **kwargs):
        """Ensure callback executes on the Tkinter main thread."""
        try:
            self.widgets.root.after(0, lambda: callback(*args, **kwargs))
        except RuntimeError:
            callback(*args, **kwargs)

    def _set_current_prompt_step(self, step):
        """Update user_data/current_prompt_step.json when batch inputs are loaded."""
        step_path = self._score_log_dir / 'current_prompt_step.json'
        try:
            if step is None:
                if step_path.exists():
                    step_path.unlink()
                return
            payload = {'step': int(step)}
            step_path.write_text(json.dumps(payload), encoding='utf-8')
        except Exception as exc:
            rospy.logwarn(f"GUI: Failed to update current_prompt_step.json: {exc}")

    def _prepare_scored_inputs_log(self, step):
        """Point scoring to the right step file and reset it before new batch scoring."""
        self._set_current_prompt_step(step)
        if step is None:
            return
        log_file = self._score_log_dir / f'scored_inputs{step}.jsonl'
        try:
            if log_file.exists():
                log_file.unlink()
        except Exception as exc:
            rospy.logwarn(f"GUI: Failed to reset scored_inputs{step}.jsonl: {exc}")
    
    def setup_ros_handlers(self):
        """Setup ROS message handlers."""
        # Model selection handler
        def on_model_change(event):
            selection = self.widgets.model_combo.get()
            model_id = self.model_id_map.get(selection, 'base-120b-low')

            import os
            gemini_available = bool(os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY'))
            
            if gemini_available or model_id.startswith('base-120b'):
                from message_bus import Publisher
                pub_model_select = Publisher('/nl_control/model_select')
                pub_model_select.publish(String(data=model_id))
                rospy.loginfo(f"Model selection published: {model_id}")
            else:
                self.widgets.model_combo.set('120B Low Reasoning')
                rospy.logwarn("GEMINI_API_KEY not set - cannot select Gemini models")
                model_id = self.model_id_map.get('120B Low Reasoning', 'base-120b-low')

            # If switching to a model that supports parallel batch inference, dispatch queued requests
            if self._is_parallel_batch_supported() and self.batch_preview_queue and not self.batch_preview_inflight:
                self._start_batch_processing()

        if hasattr(self.widgets, 'model_combo') and self.widgets.model_combo:
            self.widgets.model_combo.bind('<<ComboboxSelected>>', on_model_change)
        # Immediately publish the current model selection so NL control is in sync
        try:
            on_model_change(None)
        except Exception:
            pass

        # Prompt selection handler
        def on_prompt_change(event=None):
            selection = self.widgets.prompt_combo.get()
            prompt_name = selection.lower()

            if hasattr(self.widgets, '_available_prompts') and self.widgets._available_prompts:
                for key in self.widgets._available_prompts.keys():
                    if key.lower() == prompt_name:
                        prompt_name = key
                        break

            from message_bus import Publisher
            pub_prompt_select = Publisher('/nl_control/prompt_select')
            pub_prompt_select.publish(String(data=prompt_name))
            rospy.loginfo(f"Prompt selection published: {prompt_name}")

        if hasattr(self.widgets, 'prompt_combo') and self.widgets.prompt_combo:
            self.widgets.prompt_combo.bind('<<ComboboxSelected>>', on_prompt_change)
        on_prompt_change()
        
        # World change handler
        def on_world_change(event=None):
            selected_text = self.widgets.world_var.get()
            if selected_text:
                # Extract world_id from the selected text (format: "1. World Name")
                world_id = selected_text.split('.')[0].strip()
                if world_id != self.gui_core.current_world_id:
                    self.load_world(world_id)

        if hasattr(self.widgets, 'world_combo') and self.widgets.world_combo:
            self.widgets.world_combo.bind('<<ComboboxSelected>>', on_world_change)
        
        # Mode change handler (removed in new layout)
        if hasattr(self.widgets, 'mode_combo') and self.widgets.mode_combo:
            self.widgets.mode_combo.bind('<<ComboboxSelected>>', self._on_mode_change)
        
        # Input selection handler (removed in new layout)
        if hasattr(self.widgets, 'inputs_tree') and self.widgets.inputs_tree:
            self.widgets.inputs_tree.bind('<<TreeviewSelect>>', self._on_input_selection_change)
            self.widgets.inputs_tree.bind('<Double-1>', self._on_input_double_click)
    
    def on_robot_state(self, msg):
        """Handle robot state updates."""
        try:
            fields = dict(part.split(':', 1) for part in msg.data.split(','))
            connected = fields.get('connected', 'false').lower() == 'true'
            powered = fields.get('powered', 'false').lower() == 'true'
            standing = fields.get('standing', 'false').lower() == 'true'
            
            if not connected:
                self.gui_core.robot_state = "disconnected"
            elif not powered:
                self.gui_core.robot_state = "powered_off"
            elif standing:
                self.gui_core.robot_state = "stand"
            else:
                self.gui_core.robot_state = "sit"
            
            # Update display on main thread (only if GUI is ready)
            def update_gui():
                try:
                    if hasattr(self.widgets, 'state_label') and self.widgets.state_label:
                        self.widgets.state_label.config(text=f"State: {self.gui_core.robot_state}")
                except Exception as e:
                    rospy.logwarn(f"Failed to update GUI from robot state: {e}")
            
            try:
                self.widgets.root.after_idle(update_gui)
            except RuntimeError:
                # GUI not ready yet, skip update
                pass
            
        except Exception as e:
            rospy.logwarn(f"Failed to parse robot state: {e}")
    
    def on_execution_feedback(self, msg):
        """Handle execution feedback updates."""
        feedback = msg.data
        rospy.loginfo(f"Execution: {feedback}")
        
        # Handle specific feedback messages
        if "start_moving" in feedback:
            self.gui_core.current_action = "moving"
        elif "stop_moving" in feedback:
            self.gui_core.current_action = "idle"
        elif "grasp_object" in feedback:
            self.gui_core.current_action = "grasping"
            self.gui_core.arm_status = "deployed"
            self.gui_core.gripper_status = "open"
        elif "place_object" in feedback:
            self.gui_core.current_action = "placing"
        elif "arm_stowed" in feedback:
            self.gui_core.arm_status = "stowed"
        elif "gripper_closed" in feedback:
            self.gui_core.gripper_status = "closed"
        elif "gripper_open" in feedback:
            self.gui_core.gripper_status = "open"
        
        # Update GUI on main thread (only if GUI is ready)
        def update_gui():
            try:
                # Update action label
                if hasattr(self.widgets, 'action_label') and self.widgets.action_label:
                    self.widgets.action_label.config(text=f"Action: {feedback}")
                
                # Update arm label
                if hasattr(self.widgets, 'arm_label') and self.widgets.arm_label:
                    arm_text = f"Arm: {self.gui_core.arm_status} | Gripper: {self.gui_core.gripper_status}"
                    if self.gui_core.has_object:
                        arm_text += " | Carrying object"
                    self.widgets.arm_label.config(text=arm_text)
            except Exception as e:
                rospy.logwarn(f"Failed to update GUI from execution feedback: {e}")
        
        try:
            self.widgets.root.after_idle(update_gui)
        except RuntimeError:
            # GUI not ready yet, skip update
            pass
        
        # Handle plan completion
        if "Plan completed successfully" in feedback:
            self._update_feedback_text("Plan execution completed successfully!")
            self._update_feedback_text("Ready for new commands")
            self._update_feedback_text("Input selection disabled - load new inputs to continue")
            
            # Disable preview buttons after execution completes
            self._disable_preview_buttons_after_execution()
            
            # Clear the trajectory visualization and preview state
            self.preview_manager.clear_plan_preview_state(clear_path=True)
            
            # Schedule a display update so the cleared preview path disappears using current robot state
            with self.gui_core.update_lock:
                self.gui_core.position_changed = True
    
    def on_nl_position_update(self, msg):
        """Handle position updates from NL control."""
        try:
            # If a preview animation is playing, ignore live pose updates so the
            # animated trajectory does not get overridden by background pose sync.
            try:
                if getattr(self.preview_manager, '_preview_animation_running', False):
                    return
            except Exception:
                pass
            # Parse position data
            position_data = json.loads(msg.data)
            
            # Check if this is from robot simulator (grid-based) or NL control (world coordinates)
            if 'row' in position_data and 'col' in position_data:
                # Robot simulator format: grid coordinates with cardinal direction
                # Convert to float to maintain precision
                self.gui_core.robot_row = float(position_data.get('row', 4))
                self.gui_core.robot_col = float(position_data.get('col', 4))
                self.gui_core.robot_facing = position_data.get('facing', 'N')
                
                # Convert grid cell to meter coordinates (center of cell)
                # Cell size is 0.57m (57cm) as defined in pose_tracker.py
                # Cell (4,4) should be at (0.57 * 4.5, 0.57 * 4.5) = (2.565, 2.565)
                robot_x = (self.gui_core.robot_col + 0.5) * 0.57
                robot_y = (self.gui_core.robot_row + 0.5) * 0.57
                
                # Handle yaw if available
                if 'yaw_deg' in position_data:
                    import math
                    robot_yaw = float(position_data['yaw_deg']) * math.pi / 180.0
                else:
                    # Cardinal mapping per system convention: N=0, E=-pi/2, W=+pi/2, S=pi
                    direction_to_yaw = {'N': 0.0, 'E': -1.57, 'S': 3.14, 'W': 1.57}
                    robot_yaw = direction_to_yaw.get(self.gui_core.robot_facing, 0.0)
                
                # Update position label with meter coordinates on main thread
                def update_gui():
                    try:
                        if hasattr(self.widgets, 'position_label') and self.widgets.position_label:
                            # Show meter coordinates with float precision
                            yaw_deg = robot_yaw * 180.0 / math.pi
                            self.widgets.position_label.config(text=f"Position: ({robot_x:.2f}, {robot_y:.2f}, {yaw_deg:.1f}°)")
                    except Exception as e:
                        rospy.logwarn(f"Failed to update GUI from position: {e}")
                
                try:
                    self.widgets.root.after_idle(update_gui)
                except RuntimeError:
                    # GUI not ready yet, skip update
                    pass
            else:
                # NL control format: world coordinates with yaw angle
                x = position_data.get('x', 0.0)
                y = position_data.get('y', 0.0)
                yaw = position_data.get('yaw', 0.0)
                
                # Convert to grid coordinates
                self.gui_core.robot_row = int(round(y + 4.5))  # Convert from world coordinates
                self.gui_core.robot_col = int(round(x + 4.5))
                
                # Convert yaw to direction
                if -45 <= yaw <= 45:
                    self.gui_core.robot_facing = "E"
                elif 45 < yaw <= 135:
                    self.gui_core.robot_facing = "N"
                elif 135 < yaw <= 225:
                    self.gui_core.robot_facing = "W"
                else:
                    self.gui_core.robot_facing = "S"
                
                # Update position label with grid coordinates on main thread (like old GUI)
                def update_gui():
                    try:
                        if hasattr(self.widgets, 'position_label') and self.widgets.position_label:
                            self.widgets.position_label.config(text=f"Position: ({self.gui_core.robot_col}, {self.gui_core.robot_row}, {self.gui_core.robot_facing})")
                    except Exception as e:
                        rospy.logwarn(f"Failed to update GUI from position: {e}")
                
                try:
                    self.widgets.root.after_idle(update_gui)
                except RuntimeError:
                    # GUI not ready yet, skip update
                    pass
            
            # Mark position as changed for display update
            with self.gui_core.update_lock:
                self.gui_core.position_changed = True
                
        except Exception as e:
            rospy.logwarn(f"Failed to parse position update: {e}")
    
    def _on_interpretation(self, msg):
        """Handle natural language interpretation."""
        try:
            interpretation = msg.data.strip()
        except Exception as e:
            rospy.logwarn(f"Failed to parse interpretation: {e}")
            return

        if not interpretation:
            return

        rospy.loginfo(f"GUI: Received interpretation: {interpretation}")
        self._run_on_ui_thread(self._handle_interpretation_update, interpretation)

    def _handle_interpretation_update(self, interpretation):
        """Update GUI elements when a new interpretation arrives."""
        self._update_feedback_text("Plan generated:")
        self._update_feedback_text(interpretation)

        approve_btn, dismiss_btn = self._get_current_approve_dismiss_buttons()
        if approve_btn:
            approve_btn.config(state='normal')
        if dismiss_btn:
            dismiss_btn.config(state='normal')
    
    def _on_plan_preview(self, msg):
        """Handle plan preview updates."""
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"Failed to parse plan preview payload: {e}")
            return

        self._run_on_ui_thread(self._handle_plan_preview_payload, payload)

    def _handle_plan_preview_payload(self, payload):
        """Process plan preview payloads on the GUI thread."""
        actions = payload.get("actions") or []
        start_cell = payload.get("start_cell")
        start_facing = (payload.get("start_facing") or self.gui_core.robot_facing or "N")
        error = payload.get("error")

        # Check for errors first
        if error:
            rospy.logwarn(f"GUI: Plan generation failed with error: {error}")
            self.preview_manager.clear_plan_preview_state(clear_path=True)
            # Show error in plan status indicator (only for single command mode)
            if hasattr(self.widgets, 'control_mode') and self.widgets.control_mode.get() == "Single Command":
                if hasattr(self.widgets, 'update_plan_status'):
                    self.widgets.update_plan_status(status_text=error, show=True, is_error=True)
            return

        if not actions:
            rospy.loginfo("GUI: Clearing plan preview state")
            self.preview_manager.clear_plan_preview_state(clear_path=True)
            # Hide plan status indicator
            if hasattr(self.widgets, 'update_plan_status'):
                self.widgets.update_plan_status(show=False)
            return
        
        start_cell = [self.gui_core.robot_row, self.gui_core.robot_col]

        self.preview_manager.cancel_preview_animation()
        
        # Reset path visibility when a new plan is loaded
        self.preview_manager._path_hidden = False
        # Update toggle button text if it exists
        if hasattr(self.widgets, 'toggle_path_button') and self.widgets.toggle_path_button:
            self.widgets.toggle_path_button.config(text="Hide Path")
            self.widgets.path_visible = True

        with self.gui_core.update_lock:
            self.preview_manager.pending_plan_actions = actions
            self.preview_manager.pending_plan_start = {
                "row": round(start_cell[0]),
                "col": round(start_cell[1]),
                "facing": str(start_facing).upper()
            }
            frames = self.preview_manager.compute_preview_frames(actions, self.preview_manager.pending_plan_start)
            rospy.loginfo(f"DEBUG: _on_plan_preview computed {len(frames) if frames else 0} frames")
            self.preview_manager._preview_frames = frames
            self.preview_manager.preview_path_coords = self.preview_manager.build_preview_path(frames)
            self.preview_manager.preview_waypoints = self.preview_manager.extract_waypoints(actions)
            
            self.gui_core.pending_plan_actions = actions
            self.gui_core.pending_plan_start = {
                "row": round(start_cell[0]),
                "col": round(start_cell[1]),
                "facing": str(start_facing).upper()
            }
            self.gui_core._preview_frames = frames
            self.gui_core.preview_path_coords = self.preview_manager.preview_path_coords
            self.gui_core.preview_waypoints = self.preview_manager.preview_waypoints

        if not self.preview_manager._preview_frames:
            rospy.logwarn("GUI: Plan preview contains no frames, clearing state")
            self.preview_manager.clear_plan_preview_state(clear_path=True)
            # Hide plan status indicator
            if hasattr(self.widgets, 'update_plan_status'):
                self.widgets.update_plan_status(show=False)
            return

        rospy.loginfo("GUI: Plan preview ready")
        self._update_feedback_text("Preview ready — click Preview to watch the planned path, or Approve to execute")
        
        # Update plan status indicator (only for single command mode)
        if hasattr(self.widgets, 'control_mode') and self.widgets.control_mode.get() == "Single Command":
            if hasattr(self.widgets, 'update_plan_status'):
                num_steps = len(actions) if actions else 0
                status_text = f"Plan generated — {num_steps} step{'s' if num_steps != 1 else ''} ready for preview"
                self.widgets.update_plan_status(status_text, show=True)
        
        if hasattr(self.widgets, 'single_preview_trajectory_button') and self.widgets.single_preview_trajectory_button:
            self.widgets.single_preview_trajectory_button.config(state='normal')
        
        plan_payload = {
            "plan_id": 0,
            "actions": actions,
            "cached_plan": False
        }
        try:
            self.gui_core.pub_plan.publish(String(data=json.dumps(plan_payload)))
        except Exception as e:
            rospy.logwarn(f"Failed to publish plan for approval setup: {e}")
        
        approve_btn, dismiss_btn = self._get_current_approve_dismiss_buttons()
        if approve_btn:
            approve_btn.config(state='normal')
        if dismiss_btn:
            dismiss_btn.config(state='normal')
        
        self.gui_core.update_display()
    
    def _on_batch_plan_preview(self, msg):
        """Handle batch plan preview updates."""
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"Failed to parse batch plan preview payload: {e}")
            return
        self._run_on_ui_thread(self._handle_batch_plan_preview_payload, payload)
    
    def _handle_batch_plan_preview_payload(self, payload):
        request_id = payload.get("request_id")
        actions = payload.get("actions", [])
        start_cell = payload.get("start_cell", [4, 4])
        start_facing = payload.get("start_facing", "N")
        error = payload.get("error")

        if request_id not in self.input_plan_cache:
            return

        cache = self.input_plan_cache[request_id]
        
        if error:
            cache["status"] = f"Failed: {error}"
            cache["score"] = "N/A"
            cache["error"] = str(error)
            rospy.logwarn(f"GUI: Batch planning failed for {request_id}: {error}")
            command_index = cache.get("index", 0) + 1
            total_commands = len(self.loaded_inputs)
            self._update_feedback_text(f"Command {command_index}/{total_commands} failed: {error}")
            self._record_scored_plan(request_id, cache)
        else:
            cache["actions"] = actions
            cache["start"] = {"row": start_cell[0], "col": start_cell[1], "facing": start_facing}
            cache["frames"] = self.preview_manager.compute_preview_frames(actions, cache["start"])
            cache["path"] = self.preview_manager.build_preview_path(cache["frames"])
            cache["waypoints"] = self.preview_manager.extract_waypoints(actions)
            cache["status"] = "Preview ready"
            
            try:
                checkpoints = self._get_checkpoints_for_world(self.gui_core.current_world_id)
                if checkpoints and actions:
                    log_dir = Path(__file__).resolve().parents[1] / 'logs' / 'batch'
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log_path = log_dir / f"{request_id}_score.log"
                    initial_pose = (start_cell[0], start_cell[1])
                    score_result = self.plan_scorer.score_plan(
                        actions=actions,
                        world_id=self.gui_core.current_world_id,
                        checkpoints=checkpoints,
                        initial_pose=initial_pose,
                        log_path=log_path
                    )
                    cache["score"] = score_result.total_score
                    cache["score_result"] = score_result
                    rospy.loginfo(f"GUI: Plan scored for {request_id} - Score: {score_result.total_score}")
                    self._record_scored_plan(request_id, cache)
                else:
                    cache["score"] = "N/A"
                    rospy.loginfo(f"GUI: No checkpoints found for world {self.gui_core.current_world_id}, skipping scoring")
            except Exception as score_error:
                rospy.logwarn(f"GUI: Failed to score plan for {request_id}: {score_error}")
                cache["score"] = "Error"
                cache["error"] = str(score_error)
                self._record_scored_plan(request_id, cache)
            
            rospy.loginfo(f"GUI: Batch plan preview received for {request_id}")
            command_index = cache.get("index", 0) + 1
            total_commands = len(self.loaded_inputs)
            score_str = f" (Score: {cache.get('score', 'N/A')})" if cache.get('score') not in [None, "N/A"] else ""
            self._update_feedback_text(f"Command {command_index}/{total_commands} completed: '{cache.get('command', '')}'{score_str}")

        self._refresh_input_list()

        if self.batch_preview_inflight == request_id:
            self.batch_preview_inflight = None
            if not self._is_parallel_batch_supported():
                self.widgets.root.after(120, self._process_next_batch_preview)
            elif self.batch_preview_queue:
                self._start_batch_processing()

        self._mark_batch_request_completed(request_id)
    
    def _record_scored_plan(self, request_id, cache):
        """Persist the generated plan and score to a JSONL log."""
        if not cache or cache.get("score_logged"):
            return

        score = cache.get("score")
        actions = cache.get("actions") or []
        error = cache.get("error")

        model_id = self._current_model_id()
        prompt_template = None
        if hasattr(self.widgets, "prompt_combo"):
            try:
                prompt_template = self.widgets.prompt_combo.get()
            except Exception:
                prompt_template = None

        score_result = cache.get("score_result")
        breakdown = []
        achieved_checkpoints = []
        missed_checkpoints = []
        final_pose = None

        if score_result:
            breakdown = [
                {
                    "index": action_score.index,
                    "action": action_score.action,
                    "score_delta": action_score.score_delta,
                    "status": action_score.status,
                    "resulting_pose": list(action_score.resulting_pose),
                }
                for action_score in getattr(score_result, "breakdown", []) or []
            ]
            achieved_checkpoints = [
                list(cp) for cp in getattr(score_result, "achieved_checkpoints", []) or []
            ]
            missed_checkpoints = [
                list(cp) for cp in getattr(score_result, "missed_checkpoints", []) or []
            ]
            final_pose = list(getattr(score_result, "final_pose", [])) if getattr(score_result, "final_pose", None) else None

        record = {
            "timestamp": time.time(),
            "request_id": request_id,
            "model_id": model_id,
            "world_id": self.gui_core.current_world_id,
            "prompt_template": prompt_template,
            "input_text": cache.get("command"),
            "actions": actions,
            "score": score,
        }
        if error:
            record["error"] = error

        if breakdown:
            record["breakdown"] = breakdown
        if achieved_checkpoints:
            record["achieved_checkpoints"] = achieved_checkpoints
        if missed_checkpoints:
            record["missed_checkpoints"] = missed_checkpoints
        if final_pose is not None:
            record["final_pose"] = final_pose

        plan_snapshot = self._build_plan_snapshot(cache)
        if plan_snapshot:
            record["plan_snapshot"] = plan_snapshot

        # Determine prompt step for filename if available
        step = None
        try:
            step_info_path = self._score_log_dir / 'current_prompt_step.json'
            if step_info_path.exists():
                with step_info_path.open('r', encoding='utf-8') as sf:
                    info = json.load(sf)
                    s = info.get('step')
                    if isinstance(s, int):
                        step = s
                    elif isinstance(s, str) and s.isdigit():
                        step = int(s)
        except Exception:
            step = None

        if step is None:
            log_file_name = f"scored_input_{model_id.replace('/', '_')}.jsonl"
        else:
            log_file_name = f"scored_inputs{step}.jsonl"
        log_path = self._score_log_dir / log_file_name

        try:
            self._score_log_dir.mkdir(parents=True, exist_ok=True)
            with self._score_log_lock:
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record) + "\n")
            cache["score_logged"] = True
        except Exception as log_error:
            rospy.logwarn(f"GUI: Failed to append scored plan log for {request_id}: {log_error}")

    def _build_plan_snapshot(self, cache):
        """Build a lightweight snapshot of the planned path for downstream UIs."""
        try:
            rows, cols = self.world_manager_component.world_manager.get_world_dimensions(
                self.gui_core.current_world_id
            )
            rows = int(rows)
            cols = int(cols)
        except Exception:
            default_size = int(getattr(self.gui_core, "grid_size", 10))
            rows = cols = default_size

        start = cache.get("start") or {}
        start_row = int(round(float(start.get("row", getattr(self.gui_core, "robot_row", 4)))))
        start_col = int(round(float(start.get("col", getattr(self.gui_core, "robot_col", 4)))))
        start_facing = str(start.get("facing", getattr(self.gui_core, "robot_facing", "N"))).upper()

        path_coords = cache.get("path")
        if (not path_coords) and cache.get("frames") and self.preview_manager:
            try:
                path_coords = self.preview_manager.build_preview_path(cache["frames"])
            except Exception:
                path_coords = None

        normalized_path = []
        for coord in path_coords or []:
            row = col = None
            if isinstance(coord, dict):
                row = coord.get("row")
                col = coord.get("col")
            elif isinstance(coord, (list, tuple)) and len(coord) >= 2:
                row, col = coord[0], coord[1]

            if row is None or col is None:
                continue

            try:
                norm_row = int(round(float(row)))
                norm_col = int(round(float(col)))
            except (TypeError, ValueError):
                continue

            normalized_path.append({"row": norm_row, "col": norm_col})

        if not normalized_path or normalized_path[0]["row"] != start_row or normalized_path[0]["col"] != start_col:
            normalized_path.insert(0, {"row": start_row, "col": start_col})

        snapshot = {
            "world_id": self.gui_core.current_world_id,
            "grid": {"rows": rows, "cols": cols},
            "start": {"row": start_row, "col": start_col, "facing": start_facing},
            "path": normalized_path,
            "obstacles": self._snapshot_obstacles(),
            "checkpoints": self._snapshot_checkpoints(),
        }

        return snapshot

    def _snapshot_obstacles(self):
        """Return normalized obstacle coordinates for current world."""
        try:
            world_config = getattr(self.gui_core, "world_config", {}) or {}
            obstacle_cells = world_config.get("obstacle_cells", [])
        except Exception:
            obstacle_cells = []

        normalized = []
        for cell in obstacle_cells or []:
            row = col = None
            if isinstance(cell, dict):
                row = cell.get("row")
                col = cell.get("col")
            elif isinstance(cell, (list, tuple)) and len(cell) >= 2:
                row, col = cell[0], cell[1]
            if row is None or col is None:
                continue
            try:
                normalized.append({"row": int(row), "col": int(col)})
            except (TypeError, ValueError):
                continue
        return normalized

    def _snapshot_checkpoints(self):
        """Return normalized checkpoint coordinates for current world."""
        checkpoints = []
        try:
            checkpoint_list = self._get_checkpoints_for_world(self.gui_core.current_world_id)
        except Exception:
            checkpoint_list = []

        for cp in checkpoint_list or []:
            try:
                row, col = cp
                checkpoints.append({"row": int(row), "col": int(col)})
            except Exception:
                continue
        return checkpoints
    
    def _on_batch_interpretation(self, msg):
        """Handle batch interpretation updates."""
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"Failed to parse batch interpretation: {e}")
            return
        self._run_on_ui_thread(self._handle_batch_interpretation_payload, payload)
    
    def _handle_batch_interpretation_payload(self, payload):
        request_id = payload.get("request_id")
        interpretation = payload.get("interpretation", "")
        if request_id in self.input_plan_cache:
            self.input_plan_cache[request_id]["interpretation"] = interpretation
            rospy.loginfo(f"GUI: Batch interpretation received for {request_id}")

    def _mark_batch_request_completed(self, request_id=None):
        """Track completion of batch requests and trigger summary when done."""
        self.active_batch_requests.discard(request_id)
        if not self.active_batch_requests and not self.batch_preview_queue and not self.batch_preview_inflight:
            self._enable_show_all_trajectories_if_ready()
            self._show_batch_scoring_summary()
    
    def _on_input_selection_change(self, event=None):
        """Handle input selection change in batch mode."""
        if self.inputs_tree_disabled:
            return
        
        selection = self.widgets.inputs_tree.selection()
        if selection:
            request_id = selection[0]
            cache = self.input_plan_cache.get(request_id, {})
            
            if cache.get("status") == "Preview ready":
                self.widgets.preview_trajectory_button.config(state='normal')
            else:
                self.widgets.preview_trajectory_button.config(state='disabled')
            
            # Enable Save button if plan has actions
            if hasattr(self.widgets, 'save_baseline_button') and self.widgets.save_baseline_button:
                if cache.get("actions"):
                    self.widgets.save_baseline_button.config(state='normal')
                else:
                    self.widgets.save_baseline_button.config(state='disabled')
        else:
            self.widgets.preview_trajectory_button.config(state='disabled')
            if hasattr(self.widgets, 'save_baseline_button') and self.widgets.save_baseline_button:
                self.widgets.save_baseline_button.config(state='disabled')

    def _on_input_double_click(self, event=None):
        """Open a popup that shows the full prompt text for the selected row."""
        if self.inputs_tree_disabled or not hasattr(self.widgets, 'inputs_tree'):
            return

        tree = self.widgets.inputs_tree
        # Identify row under cursor so double-clicking empty space is ignored
        iid = tree.identify_row(event.y) if event is not None else None
        if not iid:
            selection = tree.selection()
            iid = selection[0] if selection else None
        if not iid:
            return

        try:
            idx = int(str(iid).split('-')[1])
        except Exception:
            return

        if idx < 0 or idx >= len(self.loaded_inputs):
            return

        prompt_text = self.loaded_inputs[idx]
        status = self.input_plan_cache.get(iid, {}).get("status", "")
        title = f"Prompt {idx + 1}"
        if status:
            title = f"{title} — {status}"
        self._show_prompt_popup(title, prompt_text)

    def _show_prompt_popup(self, title, prompt_text):
        """Create a simple top-level window with large prompt text for presenting."""
        try:
            popup = tk.Toplevel(self.widgets.root)
            popup.title(title)
            popup.configure(bg="#0d1117")
            popup.geometry("900x500+120+120")
            popup.attributes("-topmost", True)

            header = tk.Label(
                popup,
                text=title,
                bg="#0d1117",
                fg="#e6edf3",
                font=("Segoe UI", 35, "bold"),
                anchor="w",
                pady=12
            )
            header.pack(fill=tk.X, padx=16, pady=(12, 8))

            text_frame = tk.Frame(popup, bg="#0d1117")
            text_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))

            prompt_display = tk.Text(
                text_frame,
                wrap=tk.WORD,
                bg="#111827",
                fg="#f8fafc",
                insertbackground="#f8fafc",
                padx=16,
                pady=16,
                relief="flat",
                borderwidth=0,
                font=("Segoe UI", 35, "bold"),
                state=tk.NORMAL
            )
            prompt_display.insert("1.0", prompt_text)
            prompt_display.config(state=tk.DISABLED)
            prompt_display.pack(fill=tk.BOTH, expand=True)

            # Close on Escape for quick dismissal
            popup.bind("<Escape>", lambda _e: popup.destroy())
        except Exception as exc:
            rospy.logwarn(f"Failed to show prompt popup: {exc}")
    
    def _on_mode_change(self, event=None):
        """Handle control mode change."""
        # Forward to widgets method (which now handles the new tabbed layout)
        if hasattr(self.widgets, '_on_mode_change_plan'):
            self.widgets._on_mode_change_plan(event)
        elif hasattr(self.widgets, 'control_mode') and self.widgets.control_mode:
            # Legacy fallback
            mode = self.widgets.control_mode.get()
            if mode == "Batch Commands" and hasattr(self.widgets, '_show_batch_mode'):
                self.widgets._show_batch_mode()
            elif hasattr(self.widgets, '_show_single_mode'):
                self.widgets._show_single_mode()
    
    def _get_current_approve_dismiss_buttons(self):
        """Get current approve/dismiss buttons based on mode."""
        mode = self.widgets.control_mode.get()
        if mode == "Batch Commands":
            return self.widgets.batch_approve_button, None  # No dismiss button in batch mode
        else:
            if hasattr(self.widgets, 'single_approve_button') and hasattr(self.widgets, 'single_dismiss_button'):
                return self.widgets.single_approve_button, self.widgets.single_dismiss_button
            else:
                return None, None
    
    def _update_feedback_text(self, message):
        """Update the feedback text with a new message (append to history)."""
        def append_message():
            try:
                import datetime
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                formatted_message = f"[{timestamp}] {message}"
                
                self.widgets.feedback_text.config(state=tk.NORMAL)
                if self.widgets.feedback_text.get(1.0, tk.END).strip():
                    self.widgets.feedback_text.insert(tk.END, "\n")
                self.widgets.feedback_text.insert(tk.END, formatted_message)
                self.widgets.feedback_text.config(state=tk.DISABLED)
                self.widgets.feedback_text.see(tk.END)
            except Exception as exc:
                rospy.logwarn(f"Failed to update feedback text: {exc}")
        
        self._run_on_ui_thread(append_message)
    
    def load_world(self, world_id):
        """Load a new world configuration."""
        if self.world_manager_component.load_world(world_id):
            # Update the main GUI core's world config
            self.gui_core.world_config = self.world_manager_component.world_config
            self.gui_core.current_world_id = self.world_manager_component.current_world_id
            self.gui_core.world_name = self.world_manager_component.world_name
            
            # Update GUI elements
            self.gui_core.root.title(f"Spot Robot Simulation - {self.world_manager_component.world_name}")
            self.widgets.world_var.set(f"{world_id}. {self.world_manager_component.world_name}")
            
            # Notify NL_Control about world change
            self.gui_core.pub_world_change.publish(String(data=world_id))
            
            # Redraw the world
            self.gui_core.update_display()
            
            return True
        return False
    
    def _on_load_inputs(self):
        """Load input commands from a selected inputs*.json and schedule batch planning."""
        try:
            if hasattr(self.widgets, 'load_inputs_button') and self.widgets.load_inputs_button:
                self.widgets.load_inputs_button.config(state='disabled')
                # Re-enable after a short delay to prevent accidental double-clicks
                self.widgets.root.after(2000, lambda: self.widgets.load_inputs_button.config(state='normal'))
        except Exception:
            pass

        try:
            # Refresh available files in the dropdown
            try:
                user_data_dir = Path(__file__).resolve().parents[1] / 'user_data'
                files = [p.name for p in user_data_dir.glob('inputs*.json')]
                files = sorted([f for f in files if f != 'inputs.json'])
                if files:
                    self.widgets.inputs_file_combo['values'] = tuple(sorted(files))
                    if self.widgets.inputs_file_var.get() not in files:
                        self.widgets.inputs_file_combo.set(sorted(files)[0])
            except Exception:
                pass

            selected = None
            try:
                selected = self.widgets.inputs_file_var.get()
            except Exception:
                selected = ''
            if not selected:
                raise FileNotFoundError('No inputs file selected')
            inputs_path = Path(__file__).resolve().parents[1] / f'user_data/{selected}'
            raw_text = inputs_path.read_text(encoding='utf-8')
        except FileNotFoundError:
            from tkinter import messagebox
            messagebox.showerror("Inputs file not found", "Click 'Load files' to refresh, then select an inputs{n}.json file.")
            return
        except Exception as exc:
            from tkinter import messagebox
            messagebox.showerror("Failed to read inputs", f"Error reading inputs.json: {exc}")
            return

        try:
            inputs_payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            from tkinter import messagebox
            messagebox.showerror("Failed to parse inputs", f"inputs.json is not valid JSON: {exc}")
            return

        if not isinstance(inputs_payload, dict):
            from tkinter import messagebox
            messagebox.showerror("Invalid inputs format", "inputs.json must be a JSON object mapping user ids to prompts.")
            return

        # Update current prompt pointer and clear previous scores for this batch
        step_hint = None
        try:
            match = re.search(r'inputs(\d+)', selected)
            if match:
                step_hint = int(match.group(1))
        except Exception:
            step_hint = None
        self._prepare_scored_inputs_log(step_hint)

        commands = []
        for _user_id, prompt in inputs_payload.items():
            prompt_text = str(prompt).strip()
            if not prompt_text:
                continue
            # Send the raw prompt text to the LLM (no user_id prefix) for higher quality responses.
            commands.append(prompt_text)

        if not commands:
            from tkinter import messagebox
            messagebox.showinfo("No inputs", "inputs.json does not contain any usable commands.")
            self.loaded_inputs = []
            self.input_plan_cache = {}
            self.batch_preview_queue = []
            self.batch_preview_inflight = None
            self._refresh_input_list()
            self._update_batch_summary()
            self.widgets.preview_trajectory_button.config(state='disabled')
            self.widgets.show_all_trajectories_button.config(state='disabled')
            if self.widgets.heatmap_toggle_button:
                self.widgets.heatmap_toggle_button.config(state='disabled', text="Show Heatmap")
            if hasattr(self.widgets, 'save_baseline_button') and self.widgets.save_baseline_button:
                self.widgets.save_baseline_button.config(state='disabled')
            return

        self.loaded_inputs = commands
        self.input_plan_cache = {}
        self.batch_preview_queue = []
        self.batch_preview_inflight = None
        self.active_batch_requests.clear()
        limit = min(self.batch_preview_limit, len(commands))

        for idx, command in enumerate(commands):
            request_id = self._input_request_id(idx)
            status = "Queued" if idx < limit else "Preview not scheduled"
            self.input_plan_cache[request_id] = {
                "request_id": request_id,
                "index": idx,
                "command": command,
                "status": status,
                "actions": None,
                "start": None,
                "frames": None,
                "path": None,
                "interpretation": None,
                "score_logged": False
            }
            if idx < limit:
                self.batch_preview_queue.append(request_id)

        self.preview_manager.clear_plan_preview_state(clear_path=True)
        self.preview_manager.clear_all_preview_overlay()
        
        # Disable preview buttons when loading new inputs
        self._disable_preview_buttons_after_execution()
        
        # Re-enable input tree selection for new inputs
        if hasattr(self.widgets, 'inputs_tree'):
            # Clear disabled flag
            self.inputs_tree_disabled = False
            # Re-bind the selection event
            self.widgets.inputs_tree.bind('<<TreeviewSelect>>', self._on_input_selection_change)
            # Re-bind mouse events for interaction
            self.widgets.inputs_tree.bind('<Button-1>', lambda e: None)  # Allow selection
            self.widgets.inputs_tree.bind('<Double-1>', self._on_input_double_click)
        
        # Ensure we're in batch mode to show the inputs table
        self.widgets.control_mode.set("Batch Commands")
        self.widgets._show_batch_mode()
        
        # Immediately refresh the GUI to show the loaded inputs with their initial status
        self._refresh_input_list()
        self._update_batch_summary()
        
        # Force GUI update to ensure table is immediately visible
        self.widgets.root.update_idletasks()
        
        # Update feedback to show immediate status
        self._update_feedback_text(
            f"Loaded {len(commands)} inputs from inputs.json — {limit} commands queued for planning"
        )
        
        current_selection = self.widgets.inputs_tree.selection()
        if current_selection:
            self.widgets.inputs_tree.selection_remove(*current_selection)
        self.widgets.preview_trajectory_button.config(state='disabled')
        self.widgets.show_all_trajectories_button.config(state='disabled')
        if self.widgets.heatmap_toggle_button:
            self.widgets.heatmap_toggle_button.config(state='disabled', text="Show Heatmap")
        if hasattr(self.widgets, 'save_baseline_button') and self.widgets.save_baseline_button:
            self.widgets.save_baseline_button.config(state='disabled')
        
        # Start processing queued commands according to current model capabilities
        self.widgets.root.after(100, self._start_batch_processing)

    def _on_refresh_input_files(self):
        """Refresh the inputs file dropdown list from user_data/inputs*.json."""
        try:
            user_data_dir = Path(__file__).resolve().parents[1] / 'user_data'
            files = [p.name for p in user_data_dir.glob('inputs*.json')]
            files = sorted([f for f in files if f != 'inputs.json'])
            self.widgets.inputs_file_combo['values'] = tuple(files)
            if files:
                # Keep current selection if still present, else select first
                current = self.widgets.inputs_file_var.get()
                if current not in files:
                    self.widgets.inputs_file_combo.set(files[0])
        except Exception:
            pass
    
    def _on_copy_and_clear_user_data(self):
        """Copy all files from user_data folder to user_data_saved/timestamp and then empty user_data."""
        from tkinter import messagebox
        import shutil
        import datetime
        
        try:
            base_dir = Path(__file__).resolve().parents[1]
            user_data_dir = base_dir / 'user_data'
            
            # Check if user_data directory exists and has files
            if not user_data_dir.exists():
                messagebox.showwarning("Directory not found", "user_data directory does not exist.")
                return
            
            # Get all files in user_data
            files = list(user_data_dir.glob('*'))
            files = [f for f in files if f.is_file()]  # Only files, not directories
            
            if not files:
                messagebox.showinfo("No files", "user_data folder is already empty.")
                return
            
            # Create user_data_saved folder if it doesn't exist
            saved_base_dir = base_dir / 'user_data_saved'
            saved_base_dir.mkdir(exist_ok=True)
            
            # Create timestamp folder (format: YYYY-MM-DD_HH-MM-SS)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            timestamp_dir = saved_base_dir / timestamp
            timestamp_dir.mkdir(exist_ok=True)
            
            # Copy all files to timestamped destination
            copied_count = 0
            for file_path in files:
                try:
                    dest_file = timestamp_dir / file_path.name
                    shutil.copy2(file_path, dest_file)
                    copied_count += 1
                except Exception as e:
                    rospy.logwarn(f"Failed to copy {file_path.name}: {e}")
                    messagebox.showerror(
                        "Copy error",
                        f"Failed to copy {file_path.name}:\n{e}"
                    )
                    return
            
            # Empty the user_data folder (delete all files)
            deleted_count = 0
            for file_path in files:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    rospy.logwarn(f"Failed to delete {file_path.name}: {e}")
                    messagebox.showerror(
                        "Delete error",
                        f"Failed to delete {file_path.name}:\n{e}"
                    )
                    return
            
            # Refresh the inputs file dropdown
            self._on_refresh_input_files()
            
            # Show success message
            dest_path_str = str(timestamp_dir)
            messagebox.showinfo(
                "Success",
                f"Copied {copied_count} file(s) to:\n{dest_path_str}\n\n"
                f"Deleted {deleted_count} file(s) from user_data folder."
            )
            
            self._update_feedback_text(
                f"Copied {copied_count} file(s) from user_data to {dest_path_str} and cleared user_data folder"
            )
            
        except Exception as e:
            rospy.logerr(f"Error in copy and clear user_data: {e}")
            messagebox.showerror("Error", f"An error occurred:\n{e}")
    
    def _input_request_id(self, index):
        """Generate request ID for input."""
        return f"input-{index}"
    
    def _refresh_input_list(self):
        """Refresh the batch inputs table with current status."""
        if not hasattr(self.widgets, 'inputs_tree'):
            return

        current_selection = set(self.widgets.inputs_tree.selection())
        self.widgets.inputs_tree.delete(*self.widgets.inputs_tree.get_children())

        for idx, command in enumerate(self.loaded_inputs):
            request_id = self._input_request_id(idx)
            cache = self.input_plan_cache.get(request_id, {})
            status = cache.get("status", "") or ("Preview not scheduled" if idx >= self.batch_preview_limit else "Queued")
            score = cache.get("score", "")
            
            # Format score display
            if isinstance(score, (int, float)):
                score_display = str(int(score))
            else:
                score_display = str(score) if score else ""
            
            display_command = command if len(command) <= 60 else command[:57] + "…"
            self.widgets.inputs_tree.insert('', tk.END, iid=request_id, values=(display_command, status, score_display))
            if request_id in current_selection:
                self.widgets.inputs_tree.selection_add(request_id)
    
    def _update_batch_summary(self):
        """Update summary label for loaded inputs."""
        total = len(self.loaded_inputs)
        if total == 0:
            summary = "No inputs loaded"
        else:
            preview_count = min(self.batch_preview_limit, total)
            summary = f"Loaded {total} inputs — previews shown for first {preview_count}"
        self.widgets.inputs_summary_label.config(text=summary)
    
    def _start_batch_processing(self):
        """Kick off batch processing based on current model capabilities."""
        if self.batch_preview_inflight:
            return
        if not self.batch_preview_queue:
            if not self.active_batch_requests:
                self._enable_show_all_trajectories_if_ready()
                self._show_batch_scoring_summary()
            return

        if self._is_parallel_batch_supported():
            pending = list(self.batch_preview_queue)
            self.batch_preview_queue = []
            for request_id in pending:
                self._send_batch_preview_request(request_id)
        else:
            self._process_next_batch_preview()
    
    def _process_next_batch_preview(self):
        """Send the next batch preview request if one is queued."""
        if self._is_parallel_batch_supported() and self.batch_preview_queue:
            # Model switched to parallel mode mid-queue; flush remaining requests
            pending = list(self.batch_preview_queue)
            self.batch_preview_queue = []
            for request_id in pending:
                self._send_batch_preview_request(request_id)
            return

        if self.batch_preview_inflight or not self.batch_preview_queue:
            # Check if all batch processing is complete
            if not self.batch_preview_inflight and not self.batch_preview_queue:
                self._mark_batch_request_completed(None)  # Trigger completion check
            return

        request_id = self.batch_preview_queue.pop(0)
        self.batch_preview_inflight = request_id
        self._send_batch_preview_request(request_id, sequential=True)

    def _send_batch_preview_request(self, request_id, sequential=False):
        """Publish a batch preview request and update local state."""
        cache = self.input_plan_cache.get(request_id, {})
        command = cache.get("command", "")

        cache["status"] = "Planning"
        cache["score_logged"] = False
        self.input_plan_cache[request_id] = cache
        self._refresh_input_list()

        command_index = cache.get("index", 0) + 1
        total_commands = len(self.loaded_inputs)
        if sequential or not self._is_parallel_batch_supported():
            feedback_msg = f"Processing command {command_index}/{total_commands}: '{command}'"
        else:
            feedback_msg = f"Dispatching command {command_index}/{total_commands}: '{command}'"
        self._update_feedback_text(feedback_msg)

        payload = json.dumps({
            "request_id": request_id,
            "command": command,
            "world_id": self.gui_core.current_world_id
        })

        try:
            self.gui_core.pub_batch_preview_request.publish(String(data=payload))
            rospy.loginfo(f"GUI: Sent batch preview request for {request_id}")
            self.active_batch_requests.add(request_id)
        except Exception as exc:
            rospy.logwarn(f"Failed to publish batch preview request: {exc}")
            cache["status"] = "Failed to request plan"
            self.input_plan_cache[request_id] = cache
            self._refresh_input_list()
            self._mark_batch_request_completed(request_id)
            if sequential:
                self.batch_preview_inflight = None
                self.widgets.root.after(200, self._process_next_batch_preview)
    
    def _enable_show_all_trajectories_if_ready(self):
        """Enable 'Show All' button if enough previews are ready."""
        ready_count = sum(1 for cache in self.input_plan_cache.values() 
                         if cache.get("status") == "Preview ready")
        
        if ready_count >= 2:
            self.widgets.show_all_trajectories_button.config(state='normal')
            if self.widgets.heatmap_toggle_button:
                self.widgets.heatmap_toggle_button.config(state='normal')
            self._update_feedback_text(f"Batch planning complete — {ready_count} trajectories ready for preview")
        else:
            if self.widgets.heatmap_toggle_button:
                self.widgets.heatmap_toggle_button.config(state='disabled', text="Show Heatmap")
                self.preview_manager.heatmap_visible = False
            self._update_feedback_text(f"Batch planning in progress — {ready_count} trajectories ready")
    
    def _show_batch_scoring_summary(self):
        """Show summary of scoring results when batch processing completes."""
        if not self.input_plan_cache:
            return
        
        # Collect scores
        scores = []
        failed_count = 0
        
        for cache in self.input_plan_cache.values():
            score = cache.get("score")
            status = cache.get("status", "")
            
            if isinstance(score, (int, float)):
                scores.append(score)
            elif "Failed" in str(status):
                failed_count += 1
        
        if not scores and failed_count == 0:
            return
        
        # Calculate statistics
        summary_parts = []
        
        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            total_scored = len(scores)
            
            summary_parts.append(f"Batch scoring complete: {total_scored} plans scored")
            summary_parts.append(f"Average score: {avg_score:.1f} | Min: {min_score} | Max: {max_score}")
        
        if failed_count > 0:
            summary_parts.append(f"{failed_count} command(s) failed to generate plans")
        
        if summary_parts:
            self._update_feedback_text(" | ".join(summary_parts))
            rospy.loginfo(f"Batch scoring summary: {', '.join(summary_parts)}")
    
    def _disable_preview_buttons_after_execution(self):
        """Disable preview buttons after plan execution completes."""
        # Disable batch mode preview buttons
        self.widgets.preview_trajectory_button.config(state='disabled')
        self.widgets.show_all_trajectories_button.config(state='disabled')
        if self.widgets.heatmap_toggle_button:
            self.widgets.heatmap_toggle_button.config(state='disabled', text="Show Heatmap")
        if hasattr(self.widgets, 'save_baseline_button') and self.widgets.save_baseline_button:
            self.widgets.save_baseline_button.config(state='disabled')
        
        # Disable single mode preview button (if it exists)
        if hasattr(self.widgets, 'single_preview_trajectory_button') and self.widgets.single_preview_trajectory_button:
            self.widgets.single_preview_trajectory_button.config(state='disabled')
        
        # Disable input tree selection in batch mode to prevent clicking on old commands
        if hasattr(self.widgets, 'inputs_tree'):
            # Set flag to indicate tree is disabled
            self.inputs_tree_disabled = True
            # Unbind the selection event to prevent clicking
            self.widgets.inputs_tree.unbind('<<TreeviewSelect>>')
            # Clear any current selection
            self.widgets.inputs_tree.selection_remove(self.widgets.inputs_tree.selection())
            # Disable tree interaction by unbinding all mouse events
            self.widgets.inputs_tree.unbind('<Button-1>')
            self.widgets.inputs_tree.unbind('<Button-3>')
            self.widgets.inputs_tree.unbind('<Double-1>')
        
        # Clear any existing previews
        self.preview_manager.clear_plan_preview_state(clear_path=True)
        self.preview_manager.clear_all_preview_overlay()
        
        rospy.loginfo("Preview buttons and input selection disabled after execution completion")
    
    def _on_send_single_command(self, event=None):
        """Send single command to robot."""
        # Get command from either command_entry or prompt_entry
        # Note: command_entry is an alias for prompt_entry, so check prompt_entry first
        command = ""
        entry_widget = None
        if hasattr(self.widgets, 'prompt_entry') and self.widgets.prompt_entry:
            entry_widget = self.widgets.prompt_entry
            # Check if it's a Text widget or Entry widget
            if isinstance(entry_widget, tk.Text):
                # Text widget - get all text
                command = entry_widget.get("1.0", tk.END).strip()
            else:
                # Entry widget
                command = entry_widget.get().strip()
        elif hasattr(self.widgets, 'command_entry') and self.widgets.command_entry:
            entry_widget = self.widgets.command_entry
            # Check if it's a Text widget or Entry widget
            if isinstance(entry_widget, tk.Text):
                # Text widget - get all text
                command = entry_widget.get("1.0", tk.END).strip()
            else:
                # Entry widget
                command = entry_widget.get().strip()
        
        if not command:
            self._update_feedback_text("Please enter a command")
            return
        
        # Update the last prompt label before sending
        if hasattr(self.widgets, 'last_prompt_label') and self.widgets.last_prompt_label:
            self.widgets.last_prompt_label.config(text=f"Last prompt: {command}", fg='#c9d1d9')
            
        rospy.loginfo(f"GUI: Sending single command: {command}")
        self._update_feedback_text(f"Sending command: '{command}'")
        
        # Show loading animation when sending a new command
        if hasattr(self.widgets, 'update_plan_status'):
            self.widgets.update_plan_status(loading=True)
        
        # Send command to NL control
        self.gui_core.pub_user_speech.publish(String(data=command))
        
        # Disable preview buttons when new command is sent
        self._disable_preview_buttons_after_execution()
        
        # Disable buttons until plan is generated
        if hasattr(self.widgets, 'single_preview_trajectory_button') and self.widgets.single_preview_trajectory_button:
            self.widgets.single_preview_trajectory_button.config(state='disabled')
        if hasattr(self.widgets, 'single_approve_button') and self.widgets.single_approve_button:
            self.widgets.single_approve_button.config(state='disabled')
        if hasattr(self.widgets, 'single_dismiss_button') and self.widgets.single_dismiss_button:
            self.widgets.single_dismiss_button.config(state='disabled')
        
        # Clear the entry
        if entry_widget:
            if isinstance(entry_widget, tk.Text):
                # Text widget
                entry_widget.delete("1.0", tk.END)
            else:
                # Entry widget
                entry_widget.delete(0, tk.END)
    
    def _on_clear_command(self):
        """Clear the single command entry."""
        entry_widget = None
        if hasattr(self.widgets, 'command_entry') and self.widgets.command_entry:
            entry_widget = self.widgets.command_entry
        elif hasattr(self.widgets, 'prompt_entry') and self.widgets.prompt_entry:
            entry_widget = self.widgets.prompt_entry
        
        if entry_widget:
            if isinstance(entry_widget, tk.Text):
                entry_widget.delete("1.0", tk.END)
            else:
                entry_widget.delete(0, tk.END)
        self._update_feedback_text("Command cleared")
    
    def _on_single_preview_trajectory(self):
        """Preview trajectory for single command."""
        # Start the preview animation
        self._on_preview()
        
        # Enable approve/dismiss buttons after preview is shown
        if hasattr(self.widgets, 'single_approve_button') and self.widgets.single_approve_button:
            self.widgets.single_approve_button.config(state='normal')
        if hasattr(self.widgets, 'single_dismiss_button') and self.widgets.single_dismiss_button:
            self.widgets.single_dismiss_button.config(state='normal')
        
        self._update_feedback_text("Previewing trajectory for current command")
    
    def _on_single_approve(self):
        """Approve single command plan."""
        rospy.loginfo("GUI: Approving single command plan")
        self._update_feedback_text("Plan approved - executing...")
        import json
        self.gui_core.pub_approval.publish(String(data=json.dumps({"plan_id": 0, "approved": True})))
        
        # Disable buttons after approval
        if hasattr(self.widgets, 'single_approve_button') and self.widgets.single_approve_button:
            self.widgets.single_approve_button.config(state='disabled')
        if hasattr(self.widgets, 'single_dismiss_button') and self.widgets.single_dismiss_button:
            self.widgets.single_dismiss_button.config(state='disabled')
    
    def _on_single_dismiss(self):
        """Dismiss single command plan."""
        rospy.loginfo("GUI: Dismissing single command plan")
        self._update_feedback_text("Plan dismissed - ready for new command")
        import json
        self.gui_core.pub_approval.publish(String(data=json.dumps({"plan_id": 0, "approved": False})))
        
        # Disable buttons after dismissal
        if hasattr(self.widgets, 'single_approve_button') and self.widgets.single_approve_button:
            self.widgets.single_approve_button.config(state='disabled')
        if hasattr(self.widgets, 'single_dismiss_button') and self.widgets.single_dismiss_button:
            self.widgets.single_dismiss_button.config(state='disabled')
    
    def _on_preview(self):
        """Animate the pending plan on the grid without executing it."""
        rospy.loginfo(f"DEBUG: _on_preview called - actions: {len(self.gui_core.pending_plan_actions) if self.gui_core.pending_plan_actions else 0}, frames: {len(self.preview_manager._preview_frames) if self.preview_manager._preview_frames else 0}")
        if not self.gui_core.pending_plan_actions or not self.preview_manager._preview_frames:
            rospy.loginfo(f"DEBUG: _on_preview returning early - no actions or frames")
            return

        if self.preview_manager._preview_animation_running:
            rospy.loginfo(f"DEBUG: _on_preview returning early - animation already running")
            return

        rospy.loginfo(f"DEBUG: _on_preview calling start_preview_animation")
        self.preview_manager.start_preview_animation()
    
    def _path_is_grid_coordinates(self, path_points):
        """Determine whether stored path points are in grid units."""
        if not path_points:
            return True
        
        renderer = getattr(self.gui_core, "canvas_renderer", None)
        grid_limit = 50
        if renderer:
            grid_limit = max(renderer.grid_rows, renderer.grid_cols) + 5
        
        sample = path_points[0]
        if isinstance(sample, dict):
            row = sample.get("row")
            col = sample.get("col")
        elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
            row, col = sample[0], sample[1]
        else:
            return True
        
        if row is None or col is None:
            return True
        
        try:
            row_val = float(row)
            col_val = float(col)
        except (TypeError, ValueError):
            return True
        
        return abs(row_val) <= grid_limit and abs(col_val) <= grid_limit
    
    def _ensure_grid_path(self, cache_entry, frames):
        """Ensure cached path points are stored in grid units."""
        path = cache_entry.get("path")
        if path and self._path_is_grid_coordinates(path):
            return path
        
        if not frames:
            return []
        
        path = self.preview_manager.build_preview_path(frames)
        cache_entry["path"] = path
        return path
    
    def _build_multi_preview_data(self):
        """Aggregate cached batch previews into combined path and heatmap data."""
        ordered_entries = sorted(
            (
                data.get("index", 0),
                request_id,
                data
            )
            for request_id, data in self.input_plan_cache.items()
        )

        palette = [
            "#1F6FEB", "#D29922", "#58A6FF", "#FF7B72",
            "#34C759", "#9A7FF0", "#FFB454", "#0FA295",
            "#FF3B30", "#7C5CF5"
        ]

        combined_paths = []
        overlap_counts = {}
        missing_indices = []

        for order, request_id, data in ordered_entries:
            if not data:
                continue

            actions = data.get("actions")
            if not actions:
                status_text = (data.get("status") or "").lower()
                if status_text and ("fail" in status_text or "error" in status_text):
                    continue
                missing_indices.append(order + 1)
                continue

            start_state = data.get("start") or {
                "row": self.gui_core.robot_row,
                "col": self.gui_core.robot_col,
                "facing": self.gui_core.robot_facing
            }

            frames = data.get("frames")
            if not frames:
                frames = self.preview_manager.compute_preview_frames(actions, start_state)
                data["frames"] = frames

            if not frames:
                missing_indices.append(order + 1)
                continue

            path_coords = data.get("path")
            if not path_coords or not self._path_is_grid_coordinates(path_coords):
                path_coords = self.preview_manager.build_preview_path(frames)
                data["path"] = path_coords

            if not path_coords:
                missing_indices.append(order + 1)
                continue

            color = palette[order % len(palette)]
            combined_paths.append({
                "cells": path_coords,
                "color": color,
                "label": f"{order + 1}"
            })

            for frame in frames:
                row = int(round(frame.get("row", 0)))
                col = int(round(frame.get("col", 0)))
                overlap_counts[(row, col)] = overlap_counts.get((row, col), 0) + 1

        return combined_paths, overlap_counts, missing_indices

    def _on_show_all_previews(self):
        """Toggle visualization of all available trajectory previews on the canvas."""
        if self.preview_manager.all_preview_visible:
            self.preview_manager.all_preview_visible = False
            self.preview_manager.multi_preview_paths = []
            self.widgets.show_all_trajectories_button.config(text="Show All")
            self._update_feedback_text("Combined preview overlay hidden")
            self.gui_core.update_display()
            return

        if not self.loaded_inputs:
            self._update_feedback_text("No inputs loaded — load inputs to view combined previews")
            return

        combined_paths, overlap_counts, missing_indices = self._build_multi_preview_data()

        if not combined_paths:
            if missing_indices:
                self._update_feedback_text(f"No valid previews available (missing: {', '.join(map(str, missing_indices))})")
            else:
                self._update_feedback_text("No valid previews available")
            return

        self.preview_manager.multi_preview_paths = combined_paths
        self.preview_manager.multi_preview_counts = overlap_counts
        self.preview_manager.all_preview_visible = True

        # Update button text to "Hide All"
        self.widgets.show_all_trajectories_button.config(text="Hide All")

        # Ensure heatmap button reflects current state and available data
        if self.widgets.heatmap_toggle_button:
            if self.preview_manager.heatmap_visible and not overlap_counts:
                self.preview_manager.heatmap_visible = False
            button_text = "Hide Heatmap" if self.preview_manager.heatmap_visible else "Show Heatmap"
            self.widgets.heatmap_toggle_button.config(text=button_text)

        self.gui_core.update_display()

        if missing_indices:
            self._update_feedback_text(f"Showing combined previews (missing: {', '.join(map(str, missing_indices))})")
        else:
            self._update_feedback_text("Showing combined previews")

    def _on_toggle_heatmap(self):
        """Toggle the combined heatmap overlay for batch trajectories."""
        if not self.loaded_inputs:
            self._update_feedback_text("No inputs loaded — load inputs to view the heatmap")
            return

        display_updated = False

        if not self.preview_manager.heatmap_visible:
            combined_paths, overlap_counts, missing_indices = self._build_multi_preview_data()
            if not overlap_counts:
                if missing_indices:
                    self._update_feedback_text(f"No heatmap available yet (missing previews: {', '.join(map(str, missing_indices))})")
                else:
                    self._update_feedback_text("Heatmap unavailable — generate trajectory previews first")
                return

            self.preview_manager.multi_preview_counts = overlap_counts
            if not self.preview_manager.multi_preview_paths:
                self.preview_manager.multi_preview_paths = combined_paths

            self.preview_manager.heatmap_visible = True
            if self.widgets.heatmap_toggle_button:
                self.widgets.heatmap_toggle_button.config(text="Hide Heatmap")
            display_updated = True

            if missing_indices:
                self._update_feedback_text(f"Heatmap displayed (missing previews: {', '.join(map(str, missing_indices))})")
            else:
                self._update_feedback_text("Heatmap displayed")
        else:
            self.preview_manager.heatmap_visible = False
            if self.widgets.heatmap_toggle_button:
                self.widgets.heatmap_toggle_button.config(text="Show Heatmap")
            self._update_feedback_text("Heatmap hidden")
            display_updated = True

        if display_updated:
            self.gui_core.update_display()

    def _on_toggle_wall_enforcement(self):
        """Toggle obstacle-safe plan enforcement."""
        if getattr(self.widgets, "wall_enforcement_var", None) is None:
            return

        enabled = bool(self.widgets.wall_enforcement_var.get())
        self.wall_enforcement_enabled = enabled

        try:
            self.pub_wall_enforcement.publish(String(data="true" if enabled else "false"))
            rospy.loginfo(f"GUI: Obstacle enforcement {'enabled' if enabled else 'disabled'}")
        except Exception as exc:
            rospy.logwarn(f"GUI: Failed to publish wall enforcement toggle: {exc}")

        if enabled:
            self._update_feedback_text(
                "Obstacle-safe path enforcement enabled — plans may be expanded to avoid obstacles"
            )
        else:
            self._update_feedback_text(
                "Obstacle-safe path enforcement disabled — using raw LLM paths without adjustments"
            )

    def _on_toggle_strict_scoring(self):
        """Toggle advanced scoring rules for batch evaluation."""
        if getattr(self.widgets, "strict_scoring_var", None) is None:
            return

        enabled = bool(self.widgets.strict_scoring_var.get())
        self.plan_scorer.enable_additional_rules = enabled
        if enabled:
            self._update_feedback_text(
                "Strict scoring enabled — revisits penalized and checkpoints must be visited in order"
            )
        else:
            self._update_feedback_text(
                "Strict scoring disabled — penalty-only without checkpoint ordering or revisit penalties"
            )
    
    def _on_preview_trajectory(self):
        """Preview trajectory for the selected input - loads preview and starts animation."""
        selection = self.widgets.inputs_tree.selection()
        if not selection:
            self._update_feedback_text("No input selected")
            return

        request_id = selection[0]
        data = self.input_plan_cache.get(request_id)
        if not data or not data.get("actions"):
            self._update_feedback_text("No plan available for selected input")
            return

        # Apply the cached preview (same as Show Trajectory)
        if self._apply_cached_preview(request_id):
            self._update_feedback_text(f"Trajectory preview loaded for input {data['index'] + 1}: '{data.get('command', '')}'")
            # Start the preview animation (same as Preview button)
            self._on_preview()
    
    def _apply_cached_preview(self, request_id):
        """Apply cached preview data to the main preview pane."""
        data = self.input_plan_cache.get(request_id)
        if not data or not data.get("actions"):
            return False

        start_state = data.get("start") or {
            "row": self.gui_core.robot_row,
            "col": self.gui_core.robot_col,
            "facing": self.gui_core.robot_facing
        }

        frames = data.get("frames")
        if not frames:
            frames = self.preview_manager.compute_preview_frames(data["actions"], start_state)
            data["frames"] = frames
            data["path"] = self.preview_manager.build_preview_path(frames)
        waypoints = data.get("waypoints")
        if waypoints is None:
            waypoints = self.preview_manager.extract_waypoints(data["actions"])
            data["waypoints"] = waypoints

        path = self._ensure_grid_path(data, frames)

        # Cancel animation and clear previous state before loading new preview
        self.preview_manager.cancel_preview_animation()
        # Clear the path from canvas to ensure clean state - delete all preview path elements
        self.preview_manager.canvas.delete("preview_path")
        # Also clear any path drawn by draw_world (which uses preview_path_coords)
        # Force a display update to clear the old path before loading new one
        self.preview_manager.preview_path_coords = None
        self.gui_core.update_display()
        
        with self.gui_core.update_lock:
            # Update gui_core state with NEW plan data
            self.gui_core.pending_plan_actions = list(data["actions"])
            self.gui_core.pending_plan_start = dict(start_state)
            self.gui_core._preview_frames = list(frames)
            self.gui_core.preview_path_coords = list(path) if path else None
            self.gui_core.preview_waypoints = [dict(wp) for wp in waypoints] if waypoints else []
            
            # Also store frames and actions in preview manager for _on_preview method
            self.preview_manager._preview_frames = list(frames)
            self.preview_manager.preview_path_coords = list(path) if path else None
            self.preview_manager.pending_plan_actions = list(data["actions"])
            self.preview_manager.pending_plan_start = dict(start_state)
            self.preview_manager.preview_waypoints = [dict(wp) for wp in waypoints] if waypoints else []

        # Force display update to show new path - this will redraw the entire canvas with new path
        self.gui_core.update_display()
        # Force canvas update to ensure it's redrawn
        if hasattr(self.gui_core, 'root'):
            self.gui_core.root.update_idletasks()
        command = data.get("command", "")
        self._update_feedback_text(
            f"Preview loaded for input {data['index'] + 1}: '{command}' — ready for approval"
        )
        
        # Set up the cached plan for approval when preview is loaded
        self._setup_cached_plan_for_approval(data["actions"], command)
        
        # Enable Approve button after preview is loaded
        approve_btn, dismiss_btn = self._get_current_approve_dismiss_buttons()
        if approve_btn:
            approve_btn.config(state='normal')
        if dismiss_btn:
            dismiss_btn.config(state='normal')
        
        return True
    
    def _setup_cached_plan_for_approval(self, actions, command):
        """Set up the cached plan for approval without regenerating."""
        # Publish interpretation for display
        interpretation = f"Execute cached plan: {command}"
        self.gui_core.pub_interpretation.publish(String(data=interpretation))
        
        # Set up the NL control system to use the cached plan for approval
        import json
        plan_payload = {
            "plan_id": 0,
            "actions": actions,
            "cached_plan": True  # Flag to indicate this is a cached plan
        }
        self.gui_core.pub_plan.publish(String(data=json.dumps(plan_payload)))
        
        # Enable approve/decline buttons since we have a plan ready
        approve_btn, dismiss_btn = self._get_current_approve_dismiss_buttons()
        if approve_btn:
            approve_btn.config(state='normal')
        if dismiss_btn:
            dismiss_btn.config(state='normal')
    
    def _on_approve(self):
        """Approve pending plan"""
        rospy.loginfo("GUI: Approving plan")
        self._update_feedback_text("Plan approved - executing...")
        import json
        self.gui_core.pub_approval.publish(String(data=json.dumps({"plan_id": 0, "approved": True})))
        # Disable the current approve/dismiss buttons
        approve_btn, dismiss_btn = self._get_current_approve_dismiss_buttons()
        if approve_btn:
            approve_btn.config(state='disabled')
        if dismiss_btn:
            dismiss_btn.config(state='disabled')
        
        # Disable preview buttons to prevent previewing old plans
        self._disable_preview_buttons_after_execution()
        
        # Clear the trajectory visualization and preview state
        self.preview_manager.clear_plan_preview_state(clear_path=True)
        
        # Update the display to remove the trajectory from the canvas
        self.gui_core.update_display()
    
    def _on_decline(self):
        """Decline pending plan"""
        rospy.loginfo("GUI: Declining plan")
        self._update_feedback_text("Plan rejected - ready for new command")
        import json
        self.gui_core.pub_approval.publish(String(data=json.dumps({"plan_id": 0, "approved": False})))
        # Disable the current approve/dismiss buttons
        approve_btn, dismiss_btn = self._get_current_approve_dismiss_buttons()
        if approve_btn:
            approve_btn.config(state='disabled')
        if dismiss_btn:
            dismiss_btn.config(state='disabled')
        self.preview_manager.clear_plan_preview_state(clear_path=True)
        self.preview_manager.clear_all_preview_overlay()
    
    # Baseline mode methods
    def _on_show_baseline(self):
        """Load and display selected baseline."""
        baseline_id = self.widgets.baseline_selector_var.get()
        rospy.loginfo(f"GUI: Loading baseline {baseline_id}")
        
        # Load baseline data
        baseline_data = self._load_baseline_data(baseline_id)
        if baseline_data:
            self._display_baseline_prompt_plan(baseline_data)
            self._update_feedback_text(f"Loaded {baseline_id}")
        else:
            self._update_feedback_text(f"Failed to load {baseline_id}")
    
    def _load_baseline_data(self, baseline_id):
        """Load baseline data from JSON file."""
        try:
            baselines_dir = Path(__file__).resolve().parents[1] / 'baselines'
            
            # Try to find the baseline file by matching the name in the JSON
            for f in baselines_dir.glob('baseline_*.json'):
                try:
                    with open(f, 'r') as bf:
                        data = json.load(bf)
                        if data.get('name') == baseline_id:
                            rospy.loginfo(f"Loaded baseline from {f}")
                            return data
                except Exception:
                    continue
            
            # Fallback: try direct mapping for legacy baseline names
            baseline_map = {
                'Baseline 1': 'baseline_1',
                'Baseline 2': 'baseline_2',
                'Baseline 3': 'baseline_3'
            }
            file_id = baseline_map.get(baseline_id, None)
            
            if file_id:
                baseline_file = baselines_dir / f'{file_id}.json'
                if baseline_file.exists():
                    with open(baseline_file, 'r') as f:
                        data = json.load(f)
                    rospy.loginfo(f"Loaded baseline from {baseline_file}")
                    return data
            
            rospy.logerr(f"Baseline not found: {baseline_id}")
            return None
        except Exception as e:
            rospy.logerr(f"Error loading baseline data: {e}")
            return None
    
    def _display_baseline_prompt_plan(self, data):
        """Display prompt and plan in baseline text widgets."""
        # Display prompt
        if hasattr(self.widgets, 'baseline_prompt_text') and self.widgets.baseline_prompt_text:
            self.widgets.baseline_prompt_text.config(state='normal')
            self.widgets.baseline_prompt_text.delete('1.0', tk.END)
            self.widgets.baseline_prompt_text.insert('1.0', data.get('prompt', 'No prompt available'))
            self.widgets.baseline_prompt_text.config(state='disabled')
        
        # Display plan
        plan_text = data.get('plan', 'No plan available')
        if hasattr(self.widgets, 'baseline_plan_text') and self.widgets.baseline_plan_text:
            self.widgets.baseline_plan_text.config(state='normal')
            self.widgets.baseline_plan_text.delete('1.0', tk.END)
            self.widgets.baseline_plan_text.insert('1.0', plan_text)
            self.widgets.baseline_plan_text.config(state='disabled')
        
        # Parse plan text into actions for animation
        actions = self._parse_baseline_plan(plan_text)
        
        # Store actions in gui_core for the Play button to access
        self.gui_core.pending_plan_actions = actions
        self.preview_manager.pending_plan_actions = list(actions)
        
        # Compute preview frames for animation
        start_state = {
            "row": self.gui_core.robot_row,
            "col": self.gui_core.robot_col,
            "facing": self.gui_core.robot_facing
        }
        
        frames = self.preview_manager.compute_preview_frames(actions, start_state)
        self.preview_manager._preview_frames = frames
        waypoints = self.preview_manager.extract_waypoints(actions)
        self.preview_manager.preview_waypoints = waypoints
        self.gui_core.preview_waypoints = list(waypoints)
        # If no trajectory is provided, fall back to the computed path for display
        if not data.get("trajectory"):
            path = self.preview_manager.build_preview_path(frames)
            self.preview_manager.preview_path_coords = path
            self.gui_core.preview_path_coords = path
        
        # Display trajectory on grid
        if 'trajectory' in data and 'plan' in data:
            self._display_baseline_trajectory(data)
        
        # Enable execute button
        if hasattr(self.widgets, 'baseline_execute_button') and self.widgets.baseline_execute_button:
            self.widgets.baseline_execute_button.config(state='normal')
            rospy.loginfo("DEBUG: Enabled baseline_execute_button")
        else:
            rospy.logwarn("DEBUG: baseline_execute_button not found or None!")
        
        # Store current baseline data for execution
        self.current_baseline_data = data
        rospy.loginfo(f"DEBUG: Stored baseline data: {data.get('name', 'unknown')}")
    
    def _parse_baseline_plan(self, plan_text):
        """Parse baseline plan text into action list."""
        actions = []
        for line in plan_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Remove numbering like "1. ", "2. ", etc.
            import re
            match = re.match(r'^\d+\.\s*(.*)', line)
            if match:
                action = match.group(1).strip()
                if action:
                    actions.append(action)
            elif line and not line.startswith('#'):
                actions.append(line)
        return actions
    
    def _display_baseline_trajectory(self, data):
        """Display baseline trajectory on the grid."""
        try:
            trajectory = data.get('trajectory', [])
            plan_text = data.get('plan', '')
            
            if not trajectory:
                rospy.logwarn("No trajectory data in baseline")
                return
            
            # Don't clear preview state - we need the frames for the Play button!
            # Just clear the path visually
            self.gui_core.canvas.delete("preview_path")
            
            # Ensure waypoint markers are available for overlay
            if not self.preview_manager.preview_waypoints:
                actions = self.gui_core.pending_plan_actions or self._parse_baseline_plan(plan_text)
                waypoints = self.preview_manager.extract_waypoints(actions)
                self.preview_manager.preview_waypoints = waypoints
                self.gui_core.preview_waypoints = list(waypoints)

            # Set the path coordinates for display
            self.preview_manager.preview_path_coords = trajectory
            self.gui_core.preview_path_coords = trajectory
            
            # Update display to show the trajectory
            self.gui_core.update_display()
            
            rospy.loginfo(f"Displayed baseline trajectory with {len(trajectory)} waypoints")
        except Exception as e:
            rospy.logerr(f"Error displaying baseline trajectory: {e}")
    
    def _on_execute_baseline(self):
        """Execute the currently loaded baseline plan."""
        rospy.loginfo("DEBUG: _on_execute_baseline called")
        
        if not hasattr(self, 'current_baseline_data') or not self.current_baseline_data:
            rospy.logwarn("No baseline data loaded")
            self._update_feedback_text("No baseline loaded to execute")
            return
        
        rospy.loginfo("GUI: Executing baseline plan")
        self._update_feedback_text("Executing baseline plan...")
        
        # Parse plan into actions
        plan_text = self.current_baseline_data.get('plan', '')
        actions = self._parse_baseline_plan(plan_text)
        
        if not actions:
            rospy.logerr("No actions found in baseline plan")
            self._update_feedback_text("Error: No actions in baseline plan")
            return
        
        rospy.loginfo(f"DEBUG: Parsed {len(actions)} actions from baseline plan")
        
        # Send plan to nl_control for immediate execution (skip approval for baselines)
        try:
            # Send plan with direct_execution and bypass_approval since baselines are pre-approved
            plan_payload = {
                'actions': actions,
                'request_id': 'baseline',
                'command': f"Baseline {self.current_baseline_data.get('name', 'plan')}",
                'direct_execution': True,  # Execute immediately
                'bypass_approval': True    # Skip approval workflow
            }
            
            rospy.loginfo(f"DEBUG: Creating plan payload: direct_execution={plan_payload['direct_execution']}, bypass_approval={plan_payload['bypass_approval']}")
            
            # Use the existing publisher from gui_core instead of creating a new one
            plan_json = json.dumps(plan_payload)
            rospy.loginfo(f"DEBUG: Publishing plan to /nl_control/plan: {plan_json[:200]}...")
            
            self.gui_core.pub_plan.publish(String(data=plan_json))
            
            rospy.loginfo(f"Sent baseline plan with {len(actions)} actions for direct execution")
            
            # Disable execute button during execution
            if hasattr(self.widgets, 'baseline_execute_button') and self.widgets.baseline_execute_button:
                self.widgets.baseline_execute_button.config(state='disabled')
            
            self._update_feedback_text("Executing baseline plan...")
        except Exception as e:
            rospy.logerr(f"Error executing baseline: {e}")
            import traceback
            rospy.logerr(f"Traceback: {traceback.format_exc()}")
            self._update_feedback_text(f"Error executing baseline: {e}")
    
    def _on_save_as_baseline(self):
        """Save the selected plan as a new baseline."""
        # Get the selected item from inputs tree
        selection = self.widgets.inputs_tree.selection()
        if not selection:
            self._update_feedback_text("No input selected - please select a plan to save")
            return
        
        request_id = selection[0]
        cache = self.input_plan_cache.get(request_id)
        
        if not cache:
            self._update_feedback_text("No plan data available for selected input")
            return
        
        actions = cache.get("actions")
        command = cache.get("command", "")
        
        if not actions:
            self._update_feedback_text("No plan available - generate a plan first")
            return
        
        # Build the plan string (numbered steps)
        plan_lines = []
        for i, action in enumerate(actions, 1):
            plan_lines.append(f"{i}. {action}")
        plan_string = "\n".join(plan_lines)
        
        # Build trajectory from path if available, otherwise from frames
        trajectory = []
        if cache.get("path"):
            trajectory = cache["path"]
        elif cache.get("frames"):
            for frame in cache["frames"]:
                coord = [frame.get("row", 0), frame.get("col", 0)]
                if not trajectory or trajectory[-1] != coord:
                    trajectory.append(coord)
        
        # Find the next available baseline number
        baselines_dir = Path(__file__).resolve().parents[1] / 'baselines'
        baselines_dir.mkdir(exist_ok=True)
        
        existing_ids = []
        for f in baselines_dir.glob('baseline_*.json'):
            try:
                num = int(f.stem.replace('baseline_', ''))
                existing_ids.append(num)
            except ValueError:
                pass
        
        next_id = max(existing_ids) + 1 if existing_ids else 1
        baseline_id = f"baseline_{next_id}"
        
        # Create baseline data
        baseline_data = {
            "id": baseline_id,
            "name": f"Baseline {next_id}",
            "prompt": command,
            "plan": plan_string,
            "trajectory": trajectory,
            "world_id": self.gui_core.current_world_id
        }
        
        # Save to file
        baseline_file = baselines_dir / f'{baseline_id}.json'
        try:
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            
            self._update_feedback_text(f"Saved as {baseline_id} ({baseline_file.name})")
            rospy.loginfo(f"Saved baseline to {baseline_file}")
            
            # Update the baseline combo dropdown to include the new baseline
            self._refresh_baseline_combo()
            
        except Exception as e:
            rospy.logerr(f"Error saving baseline: {e}")
            self._update_feedback_text(f"Error saving baseline: {e}")
    
    def _refresh_baseline_combo(self):
        """Refresh the baseline combo box with available baselines."""
        if not hasattr(self.widgets, 'baseline_combo') or not self.widgets.baseline_combo:
            return
        
        try:
            baselines_dir = Path(__file__).resolve().parents[1] / 'baselines'
            baseline_names = []
            
            for f in sorted(baselines_dir.glob('baseline_*.json')):
                try:
                    with open(f, 'r') as bf:
                        data = json.load(bf)
                        name = data.get('name', f.stem)
                        baseline_names.append(name)
                except Exception:
                    baseline_names.append(f.stem.replace('_', ' ').title())
            
            if baseline_names:
                self.widgets.baseline_combo['values'] = tuple(baseline_names)
        except Exception as e:
            rospy.logwarn(f"Error refreshing baseline combo: {e}")
