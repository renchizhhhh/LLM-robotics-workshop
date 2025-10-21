#!/usr/bin/env python3
"""
Natural Language Control for Spot Robot (Grid-Based Version)
"""

import rospy
import sys
import os
import argparse
from google import genai
import threading
import time
import json
import math
from std_msgs.msg import String, Empty
from std_srvs.srv import Trigger

# Add the FSM to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'spot_hololens_llm_interface'))
from finite_state_machine import SpotStateMachine

# Import timing utilities
from timing_utils import recorder

# Import world manager
from world_manager import WorldManager

# Grid-based prompt for converting natural language to FSM commands
PROMPT = """You are a Spot robot path planner. Generate robot actions for the given task.

ACTIONS (EXACT FORMAT REQUIRED):
- stand_up
- sit_down  
- move_to_cell row=<int> col=<int>
- rotate_to direction=<N/S/E/W>
- start_automated_grasp object_type=<object_name_provided_by_user>  # MUST use the exact object name from user's command, NOT waypoint names
- start_drop_off

CRITICAL: Use ONLY these exact action formats. Do NOT use move_west, move_north, move_east, move_south, or any other movement commands.

MOVEMENT STRATEGY: Use direct movements to target cells when possible. Instead of multiple small steps, use single move_to_cell commands to reach distant cells on the same axis.

RULES:
1. If vague command (e.g. "go!") → "FAIL unclear_command"
2. If robot already standing → skip stand_up
3. Can move to any cell on same axis (horizontal or vertical only)
4. CRITICAL: Use the EXACT object name mentioned by the user in their command, NOT the waypoint name. For example, if user says "pick up the apple juice", use object_type="apple_juice", NOT object_type="PICKUP"

GRID SYSTEM:
- Robot moves between cells in a 10x10 grid (0-9 for both row and col)
- Robot starts at the position shown in CURRENT_STATE.robot_cell
- Can move to any cell on the same axis (horizontal or vertical only)
- Example: from (0,0) can move to (0,3) or (3,0) but not (3,3)
- Robot internally checks all intermediate cells for obstacles
- PREFER DIRECT MOVEMENTS: Use move_to_cell row=X col=Y to go directly to target, not step-by-step

OFFSET COMPENSATION:
- The robot has an offset compensation system that tracks accumulated position errors
- CURRENT_STATE includes accumulated_offset_row and accumulated_offset_col values
- These offsets are automatically compensated for in movement calculations
- You can ignore these offset values - they are handled internally by the system

ROTATION:
- For PICK: rotate to waypoint.pick_direction before grasping
- For DROP: rotate to zone.direction before dropping
- Use rotate_to direction=<N/S/E/W> to face cardinal directions
- Robot orientation changes after each rotation

MAZE NAVIGATION:
- WORLD_MODEL includes "wall_cells" array with cells that are walls
- Each wall cell is in format: [row, col] - this cell is a wall and cannot be entered
- CRITICAL: Check if ANY planned movement target is in wall_cells - if so, find alternative path
- Plan movements that navigate AROUND wall cells, not into them
- Use direct movements to target cells when possible (e.g., move_to_cell row=0 col=3 instead of multiple small steps)

OUTPUT: One action per line, no comments or numbering.

IMPORTANT: The robot's current position is shown in CURRENT_STATE.robot_cell below.

CURRENT_STATE = {current_state_json}
WORLD_MODEL = {world_model_json}
TASK = {command}"""

class NaturalLanguageControl:
    def __init__(self, use_speech=False):
        rospy.init_node('nl_control', anonymous=True)
        self.use_speech = use_speech
        
        # Grid position tracking
        self.current_cell = [4.0, 4.0]  # [row, col] - start in middle of 10x10 grid
        self.current_facing = "N"  # Current facing direction
        self.start_cell = [4.0, 4.0]  # Store initial position
        
        # Offset compensation tracking
        self.accumulated_offset_row = 0.0  # Accumulated offset in grid cells (row)
        self.accumulated_offset_col = 0.0  # Accumulated offset in grid cells (col)
        self.last_expected_cell = [4.0, 4.0]  # Last expected position after movement
        
        # Grid synchronization tracking
        self.grid_offset_row = None  # Offset from real position to grid center
        self.grid_offset_col = None
        
        # Publisher for position updates
        self.pub_position = rospy.Publisher('/nl_control/robot_position', String, queue_size=1)
        
        # Publisher for interpretation feedback
        self.pub_interpretation = rospy.Publisher('/llm_int/interpretation', String, queue_size=1)
        # Publisher for GUI plan (enables approve button in UI)
        self.pub_plan = rospy.Publisher('/spot/plan', String, queue_size=10)
        
        # HoloLens stop signal handling
        self.stop_requested = False
        self.sub_hl_stop = rospy.Subscriber('/hl/stop', Empty, self.on_hl_stop)
        
        # Approval handling
        self.sub_approval = rospy.Subscriber('/hl/approval', String, self.on_approval)
        self.pending_plan = []
        self.awaiting_approval = False
        
        # World change handling
        self.sub_world_change = rospy.Subscriber('/gui/world_change', String, self.on_world_change)
        
        # Check if dummy mode is set globally
        dummy_mode = rospy.get_param('/spot_fsm/dummy_mode', 
                     rospy.get_param('/spot_entrance/dummy_mode',
                     rospy.get_param('dummy_mode', True)))
        
        rospy.loginfo(f"Natural Language Control starting in {'DUMMY' if dummy_mode else 'REAL'} mode (GRID-BASED)")
        self.spot_fsm = SpotStateMachine(dummy_mode=dummy_mode)
        
        # Initialize world manager
        self.world_manager = WorldManager()
        self.current_world_id = "1"  # Default to Simple Pick & Drop
        self.world_config = self.world_manager.get_world_config(self.current_world_id)
        
        # Track yaw origin so GUI facing matches relative frame (N=0 at start)
        self.yaw_origin = None
        self.current_yaw_rel = 0.0
        # Grid normalization: offsets so real pose maps to (4,4)
        self.grid_offset_row = None
        self.grid_offset_col = None
        
        # Ensure /spot_entrance services are available before attempting FSM actions (robust like orchestrator flow)
        self._wait_for_spot_services()

        # Check current robot state and connect/power on if needed
        rospy.loginfo("Checking robot state...")
        rospy.sleep(1.0)  # Give FSM time to complete its startup sequence
        
        current_state = str(self.spot_fsm.current_state)
        rospy.loginfo(f"Current FSM state: {current_state}")
        
        # Only connect if not already connected
        if current_state in ["unknown", "disconnected"]:
            try:
                rospy.loginfo("Robot not connected, sending connect command...")
                recorder.publish_event('start_connect')
                self.spot_fsm.send("connect")
                recorder.publish_event('stop_connect')
                rospy.loginfo("Connect command completed successfully")
            except Exception as e:
                rospy.logerr(f"Connect command failed: {e}")
                raise
        else:
            rospy.loginfo("Robot already connected, skipping connect command")
        
        # Only power on if not already powered
        current_state = str(self.spot_fsm.current_state)
        if current_state in ["unknown", "disconnected", "connected"]:
            try:
                rospy.loginfo("Robot not powered, sending power_on command...")
                recorder.publish_event('start_power_on')
                self.spot_fsm.send("power_on")
                recorder.publish_event('stop_power_on')
                rospy.loginfo("Power_on command completed successfully")
            except Exception as e:
                rospy.logerr(f"Power_on command failed: {e}")
                raise
        else:
            rospy.loginfo("Robot already powered, skipping power_on command")
        
        # Stand up robot to get it ready for commands
        try:
            rospy.loginfo("Sending stand_up command to FSM...")
            recorder.publish_event('start_stand_up')
            self.spot_fsm.send("stand_up")
            recorder.publish_event('stop_stand_up')
            rospy.loginfo("Stand_up command completed successfully")
            
            rospy.loginfo(f"Current FSM state after stand_up: {self.spot_fsm.current_state}")
        except Exception as e:
            rospy.logerr(f"Stand_up command failed: {e}")
            # Fallback: try direct service if available
            try:
                rospy.wait_for_service('/spot_entrance/stand', timeout=3.0)
                stand_srv = rospy.ServiceProxy('/spot_entrance/stand', Trigger)
                resp = stand_srv()
                if not getattr(resp, 'success', False):
                    raise rospy.ROSException(f"Stand service returned failure: {getattr(resp, 'message', '')}")
                rospy.loginfo("Stand_up completed via direct service fallback")
            except Exception as e2:
                rospy.logerr(f"Fallback stand service failed: {e2}")
                raise
        
        rospy.loginfo("Robot ready for commands")
        
        # Publisher for simulation feedback
        self.pub_feedback = rospy.Publisher('/spot/execution_feedback', String, queue_size=20)
        
        # Always listen for GUI/HoloLens user speech to trigger planning
        self.gui_speech_sub = rospy.Subscriber('/hl/user_speech', String, self.on_user_speech)
        if use_speech:
            self.speech_sub = rospy.Subscriber('/hl/user_speech', String, self.speech_callback)
            self.speech_input = None
            self.speech_event = threading.Event()
            rospy.loginfo("Speech mode: listening on /hl/user_speech")
        else:
            rospy.loginfo("Terminal mode: type commands")
        
        try:
            self.setup_llm()
            rospy.loginfo("Ready")
        except Exception as e:
            rospy.logwarn(f"LLM setup failed: {e}, continuing without LLM")
            self.llm = None
            rospy.loginfo("Ready (LLM disabled)")
        
        # Plan id tracking for GUI approvals
        self._plan_counter = 0
        self.current_plan_id = None

        # Periodic real-pose sync (only in real mode)
        try:
            if not dummy_mode:
                self._sync_timer = rospy.Timer(rospy.Duration(0.5), self._sync_real_pose)
                rospy.loginfo("NL: Started periodic real-pose synchronization")
        except Exception as e:
            rospy.logwarn(f"NL: Failed to start sync timer: {e}")

    def _wait_for_spot_services(self, timeout_per=3.0, retries=5):
        """Wait for essential /spot_entrance services to be available.
        Retries a few times to tolerate slow startup when not in dummy mode.
        """
        services = [
            '/spot_entrance/connect',
            '/spot_entrance/power_on',
            '/spot_entrance/stand'
        ]
        for svc in services:
            ok = False
            for _ in range(max(1, int(retries))):
                try:
                    rospy.wait_for_service(svc, timeout=timeout_per)
                    ok = True
                    break
                except Exception:
                    rospy.logwarn(f"Waiting for service {svc} ...")
            if not ok:
                rospy.logwarn(f"Service {svc} not available after waits; continuing (FSM may handle retries)")
    
    def get_actual_robot_position(self):
        """Get actual robot position and convert to grid coordinates."""
        try:
            # In dummy mode, we'll simulate position tracking
            if hasattr(self.spot_fsm, 'dummy_mode') and self.spot_fsm.dummy_mode:
                # For dummy mode, just keep the current position as is
                if self.start_cell is None:
                    self.start_cell = [4.0, 4.0]
                    self.current_cell = [4.0, 4.0]
                    rospy.loginfo("Dummy mode: Starting position at cell (4.0, 4.0)")
                return True
            
            # Get robot pose from the service (returns vision frame position)
            response = self.spot_fsm.get_robot_pose()
            if response and response.success:
                # Extract position from the pose
                pose = response.robot_pose.pose
                x = pose.position.x
                y = pose.position.y
                
                # Convert vision frame coordinates to continuous grid coordinates
                # Robot vision frame: X=forward, Y=left
                # Grid frame: row=Y (up/down), col=X (left/right)
                # When robot moves forward (positive X), it should move up in grid (decreasing row)
                # When robot moves back (negative X), it should move down in grid (increasing row)
                row_cont = -x / 0.3  # Robot X (forward) -> Grid row (up) - NEGATE X
                col_cont = -y / 0.3  # Robot Y (left) -> Grid col (left) - NEGATE Y
                
                # Set start position on first read and map to grid center (4.0, 4.0)
                if self.start_cell is None:
                    # Store the real robot's starting position
                    self.start_cell = [row_cont, col_cont]
                    rospy.loginfo(f"Real robot starts at vision frame ({row_cont:.2f}, {col_cont:.2f}) cells - mapping to grid (4.0, 4.0) cells")
                    # Map real position to grid center (4.0, 4.0) for planning
                    self.current_cell = [4.0, 4.0]
                    # Calculate offset from real position to grid center
                    self.grid_offset_row = 4.0 - row_cont  # How many cells to add to real position
                    self.grid_offset_col = 4.0 - col_cont
                    rospy.loginfo(f"Grid offset: row={self.grid_offset_row:.2f}, col={self.grid_offset_col:.2f}")
                else:
                    # Calculate current position in grid coordinates
                    # Real position + offset = grid position
                    grid_row = row_cont + self.grid_offset_row
                    grid_col = col_cont + self.grid_offset_col
                    self.current_cell = [grid_row, grid_col]
                
                # Convert quaternion to yaw and then to cardinal direction
                from tf.transformations import euler_from_quaternion
                quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                _, _, yaw = euler_from_quaternion(quat)
                
                # Initialize yaw origin once so start heading is N (0)
                if self.yaw_origin is None:
                    self.yaw_origin = yaw
                    rospy.loginfo(f"NL: Yaw origin set to {self.yaw_origin:.2f} (GUI N=0)")
                # Relative yaw (flip sign so GUI E/W match robot): origin - yaw
                import math as _m
                rel = _m.atan2(_m.sin(self.yaw_origin - yaw), _m.cos(self.yaw_origin - yaw))
                
                # Map relative yaw to cardinal per convention: N=0, E=-pi/2, W=+pi/2, S=pi
                def _near(a, b, tol=0.2):
                    return abs(a - b) < tol
                self.current_yaw_rel = rel
                if _near(rel, 0.0):
                    self.current_facing = "N"
                elif _near(rel, -_m.pi/2):
                    self.current_facing = "E"
                elif _near(rel, _m.pi/2):
                    self.current_facing = "W"
                elif _near(abs(rel), _m.pi):  # near pi or -pi
                    self.current_facing = "S"
                
                rospy.loginfo(f"Robot at real position ({row}, {col}) -> grid cell {self.current_cell}, facing {self.current_facing}")
                return True
            else:
                rospy.logwarn("Failed to get robot pose")
                return False
        except Exception as e:
            rospy.logerr(f"Error getting robot position: {e}")
            return False
    
    def on_hl_stop(self, msg):
        """Handle HoloLens stop signal."""
        rospy.loginfo("Received stop signal from HoloLens")
        self.stop_requested = True
    
    def on_approval(self, msg):
        """Handle approval from GUI."""
        try:
            import json
            payload = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"Invalid JSON on /hl/approval: {e}")
            return
        
        if not self.awaiting_approval:
            rospy.logwarn("Plan not awaiting approval, ignoring /hl/approval message")
            return
        
        decision = payload.get("approved")
        if decision:
            rospy.loginfo("Plan approved. Executing...")
            self.awaiting_approval = False
            plan = list(self.pending_plan)
            self.pending_plan = []
            # Execute actions in a separate thread to avoid blocking GUI
            import threading
            execution_thread = threading.Thread(target=self.execute_actions, args=(plan,), daemon=True)
            execution_thread.start()
        else:
            rospy.loginfo("Plan declined by user")
            self.awaiting_approval = False
            self.pending_plan = []
    
    def on_world_change(self, msg):
        """Handle world change from GUI."""
        world_id = msg.data
        rospy.loginfo(f"World change received: {world_id}")
        
        try:
            # Update world configuration
            self.current_world_id = world_id
            self.world_config = self.world_manager.get_world_config(world_id)
            
            if self.world_config:
                world_name, world_desc = self.world_manager.get_world_info(world_id)
                rospy.loginfo(f"Switched to world: {world_name} - {world_desc}")
            else:
                rospy.logwarn(f"Invalid world ID: {world_id}")
                
        except Exception as e:
            rospy.logerr(f"Failed to switch world: {e}")
    
    def setup_llm(self):
        """Setup LLM client."""
        try:
            # Try to use Gemini API
            api_key = os.getenv('GOOGLE_API_KEY')
            if api_key:
                self.llm = genai.Client(api_key=api_key)
                rospy.loginfo("LLM: Using Gemini API")
            else:
                rospy.logwarn("GOOGLE_API_KEY not found, LLM disabled")
                self.llm = None
        except Exception as e:
            rospy.logerr(f"LLM setup failed: {e}")
            self.llm = None
    
    def parse_command(self, command):
        """Parse natural language command using LLM."""
        if not self.llm:
            rospy.logwarn("LLM not available")
            return None
        
        try:
            # Get current robot state
            current_state = self.get_current_state()
            
            # Create world model
            world_model = self.create_world_model()
            
            # Format prompt
            prompt = PROMPT.format(
                current_state_json=json.dumps(current_state),
                world_model_json=json.dumps(world_model),
                command=command
            )
            
            rospy.loginfo("Sending command to LLM...")
            # Publish GUI feedback so it shows up in NL panel
            try:
                self.pub_feedback.publish(String(data="[exec] Sending command to LLM..."))
            except Exception:
                pass
            recorder.publish_event('start_llm_processing')
            
            # Generate response
            response = self.llm.models.generate_content(
                model='gemini-2.5-pro',
                contents=prompt
            )
            actions_text = response.text.strip()
            
            recorder.publish_event('stop_llm_processing')
            
            # Parse actions
            actions = []
            for line in actions_text.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    actions.append(line)
            
            rospy.loginfo(f"LLM generated {len(actions)} actions")
            try:
                self.pub_feedback.publish(String(data=f"[exec] LLM generated {len(actions)} actions"))
            except Exception:
                pass
            return actions
            
        except Exception as e:
            rospy.logerr(f"LLM parsing failed: {e}")
            recorder.publish_event('stop_llm_processing')
            return None
    
    def get_current_state(self):
        """Get current robot state in grid format."""
        return {
            "robot_cell": self.current_cell,
            "facing": self.current_facing,
            "robot_state": str(self.spot_fsm.current_state),
            "has_object": False,  # TODO: Track object state
            "carried_object": None,
            "accumulated_offset_row": self.accumulated_offset_row,
            "accumulated_offset_col": self.accumulated_offset_col
        }
    
    def create_world_model(self):
        """Create world model from current world configuration."""
        world_model = {
            "waypoints": self.world_config.get("waypoints", {}),
            "zones": self.world_config.get("zones", {}),
            "synonyms": self.world_config.get("synonyms", {}),
            "wall_cells": self.world_config.get("wall_cells", [])
        }
        return world_model

    def _nl_from_actions_openai(self, actions, command):
        """Use OpenAI small model to convert action steps into a natural-language plan.
        Requires OPENAI_API_KEY in environment. Returns None on failure.
        """
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                return None
            import json as _json
            import requests
            # Compose a concise instruction
            sys_prompt = (
                "You will receive a list of robot actions (grid steps) and the original user command. "
                "Write a short, clear natural-language plan describing what the robot will do, in 2-6 lines. "
                "Avoid repeating raw parameters; focus on intent (move, pick, drop, rotate)."
            )
            user_content = {
                "command": command,
                "actions": actions,
            }
            payload = {
                "model": "gpt-4.1-mini",
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": _json.dumps(user_content)}
                ],
                "temperature": 0.2,
                "max_tokens": 400
            }
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                data=_json.dumps(payload),
                timeout=15
            )
            if resp.status_code != 200:
                rospy.logwarn(f"OpenAI summarization failed: {resp.status_code} {resp.text[:200]}")
                return None
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return text or None
        except Exception as e:
            rospy.logwarn(f"OpenAI NL plan generation error: {e}")
            return None
    
    def execute_actions(self, actions):
        """Execute FSM actions."""
        if not actions:
            return False
        
        for action in actions:
            # Check for stop signal before each action
            if self.stop_requested:
                rospy.loginfo("Stop signal received - finishing current task and stopping plan execution")
                self.stop_requested = False
                return False
            
            try:
                rospy.loginfo(f"Executing: {action}")
                # Publish feedback for simulation
                self.pub_feedback.publish(String(data=f"[exec] Step: {action}"))
                
                # Parse action name (remove parameters)
                if '=' in action:
                    action_name = action.split()[0]  # Get first word (action name)
                else:
                    action_name = action.strip()
                
                if '=' in action:
                    # Parse action with parameters
                    if ',' in action:
                        parts = action.split(',')
                        action_name = parts[0].strip()
                        param_parts = parts[1:]
                    else:
                        # Handle space-separated format: "move_to_cell row=2 col=3"
                        import re
                        words = re.split(r'\s+(?=\w+=)', action)
                        action_name = words[0].split()[0]  # Get first word only
                        param_parts = words[1:] if len(words) > 1 else []
                    
                    params = {}
                    for part in param_parts:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            
                            try:
                                params[key] = float(value) if '.' in value else int(value)
                            except ValueError:
                                params[key] = value
                    
                    # Execute the action and track position
                    if action_name == 'start_drop_off':
                        params['_is_drop_off'] = True
                    
                    # Handle grid-based actions
                    if action_name == 'move_to_cell':
                        # Update current cell based on movement
                        target_row = float(params.get('row', self.current_cell[0]))
                        target_col = float(params.get('col', self.current_cell[1]))
                        
                        # Calculate movement with offset compensation
                        # Start from the actual expected position (current_cell + accumulated offsets)
                        # Note: accumulated offsets are in grid cells, so we add them directly
                        actual_current_row = self.current_cell[0] + self.accumulated_offset_row
                        actual_current_col = self.current_cell[1] + self.accumulated_offset_col
                        
                        # Compute relative body-frame movement from actual current position
                        cell_size = 0.3
                        drow = target_row - actual_current_row
                        dcol = target_col - actual_current_col
                        # Robot X (forward) maps to negative row change in grid
                        # Robot Y (left) maps to negative col change in grid
                        dx = (-drow) * cell_size  # forward (robot X) = -row change
                        dy = (-dcol) * cell_size  # left (robot Y) = -col change
                        
                        rospy.loginfo(f"NL: move_to_cell with offset compensation:")
                        rospy.loginfo(f"  Expected position: ({self.current_cell[0]:.2f},{self.current_cell[1]:.2f}) cells")
                        rospy.loginfo(f"  Actual position: ({actual_current_row:.2f},{actual_current_col:.2f}) cells")
                        rospy.loginfo(f"  Target: ({target_row:.2f},{target_col:.2f}) cells => dx={dx:.3f}m, dy={dy:.3f}m")
                        rospy.loginfo(f"  Accumulated offsets: row={self.accumulated_offset_row:.2f} cells ({self.accumulated_offset_row*30:.1f}cm), col={self.accumulated_offset_col:.2f} cells ({self.accumulated_offset_col*30:.1f}cm)")
                        
                        # Send relative move in body frame to FSM
                        self.spot_fsm.send('start_moving', x=dx, y=dy, yaw=0.0, frame='body')
                        
                        # Store expected position for offset calculation
                        self.last_expected_cell = [target_row, target_col]
                        
                        # Update position after sending (this will be corrected by position verification)
                        self.current_cell = [target_row, target_col]
                        rospy.loginfo(f"Updated expected position to cell ({target_row}, {target_col})")
                        
                    elif action_name == 'rotate_to':
                        # Update facing direction
                        direction = params.get('direction', self.current_facing)
                        if direction in ["N", "S", "E", "W"]:
                            self.current_facing = direction
                            rospy.loginfo(f"Rotated to face {direction}")
                        else:
                            rospy.logwarn(f"Invalid direction: {direction}")
                        
                        # Ensure robot is in stand state before rotating
                        current_state = str(self.spot_fsm.current_state)
                        if current_state != "stand":
                            rospy.loginfo(f"Robot in {current_state} state, waiting for stand state before rotating...")
                            # Wait longer for state transition to complete
                            rospy.sleep(3.0)  # Give more time for movement to complete
                        
                        # Update internal yaw (relative-to-start convention: N=0, E=-pi/2, W=+pi/2, S=pi)
                        try:
                            if direction == "N":
                                self.current_yaw_rel = 0.0
                            elif direction == "E":
                                self.current_yaw_rel = -math.pi / 2.0
                            elif direction == "W":
                                self.current_yaw_rel = math.pi / 2.0
                            elif direction == "S":
                                self.current_yaw_rel = math.pi
                        except Exception:
                            pass

                        # Send to FSM
                        self.spot_fsm.send('start_rotating', **params)
                    else:
                        # For other actions, send directly
                        self.spot_fsm.send(action_name, **params)
                else:
                    # Handle actions without parameters
                    action_name = action.strip()
                    if action_name == 'start_drop_off':
                        self.spot_fsm.send(action_name, _is_drop_off=True)
                    else:
                        self.spot_fsm.send(action_name)
                
                # Publish completion feedback
                self.pub_feedback.publish(String(data=f"[exec] ✓ {action}"))
                
                # Publish position update
                self.publish_position()
                
                # Check for stop signal after action completion
                if self.stop_requested:
                    rospy.loginfo("Stop signal received after action completion - stopping plan execution")
                    self.stop_requested = False
                    return False
                
                # Wait for FSM to complete the action before proceeding
                rospy.sleep(2.0)  # Give FSM time to complete the action
                
                # Verify position after movement and update offsets
                if action_name == 'move_to_cell':
                    self._verify_and_correct_position()
                    
            except Exception as e:
                rospy.logerr(f"Action failed '{action}': {e}")
                return False
        
        # Publish plan completion signal
        self.pub_feedback.publish(String(data="[exec] Plan completed successfully"))
        return True
    
    def publish_position(self):
        """Publish current robot position in grid format."""
        # Include numeric yaw (degrees) relative to start so GUI can render arrow consistently
        yaw_deg = float(self.current_yaw_rel) * 180.0 / math.pi
        # Provide continuous meter-level position normalized to the (4,4) origin when available
        x_m = None
        y_m = None
        try:
            if self.grid_offset_col is not None and self.grid_offset_row is not None:
                # Set meter coordinates to show robot at center of cell (4,4) in the grid
                # (1.35, 1.35) meters corresponds to center of cell (4,4) in the GUI
                x_m = 1.35  # (4 + 0.5) cells * 0.3m = 1.35m
                y_m = 1.35  # (4 + 0.5) cells * 0.3m = 1.35m
        except Exception:
            pass
        
        # Calculate GUI position (current_cell is already at (4.0,4.0))
        # This makes the starting position appear as (4.0,4.0) and all movements are relative to that
        gui_row = float(self.current_cell[0])
        gui_col = float(self.current_cell[1])
        
        position_data = {
            'row': gui_row,
            'col': gui_col,
            'facing': self.current_facing,
            'yaw_deg': yaw_deg,
            'x_m': x_m,
            'y_m': y_m
        }
        self.pub_position.publish(String(data=json.dumps(position_data)))

    def _verify_and_correct_position(self):
        """Verify robot position after movement and update accumulated offsets."""
        try:
            # Only verify in real mode (not dummy mode)
            if hasattr(self.spot_fsm, 'dummy_mode') and self.spot_fsm.dummy_mode:
                return
            
            # Get actual robot position
            response = self.spot_fsm.get_robot_pose()
            if not response or not getattr(response, 'success', False):
                rospy.logwarn("Could not get robot pose for position verification")
                return
            
            pose = response.robot_pose.pose
            x = pose.position.x
            y = pose.position.y
            
            # Convert to continuous grid coordinates
            # Robot vision frame: X=forward, Y=left
            # Grid frame: row=Y (up/down), col=X (left/right)
            # So: row = -Y (negative Y in robot frame = positive row in grid)
            #     col = -X (negative X in robot frame = positive col in grid)
            row_cont = -x / 0.3  # Robot X (forward) -> Grid row (up) - NEGATE X
            col_cont = -y / 0.3  # Robot Y (left) -> Grid col (left) - NEGATE Y
            
            # Calculate actual position in grid coordinates
            if self.grid_offset_row is not None and self.grid_offset_col is not None:
                # Convert real position to grid coordinates
                actual_row = row_cont + self.grid_offset_row
                actual_col = col_cont + self.grid_offset_col
                
                # Calculate the offset from expected position
                expected_row = self.last_expected_cell[0]
                expected_col = self.last_expected_cell[1]
                
                offset_row = actual_row - expected_row
                offset_col = actual_col - expected_col
                
                # Update accumulated offsets
                self.accumulated_offset_row += offset_row
                self.accumulated_offset_col += offset_col
                
                rospy.loginfo(f"Position verification:")
                rospy.loginfo(f"  Expected: ({expected_row:.2f}, {expected_col:.2f}) cells")
                rospy.loginfo(f"  Actual: ({actual_row:.2f}, {actual_col:.2f}) cells")
                rospy.loginfo(f"  This offset: ({offset_row:.2f}, {offset_col:.2f}) cells = ({offset_row*30:.1f}, {offset_col*30:.1f}) cm")
                rospy.loginfo(f"  Accumulated offset: ({self.accumulated_offset_row:.2f}, {self.accumulated_offset_col:.2f}) cells = ({self.accumulated_offset_row*30:.1f}, {self.accumulated_offset_col*30:.1f}) cm")
                
                # If offset is significant, log a warning
                if abs(offset_row) > 0.1 or abs(offset_col) > 0.1:
                    rospy.logwarn(f"Significant position offset detected: ({offset_row:.2f}, {offset_col:.2f}) cells = ({offset_row*30:.1f}, {offset_col*30:.1f}) cm")
                
        except Exception as e:
            rospy.logwarn(f"Position verification failed: {e}")
    
    def _sync_real_pose(self, _event):
        """Periodically sync GUI position with the real robot pose (vision frame -> grid)."""
        try:
            resp = self.spot_fsm.get_robot_pose()
            if not resp or not getattr(resp, 'success', False):
                return
            pose = resp.robot_pose.pose
            x = pose.position.x
            y = pose.position.y
            # Compute continuous grid coordinates (floats)
            # Robot vision frame: X=forward, Y=left
            # Grid frame: row=Y (up/down), col=X (left/right)
            row_cont = -x / 0.3  # Robot X (forward) -> Grid row (up) - NEGATE X
            col_cont = -y / 0.3  # Robot Y (left) -> Grid col (left) - NEGATE Y

            # Initialize grid offset on first read
            if self.grid_offset_row is None or self.grid_offset_col is None:
                # Store the real robot's starting position for reference
                self.grid_offset_row = 4.0 - row_cont  # Offset to map to grid center (4.0, 4.0)
                self.grid_offset_col = 4.0 - col_cont
                rospy.loginfo(f"NL: Robot starting at real position ({row_cont:.2f}, {col_cont:.2f}) - mapping to grid (4.0, 4.0)")
            
            # Convert real position to grid coordinates for GUI display
            grid_row = row_cont + self.grid_offset_row
            grid_col = col_cont + self.grid_offset_col
            self.current_cell = [grid_row, grid_col]

            # Update yaw relative to origin
            from tf.transformations import euler_from_quaternion
            quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            _, _, yaw = euler_from_quaternion(quat)
            if self.yaw_origin is None:
                # Normalize startup heading to North (0)
                self.yaw_origin = yaw
                self.current_yaw_rel = 0.0
                self.current_facing = "N"
            import math as _m
            self.current_yaw_rel = _m.atan2(_m.sin(self.yaw_origin - yaw), _m.cos(self.yaw_origin - yaw))

            # Update facing for cardinal display
            def _near(a, b, tol=0.2):
                return abs(a - b) < tol
            if _near(self.current_yaw_rel, 0.0):
                self.current_facing = "N"
            elif _near(self.current_yaw_rel, -math.pi/2):
                self.current_facing = "E"
            elif _near(self.current_yaw_rel, math.pi/2):
                self.current_facing = "W"
            elif _near(abs(self.current_yaw_rel), math.pi):
                self.current_facing = "S"

            # Publish updated position for GUI
            self.publish_position()
        except Exception:
            pass
    
    def process_command(self, command, auto_execute=False, require_approval=False):
        """Process natural language command."""
        if self.stop_requested:
            rospy.loginfo("Stop signal received - ignoring new command until stop is cleared")
            return False
        
        self.stop_requested = False
            
        current_state = self.spot_fsm.current_state.name
        print(f"\nProcessing: {command}")
        print(f"Current robot state: {current_state}")
        
        # Publish timing event for HoloLens input received
        recorder.publish_event('received_hololens_input')
        
        # Process with LLM
        actions = self.parse_command(command)
        if not actions:
            print("Parse failed - LLM is not available or command could not be understood")
            self.pub_feedback.publish(String(data="[exec] Command parsing failed - LLM unavailable or command unclear"))
            return
        
        print(f"\nLLM Generated Plan:")
        for i, action in enumerate(actions, 1):
            print(f"  {i}. {action}")
        
        # Prefer OpenAI small model to convert steps into NL; fallback to simple list
        interpretation_text = None
        try:
            interpretation_text = self._nl_from_actions_openai(actions, command)
        except Exception:
            interpretation_text = None
        if not interpretation_text:
            # Fallback formatting if OpenAI not available
            interpretation_text = f"I understand you want me to: {command}\n\nI'll execute this plan:"
            for i, action in enumerate(actions, 1):
                interpretation_text += f"\n{i}. {action}"
        # Publish interpretation for GUI feedback
        self.pub_interpretation.publish(String(data=interpretation_text))
        
        # Wait for approval when requested (GUI) or in speech mode
        if require_approval or (self.use_speech and not auto_execute):
            # Assign and publish plan id and actions for GUI
            try:
                self._plan_counter += 1
                self.current_plan_id = self._plan_counter
                payload = {"plan_id": self.current_plan_id, "actions": actions}
                self.pub_plan.publish(String(data=json.dumps(payload)))
            except Exception:
                pass
            rospy.loginfo("Speech mode: waiting for approval")
            self.pending_plan = actions
            self.awaiting_approval = True
            try:
                self.pub_feedback.publish(String(data="[approval] Plan ready; awaiting approval in GUI"))
            except Exception:
                pass
        else:
            # Auto-exec for GUI or prompt in terminal
            if auto_execute:
                try:
                    self.pub_feedback.publish(String(data="[exec] Executing plan from GUI command"))
                except Exception:
                    pass
                self.execute_actions(actions)
            else:
                confirm = input("\nExecute this plan? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    print("Executing...")
                    self.execute_actions(actions)
                else:
                    print("Cancelled - try a new command")
    
    def speech_callback(self, msg):
        """Handle speech input from HoloLens."""
        rospy.loginfo(f"Received speech: {msg.data}")
        self.speech_input = msg.data
        self.speech_event.set()
    
    def on_user_speech(self, msg):
        """Handle GUI/HoloLens commands: plan, then wait for GUI approval."""
        try:
            text = (msg.data or '').strip()
            if not text:
                return
            self.pub_feedback.publish(String(data=f"[plan] GUI command received: {text}"))
            # For GUI flow: require approval before executing
            self.process_command(text, auto_execute=False, require_approval=True)
        except Exception as e:
            rospy.logwarn(f"GUI speech handler error: {e}")
    
    def run(self):
        """Main loop."""
        print("\nNatural Language Control Ready! (Grid-Based)")
        if self.use_speech:
            print("Speech mode: listening on /hl/user_speech")
        else:
            print("Terminal mode: type commands or 'quit' to exit")
        print()
        
        while not rospy.is_shutdown():
            try:
                if self.use_speech:
                    # Wait for speech input
                    if self.speech_event.wait(timeout=1.0):
                        command = self.speech_input
                        self.speech_input = None
                        self.speech_event.clear()
                        
                        if command.lower() in ['quit', 'exit', 'stop']:
                            break
                        
                        self.process_command(command)
                else:
                    # Terminal input
                    command = input("Enter command: ").strip()
                    
                    if command.lower() in ['quit', 'exit', 'stop']:
                        break
                    
                    if command:
                        self.process_command(command)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                rospy.logerr(f"Error in main loop: {e}")
                rospy.sleep(1.0)
        
        rospy.loginfo("Shutting down...")

def main():
    """Main function."""
    import argparse
    
    # Filter out ROS launch arguments
    filtered_args = []
    for arg in sys.argv[1:]:
        if not arg.startswith('__') and not arg.startswith('_log:='):
            filtered_args.append(arg)
    
    parser = argparse.ArgumentParser(description='Natural Language Control for Spot Robot (Grid-Based)')
    parser.add_argument('--speech', action='store_true', help='Enable speech mode')
    args = parser.parse_args(filtered_args)
    
    try:
        nl_control = NaturalLanguageControl(use_speech=args.speech)
        nl_control.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")

if __name__ == '__main__':
    main()
