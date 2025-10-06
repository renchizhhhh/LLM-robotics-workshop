#!/usr/bin/env python3
import rospy
import sys
import os
from google import genai
import threading
import time
import json
import math
from std_msgs.msg import String, Empty

# Add the FSM to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'spot_hololens_llm_interface'))
from finite_state_machine import SpotStateMachine

# Import timing utilities
from timing_utils import recorder

# Import world manager
from world_manager import WorldManager

# Universal Spot Path Planner Prompt v2
PROMPT = """You are a path planning and action sequencing system for a Boston Dynamics Spot robot.

CRITICAL: If the robot state is "Stand", NEVER use stand_up as the first action. The robot is already standing!

INPUTS
- CURRENT_STATE: JSON with robot pose and mode in the vision frame.
- WORLD_MODEL: JSON with named waypoints, semantic zones and per-location orientation or drop rules.
- TASK: natural language command with destinations in order.

ACTION PRIMITIVES
Allowed actions, one per line, no quotes, no numbering:
- stand_up
- sit_down
- start_moving x=<float> y=<float> yaw=<float> frame=body
- get_image image_source=<camera_name>
- get_initial_pose
- start_automated_grasp object_type=<exact_object_name_from_user>
- start_move_arm_pose x=<float> y=<float> z=<float> qw=<float> qx=<float> qy=<float> qz=<float> duration=<float> open_gripper=<true|false>
- start_arm_command command_type=<open|close|stow|carry>

HARD RULES
1. Execute destinations in the exact order given by the user.
2. Output only the action list. One action per line. No comments.
3. CRITICAL: If robot state is "Stand", NEVER use stand_up. Only use stand_up if robot state is "Sit" or "Powered off".
4. Use relative body-frame moves. Each start_moving is relative to the current robot body frame.
5. One degree of freedom per move line. Either x or y or yaw is non-zero.
6. NEVER start with stand_up if robot is already standing.
6. For navigation to a target T in vision frame:
   6.1 First resolve T to numeric coordinates and a yaw_hint using WORLD_MODEL and synonyms.
   6.2 If a yaw is required at T, add a separate yaw-only rotation to that yaw.
   6.3 Move along body x and body y in separate steps to reach T. Keep each step axis-aligned.
7. For pick-up targets, finish aligned to the pick yaw defined for that target. Default yaw 0.0 if unspecified.
8. For drop-off targets, finish aligned to the drop yaw defined for that target. Default yaw 1.57 if unspecified.
9. Use the exact user object name in start_automated_grasp object_type.
10. Standard drop-off procedure after arriving and setting drop yaw:
    - start_move_arm_pose x=0.8 y=0.0 z=0.3 qw=0.7071 qx=0.7071 qy=0.0 qz=0.0 duration=1.0 open_gripper=false
    - start_arm_command command_type=open
    - start_arm_command command_type=close
    - start_arm_command command_type=stow
11. Keep a safe final approach. Stop body motion with at least 0.30 m clearance before grasping when possible.
12. If a referenced location or object cannot be resolved, output a single line: FAIL unresolved=<what>

PLANNING STEPS
A. Parse TASK into an ordered list of subgoals with types: pick, place, visit, look.
B. Resolve each natural language location via WORLD_MODEL.names and WORLD_MODEL.synonyms. Use canonical names.
C. For each subgoal:
   C1. CRITICAL: If robot state is "Stand", NEVER use stand_up. Only use stand_up if robot state is "Sit" or "Powered off".
   C2. For PICK tasks: 
       - ALWAYS navigate to the pickup location FIRST using start_moving
       - Use WORLD_MODEL to find pickup location coordinates
       - Then use start_automated_grasp with the exact object name
       - NEVER grasp without first navigating to the pickup location
   C3. For navigation: Rotate in yaw-only steps first if orientation helps axis-aligned approach.
   C4. Plan axis-aligned moves in body frame toward the target. Use relative steps.
   C5. On arrival, set required yaw using a yaw-only step.
   C6. If pick: call start_automated_grasp with exact object_type.
   C7. If drop: run the standard drop-off procedure.
D. Do not repeat locations already satisfied unless the order requires a revisit.

OUTPUT FORMAT
Only the actions. One per line.

CRITICAL RULES:
- If robot state is "Stand", NEVER use stand_up
- Always navigate to pickup location FIRST, then grasp
- Start with navigation, not stand_up

VARIABLES TO FILL

CURRENT_STATE = {current_state_json}

WORLD_MODEL = {world_model_json}

TASK = {command}"""

class NaturalLanguageControl:
    def __init__(self, use_speech=False, world_id=None):
        rospy.init_node('nl_control', anonymous=True)
        self.use_speech = use_speech
        
        # Get world_id from ROS parameter if not provided
        if world_id is None:
            world_id = rospy.get_param('~world_id', '1')
        
        # Position tracking - will get actual position from robot
        self.current_position = [0.0, 0.0, 0.0]  # [x, y, yaw] relative to start position
        self.start_position = None  # Will be set on first position read
        
        # Initialize world manager
        self.world_manager = WorldManager()
        
        # Initialize LLM as None first to prevent AttributeError
        self.llm = None
        
        # World model configuration - can be loaded from file or set programmatically
        rospy.loginfo("Loading world model...")
        try:
            if world_id:
                self.world_model = self.world_manager.get_world_config(world_id)
                if self.world_model:
                    world_name, world_desc = self.world_manager.get_world_info(world_id)
                    rospy.loginfo(f"Loaded world: {world_name} - {world_desc}")
                else:
                    rospy.logwarn(f"Invalid world ID: {world_id}, using default")
                    self.world_model = self.load_world_model()
            else:
                self.world_model = self.load_world_model()
            rospy.loginfo("World model loaded successfully")
            
            # Initialize prompt template with current world model
            self._update_prompt_template()
        except Exception as e:
            rospy.logerr(f"Error loading world model: {e}")
            self.world_model = self.load_world_model()
            self._update_prompt_template()
        
        # Publisher for position updates
        self.pub_position = rospy.Publisher('/nl_control/robot_position', String, queue_size=1)
        
        # Publisher for interpretation feedback
        self.pub_interpretation = rospy.Publisher('/llm_int/interpretation', String, queue_size=10)
        
        # HoloLens stop signal handling
        self.stop_requested = False
        self.sub_hl_stop = rospy.Subscriber('/hl/stop', Empty, self.on_hl_stop)
        
        # World change handling
        self.sub_world_change = rospy.Subscriber('/gui/world_change', String, self.on_world_change)
        
        # Approval handling
        self.sub_approval = rospy.Subscriber('/hl/approval', String, self.on_approval)
        self.pending_plan = []
        self.awaiting_approval = False
        
        # Check if dummy mode is set globally (try multiple locations)
        dummy_mode = rospy.get_param('/spot_fsm/dummy_mode', 
                     rospy.get_param('/spot_entrance/dummy_mode',
                     rospy.get_param('dummy_mode', True)))
        
        rospy.loginfo(f"Natural Language Control starting in {'DUMMY' if dummy_mode else 'REAL'} mode")
        self.spot_fsm = SpotStateMachine(dummy_mode=dummy_mode)
        
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
        
        # Stand up robot to get it ready for commands (only if not already standing)
        current_state = str(self.spot_fsm.current_state)
        if current_state not in ["Stand", "Moving", "Grasping"]:
            try:
                rospy.loginfo("Sending stand_up command to FSM...")
                recorder.publish_event('start_stand_up')
                self.spot_fsm.send("stand_up")
                recorder.publish_event('stop_stand_up')
                rospy.loginfo("Stand_up command completed successfully")
                
                rospy.loginfo(f"Current FSM state after stand_up: {self.spot_fsm.current_state}")
            except Exception as e:
                rospy.logerr(f"Stand_up command failed: {e}")
                raise
        else:
            rospy.loginfo(f"Robot already in {current_state} state, skipping stand_up command")
        
        rospy.loginfo("Robot ready for commands")
        
        # Publisher for simulation feedback
        self.pub_feedback = rospy.Publisher('/spot/execution_feedback', String, queue_size=20)
        
        if use_speech:
            self.speech_sub = rospy.Subscriber('/hl/user_speech', String, self.speech_callback)
            self.speech_input = None
            self.speech_event = threading.Event()
            rospy.loginfo("Speech mode: listening on /hl/user_speech")
        else:
            rospy.loginfo("Terminal mode: type commands")
        
        rospy.loginfo("About to setup LLM...")
        try:
            self.setup_llm()
            rospy.loginfo("Ready")
        except Exception as e:
            rospy.logwarn(f"LLM setup failed: {e}, continuing without LLM")
            self.llm = None
            rospy.loginfo("Ready (LLM disabled)")
    
    def load_world_model(self):
        """Load world model configuration using WorldManager."""
        # Use the same world as the GUI if available, otherwise default to world 1
        try:
            if hasattr(self, 'world_manager') and self.world_manager:
                # Use the same world configuration as the GUI
                return self.world_manager.get_world_config("1")  # Default to world 1
            else:
                # Fallback to simple world if WorldManager not available
                return {
                    "waypoints": {
                        "Beverages": {"x": 2.0, "y": -1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None}
                    },
                    "zones": {
                        "DeliveryArea": {
                            "centroid": {"x": 2.0, "y": 0.5, "z": 0.0},
                            "yaw_hint": 1.57,
                            "tags": ["delivery", "drop-off", "destination"]
                        }
                    },
                    "synonyms": {
                        "beverages": "Beverages",
                        "drinks": "Beverages",
                        "delivery area": "DeliveryArea",
                        "drop-off area": "DeliveryArea"
                    }
                }
        except Exception as e:
            rospy.logwarn(f"Failed to load world model from WorldManager: {e}")
            # Fallback to simple world
            return {
                "waypoints": {
                    "Beverages": {"x": 2.0, "y": -1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None}
                },
                "zones": {
                    "DeliveryArea": {
                        "centroid": {"x": 2.0, "y": 0.5, "z": 0.0},
                        "yaw_hint": 1.57,
                        "tags": ["delivery", "drop-off", "destination"]
                    }
                },
                "synonyms": {
                    "beverages": "Beverages",
                    "drinks": "Beverages",
                    "delivery area": "DeliveryArea",
                    "drop-off area": "DeliveryArea"
                }
            }
    
    def update_world_model(self, new_world_model):
        """Update the world model configuration."""
        self.world_model = new_world_model
        rospy.loginfo("World model updated")
    
    def switch_world(self, world_id):
        """Switch to a different world configuration."""
        try:
            if world_id in self.world_manager.worlds:
                self.world_model = self.world_manager.get_world_config(world_id)
                world_name, world_desc = self.world_manager.get_world_info(world_id)
                rospy.loginfo(f"Switched to world: {world_name} - {world_desc}")
                return True
            else:
                rospy.logwarn(f"Invalid world ID: {world_id}")
                return False
        except Exception as e:
            rospy.logerr(f"Failed to switch world: {e}")
            return False
    
    def load_world_model_from_file(self, filepath):
        """Load world model from JSON file."""
        try:
            with open(filepath, 'r') as f:
                self.world_model = json.load(f)
            rospy.loginfo(f"World model loaded from {filepath}")
        except Exception as e:
            rospy.logerr(f"Failed to load world model from {filepath}: {e}")
    
        
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
        
    def get_actual_robot_position(self):
        """Get actual robot position and convert to body frame coordinates."""
        try:
            # In dummy mode, we'll simulate position tracking
            if hasattr(self.spot_fsm, 'dummy_mode') and self.spot_fsm.dummy_mode:
                # For dummy mode, just keep the current position as is
                if self.start_position is None:
                    self.start_position = [0.0, 0.0, 0.0]
                    self.current_position = [0.0, 0.0, 0.0]
                    rospy.loginfo("Dummy mode: Starting position at (0, 0, 0)")
                return True
            
            # Get robot pose from the service (returns vision frame position)
            response = self.spot_fsm.get_robot_pose()
            if response and response.success:
                # Extract position from the pose
                pose = response.robot_pose.pose
                x = pose.position.x
                y = pose.position.y
                z = pose.position.z
                
                # Convert quaternion to yaw
                from tf.transformations import euler_from_quaternion
                quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                _, _, yaw = euler_from_quaternion(quat)
                
                # Set start position on first read
                if self.start_position is None:
                    self.start_position = [x, y, yaw]
                    rospy.loginfo(f"Set start position (vision frame): {self.start_position}")
                    self.current_position = [0.0, 0.0, 0.0]  # Start at origin
                else:
                    # Calculate relative position from start in vision frame
                    vision_relative = [
                        x - self.start_position[0],
                        y - self.start_position[1], 
                        yaw - self.start_position[2]
                    ]
                    
                    # Transform vision frame relative position to body frame
                    # The service gives us vision_tform_body (body pose in vision frame)
                    # To get relative motion in body frame, we need to "undo" the rotation
                    # that vision frame has relative to the starting body orientation
                    
                    # Use the starting yaw to transform coordinates back to body frame
                    start_yaw_offset = self.start_position[2]  # Initial robot orientation in vision frame
                    
                    # Rotate vision frame coordinates to body frame using inverse rotation
                    cos_offset = math.cos(-start_yaw_offset)  # Negative for inverse rotation
                    sin_offset = math.sin(-start_yaw_offset)
                    
                    # Apply rotation matrix to transform vision coordinates to body coordinates
                    self.current_position = [
                        cos_offset * vision_relative[0] - sin_offset * vision_relative[1],  # body frame X (forward)
                        sin_offset * vision_relative[0] + cos_offset * vision_relative[1],  # body frame Y (left)
                        vision_relative[2]  # yaw rotation is the same
                    ]
                
                rospy.loginfo(f"Current position relative to start (body frame): {self.current_position}")
                return True
            else:
                rospy.logwarn(f"Failed to get robot pose: {response.message}")
                return False
        except Exception as e:
            rospy.logerr(f"Error getting robot position: {e}")
            return False
        
    def setup_llm(self):
        """Setup Gemini API for planning and OpenAI for interpretation."""
        try:
            # Setup Gemini for plan generation
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                rospy.logwarn("No GOOGLE_API_KEY environment variable found - LLM disabled")
                self.llm = None
                return
            
            rospy.loginfo("Setting up Gemini LLM...")
            rospy.loginfo(f"API key found: {api_key[:10]}...")
            self.llm = genai.Client(api_key=api_key)
            rospy.loginfo("Gemini LLM ready")
            
            # Setup OpenAI for plan interpretation (like orchestrator)
            import openai
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if openai_api_key:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                rospy.loginfo("OpenAI client ready for plan interpretation")
            else:
                rospy.logwarn("No OPENAI_API_KEY - plan interpretation will use fallback")
                self.openai_client = None
                
        except Exception as e:
            rospy.logerr(f"Failed to setup LLM client: {e}")
            rospy.logerr(f"LLM will be disabled. Error details: {str(e)}")
            self.llm = None
            raise
    
    def speech_callback(self, msg):
        """Callback for speech input."""
        self.speech_input = msg.data
        self.speech_event.set()
        rospy.loginfo(f"Speech: {msg.data}")
        
        # Note: received_hololens_input will be published in process_command to ensure correct order
    
    def parse_command(self, command):
        """Parse natural language command using LLM."""
        if not self.llm:
            rospy.logwarn("LLM is not available. Cannot parse natural language commands.")
            rospy.logwarn("Please check your GOOGLE_API_KEY environment variable.")
            return None
        
        # Publish LLM processing start
        recorder.publish_event('start_llm_processing')
        
        try:
            # Get current robot state
            current_state = self.spot_fsm.current_state.name
            
            # Get actual robot position from vision frame
            if not self.get_actual_robot_position():
                rospy.logwarn("Using last known position")
            
            # Create current state JSON
            current_state_json = {
                "robot_state": current_state,
                "standing": current_state in ["Stand", "Moving", "Grasping"],
                "pose_vision": {
                    "x": self.current_position[0],
                    "y": self.current_position[1], 
                    "yaw": self.current_position[2]
                },
                "arm": "stowed",  # Default - could be enhanced to track actual arm state
                "held_object": None  # Default - could be enhanced to track held objects
            }
            
            # Format prompt with current state, world model, and command
            rospy.loginfo(f"[LLM DEBUG] Sending position to LLM: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f}, {self.current_position[2]:.2f})")
            rospy.loginfo(f"[LLM DEBUG] Robot state: {current_state}")
            rospy.loginfo(f"[LLM DEBUG] Current state JSON: {json.dumps(current_state_json, indent=2)}")
            
            # Publish initial position to GUI
            self.publish_position_update()
            formatted_prompt = PROMPT.format(
                current_state_json=json.dumps(current_state_json, indent=2),
                world_model_json=json.dumps(self.world_model, indent=2),
                command=command
            )
            
            response = self.llm.models.generate_content(
                model="gemini-2.5-pro",
                contents=formatted_prompt
            )
            
            result = response.text.strip()
            actions = [line.strip() for line in result.split('\n') if line.strip()]
            
            # Publish LLM processing complete
            recorder.publish_event('stop_llm_processing')
            
            return actions
            
        except Exception as e:
            rospy.logerr(f"LLM failed: {e}")
            recorder.publish_event('stop_llm_processing')  # Make sure to stop timing even on error
            return None
    
    def execute_actions(self, actions):
        """Execute FSM actions."""
        if not actions:
            return False
        
        for action in actions:
            # Check for stop signal before each action
            if self.stop_requested:
                rospy.loginfo("Stop signal received - finishing current task and stopping plan execution")
                # Reset stop flag for next plan
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
                        # Handle space-separated format: "start_moving x=0 y=0 yaw=1.5708 frame=body"
                        words = action.split()
                        action_name = words[0]
                        param_parts = words[1:]
                    
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
                    
                    # Execute the action and track position in dummy mode
                    self.spot_fsm.send(action_name, **params)
                    
                    # Update position tracking for dummy mode
                    if hasattr(self.spot_fsm, 'dummy_mode') and self.spot_fsm.dummy_mode:
                        if action_name == 'start_moving':
                            # Update current position based on movement
                            move_x = params.get('x', 0.0)
                            move_y = params.get('y', 0.0)
                            move_yaw = params.get('yaw', 0.0)
                            
                            # Transform body frame movement to world frame using 2D rotation
                            # Body frame: x=forward, y=left
                            # World frame: x=forward, y=left
                            # At yaw=0: body x -> world x, body y -> world y
                            # At yaw=π/2: body x -> world y, body y -> world -x
                            # Standard 2D rotation matrix: [cos -sin; sin cos]
                            cos_yaw = math.cos(self.current_position[2])
                            sin_yaw = math.sin(self.current_position[2])
                            
                            # Apply rotation matrix to transform body movement to world frame
                            world_dx = move_x * cos_yaw - move_y * sin_yaw
                            world_dy = move_x * sin_yaw + move_y * cos_yaw
                            
                            self.current_position[0] += world_dx
                            self.current_position[1] += world_dy
                            self.current_position[2] += move_yaw
                            
                            # Publish position update to GUI
                            self.publish_position_update()
                            
                            rospy.loginfo(f"[NL_CONTROL DEBUG] Body movement: ({move_x}, {move_y}, {move_yaw}) with robot yaw: {self.current_position[2]:.2f}")
                            rospy.loginfo(f"[NL_CONTROL DEBUG] World displacement: ({world_dx:.2f}, {world_dy:.2f})")
                            rospy.loginfo(f"[NL_CONTROL DEBUG] NL_Control position updated to: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f}, {self.current_position[2]:.2f})")
                else:
                    self.spot_fsm.send(action.strip())
                
                # Publish completion feedback for simulation
                self.pub_feedback.publish(String(data=f"[exec] ✓ {action}"))
                
                # Check for stop signal after action completion
                if self.stop_requested:
                    rospy.loginfo("Stop signal received after action completion - stopping plan execution")
                    # Reset stop flag for next plan
                    self.stop_requested = False
                    return False
                
                rospy.sleep(0.5)
                
            except Exception as e:
                rospy.logerr(f"Action failed '{action}': {e}")
                return False
        
        return True
    
    
    def process_command(self, command):
        """Process natural language command."""
        # Check for stop signal before processing new command
        if self.stop_requested:
            rospy.loginfo("Stop signal received - ignoring new command until stop is cleared")
            return False
        
        # Reset stop flag when processing new command
        self.stop_requested = False
            
        current_state = self.spot_fsm.current_state.name
        print(f"\nProcessing: {command}")
        print(f"Current robot state: {current_state}")
        
        # Publish timing event for HoloLens input received (when ROS starts processing)
        recorder.publish_event('received_hololens_input')
        
        # Process with LLM immediately after receiving command
        actions = self.parse_command(command)
        if not actions:
            print("Parse failed - LLM is not available or command could not be understood")
            print("Please check your GOOGLE_API_KEY environment variable and try again")
            return
        
        print(f"\nLLM Generated Plan:")
        for i, action in enumerate(actions, 1):
            print(f"  {i}. {action}")
        
        # Generate natural language interpretation
        natural_plan = self._interpret_plan(actions)
        
        # Create plan text with both technical and natural language (like orchestrator)
        plan_text = f"NL Control plan (awaiting approval):\n\nNatural Language: {natural_plan}\n\nTechnical Steps:\n" + "\n".join(f"{i+1}. {a}" for i, a in enumerate(actions))
        
        # Publish interpretation to GUI
        self.pub_interpretation.publish(String(data=plan_text))
        rospy.loginfo(f"GUI: Received interpretation: {plan_text}")
        
        # Publish timing event for user confirmation start
        recorder.publish_event('start_user_confirmation')
        
        # In speech mode, wait for approval like orchestrator HA mode
        if self.use_speech:
            rospy.loginfo("Speech mode: waiting for approval")
            self.pending_plan = actions
            self.awaiting_approval = True
        else:
            # Terminal mode: ask for confirmation
            confirm = input("\nExecute this plan? (y/n): ").strip().lower()
            
            # Publish timing event for user confirmation end
            recorder.publish_event('stop_user_confirmation')
            
            if confirm in ['y', 'yes']:
                print("Executing...")
                self.execute_actions(actions)
            else:
                print("Cancelled - try a new command")
    
    def run(self):
        """Main loop."""
        print("\nNatural Language Control Ready!")
        if self.use_speech:
            print("Speech mode: listening on /hl/user_speech")
        else:
            print("Terminal mode: type commands or 'quit' to exit")
        print()
        
        while not rospy.is_shutdown():
            try:
                if self.use_speech:
                    if self.speech_event.wait(timeout=1.0):
                        command = self.speech_input
                        self.speech_event.clear()
                        if command:
                            self.process_command(command)
                else:
                    try:
                        command = input("Command: ").strip()
                        if command.lower() == 'quit':
                            print("Goodbye!")
                            break
                        if command:
                            self.process_command(command)
                    except (EOFError, KeyboardInterrupt):
                        print("\nGoodbye!")
                        break
                        
            except rospy.ROSInterruptException:
                break
            except Exception as e:
                print(f"Error: {e}")
                rospy.sleep(1.0)
    
    def on_hl_stop(self, msg):
        """Handle HoloLens stop signal"""
        rospy.loginfo("HoloLens stop signal received - will finish current task and stop plan execution")
        self.stop_requested = True
    
    def on_world_change(self, msg):
        """Handle world change from GUI."""
        world_id = msg.data
        rospy.loginfo(f"World change received: {world_id}")
        if self.switch_world(world_id):
            rospy.loginfo(f"Successfully switched to world {world_id}")
            # Update the prompt template with new world model
            self._update_prompt_template()
        else:
            rospy.logwarn(f"Failed to switch to world {world_id}")
    
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
            self.execute_actions(plan)
        else:
            rospy.loginfo("Plan rejected. Describe a new plan.")
            self.awaiting_approval = False
            self.pending_plan = []
    
    def _update_prompt_template(self):
        """Store reference to current world model for prompt formatting."""
        try:
            rospy.loginfo("Prompt template updated with new world model")
        except Exception as e:
            rospy.logwarn(f"Failed to update prompt template: {e}")
    
    def _interpret_plan(self, plan):
        """Convert programmatic plan to natural language using LLM like orchestrator."""
        try:
            if not plan:
                return "No plan generated"
            
            # Check if we have OpenAI client for interpretation
            if not hasattr(self, 'openai_client') or self.openai_client is None:
                # Fallback to simple interpretation if no OpenAI
                return f"Robot will execute {len(plan)} actions: {', '.join(plan[:3])}{'...' if len(plan) > 3 else ''}"
            
            # Format plan as string
            plan_str = "\n".join([f"{i+1}. {action}" for i, action in enumerate(plan)])
            
            # Plan interpretation prompt (same as orchestrator)
            PLAN_INTERPRETATION_PROMPT = """You are a robot plan interpreter. Convert a programmatic robot plan into natural language.

COORDINATE SYSTEM:
- X positive = forward movement
- Y positive = left movement  
- Y negative = right movement
- Yaw positive = rotation left
- Yaw negative = rotation right

ROBOT ACTIONS:
- stand_up: Robot stands up
- sit_down: Robot sits down  
- start_moving, x=X, y=Y, yaw=YAW: Robot moves X meters forward, Y meters left (if Y>0) or right (if Y<0), rotates YAW radians left (if YAW>0) or right (if YAW<0)
- start_automated_grasp, object_type="OBJECT": Robot grasps the OBJECT
- start_move_arm_pose, x=X, y=Y, z=Z: Robot moves arm to position (X,Y,Z)
- start_arm_command, command_type=COMMAND: Robot executes arm command (open/close/stow/carry)

PLAN: {plan}

INSTRUCTIONS:
1. Convert the plan into 2-3 clear sentences
2. Use correct directional language: "left" for Y>0, "right" for Y<0, "forward" for X>0, "backward" for X<0
3. For rotations: "left" for positive yaw, "right" for negative yaw
4. Group similar movements together (e.g., "moves forward 2 meters, then right 1 meter")
5. For repeated movements, say "repeats these movements"
6. Keep it concise and user-friendly
7. Focus on the main actions, not technical details

NATURAL LANGUAGE DESCRIPTION:"""
            
            # Use small model for interpretation (same as orchestrator)
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Small, fast model
                messages=[
                    {"role": "system", "content": "You are a helpful robot plan interpreter."},
                    {"role": "user", "content": PLAN_INTERPRETATION_PROMPT.format(plan=plan_str)}
                ],
                max_tokens=150,  # Keep it short
                temperature=0.3  # Consistent output
            )
            
            interpretation = response.choices[0].message.content.strip()
            rospy.loginfo(f"Plan interpretation: {interpretation}")
            return interpretation
            
        except Exception as e:
            rospy.logwarn(f"Plan interpretation failed: {e}")
            return f"Robot will execute {len(plan)} actions: {', '.join(plan[:3])}{'...' if len(plan) > 3 else ''}"
    
    def publish_position_update(self):
        """Publish current robot position to GUI"""
        try:
            import json
            position_data = {
                'x': self.current_position[0],
                'y': self.current_position[1], 
                'yaw': self.current_position[2]
            }
            self.pub_position.publish(json.dumps(position_data))
        except Exception as e:
            rospy.logwarn(f"Failed to publish position update: {e}")

def main():
    # Configuration: True=speech, False=terminal
    USE_SPEECH = True  # Changed to True to listen to GUI commands
    
    try:
        # Check if we're running in launch mode (with ROS parameters)
        try:
            rospy.init_node('nl_control', anonymous=True)
            world_id = rospy.get_param('~world_id', None)
            if world_id:
                # Running in launch mode - use ROS parameter
                print(f"Launch mode: Using world {world_id}")
                controller = NaturalLanguageControl(use_speech=USE_SPEECH, world_id=world_id)
                controller.run()
                return
        except:
            # Not running in ROS mode, continue with interactive selection
            pass
        
        # Interactive world selection mode
        # Create world manager for selection without full controller
        wm = WorldManager()
        
        # Show world selection
        print("\n" + "="*60)
        print("SPOT ROBOT - WORLD SELECTION")
        print("="*60)
        print("Available Worlds:")
        print()
        
        world_list = wm.get_world_list()
        for world_id, name, description in world_list:
            print(f"{world_id}. {name}")
            print(f"   {description}")
            print()
        
        while True:
            try:
                choice = input("Select a world (1-10) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    print("No world selected. Exiting...")
                    return
                
                if choice in wm.worlds:
                    world_name, world_desc = wm.get_world_info(choice)
                    print(f"\nSelected: {world_name}")
                    print(f"Description: {world_desc}")
                    confirm = input("Continue with this world? (y/n): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        selected_world = choice
                        break
                    else:
                        continue
                else:
                    print("Invalid selection. Please choose 1-10 or 'q' to quit.")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                return
        
        # Create the controller with selected world
        controller = NaturalLanguageControl(use_speech=USE_SPEECH, world_id=selected_world)
        controller.run()
    except KeyboardInterrupt:
        rospy.loginfo("Shutdown")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
    finally:
        rospy.loginfo("Natural Language Control shutdown complete")

if __name__ == "__main__":
    main()
