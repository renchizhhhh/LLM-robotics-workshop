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

# LLM prompt for converting natural language to FSM commands
PROMPT = """You are a control system for a Boston Dynamics Spot robot.

CURRENT STATE:
- Robot state: {current_state}
- Current position (body frame): {current_position}

ROBOT SPECS:
- Size: 0.7m wide × 1.4m long
- Movement: Uses body frame for positioning
- Coordinate system: x=forward, y=left, yaw=rotation (body frame)

AVAILABLE ACTIONS:
- stand_up
- sit_down  
- start_moving, x=float, y=float, yaw=float, frame="body"
- get_image, image_source="camera_name"
- get_initial_pose
- start_automated_grasp, object_type="exact_object_name_from_user" (arm will be stowed after grasping)
- start_move_arm_pose, x=float, y=float, z=float, qw=float, qx=float, qy=float, qz=float, duration=float, open_gripper=bool
- start_arm_command, command_type=open|close|stow|carry

RULES:
1. One action per line, no quotes/brackets, no numbering
2. Robot must be in standing mode before moving and grasping
3. Only use "stand_up" if robot is not already standing, moving or grasping
4. In one move, the robot can either move in the x direction, the y direction, or the yaw direction, not two or three at once.
5. Use RELATIVE body frame coordinates for movements - each movement is relative to current position
6. For movements, calculate: move_x = target_x - current_x, move_y = target_y - current_y
7. Use EXACT object name from user command for object_type (e.g., "tomato can" not "tomato")
8. For object manipulation: use start_automated_grasp to pick up objects, then use start_move_arm_pose and start_arm_command to place/release objects

PLANNING PROCESS:
1. Parse user command to understand the task
2. Calculate relative body frame movements from current position to target locations
3. Plan path with appropriate movements and manipulations
4. Generate action sequence with relative body frame movements

OUTPUT FORMAT:
Return only the action list, one action per line.

Task: {command}

Actions:"""

class NaturalLanguageControl:
    def __init__(self, use_speech=False):
        rospy.init_node('nl_control', anonymous=True)
        self.use_speech = use_speech
        
        # Position tracking - will get actual position from robot
        self.current_position = [0.0, 0.0, 0.0]  # [x, y, yaw] relative to start position
        self.start_position = None  # Will be set on first position read
        
        # Publisher for position updates
        self.pub_position = rospy.Publisher('/nl_control/robot_position', String, queue_size=1)
        
        # HoloLens stop signal handling
        self.stop_requested = False
        self.sub_hl_stop = rospy.Subscriber('/hl/stop', Empty, self.on_hl_stop)
        
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
            raise
        
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
        """Setup Gemini API."""
        try:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                rospy.logwarn("No GOOGLE_API_KEY environment variable found - LLM disabled")
                self.llm = None
                return
            
            rospy.loginfo("Setting up Gemini LLM...")
            self.llm = genai.Client(api_key=api_key)
            rospy.loginfo("LLM ready")
        except Exception as e:
            rospy.logwarn(f"Failed to setup Gemini client: {e}")
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
            return None
        
        # Publish LLM processing start
        recorder.publish_event('start_llm_processing')
        
        try:
            # Get current robot state
            current_state = self.spot_fsm.current_state.name
            
            # Get actual robot position from vision frame
            if not self.get_actual_robot_position():
                rospy.logwarn("Using last known position")
            
            # Format prompt with current state, position, and command
            rospy.loginfo(f"[LLM DEBUG] Sending position to LLM: ({self.current_position[0]:.2f}, {self.current_position[1]:.2f}, {self.current_position[2]:.2f})")
            
            # Publish initial position to GUI
            self.publish_position_update()
            formatted_prompt = PROMPT.format(
                current_state=current_state,
                current_position=f"({self.current_position[0]:.2f}, {self.current_position[1]:.2f}, {self.current_position[2]:.2f})",
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
                            
                            # Transform body frame movement to world frame
                            # Body frame: x=forward, y=left
                            # World frame: x=forward, y=left (same in this case)
                            cos_yaw = math.cos(self.current_position[2])
                            sin_yaw = math.sin(self.current_position[2])
                            
                            # Transform body frame movement to world frame
                            # When robot is rotated by yaw, body frame movements need to be rotated
                            # For a robot facing direction yaw, body frame (x,y) becomes world frame:
                            # Standard rotation matrix but with corrected coordinate system
                            # Body frame: x=forward, y=left (positive left)
                            # World frame: x=forward, y=left (positive left) 
                            world_dx = move_x * cos_yaw + move_y * sin_yaw
                            world_dy = -move_x * sin_yaw + move_y * cos_yaw
                            
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
        """Process command - either fixed plan or natural language."""
        # Check for stop signal before processing new command
        if self.stop_requested:
            rospy.loginfo("Stop signal received - ignoring new command until stop is cleared")
            return False
        
        # Reset stop flag when processing new command
        self.stop_requested = False
            
        current_state = self.spot_fsm.current_state.name
        print(f"\nCurrent robot state: {current_state}")
        
        # Check if user wants standard plan or natural language
        if command.lower() == "standard":
            # Fixed plan - always the same sequence
            actions = [
                "start_automated_grasp, object_type=\"tomato can\"",
                "start_arm_command, command_type=open",
                "start_arm_command, command_type=close", 
                "start_arm_command, command_type=stow"
            ]
            
            print(f"\nFixed Plan:")
            for i, action in enumerate(actions, 1):
                print(f"  {i}. {action}")
        elif command.lower().startswith("nl "):
            # Natural language processing - remove "nl " prefix
            nl_command = command[3:].strip()
            print(f"\nProcessing: {nl_command}")
            
            # Publish timing event for HoloLens input received (when ROS starts processing)
            recorder.publish_event('received_hololens_input')
            
            # Process with LLM
            actions = self.parse_command(nl_command)
            if not actions:
                print("Parse failed - try a different command")
                return
            
            print(f"\nLLM Generated Plan:")
            for i, action in enumerate(actions, 1):
                print(f"  {i}. {action}")
        else:
            # Direct natural language command (for backward compatibility)
            print(f"\nProcessing: {command}")
            
            # Publish timing event for HoloLens input received (when ROS starts processing)
            recorder.publish_event('received_hololens_input')
            
            # Process with LLM
            actions = self.parse_command(command)
            if not actions:
                print("Parse failed - try a different command")
                return
            
            print(f"\nLLM Generated Plan:")
            for i, action in enumerate(actions, 1):
                print(f"  {i}. {action}")
        
        # Publish timing event for user confirmation start
        recorder.publish_event('start_user_confirmation')
        
        confirm = input("\nExecute this plan? (y/n): ").strip().lower()
        
        # Publish timing event for user confirmation end
        recorder.publish_event('stop_user_confirmation')
        
        if confirm in ['y', 'yes']:
            print("Executing...")
            self.execute_actions(actions)
        else:
            print("Cancelled - try again")
    
    def run(self):
        """Main loop."""
        print("\nRobot Control Ready!")
        print("Type 'standard' for fixed plan, 'nl' for natural language, or 'quit' to exit")
        print("Examples:")
        print("  standard - Execute the fixed tomato can plan")
        print("  nl 'Pick up the bottle' - Use natural language control")
        print("  quit - Exit the program\n")
        
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
    USE_SPEECH = False
    
    try:
        controller = NaturalLanguageControl(use_speech=USE_SPEECH)
        controller.run()
    except KeyboardInterrupt:
        rospy.loginfo("Shutdown")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
    finally:
        rospy.loginfo("Natural Language Control shutdown complete")

if __name__ == "__main__":
    main()