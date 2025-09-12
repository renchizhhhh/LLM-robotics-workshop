#!/usr/bin/env python3
import rospy
import sys
import os
from google import genai
import threading
import time
import json
import math
from std_msgs.msg import String

# Add the FSM to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'spot_hololens_llm_interface'))
from finite_state_machine import SpotStateMachine

# Import timing utilities
from timing_utils import recorder

# LLM prompt for converting natural language to FSM commands
PROMPT = """You are a path planning system for a Boston Dynamics Spot robot.

CURRENT STATE:
- Robot state: {current_state}
- Current position (vision frame): {current_position}

WORLD LAYOUT (vision frame coordinates):
- Vegetables: (1.0, 0.0, 0.0)
- Fruits: (3.0, 0.0, 0.0)  
- Meat: (5.0, 0.0, 0.0)
- OBSTACLE: Square at (2.0, 0.0, 0.0) - AVOID x: 1.85-2.15m, y: -0.15 to 0.15m

ROBOT SPECS:
- Size: 0.6m wide × 1.2m long
- Movement: Uses vision frame for precise positioning
- Coordinate system: x=forward, y=left, yaw=rotation (vision frame)

AVAILABLE ACTIONS:
- stand_up
- sit_down  
- start_moving, x=float, y=float, yaw=float, frame="vision"
- get_image, image_source="camera_name"
- get_initial_pose
- arm_command, command_type="open|close|stow|carry"

RULES:
1. Visit destinations in EXACT order specified by user
2. AVOID obstacle at (2.0, 0.0) - stay outside x: 1.85-2.15, y: -0.15 to 0.15
3. Account for full robot body (1.1m long, 0.5m wide) when checking collisions
4. One action per line, no quotes/brackets, no numbering
5. Only use "stand_up" if robot is not already standing or moving
6. Use ABSOLUTE vision frame coordinates for movements - calculate target position from current position
7. For movements, calculate: target_x = current_x + desired_movement_x, target_y = current_y + desired_movement_y

PLANNING PROCESS:
1. Parse user command to identify destinations in order
2. Calculate absolute vision frame coordinates for each destination
3. Plan collision-free path visiting each destination once in order
4. Check each movement segment for robot-obstacle collision
5. Generate action sequence with absolute vision frame coordinates

OUTPUT FORMAT:
Return only the action list, one action per line.

Actions:"""

class NaturalLanguageControl:
    def __init__(self, use_speech=False):
        rospy.init_node('nl_control', anonymous=True)
        self.use_speech = use_speech
        
        # Position tracking - will get actual position from robot
        self.current_position = [0.0, 0.0, 0.0]  # [x, y, z] in meters
        
        # Check if dummy mode is set globally
        dummy_mode = rospy.get_param('/spot_fsm/dummy_mode', False)
        self.spot_fsm = SpotStateMachine(dummy_mode=dummy_mode)
        
        # Auto-connect and power on robot
        rospy.loginfo("Auto-connecting robot...")
        
        # Connect robot
        recorder.publish_event('start_connect')
        self.spot_fsm.send("connect")
        recorder.publish_event('stop_connect')
        
        # Power on robot
        recorder.publish_event('start_power_on')
        self.spot_fsm.send("power_on")
        recorder.publish_event('stop_power_on')
        
        rospy.loginfo("Robot ready for commands")
        
        if use_speech:
            self.speech_sub = rospy.Subscriber('/hl/user_speech', String, self.speech_callback)
            self.speech_input = None
            self.speech_event = threading.Event()
            rospy.loginfo("Speech mode: listening on /hl/user_speech")
        else:
            rospy.loginfo("Terminal mode: type commands")
        
        self.setup_llm()
        rospy.loginfo("Ready")
        
    def get_actual_robot_position(self):
        """Get actual robot position from vision frame (camera-based odometry)."""
        try:
            # Get robot pose from the service (returns vision frame position)
            response = self.spot_fsm.get_robot_pose()
            if response.success:
                # Extract position from the pose
                pose = response.robot_pose.pose
                x = pose.position.x
                y = pose.position.y
                z = pose.position.z
                
                # Convert quaternion to yaw
                from tf.transformations import euler_from_quaternion
                quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
                _, _, yaw = euler_from_quaternion(quat)
                
                self.current_position = [x, y, yaw]
                rospy.loginfo(f"Actual robot position (vision frame): {self.current_position}")
                return True
            else:
                rospy.logwarn(f"Failed to get robot pose: {response.message}")
                return False
        except Exception as e:
            rospy.logerr(f"Error getting robot position: {e}")
            return False
        
    def setup_llm(self):
        """Setup Gemini API."""
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            rospy.logwarn("No GOOGLE_API_KEY - LLM disabled")
            self.llm = None
            return
        
        self.llm = genai.Client(api_key=api_key)
        rospy.loginfo("LLM ready")
    
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
            
            # Format prompt with current state and actual position
            formatted_prompt = PROMPT.format(
                current_state=current_state,
                current_position=f"({self.current_position[0]:.2f}, {self.current_position[1]:.2f}, {self.current_position[2]:.2f})"
            )
            
            response = self.llm.models.generate_content(
                model="gemini-2.5-pro",
                contents=f"{formatted_prompt}\n\nCommand: {command}\n\nActions:"
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
            try:
                rospy.loginfo(f"Executing: {action}")
                
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
                    
                    # Execute the action (position will be tracked by actual robot pose)
                    self.spot_fsm.send(action_name, **params)
                else:
                    self.spot_fsm.send(action.strip())
                
                rospy.sleep(0.5)
                
            except Exception as e:
                rospy.logerr(f"Action failed '{action}': {e}")
                return False
        
        return True
    
    
    def process_command(self, command):
        """Process natural language command."""
        current_state = self.spot_fsm.current_state.name
        print(f"\nProcessing: {command}")
        print(f"Current robot state: {current_state}")
        
        # Publish timing event for HoloLens input received (when ROS starts processing)
        recorder.publish_event('received_hololens_input')
        
        # Process with LLM immediately after receiving command
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
            print("Cancelled - try a new command")
    
    def run(self):
        """Main loop."""
        print("\nNatural Language Control Ready!")
        print("Type commands or 'quit' to exit\n")
        
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
