#!/usr/bin/env python3
import rospy
import sys
import os
import openai
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
- Current position: {current_position}

WORLD LAYOUT:
- Vegetables: (1.0, 0.0, 0.0)
- Fruits: (3.0, 0.0, 0.0)  
- Meat: (5.0, 0.0, 0.0)
- OBSTACLE: Square at (2.0, 0.0, 0.0) - AVOID x: 1.9-2.1m, y: -0.1 to 0.1m

ROBOT SPECS:
- Size: 0.5m wide × 1.1m long
- Movement: Relative to current position
- Coordinate system: x=forward, y=right, yaw=rotation

AVAILABLE ACTIONS:
- stand_up
- sit_down  
- start_moving, x=float, y=float, yaw=float, frame="body"
- get_image, image_source="camera_name"
- get_initial_pose
- arm_command, command_type="open|close|stow|carry"

RULES:
1. Visit destinations in EXACT order specified by user
2. AVOID obstacle at (2.0, 0.0) - stay outside x: 1.9-2.1, y: -0.1 to 0.1
3. Account for full robot body (1.1m long, 0.5m wide) when checking collisions
4. One action per line, no quotes/brackets, no numbering
5. Only use "stand_up" if robot is not already standing
6. In one move, the robot can either move in the x direction, the y direction, or the yaw direction, but not all three at once.

PLANNING PROCESS:
1. Parse user command to identify destinations in order
2. Plan collision-free path visiting each destination once in order
3. Check each movement segment for robot-obstacle collision
4. Generate action sequence

OUTPUT FORMAT:
Return only the action list, one action per line.

Actions:"""

class NaturalLanguageControl:
    def __init__(self, use_speech=False):
        rospy.init_node('nl_control', anonymous=True)
        self.use_speech = use_speech
        
        # Position tracking
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
        
    def update_position(self, x, y, yaw):
        """Update current position based on movement."""
        self.current_position[0] += x
        self.current_position[1] += y
        rospy.loginfo(f"Position updated: {self.current_position}")
        
    def setup_llm(self):
        """Setup OpenAI API."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            rospy.logwarn("No OPENAI_API_KEY - LLM disabled")
            self.llm = None
            return
        
        self.llm = openai.OpenAI(api_key=api_key)
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
            
            # Format prompt with current state and position
            formatted_prompt = PROMPT.format(
                current_state=current_state,
                current_position=f"({self.current_position[0]:.2f}, {self.current_position[1]:.2f}, {self.current_position[2]:.2f})"
            )
            
            response = self.llm.responses.create(
                model="gpt-5",
                input=f"{formatted_prompt}\n\nCommand: {command}\n\nActions:",
                reasoning={"effort": "minimal"}
            )
            
            result = response.output_text.strip()
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
                    
                    # Track position changes for movement commands
                    if action_name == "start_moving" and 'x' in params and 'y' in params:
                        self.update_position(params['x'], params['y'], params.get('yaw', 0))
                    
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
