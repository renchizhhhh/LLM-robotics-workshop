#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import json
import re
import math
import time
import threading
import openai
import os

# LLM System Guide for the planner - Updated for V2 with exact FSM capabilities
LLM_SYSTEM_GUIDE = """BEGIN_GUIDE
You are controlling a Boston Dynamics Spot with arm, hand camera and depth. You build short, safe, parameterized actions. Each JSON message on /fsm_commands triggers exactly one atomic action. Actions are blocking. The FSM returns to stand after each action.

STATE AWARENESS:
- The robot can be in 'sit' or 'stand' state
- Consider the current robot state when planning actions
- Only include 'stand' action if robot is currently sitting and action requires standing
- After any action (except sit), robot automatically returns to 'stand' state

EXACT FSM CAPABILITIES AND LIMITS:

Movement limits (move_relative):
- x: [-5.0, 5.0] meters (forward/backward)
- y: [-3.0, 3.0] meters (left/right) 
- yaw: any value in radians (will be normalized to [-π, π])
- timeout: [1.0, 60.0] seconds (default: 10.0)
- For 360° rotation: use yaw=6.28 (2π) with timeout=30.0 seconds

Standing limits:
- height: [-0.2, 0.5] meters (default: 0.0)

Hand camera limits (aim_hand):
- x: [0.60, 0.90] meters forward from BODY
- y: [-0.5, 0.5] meters left/right
- z: [0.55, 0.75] meters height

Grasp limits (grasp_body_point):
- Radial distance r = sqrt(x² + y²): [0.30, 0.85] meters
- z: [-0.45, 1.0] meters
- approach: [0.05, 0.30] meters (default: 0.12)
- lift: [0.05, 0.30] meters (default: 0.15)

Pick limits (pick_from_pixel):
- x: [0, 1920] pixels
- y: [0, 1080] pixels
- timeout: [5.0, 60.0] seconds (default: 25.0)

Detection limits (detect_label):
- confidence: [0.0, 1.0] (default: 0.25)

Robot facts and safe envelopes:
- The arm is on the front torso. Avoid grasp targets that are too close to the chest.
- Hand camera observation pose keeps the wrist clear and gives a good view.
- If a target is too close to the chest, first move the base a little, then aim the hand and pick.

Frames and picking:
- move_relative uses BODY frame. x forward, y left, yaw in radians.
- pick_from_pixel uses the vision chain. Prefer image_source "hand_color_image". Set top_down true and auto_walk false by default.

Standing and posture:
- Stand before movement or manipulation. Use stand(height=0.0) unless stated otherwise.
- Open the gripper before approach, close to grasp, lift slightly after grasp if you script a BODY grasp.

Atomic actions:
- {"op":"stand","height":float}
- {"op":"sit"}
- {"op":"move_relative","x":float,"y":float,"yaw":float,"timeout":float}
- {"op":"arm_ready"}
- {"op":"arm_stow"}
- {"op":"aim_hand","x":float,"y":float,"z":float}
- {"op":"gripper","mode":"open"|"close","fraction":float}
- {"op":"detect_label","label":str,"provider":"ultralytics","conf":float,"image_source":"hand_color_image"}
- {"op":"pick_from_pixel","image_source":"hand_color_image","x":int,"y":int,"top_down":bool,"auto_walk":bool,"timeout":float}
- {"op":"grasp_body_point","x":float,"y":float,"z":float,"approach":float,"lift":float}
- {"op":"sleep","seconds":float}
- {"op":"pick_apriltag","tag_id":int|null,"z_offset":float,"approach":float,"lift":float}

Apriltag picking:
- The robot queries the World Object service for Apriltags. It picks the nearest tag if tag_id is null. Compute a grasp point z_offset above the tag. Transform to BODY, then perform a scripted grasp. Defaults: z_offset 0.20 m, approach 0.12 m, lift 0.15 m.

Safety and preconditions:
- Arm operations (arm_ready, arm_stow, aim_hand, gripper, pick_from_pixel, grasp_body_point, pick_apriltag) require robot to be in 'stand' state
- Movement operations (move_relative) require robot to be in 'stand' state
- Detection operations (detect_label) can be done from any state
- Always check if robot is standing before manipulation tasks

Planning patterns:
A) Safe forward pick on floor:
{"op":"stand","height":0.0}
{"op":"aim_hand","x":0.70,"y":-0.10,"z":0.60}
{"op":"detect_label","label":"bottle","conf":0.25,"image_source":"hand_color_image"}
{"op":"pick_from_pixel","image_source":"hand_color_image","x":960,"y":540,"top_down":true,"auto_walk":false,"timeout":25}
{"op":"arm_stow"}

B) Apriltag picking:
{"op":"stand","height":0.0}
{"op":"pick_apriltag","tag_id":null,"z_offset":0.20,"approach":0.12,"lift":0.15}
{"op":"arm_stow"}

C) Complex navigation:
{"op":"stand","height":0.0}
{"op":"move_relative","x":1.0,"y":0.0,"yaw":0.0,"timeout":10.0}
{"op":"move_relative","x":0.0,"y":0.5,"yaw":1.57,"timeout":10.0}
{"op":"sit"}

D) 360° rotation:
{"op":"stand","height":0.0}
{"op":"move_relative","x":0.0,"y":0.0,"yaw":6.28,"timeout":30.0}

E) Object manipulation sequence:
{"op":"stand","height":0.0}
{"op":"arm_ready"}
{"op":"aim_hand","x":0.75,"y":0.0,"z":0.65}
{"op":"gripper","mode":"open"}
{"op":"detect_label","label":"cup","conf":0.3,"image_source":"hand_color_image"}
{"op":"pick_from_pixel","image_source":"hand_color_image","x":960,"y":540,"top_down":true,"auto_walk":false,"timeout":25}
{"op":"arm_stow"}

Response format: Return only a JSON array of action objects, no explanations.
END_GUIDE"""

class NaturalLanguageControl:
    def __init__(self, use_llm=True, use_speech=False):
        rospy.init_node('natural_language_control', anonymous=True)
        
        # Configuration
        self.use_llm = use_llm
        self.use_speech = use_speech
        
        # Publisher for sending commands to FSM
        self.fsm_publisher = rospy.Publisher('/fsm_commands', String, queue_size=10)
        
        # Subscriber for feedback from FSM
        self.feedback_subscriber = rospy.Subscriber('/fsm_feedback', String, self.feedback_callback)
        self.last_feedback = None
        
        # Speech input subscriber (only used when use_speech=True)
        if self.use_speech:
            self.speech_subscriber = rospy.Subscriber('/user_speech', String, self.speech_callback)
            self.latest_speech_input = None
            self.speech_received = threading.Event()
            print("SPEECH MODE ENABLED - Listening for speech input on /user_speech topic")
        else:
            print("TERMINAL MODE ENABLED - Type commands in terminal")
        
        # Robot state tracking
        self._current_state = 'sit'
        self._last_fsm_feedback = None
        
        # OpenAI configuration
        if self.use_llm:
            self.setup_openai()
        
        print("=== NATURAL LANGUAGE ROBOT CONTROL V2 ===")
        if self.use_llm:
            print("LLM MODE: Using GPT-5 for complex instruction parsing")
            print("Example instructions:")
            print("- 'I want the robot to make a 360'")
            print("- 'Move forward and then turn left'")
            print("- 'Stand up high and take a picture'")
            print("- 'Walk in a square pattern'")
            print("- 'Pick the green sphere'")
            print("- 'Grab the cup'")
            print("- 'Pick up the object'")
            print("- 'Find and pick the apriltag'")
            print("- 'Navigate to the corner and sit down'")
        else:
            print("SIMPLE MODE: Using regex patterns for basic commands")
            print("Supported patterns:")
            print("- 'loop X meter' or 'loop X m' -> move_relative(x=X, y=0, yaw=0)")
            print("- 'draai Y graden links' -> move_relative(yaw=+Y in radians)")
            print("- 'draai Y graden rechts' -> move_relative(yaw=-Y in radians)")
            print("- 'sta op hoog' -> stand(height=0.1)")
            print("- 'ga zitten' -> sit")
            print("- 'richt hand naar beneden op x y z' -> aim_hand(x,y,z)")
            print("- 'pak <label>' -> detect_label + pick_from_pixel sequence")
        
        print("Type 'quit' to exit")
        print("Type 'reset_state' to reset robot state")
        print("--------------------------------------------------")
    
    def setup_openai(self):
        """Setup OpenAI API for GPT-5 integration."""
        try:
            # Try to get API key from environment
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("Warning: OPENAI_API_KEY not found in environment")
                print("LLM functionality will be disabled")
                self.use_llm = False
                return
            
            # No need to set api_key globally in new version
            print("OpenAI API configured successfully")
        except Exception as e:
            print(f"Error setting up OpenAI: {e}")
            self.use_llm = False
    
    def feedback_callback(self, msg):
        """Handle feedback from FSM."""
        try:
            self.last_feedback = json.loads(msg.data)
            self._last_fsm_feedback = self.last_feedback
            print(f"FSM Feedback: {self.last_feedback}")
            
            # Update robot state based on feedback
            self._update_robot_state_from_feedback(self.last_feedback)
            
        except json.JSONDecodeError:
            print(f"Invalid feedback JSON: {msg.data}")
    
    def get_current_robot_state(self):
        """Get current robot state."""
        return self._current_state
    
    def _update_robot_state_from_feedback(self, feedback):
        """Update robot state based on FSM feedback."""
        op = feedback.get('op', '')
        status = feedback.get('status', '')
        
        if status == 'ok':
            if op == 'stand':
                self._current_state = 'stand'
            elif op == 'sit':
                self._current_state = 'sit'
            elif op in ['move_relative', 'arm_ready', 'arm_stow', 'aim_hand', 'gripper', 'detect_label', 'pick_from_pixel', 'grasp_body_point', 'pick_apriltag']:
                # After any action, robot returns to stand state (except sit)
                self._current_state = 'stand'
    
    def speech_callback(self, msg):
        """Handle speech input."""
        self.latest_speech_input = msg.data
        self.speech_received.set()
        print(f"Speech received: {msg.data}")
    
    def get_command_from_speech(self):
        """Get command from speech input."""
        if not self.use_speech:
            return None
        
        print("Waiting for speech input...")
        self.speech_received.wait(timeout=30.0)  # 30 second timeout
        
        if self.speech_received.is_set():
            command = self.latest_speech_input
            self.speech_received.clear()
            self.latest_speech_input = None
            return command
        else:
            print("No speech input received within timeout")
            return None
    
    def get_command_from_terminal(self):
        """Get command from terminal input."""
        try:
            return input("\nNL Command: ").strip()
        except (EOFError, KeyboardInterrupt):
            return "quit"
    
    def parse_simple_command(self, command):
        """Parse simple commands using regex patterns."""
        command = command.lower()
        
        # Movement patterns
        move_match = re.match(r'loop\s+([\d.]+)\s*(?:meter|m)', command)
        if move_match:
            distance = float(move_match.group(1))
            return [{"op": "move_relative", "x": distance, "y": 0.0, "yaw": 0.0, "timeout": 10.0}]
        
        # Rotation patterns
        rot_left_match = re.match(r'draai\s+([\d.]+)\s*graden\s*links', command)
        if rot_left_match:
            degrees = float(rot_left_match.group(1))
            radians = math.radians(degrees)
            return [{"op": "move_relative", "x": 0.0, "y": 0.0, "yaw": radians, "timeout": 10.0}]
        
        rot_right_match = re.match(r'draai\s+([\d.]+)\s*graden\s*rechts', command)
        if rot_right_match:
            degrees = float(rot_right_match.group(1))
            radians = -math.radians(degrees)
            return [{"op": "move_relative", "x": 0.0, "y": 0.0, "yaw": radians, "timeout": 10.0}]
        
        # Standing patterns
        if 'sta op hoog' in command:
            return [{"op": "stand", "height": 0.1}]
        
        if 'sta op' in command or 'stand up' in command:
            return [{"op": "stand", "height": 0.0}]
        
        # Sitting patterns
        if 'ga zitten' in command or 'sit' in command:
            return [{"op": "sit"}]
        
        # Hand aiming patterns
        aim_match = re.match(r'richt hand naar beneden op ([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)', command)
        if aim_match:
            x, y, z = map(float, aim_match.groups())
            return [{"op": "aim_hand", "x": x, "y": y, "z": z}]
        
        # Pick patterns
        pick_match = re.match(r'pak\s+(\w+)', command)
        if pick_match:
            label = pick_match.group(1)
            return [
                {"op": "stand", "height": 0.0},
                {"op": "aim_hand", "x": 0.70, "y": -0.10, "z": 0.60},
                {"op": "detect_label", "label": label, "conf": 0.25, "image_source": "hand_color_image"},
                {"op": "pick_from_pixel", "image_source": "hand_color_image", "x": 960, "y": 540, "top_down": True, "auto_walk": False, "timeout": 25.0},
                {"op": "arm_stow"}
            ]
        
        return None
    
    def parse_llm_command(self, command):
        """Parse complex commands using GPT-5."""
        if not self.use_llm:
            return None
        
        try:
            # Create the prompt with system guide, current state, and user command
            current_state = self.get_current_robot_state()
            state_aware_prompt = f"""CURRENT ROBOT STATE: {current_state}

{LLM_SYSTEM_GUIDE}

IMPORTANT: Consider the current robot state when planning actions:
- If robot is already standing and command requires standing, skip the stand action
- If robot is sitting and command requires movement/manipulation, start with stand action
- After any action (except sit), robot returns to 'stand' state"""
            
            messages = [
                {"role": "system", "content": state_aware_prompt},
                {"role": "user", "content": f"Parse this natural language command into a sequence of robot actions: {command}"}
            ]
            
            # Call GPT-5 using the correct API format
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # Combine system prompt and user input for GPT-5
            full_input = f"""{LLM_SYSTEM_GUIDE}

User instruction: '{command}'

Please convert this instruction to robot actions:"""
            
            response = client.chat.completions.create(
                model="gpt-4",  # Use gpt-4 as gpt-5 might not be available
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_GUIDE},
                    {"role": "user", "content": f"Parse this natural language command into a sequence of robot actions: {command}"}
                ],
                temperature=0.1,  # Low temperature for consistent output
                max_tokens=1000
            )
            
            # Extract the response
            llm_response = response.choices[0].message.content.strip()
            
            # Try to parse as JSON
            try:
                actions = json.loads(llm_response)
                if isinstance(actions, list):
                    return actions
                else:
                    print(f"LLM returned non-list: {llm_response}")
                    return None
            except json.JSONDecodeError:
                print(f"LLM response is not valid JSON: {llm_response}")
                return None
                
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return None
    
    def execute_action_sequence(self, actions):
        """Execute a sequence of actions with feedback waiting."""
        if not actions:
            print("No actions to execute")
            return False
        
        print(f"\nExecuting plan...")
        success = True
        
        for i, action in enumerate(actions):
            print(f"Executing action {i+1}/{len(actions)}: {action}")
            
            # Send action to FSM
            action_json = json.dumps(action)
            self.fsm_publisher.publish(String(action_json))
            
            # Wait for feedback
            start_time = time.time()
            timeout = 30.0  # 30 second timeout per action
            
            while time.time() - start_time < timeout:
                if self.last_feedback:
                    feedback = self.last_feedback
                    self.last_feedback = None  # Clear for next action
                    
                    if feedback.get('status') == 'ok':
                        print(f"Action {i+1} completed successfully")
                        break
                    elif feedback.get('status') == 'error':
                        print(f"Action {i+1} failed: {feedback.get('msg', 'Unknown error')}")
                        success = False
                        break
                
                time.sleep(0.1)
            else:
                print(f"Action {i+1} timed out")
                success = False
            
            # Add delay between actions for robot stability
            if i < len(actions) - 1:  # Don't delay after the last action
                action_type = action.get('op', '')
                next_action_type = actions[i + 1].get('op', '') if i + 1 < len(actions) else ''
                
                # Special delay before detect_label to ensure stable camera view
                if next_action_type == 'detect_label':
                    print("Waiting 2 seconds before detection for stable camera view...")
                    time.sleep(2.0)
                elif action_type in ['stand', 'sit']:
                    print("Waiting 3 seconds for robot to stabilize...")
                    time.sleep(3.0)
                elif action_type in ['aim_hand', 'arm_ready', 'arm_stow']:
                    print("Waiting 2 seconds for arm to stabilize...")
                    time.sleep(2.0)
                elif action_type in ['move_relative']:
                    print("Waiting 2 seconds for movement to complete...")
                    time.sleep(2.0)
                elif action_type in ['detect_label']:
                    print("Waiting 1 second before next action...")
                    time.sleep(1.0)
                else:
                    print("Waiting 1 second between actions...")
                    time.sleep(1.0)
        
        if success:
            print("All actions completed successfully")
        else:
            print("Some actions failed")
        
        return success
    
    def run(self):
        """Main run loop."""
        while not rospy.is_shutdown():
            try:
                # Get command from appropriate source
                if self.use_speech:
                    command = self.get_command_from_speech()
                else:
                    command = self.get_command_from_terminal()
                
                if not command:
                    continue
                
                # Handle special commands
                if command.lower() == 'quit':
                    print("Exiting...")
                    break
                
                if command.lower() == 'reset_state':
                    print("Resetting robot state...")
                    self._current_state = 'sit'
                    continue
                
                # Parse command
                current_state = self.get_current_robot_state()
                print(f"\nProcessing: '{command}' (Current robot state: {current_state})")
                
                actions = None
                
                # Try LLM parsing first if enabled
                if self.use_llm:
                    print("Trying LLM parsing...")
                    actions = self.parse_llm_command(command)
                    if actions:
                        print("LLM parsing successful!")
                    else:
                        print("LLM parsing failed, falling back to simple parsing...")
                
                # Fall back to simple parsing if LLM fails or is disabled
                if not actions:
                    print("Using simple parsing...")
                    actions = self.parse_simple_command(command)
                
                if not actions:
                    print("Could not parse command. Try a different formulation.")
                    continue
                
                # Display action plan
                print(f"\nAction plan ({len(actions)} actions):")
                for i, action in enumerate(actions):
                    print(f"  {i+1}. {action}")
                
                # Get user approval
                response = input("\nExecute this plan? (yes/no): ").strip().lower()
                if response not in ['yes', 'y']:
                    print("Plan cancelled.")
                    continue
                
                # Execute actions
                self.execute_action_sequence(actions)
                
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue

def main():
    # Configuration
    USE_LLM = True  # Set to True to use GPT-5, False for simple parsing
    USE_SPEECH = False  # Set to True to use speech input, False for terminal
    
    try:
        controller = NaturalLanguageControl(use_llm=USE_LLM, use_speech=USE_SPEECH)
        controller.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
