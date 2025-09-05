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
- IMPORTANT: The FSM automatically handles state transitions - no need to manually manage states
- For reliable arm operations, the robot will automatically ensure it's in 'stand' state before arm commands

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
- z: [-0.2, 0.65] meters height (allows pointing down to ground level)

IMPORTANT ARM COMMAND LIMITATIONS:
- arm_ready: UNRELIABLE - may fail during execution
- arm_stow: UNRELIABLE - may fail during execution  
- aim_hand: RELIABLE - use this instead of arm_ready/arm_stow
- For arm positioning, prefer aim_hand(x,y,z) over arm_ready + arm_stow sequences

Grasp limits (grasp_body_point):
- Radial distance r = sqrt(x² + y²): [0.30, 0.85] meters
- z: [-0.45, 1.0] meters
- approach: [0.05, 0.30] meters (default: 0.12)
- lift: [0.05, 0.30] meters (default: 0.15)

Pick limits (pick_from_pixel):
- x: [0, 1920] pixels (will be automatically set from detection)
- y: [0, 1080] pixels (will be automatically set from detection)
- timeout: [5.0, 60.0] seconds (default: 25.0)

Detection limits (detect_label):
- confidence: [0.0, 1.0] (default: 0.01)

Robot facts and safe envelopes:
- The arm is on the front torso. Avoid grasp targets that are too close to the chest.
- Hand camera observation pose keeps the wrist clear and gives a good view.
- If a target is too close to the chest, first move the base a little, then aim the hand and pick.
- WARNING: Keep z-coordinate between -0.2m to 0.65m for reliable arm positioning. Lower values (negative) point the hand down toward the ground, higher values may cause the arm to reach too far and look unstable.

Frames and picking:
- move_relative uses BODY frame. x forward, y left, yaw in radians.
- pick_from_pixel uses the vision chain. Prefer image_source "hand_color_image". Set top_down true and auto_walk false by default.
- IMPORTANT: For pick_from_pixel, use placeholder coordinates like x=0, y=0 - the system will automatically replace them with actual detection coordinates.

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
- {"op":"pick_from_pixel","image_source":"hand_color_image","x":0,"y":0,"top_down":bool,"auto_walk":bool,"timeout":float} (default: auto_walk=true for better grasp success)
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

IMPORTANT: For arm movements and gestures (like waving), use aim_hand sequences instead of arm_ready/arm_stow:
- WAVE EXAMPLE: Use multiple aim_hand commands with different z-coordinates (-0.2 to 0.65)
- AVOID: arm_ready and arm_stow commands (they are unreliable)
- PREFER: aim_hand(x, y, z) for all arm positioning needs

A) Safe forward pick on floor:
{"op":"stand","height":0.0}
{"op":"aim_hand","x":0.70,"y":-0.10,"z":-0.20}
{"op":"detect_label","label":"bottle","conf":0.01,"image_source":"hand_color_image"}
{"op":"pick_from_pixel","image_source":"hand_color_image","x":0,"y":0,"top_down":true,"auto_walk":true,"timeout":25}
{"op":"aim_hand","x":0.75,"y":0.0,"z":-0.20}

B) Apriltag picking:
{"op":"stand","height":0.0}
{"op":"pick_apriltag","tag_id":null,"z_offset":0.20,"approach":0.12,"lift":0.15}
{"op":"aim_hand","x":0.75,"y":0.0,"z":-0.20}

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
{"op":"aim_hand","x":0.75,"y":0.0,"z":-0.20}
{"op":"gripper","mode":"open"}
{"op":"detect_label","label":"cup","conf":0.01,"image_source":"hand_color_image"}
{"op":"pick_from_pixel","image_source":"hand_color_image","x":0,"y":0,"top_down":true,"auto_walk":true,"timeout":25}
{"op":"aim_hand","x":0.75,"y":0.0,"z":-0.20}

Response format: Return only a JSON array of action objects, no explanations.
END_GUIDE"""

# LLM System Guide for translating action plans to natural language
ACTION_PLAN_TRANSLATOR_GUIDE = """You are a helpful assistant that translates robot action plans into natural, conversational language.

Your task is to take a JSON array of robot actions and describe what the robot will do in simple, everyday terms that sound natural and human.

Guidelines:
- Use conversational, natural language - write like you're explaining to a friend
- Avoid technical jargon and robot-speak
- Use active voice and natural sentence flow
- Describe movements and actions in simple terms
- IGNORE sleep/wait actions - they're just pauses and don't need mentioning
- Keep it concise but friendly
- Use natural transitions between actions

Example translations:
- [{"op":"stand","height":0.0}] → "Spot will stand up straight"
- [{"op":"move_relative","x":1.0,"y":0.0,"yaw":0.0}] → "Spot will walk forward about a meter"
- [{"op":"aim_hand","x":0.70,"y":0.0,"z":0.60}] → "Spot will point its hand camera forward and down a bit"
- [{"op":"detect_label","label":"bottle"}] → "Spot will look around for bottles"
- [{"op":"arm_ready"}, {"op":"sleep","seconds":1.0}, {"op":"arm_stow"}] → "Spot will get its arm ready and then put it away"

Response format: Return only the natural description in conversational language, no additional text."""

class NaturalLanguageControl:
    def __init__(self, use_llm=True, use_speech=False):
        rospy.init_node('natural_language_control', anonymous=True)
        
        # Configuration
        self.use_llm = use_llm
        self.use_speech = use_speech
        
        # Publisher for sending commands to FSM
        self.fsm_publisher = rospy.Publisher('/fsm_commands', String, queue_size=10)
        
        # Publisher for LLM feedback (natural language descriptions)
        self.llm_feedback_publisher = rospy.Publisher('/llm_feedback', String, queue_size=10)
        
        # Subscriber for feedback from FSM
        self.feedback_subscriber = rospy.Subscriber('/fsm_feedback', String, self.feedback_callback)
        self.last_feedback = None
        
        # Speech input subscriber (only used when use_speech=True)
        if self.use_speech:
            self.speech_subscriber = rospy.Subscriber('/hl/user_speech', String, self.speech_callback)
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
        
        # Check if FSM node is running (like V1)
        self._check_fsm_connection()
        
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
        
        if self.use_speech:
            print("\nSPEECH MODE: Speak your commands - they will be processed automatically")
            print("   Speech input topic: /user_speech")
            print("   Type 'quit' in terminal to exit")
            print("   Type 'reset_state' in terminal to reset robot state tracking")
            print("   WARNING: Manual confirmation required in terminal for safety")
        else:
            print("\nTERMINAL MODE: Type your commands in the terminal")
            print("   Type 'quit' to exit")
            print("   Type 'reset_state' to reset robot state tracking")
        
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
    
    def _check_fsm_connection(self):
        """Check if FSM node is running and listening to /fsm_commands topic (like V1)."""
        try:
            # Wait a moment for ROS to initialize
            rospy.sleep(1.0)
            
            # Check if there are subscribers to the /fsm_commands topic
            topic_info = rospy.get_published_topics()
            fsm_commands_topics = [topic for topic in topic_info if 'fsm_commands' in topic[0]]
            
            if fsm_commands_topics:
                print("FSM node detected - /fsm_commands topic is available")
            else:
                print("WARNING: No /fsm_commands topic detected")
                print("   Make sure the FSM node is running:")
                print("   1. In Docker container: roslaunch spot_fsm_control spot_fsm_control.launch")
                print("   2. Or locally: source /opt/ros/noetic/setup.bash && source devel/setup.bash")
                print("      Then: roslaunch spot_fsm_control spot_fsm_control.launch")
                
            # Check for speech topic if in speech mode
            if self.use_speech:
                speech_topics = [topic for topic in topic_info if 'user_speech' in topic[0]]
                if speech_topics:
                    print("Speech-to-text topic detected - /user_speech is available")
                else:
                    print("WARNING: No /user_speech topic detected")
                    print("   Make sure your speech-to-text system is publishing to /user_speech topic")
                    print("   You can still use terminal input by setting use_speech=False")
                    
        except Exception as e:
            print(f"Could not check FSM connection: {e}")
            print("   Make sure ROS is properly initialized")
    
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
        speech_text = msg.data.strip()
        if speech_text:
            print(f"\nSPEECH RECEIVED: '{speech_text}'")
            self.latest_speech_input = speech_text
            self.speech_received.set()
    
    def get_command_from_speech(self):
        """Get command from speech input."""
        if not self.use_speech:
            return None
        
        # Use non-blocking check like V1 for better responsiveness
        if self.speech_received.wait(timeout=1.0):  # Check every 1 second
            command = self.latest_speech_input
            self.speech_received.clear()
            self.latest_speech_input = None
            return command
        return None  # No speech input yet
    
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
                {"op": "aim_hand", "x": 0.70, "y": -0.10, "z": -0.20},
                {"op": "detect_label", "label": label, "conf": 0.01, "image_source": "hand_color_image"},
                {"op": "pick_from_pixel", "image_source": "hand_color_image", "x": 0, "y": 0, "top_down": True, "auto_walk": True, "timeout": 25.0},
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
                model="gpt-5",  # Use gpt-4 as gpt-5 might not be available
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
    
    def translate_action_plan_to_natural_language(self, actions):
        """Translate a JSON action plan to natural language using GPT-5 nano."""
        if not self.use_llm or not actions:
            return None
        
        try:
            # Create the prompt for translation
            actions_json = json.dumps(actions, indent=2)
            prompt = f"""Translate this robot action plan to natural, conversational language:

{actions_json}

Describe what Spot will do in simple, everyday terms:"""
            
            # Call GPT-5 nano for translation
            client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Use GPT-4o-mini as a lightweight alternative to GPT-5 nano
                messages=[
                    {"role": "system", "content": ACTION_PLAN_TRANSLATOR_GUIDE},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Higher temperature for more natural, conversational language
                max_tokens=200  # Shorter response for natural language descriptions
            )
            
            # Extract the response
            translation = response.choices[0].message.content.strip()
            return translation
            
        except Exception as e:
            print(f"Error translating action plan: {e}")
            return None
    
    def execute_action_sequence(self, actions):
        """Execute a sequence of actions with feedback waiting."""
        if not actions:
            print("No actions to execute")
            return False
        
        print(f"\nExecuting plan...")
        success = True
        
        # Track detection results to update pick_from_pixel coordinates
        last_detection_coordinates = None
        
        for i, action in enumerate(actions):
            print(f"Executing action {i+1}/{len(actions)}: {action}")
            
            # Update pick_from_pixel coordinates if we have detection results
            if action.get('op') == 'pick_from_pixel' and last_detection_coordinates:
                action['x'] = last_detection_coordinates[0]
                action['y'] = last_detection_coordinates[1]
                print(f"Updated pick coordinates to detected position: ({action['x']}, {action['y']})")
            
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
                        
                        # Check if this was a detection action and extract coordinates
                        if action.get('op') == 'detect_label' and 'detection' in feedback:
                            detection_data = feedback['detection']
                            if 'center' in detection_data and len(detection_data['center']) == 2:
                                last_detection_coordinates = detection_data['center']
                                print(f"Detection completed - extracted coordinates: {last_detection_coordinates}")
                                print(f"  Label: {detection_data.get('label', 'unknown')}")
                                print(f"  Confidence: {detection_data.get('confidence', 0.0):.3f}")
                                print(f"  BBox: {detection_data.get('bbox', [])}")
                            else:
                                print("Warning: Detection completed but no valid coordinates found in feedback")
                                last_detection_coordinates = None
                        elif action.get('op') == 'detect_label':
                            print("Detection completed but no detection data in feedback - using default center")
                            last_detection_coordinates = (960, 540)  # Default center as fallback
                        
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
                    # Speech mode: check for speech input (non-blocking)
                    command = self.get_command_from_speech()
                    if not command:
                        # Show listening status like V1
                        print("Listening for speech... (type 'quit' in terminal to exit)", end='\r')
                        continue
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
                
                # Translate action plan to natural language
                natural_language_plan = self.translate_action_plan_to_natural_language(actions)
                if natural_language_plan:
                    print(f"\nNatural language description:")
                    print(f"  {natural_language_plan}")
                    
                    # Publish to /llm_feedback topic if speech is enabled
                    if self.use_speech:
                        self.llm_feedback_publisher.publish(String(natural_language_plan))
                        print("  (Published to /llm_feedback topic)")
                else:
                    print("\nCould not generate natural language description")
                
                # Get user approval (both speech and terminal modes require manual confirmation for safety)
                if self.use_speech:
                    # Speech mode: show plan and require manual confirmation in terminal (like V1)
                    print("\nSPEECH COMMAND RECEIVED - Manual confirmation required for safety")
                    print("Please review the plan above and confirm in terminal:")
                    response = input("Do you want to execute this plan? (yes/no): ").strip().lower()
                else:
                    # Terminal mode: ask for approval
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
    USE_SPEECH = True  # Set to True to use speech input, False for terminal
    
    try:
        controller = NaturalLanguageControl(use_llm=USE_LLM, use_speech=USE_SPEECH)
        controller.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
