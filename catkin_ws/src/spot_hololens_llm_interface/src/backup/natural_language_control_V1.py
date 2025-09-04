#!/usr/bin/env python

import openai
import os
import time
import rospy
from std_msgs.msg import String
import math
import re
import threading

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

class NaturalLanguageControlV8:
    def __init__(self, use_speech=False):
        rospy.init_node('natural_language_control_v8', anonymous=True)
        
        # Configuration: Set to True to use speech input, False for terminal input
        self.use_speech = use_speech
        
        # Publisher for sending commands to FSM
        self.fsm_publisher = rospy.Publisher('/fsm_commands', String, queue_size=10)
        
        # Subscriber for robot status from FSM
        self.robot_status_subscriber = rospy.Subscriber('/robot_status', String, self.robot_status_callback)
        self.last_robot_status = "unknown"
        
        # Speech input subscriber (only used when use_speech=True)
        if self.use_speech:
            self.speech_subscriber = rospy.Subscriber('/user_speech', String, self.speech_callback)
            self.latest_speech_input = None
            self.speech_received = threading.Event()
            print("🎤 SPEECH MODE ENABLED - Listening for speech input on /user_speech topic")
        else:
            print("⌨️  TERMINAL MODE ENABLED - Type commands in terminal")
        
        # Check if FSM node is running
        self._check_fsm_connection()
        
        print("=== NATURAL LANGUAGE ROBOT CONTROL V8 (REAL FSM-BASED) ===")
        print("This system uses GPT-5 to convert natural language to FSM-based robot actions.")
        print("- SAFETY: All actions go through the Finite State Machine!")
        print("- SPEED: Uses supervisor's safe speed (0.3 rad/s, 0.3 m/s)")
        print("- VALIDATION: State transitions and error handling")
        print("- ERROR HANDLING: Detects infeasible requests and prevents execution")
        print("SAFETY: You will see a plan first and must approve before execution.")
        print()
        print("Example instructions:")
        print("- 'I want the robot to make a 360'")
        print("- 'Move forward and then turn left'")
        print("- 'Stand up high and take a picture'")
        print("- 'Walk in a square pattern'")
        print("- 'Sit down and then stand up'")
        print("- 'Turn left'")
        print("- 'Pick the green sphere'")
        print("- 'Pick up the object'")
        print("- 'Grab the sphere'")
        print("- 'Pick up the red sphere'")
        print("- 'Pick up the blue sphere'")
        print("- 'Grab the red object'")
        print("- 'Grab the blue object'")
        print("- 'Pick up the QR code'")
        print("- 'Pick up the marker'")
        print("- 'Grab the tag'")
        print()
        if self.use_speech:
            print("🎤 SPEECH MODE: Speak your commands - they will be processed automatically")
            print("   Speech input topic: /user_speech")
            print("   Type 'quit' in terminal to exit")
            print("   Type 'reset_state' in terminal to reset robot state tracking")
        else:
            print("⌨️  TERMINAL MODE: Type your commands in the terminal")
            print("   Type 'quit' to exit")
            print("   Type 'reset_state' to reset robot state tracking")
        print("-" * 50)
        print()

    def speech_callback(self, data):
        """Callback for speech input from speech-to-text system"""
        speech_text = data.data.strip()
        if speech_text:
            print(f"\n🎤 SPEECH RECEIVED: '{speech_text}'")
            self.latest_speech_input = speech_text
            self.speech_received.set()

    def _check_fsm_connection(self):
        """Check if FSM node is running and listening to /user_speech topic"""
        try:
            # Wait a moment for ROS to initialize
            rospy.sleep(1.0)
            
            # Check if there are subscribers to the /fsm_commands topic
            topic_info = rospy.get_published_topics()
            fsm_commands_topics = [topic for topic in topic_info if 'fsm_commands' in topic[0]]
            
            if fsm_commands_topics:
                print("✅ FSM node detected - /fsm_commands topic is available")
            else:
                print("⚠️  WARNING: No /fsm_commands topic detected")
                print("   Make sure the FSM node is running:")
                print("   1. In Docker container: roslaunch spot_hololens_llm_interface spot_hololens_llm_interface.launch")
                print("   2. Or locally: source /opt/ros/noetic/setup.bash && source devel/setup.bash")
                print("      Then: roslaunch spot_hololens_llm_interface spot_hololens_llm_interface.launch")
                
            # Check for speech topic if in speech mode
            if self.use_speech:
                speech_topics = [topic for topic in topic_info if 'user_speech' in topic[0]]
                if speech_topics:
                    print("✅ Speech-to-text topic detected - /user_speech is available")
                else:
                    print("⚠️  WARNING: No /user_speech topic detected")
                    print("   Make sure your speech-to-text system is publishing to /user_speech topic")
                    print("   You can still use terminal input by setting use_speech=False")
                    
        except Exception as e:
            print(f"⚠️  Could not check FSM connection: {e}")
            print("   Make sure ROS is properly initialized")

    def get_fsm_movement_parameters(self):
        """Get FSM movement parameters based on actual FSM implementation"""
        return {
            'forward_distance': 1.5,      # meters (from two_d_location_body_frame_command(1.5, 0, 0))
            'strafe_distance': 0.75,      # meters (from two_d_location_body_frame_command(0, 0.75, 0))
            'rotation_angle_rad': math.pi/2,  # radians (from two_d_location_body_frame_command(0, 0, math.pi/2))
            'rotation_angle_deg': 90,     # degrees (π/2 radians)
            'movement_speed': 0.3,        # m/s (from self.robot_speed = 0.3)
            'rotation_speed': 0.3,        # rad/s (from self.robot_speed = 0.3)
            'forward_time': 4,            # seconds (reduced from 6s for faster response)
            'strafe_time': 3,             # seconds (reduced from 4s for faster response)
            'rotation_time': 4,           # seconds (reduced from 6s for faster response)
            'stand_up_time': 5,           # seconds (reduced from 8s for faster response)
            'sit_down_time': 4,           # seconds (reduced from 6s for faster response)
            'stand_height_time': 2,       # seconds (reduced from 3s for faster response)
            'stop_action_time': 1,        # seconds (reduced from 2s for faster response)
            'pick_object_time': 10,       # seconds (reduced from 15s for faster response)
            'command_delay': 0.5,         # seconds between commands (reduced from 1s for faster response)
            'max_timeout': 8              # seconds (reduced from 10s for faster response)
        }

    def detect_infeasible_requests(self, instruction):
        """Detect if the user request is infeasible with current FSM parameters"""
        fsm_params = self.get_fsm_movement_parameters()
        instruction_lower = instruction.lower()
        
        # Patterns to detect specific distance/angle requests
        distance_patterns = [
            r'move\s+forward\s+(\d+(?:\.\d+)?)\s*m(?:eters?)?',
            r'walk\s+forward\s+(\d+(?:\.\d+)?)\s*m(?:eters?)?',
            r'go\s+forward\s+(\d+(?:\.\d+)?)\s*m(?:eters?)?',
            r'move\s+backward\s+(\d+(?:\.\d+)?)\s*m(?:eters?)?',
            r'walk\s+backward\s+(\d+(?:\.\d+)?)\s*m(?:eters?)?',
            r'go\s+backward\s+(\d+(?:\.\d+)?)\s*m(?:eters?)?',
            r'move\s+left\s+(\d+(?:\.\d+)?)\s*m(?:eters?)?',
            r'walk\s+left\s+(\d+(?:\.\d+)?)\s*m(?:eters?)?',
            r'strafe\s+left\s+(\d+(?:\.\d+)?)\s*m(?:eters?)?',
            r'move\s+right\s+(\d+(?:\.\d+)?)\s*m(?:eters?)?',
            r'walk\s+right\s+(\d+(?:\.\d+)?)\s*m(?:eters?)?',
            r'strafe\s+right\s+(\d+(?:\.\d+)?)\s*m(?:eters?)?',
        ]
        
        angle_patterns = [
            r'turn\s+(\d+(?:\.\d+)?)\s*degrees?',
            r'rotate\s+(\d+(?:\.\d+)?)\s*degrees?',
            r'make\s+a\s+(\d+(?:\.\d+)?)\s*degree\s+turn',
            r'turn\s+left\s+(\d+(?:\.\d+)?)\s*degrees?',
            r'turn\s+right\s+(\d+(?:\.\d+)?)\s*degrees?',
        ]
        
        # Check for pick object requests that might be infeasible
        pick_patterns = [
            r'pick\s+up\s+(\d+(?:\.\d+)?)\s*objects?',
            r'grab\s+(\d+(?:\.\d+)?)\s*objects?',
            r'pick\s+(\d+(?:\.\d+)?)\s*spheres?',
        ]
        
        for pattern in pick_patterns:
            match = re.search(pattern, instruction_lower)
            if match:
                requested_count = float(match.group(1))
                if requested_count > 1:
                    return {
                        'infeasible': True,
                        'message': f"Cannot pick {requested_count} objects. FSM only supports picking one object at a time."
                    }
        
        # Check for distance requests
        for pattern in distance_patterns:
            match = re.search(pattern, instruction_lower)
            if match:
                requested_distance = float(match.group(1))
                if 'forward' in pattern or 'backward' in pattern:
                    if abs(requested_distance - fsm_params['forward_distance']) > 0.1:  # Allow small tolerance
                        return {
                            'infeasible': True,
                            'type': 'distance',
                            'requested': requested_distance,
                            'available': fsm_params['forward_distance'],
                            'direction': 'forward/backward',
                            'message': f"ERROR: Cannot move {requested_distance}m. FSM only supports exactly {fsm_params['forward_distance']}m forward/backward movements."
                        }
                elif 'left' in pattern or 'right' in pattern:
                    if abs(requested_distance - fsm_params['strafe_distance']) > 0.1:  # Allow small tolerance
                        return {
                            'infeasible': True,
                            'type': 'distance',
                            'requested': requested_distance,
                            'available': fsm_params['strafe_distance'],
                            'direction': 'left/right',
                            'message': f"ERROR: Cannot strafe {requested_distance}m. FSM only supports exactly {fsm_params['strafe_distance']}m left/right movements."
                        }
        
        # Check for angle requests
        for pattern in angle_patterns:
            match = re.search(pattern, instruction_lower)
            if match:
                requested_angle = float(match.group(1))
                # Check if angle is not a multiple of 90 degrees
                if abs(requested_angle % 90) > 1:  # Allow 1 degree tolerance
                    return {
                        'infeasible': True,
                        'type': 'angle',
                        'requested': requested_angle,
                        'available': fsm_params['rotation_angle_deg'],
                        'message': f"ERROR: Cannot turn {requested_angle}°. FSM only supports exactly {fsm_params['rotation_angle_deg']}° rotations (multiples of 90°)."
                    }
        
        # Check for speed requests
        speed_patterns = [
            r'at\s+(\d+(?:\.\d+)?)\s*m/s',
            r'with\s+speed\s+(\d+(?:\.\d+)?)\s*m/s',
            r'(\d+(?:\.\d+)?)\s*m/s',
        ]
        
        for pattern in speed_patterns:
            match = re.search(pattern, instruction_lower)
            if match:
                requested_speed = float(match.group(1))
                if abs(requested_speed - fsm_params['movement_speed']) > 0.05:  # Allow small tolerance
                    return {
                        'infeasible': True,
                        'type': 'speed',
                        'requested': requested_speed,
                        'available': fsm_params['movement_speed'],
                        'message': f"ERROR: Cannot move at {requested_speed}m/s. FSM only supports exactly {fsm_params['movement_speed']}m/s movement speed."
                    }
        
        # Check for unsupported complex movements
        unsupported_patterns = [
            r'arc',
            r'curve',
            r'diagonal',
            r'crab\s+walk',
            r'backward\s+and\s+left',
            r'forward\s+and\s+right',
        ]
        
        for pattern in unsupported_patterns:
            if re.search(pattern, instruction_lower):
                return {
                    'infeasible': True,
                    'type': 'movement_pattern',
                    'message': f"ERROR: '{pattern}' movement is not supported. FSM only supports straight forward/backward, left/right strafe, and 90° rotations."
                }
        
        return {'infeasible': False}

    def get_current_robot_state(self):
        """Get current robot state from FSM"""
        # Use our tracked state (which is updated based on commands we send)
        if hasattr(self, '_current_state'):
            return self._current_state
        else:
            self._current_state = 'sit'  # Default initial state
            return self._current_state

    def robot_status_callback(self, data):
        """Callback for robot status updates from FSM"""
        self.last_robot_status = data.data
        print(f"Robot status update: {data.data}")

    def update_robot_state(self, new_state):
        """Update the tracked robot state based on commands sent"""
        self._current_state = new_state

    def _set_target_label(self, label):
        """
        Set the target label for label-based picking.
        This publishes a special message to the FSM to configure the target label.
        
        Args:
            label: The target label (e.g., 'cup', 'bottle', 'person')
        """
        print(f"Setting target label to: {label}")
        try:
            # Publish a special configuration message
            config_message = f"CONFIG_TARGET_LABEL:{label}"
            self.fsm_publisher.publish(config_message)
            print(f"✅ Target label '{label}' configured successfully")
            time.sleep(0.5)  # Brief delay to ensure message is processed
        except Exception as e:
            print(f"❌ Error setting target label: {e}")

    def _execute_and_track_state(self, command, new_state):
        """
        Execute a command and update the tracked robot state.
        
        Args:
            command: The command to execute
            new_state: The expected new state after command execution
        """
        print(f"Executing: {command} (state: {self._current_state} -> {new_state})")
        
        # Prevent duplicate pick commands when already in pick state
        if "pick" in command and self._current_state in ['pick_green', 'pick_red', 'pick_blue', 'pick_object', 'pick_qr', 'pick_label']:
            print(f"⚠️  Skipping {command} - robot is already in pick state: {self._current_state}")
            return
        
        try:
            self.fsm_publisher.publish(command)
            print(f"✅ Command '{command}' published to FSM successfully")
            self.update_robot_state(new_state)
            
            # Add appropriate delays for different command types (reduced for faster response)
            if "pick" in command:
                time.sleep(2.0)  # Reduced delay for pick operations (was 5.0)
            elif "stand_up" in command:
                time.sleep(1.5)  # Reduced delay for stand up operations (was 3.0)
            elif "stop_action" in command:
                time.sleep(0.5)  # Reduced delay for stop operations (was 2.0)
            else:
                time.sleep(0.3)  # Reduced standard delay for other operations (was 1.0)
                
        except Exception as e:
            print(f"❌ Error publishing command '{command}' to FSM: {e}")
            print("⚠️  Make sure the FSM node is running in the same ROS environment")
            print("   Run: source /opt/ros/noetic/setup.bash && source devel/setup.bash")
            print("   Then: roslaunch spot_hololens_llm_interface spot_hololens_llm_interface.launch")
            print("   The FSM node should be listening to /fsm_commands topic")

    def get_action_plan(self, instruction):
        """Get action plan from GPT-5"""
        
        # Dynamic FSM movement parameters based on actual FSM implementation
        fsm_params = self.get_fsm_movement_parameters()
        
        system_prompt = f"""You are a robot control system that converts natural language instructions into FSM-based robot actions.

ROBOT STATE AWARENESS:
The robot can be in different states, and only certain actions are allowed from each state:

CURRENT ROBOT STATE: {self.get_current_robot_state()}

EXACT FSM STATE TRANSITIONS (from finite_state_machine.py):

INITIAL STATE:
- sit = State(initial=True)

STANDING STATES:
- stand = State()
- stand_high = State() 
- stand_low = State()

MOVEMENT STATES:
- walk_forward = State()
- walk_backward = State()
- walk_left = State()
- walk_right = State()
- turn_left = State()
- turn_right = State()

ACTION STATES:
- deitic_location_movement = State()
- face_operator = State()
- pick_object = State()
- pick_green = State()
- pick_red = State()
- pick_blue = State()
- gaze_control = State()
- arm_trajectory = State()
- direct_arm_control = State()

EXACT TRANSITION DEFINITIONS:
- sit_down = (stand.to(sit) | stand_high.to(sit) | stand_low.to(sit))
- stand_up = (sit.to(stand) | stand_high.to(stand) | stand_low.to(stand))
- stand_up_high = (stand.to(stand_high) | sit.to(stand_high) | stand_low.to(stand_high))
- stand_up_low = (stand.to(stand_low) | sit.to(stand_low) | stand_high.to(stand_low))

- walk_to_forward = stand.to(walk_forward)
- walk_to_backward = stand.to(walk_backward)
- walk_to_left = stand.to(walk_left)
- walk_to_right = stand.to(walk_right)

- turn_to_left = stand.to(turn_left)
- turn_to_right = stand.to(turn_right)

- move_to_location = stand.to(deitic_location_movement)
- face_to_operator = stand.to(face_operator)
- pick_up_object = stand.to(pick_object)
- pick_up_green = stand.to(pick_green)
- pick_up_red = stand.to(pick_red)
- pick_up_blue = stand.to(pick_blue)
- pick_up_qr = stand.to(pick_qr)
- pick_up_label = stand.to(pick_label)
- start_gaze = stand.to(gaze_control)
- start_trajectory = stand.to(arm_trajectory)
- start_direct_arm_control = stand.to(direct_arm_control)

- stop_action = (
    walk_forward.to(stand) |
    walk_backward.to(stand) |
    walk_left.to(stand) |
    walk_right.to(stand) |
    turn_right.to(stand) |
    turn_left.to(stand) |
    deitic_location_movement.to(stand) |
    face_operator.to(stand) |
    pick_object.to(stand) |
    pick_green.to(stand) |
    pick_red.to(stand) |
    pick_blue.to(stand) |
    pick_qr.to(stand) |
    pick_label.to(stand) |
    arm_trajectory.to(stand) |
    gaze_control.to(stand) |
    direct_arm_control.to(stand)
)

CRITICAL PRECONDITIONS:
1. ALL movement commands (walk_to_*, turn_to_*) require robot to be in 'stand' state
2. stop_action() can ONLY be called from movement/action states, NEVER from sit/stand/stand_high/stand_low
3. stand_up() can be called from sit, stand_high, or stand_low states
4. sit_down() can be called from stand, stand_high, or stand_low states
5. stand_up_high/low() can be called from sit, stand, or the other stand height state
6. If robot is in 'sit' state and you need movement, you MUST call stand_up() first
7. If robot is in movement state and you need to stop, you MUST call stop_action() first
8. NEVER call stop_action() when robot is in sit, stand, stand_high, or stand_low states
9. pick_up_green(), pick_up_red(), pick_up_blue(), pick_up_qr(), pick_up_label() can ONLY be called from 'stand' state
10. NEVER call the same pick command twice in a sequence - each pick command should only appear once
11. After a pick command, always call stop_action() to return to stand state
12. After ANY movement command (walk_to_*, turn_to_*), you MUST call stop_action() to return to stand state before doing anything else
13. Movement commands (walk_to_*, turn_to_*) are BLOCKING - they stay in movement state until stop_action() is called

Available robot actions (use these EXACT names - these are the only ones that work):
- stand_up(): Stand up from sitting position
- sit_down(): Sit down from standing position
- stand_up_high(): Stand up high
- stand_up_low(): Stand up low
- stop_action(): Stop all movement and return to stand state
- pick_up_object(): Pick up object (creates green sphere and picks it)
- pick_up_green(): Pick up green sphere (center position)
- pick_up_red(): Pick up red sphere (left position)
- pick_up_blue(): Pick up blue sphere (right position)
- pick_up_qr(): Pick up QR code/marker (detects nearest AprilTag and picks 20cm above it)
- set_target_label(label): Configure target label for label-based picking (e.g., set_target_label("cup"))
- pick_up_label(): Pick up object by label (detects object using YOLO/OpenCV and picks it using Manipulation API)

ACTION COMBINATIONS (these can be created by combining basic actions):
- bounce/jump: stand_up() + sit_down()
- wave: turn_to_left() + stop_action() + turn_to_right() + stop_action()
- dance: turn_to_left() + stop_action() + turn_to_right() + stop_action() + turn_to_left() + stop_action() + turn_to_right() + stop_action()
- square: walk_to_forward() + stop_action() + turn_to_left() + stop_action() + walk_to_forward() + stop_action() + turn_to_left() + stop_action() + walk_to_forward() + stop_action() + turn_to_left() + stop_action() + walk_to_forward() + stop_action()
- 360/circle: turn_to_left() + stop_action() + turn_to_left() + stop_action() + turn_to_left() + stop_action() + turn_to_left() + stop_action()

FSM movement commands (these are the ONLY movement commands available):
- walk_to_forward(): Walk forward exactly {fsm_params['forward_distance']} meters (uses FSM safe speed: {fsm_params['movement_speed']} m/s)
- walk_to_backward(): Walk backward exactly {fsm_params['forward_distance']} meters (uses FSM safe speed: {fsm_params['movement_speed']} m/s)
- walk_to_left(): Walk left exactly {fsm_params['strafe_distance']} meters (uses FSM safe speed: {fsm_params['movement_speed']} m/s)
- walk_to_right(): Walk right exactly {fsm_params['strafe_distance']} meters (uses FSM safe speed: {fsm_params['movement_speed']} m/s)
- turn_to_left(): Turn left exactly {fsm_params['rotation_angle_deg']} degrees/{fsm_params['rotation_angle_rad']} radians (uses FSM safe speed: {fsm_params['rotation_speed']} rad/s)
- turn_to_right(): Turn right exactly {fsm_params['rotation_angle_deg']} degrees/{fsm_params['rotation_angle_rad']} radians (uses FSM safe speed: {fsm_params['rotation_speed']} rad/s)

IMPORTANT: These are the ONLY commands that work with the real FSM. No custom distances or angles are supported.

REAL FSM MOVEMENT PARAMETERS:
- Forward/backward: {fsm_params['forward_distance']} meters (exact distance, position-based)
- Left/right strafe: {fsm_params['strafe_distance']} meters (exact distance, position-based)
- Rotation: {fsm_params['rotation_angle_deg']} degrees/{fsm_params['rotation_angle_rad']} radians (exact angle, position-based)
- Speed: {fsm_params['movement_speed']} m/s for movement, {fsm_params['rotation_speed']} rad/s for rotation
- All movements use trajectory commands to exact positions
- FSM waits until robot reaches goal and is settled (no fixed timing)

PICK OBJECT FUNCTIONALITY:
- pick_up_object(): Creates a green sphere at 0.45m forward, 0.20m height and picks it up
- pick_up_green(): Creates a green sphere at center position (0.45m forward, 0.0m left/right, 0.20m height)
- pick_up_red(): Creates a red sphere at left position (0.45m forward, -0.5m left, 0.20m height)
- pick_up_blue(): Creates a blue sphere at right position (0.45m forward, 0.5m right, 0.20m height)
- pick_up_qr(): Detects nearest AprilTag in World Objects, creates orange sphere 20cm above tag, and picks it
- pick_up_label(): Detects object by label using YOLO/OpenCV and picks using Manipulation API PickObjectInImage
- Sphere radius: 0.05m (5cm) for colored spheres, 0.06m (6cm) for QR visualization
- Pick operation includes: arm ready → pick → arm stow
- Timeout: 30 seconds for pick completion
- Only works when robot is in 'stand' state
- Creates visual colored spheres for target visualization
- QR picking requires AprilTags to be detected by World Object service
- Label-based picking requires object to be visible in body stereo cameras and detectable by YOLO model

REALISTIC TIMING (based on actual robot capabilities):
- Forward/backward movements: {fsm_params['forward_time']} seconds (1.5m ÷ 0.3 m/s = 5s + buffer)
- Left/right strafe movements: {fsm_params['strafe_time']} seconds (0.75m ÷ 0.3 m/s = 2.5s + buffer)
- Rotation movements: {fsm_params['rotation_time']} seconds (π/2 ÷ 0.3 rad/s = 5.2s + buffer)
- Stand up: {fsm_params['stand_up_time']} seconds (blocking_stand typically 6-8s)
- Sit down: {fsm_params['sit_down_time']} seconds (blocking_sit typically 4-6s)
- Stand high/low: {fsm_params['stand_height_time']} seconds (height change typically 2-3s)
- Stop action: {fsm_params['stop_action_time']} seconds (stop command typically 1-2s)

Timing rules:
1. Always start with stop_action() to ensure clean state
2. FSM commands are position-based - they move to exact positions
3. For {fsm_params['rotation_angle_deg']}° turns: use turn_to_left() or turn_to_right()
4. For {fsm_params['rotation_angle_deg']*2}° turns: use 2 × turn_to_left() 
5. For {fsm_params['rotation_angle_deg']*4}° turns: use 4 × turn_to_left()
6. For {fsm_params['forward_distance']}m movements: use walk_to_forward/backward
7. For {fsm_params['strafe_distance']}m movements: use walk_to_left/right
8. Always end with stop_action() to stop movement
9. Use time.sleep({fsm_params['forward_time']}) after forward/backward movements
10. Use time.sleep({fsm_params['strafe_time']}) after left/right strafe movements
11. Use time.sleep({fsm_params['rotation_time']}) after rotation movements
12. Use time.sleep({fsm_params['stand_up_time']}) after stand_up() commands
13. Use time.sleep({fsm_params['sit_down_time']}) after sit_down() commands
14. Use time.sleep({fsm_params['stand_height_time']}) after stand_up_high/low() commands
15. Use time.sleep({fsm_params['stop_action_time']}) after stop_action() commands
16. Use time.sleep({fsm_params['pick_object_time']}) after pick_up_object() commands
17. Use time.sleep({fsm_params['command_delay']}) between different command types

ERROR HANDLING:
If the user requests something that cannot be achieved with the available FSM parameters, return "ERROR: [specific reason why the request is infeasible]" instead of action code.

Examples of infeasible requests:
- "Move forward 4 meters" → ERROR: Cannot move 4m. FSM only supports exactly 1.5m forward/backward movements.
- "Turn 45 degrees" → ERROR: Cannot turn 45°. FSM only supports exactly 90° rotations (multiples of 90°).
- "Move at 2 m/s" → ERROR: Cannot move at 2m/s. FSM only supports exactly 0.3m/s movement speed.
- "Walk in a circle" → ERROR: Circle movement is not supported. FSM only supports straight movements and 90° rotations.
- "Pick up 3 objects" → ERROR: Cannot pick 3 objects. FSM only supports picking one object at a time.

Example:
Input: "make a 180" (robot in 'sit' state)
Output: 
stand_up()
time.sleep({fsm_params['stand_up_time']})
turn_to_left()
time.sleep({fsm_params['rotation_time']})
turn_to_left()
time.sleep({fsm_params['rotation_time']})
stop_action()

Input: "move forward and then turn right" (robot in 'stand' state)
Output:
walk_to_forward()
time.sleep({fsm_params['forward_time']})
turn_to_right()
time.sleep({fsm_params['rotation_time']})
stop_action()

Input: "turn left" (robot in 'sit' state)
Output:
stand_up()
time.sleep({fsm_params['stand_up_time']})
turn_to_left()
time.sleep({fsm_params['rotation_time']})
stop_action()

Input: "walk in a square" (robot in 'stand' state)
Output:
walk_to_forward()
time.sleep({fsm_params['forward_time']})
turn_to_left()
time.sleep({fsm_params['rotation_time']})
walk_to_forward()
time.sleep({fsm_params['forward_time']})
turn_to_left()
time.sleep({fsm_params['rotation_time']})
walk_to_forward()
time.sleep({fsm_params['forward_time']})
turn_to_left()
time.sleep({fsm_params['rotation_time']})
walk_to_forward()
time.sleep({fsm_params['forward_time']})
stop_action()

Input: "sit down and then walk forward" (robot in 'stand' state)
Output:
sit_down()
time.sleep({fsm_params['sit_down_time']})
stand_up()
time.sleep({fsm_params['stand_up_time']})
walk_to_forward()
time.sleep({fsm_params['forward_time']})
stop_action()

Input: "stand up high and turn left" (robot in 'sit' state)
Output:
stand_up_high()
time.sleep({fsm_params['stand_height_time']})
stand_up()
time.sleep({fsm_params['stand_up_time']})
turn_to_left()
time.sleep({fsm_params['rotation_time']})
stop_action()

Input: "pick the green sphere" (robot in 'sit' state)
Output:
stand_up()
time.sleep({fsm_params['stand_up_time']})
pick_up_green()
time.sleep({fsm_params['pick_object_time']})
stop_action()

Input: "pick the green sphere" (robot in 'stand' state)
Output:
pick_up_green()
time.sleep({fsm_params['pick_object_time']})
stop_action()

Input: "pick up the object" (robot in 'stand' state)
Output:
pick_up_object()
time.sleep({fsm_params['pick_object_time']})
stop_action()

Input: "grab the sphere" (robot in 'stand' state)
Output:
pick_up_object()
time.sleep({fsm_params['pick_object_time']})
stop_action()

Input: "pick up the red sphere" (robot in 'stand' state)
Output:
pick_up_red()
time.sleep({fsm_params['pick_object_time']})
stop_action()

Input: "pick up the blue sphere" (robot in 'stand' state)
Output:
pick_up_blue()
time.sleep({fsm_params['pick_object_time']})
stop_action()

Input: "grab the red object" (robot in 'sit' state)
Output:
stand_up()
time.sleep({fsm_params['stand_up_time']})
pick_up_red()
time.sleep({fsm_params['pick_object_time']})
stop_action()

Input: "pick the red ball" (robot in 'stand' state)
Output:
pick_up_red()
time.sleep({fsm_params['pick_object_time']})
stop_action()

Input: "grab the blue ball" (robot in 'stand' state)
Output:
pick_up_blue()
time.sleep({fsm_params['pick_object_time']})
stop_action()

Input: "pick the green ball" (robot in 'stand' state)
Output:
pick_up_green()
time.sleep({fsm_params['pick_object_time']})
stop_action()

Input: "pick the cup" (robot in 'stand' state)
Output:
set_target_label("cup")
pick_up_label()
time.sleep({fsm_params['pick_object_time']})
stop_action()

Input: "pick up the bottle" (robot in 'stand' state)
Output:
set_target_label("bottle")
pick_up_label()
time.sleep({fsm_params['pick_object_time']})
stop_action()

Input: "grab the person" (robot in 'stand' state)
Output:
set_target_label("person")
pick_up_label()
time.sleep({fsm_params['pick_object_time']})
stop_action()



Input: "pick the cup" (robot in 'sit' state)
Output:
stand_up()
time.sleep({fsm_params['stand_up_time']})
set_target_label("cup")
pick_up_label()
time.sleep({fsm_params['pick_object_time']})
stop_action()

Input: "stop moving" (robot in 'turn_left' state)
Output:
stop_action()

Input: "stop moving" (robot in 'sit' state)
Output:
ERROR: Cannot call stop_action() from 'sit' state. stop_action() can only be called from movement/action states.

Input: "move forward 4 meters" (robot in 'stand' state)
Output:
ERROR: Cannot move 4m. FSM only supports exactly 1.5m forward/backward movements.

Input: "turn 45 degrees" (robot in 'stand' state)
Output:
ERROR: Cannot turn 45°. FSM only supports exactly 90° rotations (multiples of 90°).

Return only the action code or ERROR message, no explanations.
"""

        try:
            # Use GPT-5 API format
            client = openai.OpenAI()
            
            # Combine system prompt and user input for GPT-5
            full_input = f"""{system_prompt}

User instruction: '{instruction}'

Please convert this instruction to robot actions:"""
            
            response = client.responses.create(
                model="gpt-5",
                input=full_input,
                reasoning={"effort": "minimal"}
            )
            
            # Extract the action code from the response
            content = response.output_text.strip()
            
            # Remove any markdown formatting
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content.rsplit("\n", 1)[0]
            
            return content
            
        except Exception as e:
            print(f"Error getting action plan: {e}")
            return "stop_action()"

    def execute_actions(self, actions_text):
        """Execute actions using the FSM-based controller"""
        if not actions_text:
            print("No actions to execute")
            return
            
        print(f"Executing SAFE actions via FSM:\n{actions_text}")
        print("-" * 50)

        # Create safe namespace with ONLY the FSM commands that actually exist
        local_namespace = {
            'stand_up': lambda: self._execute_and_track_state("stand_up", "stand"),
            'stand_up_high': lambda: self._execute_and_track_state("stand_up_high", "stand_high"),
            'stand_up_low': lambda: self._execute_and_track_state("stand_up_low", "stand_low"),
            'sit_down': lambda: self._execute_and_track_state("sit_down", "sit"),
            'stop_action': lambda: self._execute_and_track_state("stop_action", "stand"),
            'walk_to_forward': lambda: self._execute_and_track_state("walk_to_forward", "walk_forward"),
            'walk_to_backward': lambda: self._execute_and_track_state("walk_to_backward", "walk_backward"),
            'walk_to_left': lambda: self._execute_and_track_state("walk_to_left", "walk_left"),
            'walk_to_right': lambda: self._execute_and_track_state("walk_to_right", "walk_right"),
            'turn_to_left': lambda: self._execute_and_track_state("turn_to_left", "turn_left"),
            'turn_to_right': lambda: self._execute_and_track_state("turn_to_right", "turn_right"),
            'face_to_operator': lambda: self._execute_and_track_state("face_to_operator", "face_operator"),
            'pick_up_object': lambda: self._execute_and_track_state("pick_up_object", "pick_object"),
            'pick_up_green': lambda: self._execute_and_track_state("pick_up_green", "pick_green"),
            'pick_up_red': lambda: self._execute_and_track_state("pick_up_red", "pick_red"),
            'pick_up_blue': lambda: self._execute_and_track_state("pick_up_blue", "pick_blue"),
            'pick_up_qr': lambda: self._execute_and_track_state("pick_up_qr", "pick_qr"),
            'pick_up_label': lambda: self._execute_and_track_state("pick_up_label", "pick_label"),

            'set_target_label': lambda label: self._set_target_label(label),
            'gaze_control': lambda: self._execute_and_track_state("gaze_control", "gaze_control"),
            'start_trajectory': lambda: self._execute_and_track_state("start_trajectory", "arm_trajectory"),
            'start_direct_arm_control': lambda: self._execute_and_track_state("start_direct_arm_control", "direct_arm_control"),
            'time': time,
            'print': print,
            'pi': math.pi,
            'math': math
        }
        
        safe_globals = {"__builtins__": {}}

        try:
            exec(actions_text, safe_globals, local_namespace)
            print("-" * 50)
            print("SAFE actions completed successfully via FSM!")
        except Exception as e:
            print(f"ERROR: Failed to execute actions: {e}")
            print("Please check the action format and try again.")
            # Add a small delay to prevent rapid command sending
            time.sleep(1.0)

    def run(self):
        """Main loop for natural language control"""
        
        while not rospy.is_shutdown():
            try:
                # Get instruction based on input mode
                if self.use_speech:
                    # Speech mode: wait for speech input
                    print("🎤 Listening for speech... (type 'quit' in terminal to exit)")
                    if not self.speech_received.wait(timeout=1.0):  # Check for speech every 1 second
                        continue
                        
                    instruction = self.latest_speech_input
                    self.latest_speech_input = None  # Clear the input after processing
                    self.speech_received.clear()  # Reset the event
                    
                    # In speech mode, we can't get terminal input for quit/reset, so handle them differently
                    if instruction.lower() in ['quit', 'exit', 'q', 'stop']:
                        print("🎤 Speech command 'quit' received. Exiting...")
                        break
                        
                    if instruction.lower() in ['reset', 'reset state', 'reset robot']:
                        self._current_state = 'sit'
                        print("🎤 Speech command 'reset' received. Robot state reset to 'sit'")
                        continue
                else:
                    # Terminal mode: get input from terminal
                    instruction = input("\nEnter your instruction: ").strip()
                    
                    if instruction.lower() in ['quit', 'exit', 'q']:
                        print("Exiting...")
                        break
                        
                    if instruction.lower() == 'reset_state':
                        self._current_state = 'sit'
                        print("Robot state reset to 'sit'")
                        continue
                
                if not instruction:
                    continue
                
                print(f"\nProcessing: '{instruction}'")
                
                # First, check if the request is feasible
                feasibility_check = self.detect_infeasible_requests(instruction)
                if feasibility_check['infeasible']:
                    print("\n" + "=" * 60)
                    print("INFEASIBLE REQUEST DETECTED")
                    print("=" * 60)
                    print(feasibility_check['message'])
                    print()
                    print("Available FSM parameters:")
                    fsm_params = self.get_fsm_movement_parameters()
                    print(f"- Forward/backward distance: {fsm_params['forward_distance']}m")
                    print(f"- Left/right strafe distance: {fsm_params['strafe_distance']}m")
                    print(f"- Rotation angle: {fsm_params['rotation_angle_deg']}°")
                    print(f"- Movement speed: {fsm_params['movement_speed']}m/s")
                    print(f"- Rotation speed: {fsm_params['rotation_speed']}rad/s")
                    print()
                    print("Robot state unchanged. Please try a different instruction.")
                    continue
                
                # Get action plan from GPT-5
                actions = self.get_action_plan(instruction)
                
                # Check if GPT-5 returned an error
                if actions.strip().startswith("ERROR:"):
                    print("\n" + "=" * 60)
                    print("GPT-5 DETECTED INFEASIBLE REQUEST")
                    print("=" * 60)
                    print(actions)
                    print()
                    print("Robot state unchanged. Please try a different instruction.")
                    continue
                
                print("\n" + "=" * 60)
                print("SAFE ACTION PLAN (via FSM):")
                print("=" * 60)
                print(f"Instruction: '{instruction}'")
                print()
                print("Actions to be executed (all via FSM):")
                print("-" * 40)
                print(actions)
                print("-" * 40)
                print()
                
                # Get user approval (both speech and terminal modes require manual confirmation for safety)
                if self.use_speech:
                    # Speech mode: show plan and require manual confirmation in terminal
                    print("🎤 SPEECH COMMAND RECEIVED - Manual confirmation required for safety")
                    print("Please review the plan above and confirm in terminal:")
                    response = input("Do you want to execute this SAFE plan? (yes/no/modify): ").strip().lower()
                else:
                    # Terminal mode: ask for approval
                    response = input("Do you want to execute this SAFE plan? (yes/no/modify): ").strip().lower()
                
                if response in ['no', 'n']:
                    print("Plan cancelled.")
                    continue
                elif response in ['modify', 'm']:
                    print("Modification not implemented yet. Please try a different instruction.")
                    continue
                elif response not in ['yes', 'y']:
                    print("Invalid response. Plan cancelled.")
                    continue
                
                print("\nExecuting SAFE plan via FSM...")
                
                # Execute the actions
                self.execute_actions(actions)
                
            except KeyboardInterrupt:
                print("\nInterrupted by user. Stopping robot...")
                self.fsm_publisher.publish("stop_action")
                break
            except Exception as e:
                print(f"Error: {e}")
                self.fsm_publisher.publish("stop_action")

if __name__ == "__main__":
    try:
        # Configuration: Set to True for speech input, False for terminal input
        USE_SPEECH = False # Change this to True to enable speech mode
        
        if USE_SPEECH:
            print("🎤 Starting in SPEECH MODE")
            controller = NaturalLanguageControlV8(use_speech=True)
        else:
            print("⌨️  Starting in TERMINAL MODE")
            controller = NaturalLanguageControlV8(use_speech=False)
            
        controller.run()
    except rospy.ROSInterruptException:
        pass
