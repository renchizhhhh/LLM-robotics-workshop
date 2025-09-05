#!/usr/bin/env python3
import rospy
import sys
import os
import openai
import threading
from std_msgs.msg import String

# Add the FSM to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'spot_hololens_llm_interface'))
from finite_state_machine import SpotStateMachine

# LLM prompt for converting natural language to FSM commands
PROMPT = """Convert natural language to FSM actions for Boston Dynamics Spot robot.

Available FSM actions:
- "stand_up" - Stand up
- "sit_down" - Sit down
- "start_moving", x=float, y=float, yaw=float, frame="body" - Move robot
- "get_image", image_source="camera_name" - Take image

Constraints:
- Stand before movement/manipulation
- Movement: x[-5,5], y[-3,3] meters, yaw in radians, movements and rotations should happen seperately
- Image sources: frontleft_fisheye_image, frontright_fisheye_image, left_fisheye_image, right_fisheye_image, back_fisheye_image
- Format: one action per line, no quotes/brackets, no numbering

Return action list:"""

class NaturalLanguageControl:
    def __init__(self, use_speech=False):
        rospy.init_node('nl_control', anonymous=True)
        self.use_speech = use_speech
        
        # Check if dummy mode is set globally
        dummy_mode = rospy.get_param('/spot_fsm/dummy_mode', False)
        self.spot_fsm = SpotStateMachine(dummy_mode=dummy_mode)
        
        # Auto-connect and power on robot
        rospy.loginfo("Auto-connecting robot...")
        self.spot_fsm.send("connect")
        rospy.sleep(1.0)
        self.spot_fsm.send("power_on")
        rospy.sleep(1.0)
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
    
    def parse_command(self, command):
        """Parse natural language command using LLM."""
        if not self.llm:
            return None
            
        try:
            response = self.llm.responses.create(
                model="gpt-5",
                input=f"{PROMPT}\n\nCommand: {command}",
                reasoning={"effort": "minimal"}
            )
            
            result = response.output_text.strip()
            return [line.strip() for line in result.split('\n') if line.strip()]
            
        except Exception as e:
            rospy.logerr(f"LLM failed: {e}")
            return None
    
    def execute_actions(self, actions):
        """Execute FSM actions."""
        if not actions:
            return False
        
        for action in actions:
            try:
                rospy.loginfo(f"Executing: {action}")
                
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
        print(f"\nProcessing: {command}")
        
        actions = self.parse_command(command)
        if not actions:
            print("Parse failed - try a different command")
            return
        
        print(f"\nLLM Generated Plan:")
        for i, action in enumerate(actions, 1):
            print(f"  {i}. {action}")
        
        confirm = input("\nExecute this plan? (y/n): ").strip().lower()
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

if __name__ == "__main__":
    main()
