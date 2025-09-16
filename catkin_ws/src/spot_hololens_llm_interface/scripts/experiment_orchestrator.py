#!/usr/bin/env python3
import os
import re
import json
import math
import threading

import rospy
from std_msgs.msg import String

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'spot_hololens_llm_interface'))
from finite_state_machine import SpotStateMachine 
from timing_utils import recorder          
import openai
from google import genai 

LA_PROMPT = """\
You are converting a natural-language command into 1-2 Spot FSM actions.
Only include actions necessary to fulfill the command.
Return one action per line, format:
- "stand_up"
- "sit_down"
- "start_moving", x=<float> y=<float> yaw=<float> frame=vision
- "arm_command", command_type="open|close|stow|carry

Command: {command}
Current robot state: {current_state}

Constraints:
- Only include "stand_up" if robot is not already standing or moving
- Movement and manipulation require standing or moving state
- Movement: x[-5,5], y[-3,3] meters, yaw in radians, movements and rotations should happen seperately
- Image sources: frontleft_fisheye_image, frontright_fisheye_image, left_fisheye_image, right_fisheye_image, back_fisheye_image
- Format: one action per line, no quotes/brackets, no numbering

Return action(s):
"""

HA_PROMPT = """You are a path planning system for a Boston Dynamics Spot robot.

CURRENT STATE:
- Robot state: {current_state}
- Current position (vision frame): {current_position}

WORLD LAYOUT (vision frame coordinates):
- Vegetables: (1.0, 0.0, 0.0)
- Fruits: (3.0, 0.0, 0.0)  
- Meat: (4.0, 0.0, 0.0)
- OBSTACLE: Square at (2.0, 0.0, 0.0) - AVOID x: 1.85-2.15m, y: -0.15 to 0.15m

ROBOT SPECS:
- Size: 0.7m wide × 1.4m long
- Movement: Uses vision frame for positioning
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
6. In one move, the robot can either move in the x direction, the y direction, or the yaw direction, not two or three at once.
7. Use RELATIVE vision frame coordinates for movements - each movement is relative to current position
8. For movements, calculate: move_x = target_x - current_x, move_y = target_y - current_y

PLANNING PROCESS:
1. Parse user command to identify destinations in order
2. Calculate relative vision frame movements from current position to each destination
3. Plan collision-free path visiting each destination once in order
4. Check each movement segment for robot-obstacle collision
5. Generate action sequence with relative vision frame movements

OUTPUT FORMAT:
Return only the action list, one action per line.

Task: {task}

Actions:"""

def deg2rad(deg):
    try:
        return float(deg) * math.pi / 180.0
    except Exception:
        return 0.0

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class ExperimentState(object):
    def __init__(self, mode='LA'):
        self.mode = mode  # 'LA' or 'HA'
        self.current_task = None
        self.current_plan_id = 0
        self.pending_plan = [] 
        self.lock = threading.Lock()
        self.awaiting_approval = False
        # Position tracking for HA mode  
        self.current_position = [0.0, 0.0, 0.0]  # [x, y, yaw] relative to start position
        self.start_position = None  # Will be set on first position read

class ExperimentOrchestrator(object):
    def __init__(self):
        rospy.init_node('experiment_orchestrator', anonymous=True)
        self.state = ExperimentState(mode=rospy.get_param('~mode', 'HA').upper())

        # FSM wrapper (service/action client under the hood)
        self.dummy_mode = rospy.get_param('~dummy_mode', False)
        self.spot_fsm = SpotStateMachine(dummy_mode=self.dummy_mode)

        # Connect + power on (same flow as nl_control)
        rospy.loginfo("Orchestrator: auto-connecting Spot ...")
        recorder.publish_event('start_connect')
        self.spot_fsm.send("connect")
        recorder.publish_event('stop_connect')

        recorder.publish_event('start_power_on')
        self.spot_fsm.send("power_on")
        recorder.publish_event('stop_power_on')
        rospy.loginfo("Orchestrator: Spot ready.")

        # Publishers to HoloLens
        self.pub_interpretation = rospy.Publisher('/llm_int/interpretation', String, queue_size=10)
        self.pub_feedback = rospy.Publisher('/spot/execution_feedback', String, queue_size=20)

        # Subscribers
        self.sub_speech = rospy.Subscriber('/hl/user_speech', String, self._on_user_speech)
        self.sub_approval = rospy.Subscriber('/hl/approval', String, self._on_approval)
        # TODO: mode switch in the interface
        self.sub_mode = rospy.Subscriber('/experiment/mode', String, self._on_mode_switch)

        # LLM
        self._setup_llm()

        # Thread safety
        self._exec_lock = threading.Lock()

        rospy.loginfo("Experiment Orchestrator started in %s mode.", self.state.mode)

    # ---------- LLM ----------
    def _setup_llm(self):
        # Setup OpenAI for LA mode (GPT-5)
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai is None or not openai_api_key:
            raise Exception("No OpenAI service available. Please check the API key.")
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Setup Gemini for HA mode (Gemini Pro 2.5)
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            rospy.logwarn("No GOOGLE_API_KEY - HA mode will use OpenAI as fallback")
            self.gemini_client = None
        else:
            self.gemini_client = genai.Client(api_key=google_api_key)

    def _get_actual_robot_position(self):
        """Get actual robot position and convert to body frame coordinates."""
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
                
                # Set start position on first read
                if self.state.start_position is None:
                    self.state.start_position = [x, y, yaw]
                    rospy.loginfo(f"Set start position (vision frame): {self.state.start_position}")
                    self.state.current_position = [0.0, 0.0, 0.0]  # Start at origin
                else:
                    # Calculate relative position from start in vision frame
                    vision_relative = [
                        x - self.state.start_position[0],
                        y - self.state.start_position[1], 
                        yaw - self.state.start_position[2]
                    ]
                    
                    # Transform vision frame relative position to body frame
                    # Use the starting yaw to transform coordinates back to body frame
                    start_yaw_offset = self.state.start_position[2]  # Initial robot orientation in vision frame
                    
                    # Rotate vision frame coordinates to body frame using inverse rotation
                    cos_offset = math.cos(-start_yaw_offset)  # Negative for inverse rotation
                    sin_offset = math.sin(-start_yaw_offset)
                    
                    # Apply rotation matrix to transform vision coordinates to body coordinates
                    self.state.current_position = [
                        cos_offset * vision_relative[0] - sin_offset * vision_relative[1],  # body frame X (forward)
                        sin_offset * vision_relative[0] + cos_offset * vision_relative[1],  # body frame Y (left)
                        vision_relative[2]  # yaw rotation is the same
                    ]
                
                rospy.loginfo(f"Current position relative to start (body frame): {self.state.current_position}")
                return True
            else:
                rospy.logwarn(f"Failed to get robot pose: {response.message}")
                return False
        except Exception as e:
            rospy.logerr(f"Error getting robot position: {e}")
            return False

    def _llm_actions(self, prompt_tmpl, use_gemini=False, **fmt):
        if use_gemini and self.gemini_client:
            # Use Gemini Pro 2.5 for HA mode
            try:
                formatted = prompt_tmpl.format(**fmt)
                recorder.publish_event('start_llm_processing')
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=formatted
                )
                recorder.publish_event('stop_llm_processing')
                txt = response.text.strip()
                actions = [ln.strip() for ln in txt.splitlines() if ln.strip()]
                return actions or None
            except Exception as e:
                rospy.logerr("Gemini LLM failure: %s", str(e))
                recorder.publish_event('stop_llm_processing')
                return None
        else:
            # Use GPT-5 for LA mode or fallback
            if not self.openai_client:
                return None
            try:
                formatted = prompt_tmpl.format(**fmt)
                recorder.publish_event('start_llm_processing')
                resp = self.openai_client.responses.create(
                    model="gpt-5",
                    input=formatted,
                    reasoning={"effort": "minimal"},
                )
                recorder.publish_event('stop_llm_processing')
                txt = (resp.output_text or "").strip()
                actions = [ln.strip() for ln in txt.splitlines() if ln.strip()]
                return actions or None
            except Exception as e:
                rospy.logerr("OpenAI LLM failure: %s", str(e))
                recorder.publish_event('stop_llm_processing')
                return None

    # ---------- Parsing (LA rule-based) ----------
    def _parse_la_to_actions(self, command):
        """Very lightweight parser for LA commands. Returns list[str] of FSM actions or None."""
        # c = command.lower().strip()

        # # stand / sit
        # if re.search(r'\bstand( up)?\b', c):
        #     return ['stand_up']
        # if re.search(r'\bsit( down)?\b', c):
        #     return ['sit_down']

        # # open/close/stow/carry
        # if 'open' in c and ('gripper' in c or 'hand' in c):
        #     return ['arm_command command_type=open']
        # if 'close' in c and ('gripper' in c or 'hand' in c):
        #     return ['arm_command command_type=close']
        # if 'stow' in c and ('arm' in c or 'hand' in c):
        #     return ['arm_command command_type=stow']
        # if 'carry' in c and ('arm' in c or 'hand' in c):
        #     return ['arm_command command_type=carry']

        # # rotate
        # m = re.search(r'(rotate|turn)\s+(-?\d+(\.\d+)?)\s*(deg|degree|degrees)?\s*(left|right)?', c)
        # if m:
        #     deg = float(m.group(2))
        #     side = (m.group(5) or '').lower()
        #     sign = +1.0 if side in ('', 'left') else -1.0
        #     yaw = clamp(sign * deg2rad(deg), -math.pi, math.pi)
        #     return [f'start_moving x=0 y=0 yaw={yaw:.5f} frame=body']

        # # walk/move with distance + direction keywords
        # # e.g., "walk 2 meters forward", "move left 1.5 m", "go back 1 m"
        # m = re.search(r'(walk|go|move)\s+(-?\d+(\.\d+)?)\s*(m|meter|meters)?\s*(forward|back|backward|left|right)?', c)
        # if m:
        #     dist = float(m.group(2))
        #     dirw = (m.group(5) or 'forward').lower()
        #     x, y, yaw = 0.0, 0.0, 0.0
        #     if dirw in ('forward', ''):
        #         x = dist
        #     elif dirw in ('back', 'backward'):
        #         x = -dist
        #     elif dirw == 'left':
        #         y = +dist   # Spot body frame: +y left
        #     elif dirw == 'right':
        #         y = -dist   # Spot body frame: -y right
        #     x = clamp(x, -5.0, 5.0)
        #     y = clamp(y, -3.0, 3.0)
        #     return [f'start_moving x={x:.3f} y={y:.3f} yaw={yaw:.5f} frame=body']

        # fallback to LLM (single-command) - use GPT-5 for LA
        actions = self._llm_actions(
            LA_PROMPT,
            use_gemini=False,
            command=command,
            current_state=self.spot_fsm.current_state.name,
        )
        return actions

    # ---------- Mode switching ----------

    def _on_mode_switch(self, msg):
        val = (msg.data or '').strip().upper()
        if val not in ('LA', 'HA'):
            self._publish_feedback(f"[mode] Ignored invalid mode '{msg.data}'. Use 'LA' or 'HA'.")
            return
        with self.state.lock:
            if self.state.mode == val:
                return
            self.state.mode = val
            # self.state.num_interventions += 1
            self.state.awaiting_approval = False
            self.state.pending_plan = []
        self._publish_feedback(f"[mode] Switched to {val}.")

    # ---------- HoloLens I/O ----------
    def _on_user_speech(self, msg):
        utterance = (msg.data or '').strip()
        if not utterance:
            return

        # Add duplicate command detection
        import time
        if hasattr(self, '_last_command') and self._last_command == utterance:
            if hasattr(self, '_last_command_time') and (time.time() - self._last_command_time) < 5.0:
                rospy.logwarn(f"Ignoring duplicate command within 5 seconds: '{utterance}'")
                return
        
        self._last_command = utterance
        self._last_command_time = time.time()
        
        rospy.loginfo(f"Processing command: '{utterance}'")
        recorder.publish_event('received_hololens_input')
        mode = self.state.mode

        if mode == 'LA':
            self._handle_la(utterance)
        else:
            self._handle_ha(utterance)


    # Handle approval/rejection of HA plan
    def _on_approval(self, msg):
       
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn("Invalid JSON on /hl/approval: %s", e)
            return
        with self.state.lock:
            awaiting = self.state.awaiting_approval
        if not awaiting:
            rospy.logwarn("Plan not awaiting approval, ignoring /hl/approval message.")
            return
        plan_id = payload.get("plan_id")
        decision = payload.get("approved")
        # TODO: verify plan_id matches current_plan_id?

        if decision:
            self._publish_feedback(f"[approval] Plan {plan_id} approved. Executing plan...")
            with self.state.lock:
                plan = list(self.state.pending_plan)
                self.state.awaiting_approval = False
                # self.state.num_interventions += 1
            self._execute_actions(plan, label="HA plan")
            # End trial on successful plan execution
            # summary = self.state.complete_trial(outcome='success')
            # if summary:
            self._publish_feedback("[summary] Successful execution.")
        else:
            with self.state.lock:
                self.state.pending_plan = []
                self.state.awaiting_approval = False
                # self.state.num_interventions += 1
            self._publish_feedback("[approval] Rejected. Describe a new plan.")


    # ---------- LA / HA flows ----------
    def _handle_la(self, utterance):
        """Low Autonomy: map explicit command to immediate actions and run with no extra approval."""
        actions = self._parse_la_to_actions(utterance)
        if not actions:
            self._publish_interpretation(f"[LA] Could not parse: '{utterance}'. Try a simple command.")
            return

        payload = {
            "plan_id": self.state.current_plan_id,
            "action": actions,
        }
        self._publish_interpretation(json.dumps(payload))
        self._execute_actions(actions, label="LA command")

    def _handle_ha(self, utterance):
        """High Autonomy: generate a plan, show it, and wait for approval to execute."""
        # Get actual robot position for HA planning
        if not self._get_actual_robot_position():
            rospy.logwarn("Using last known position for HA planning")
        
        actions = self._llm_actions(
            HA_PROMPT,
            use_gemini=True,
            task=utterance,
            current_state=self.spot_fsm.current_state.name,
            current_position=f"({self.state.current_position[0]:.2f}, {self.state.current_position[1]:.2f}, {self.state.current_position[2]:.2f})"
        )

        payload = {
            "plan_id": self.state.current_plan_id,
            "action": actions,
        }
        if not actions:
            payload["action"] = "Cannot generate a plan. Please rephrase the task."
            self._publish_interpretation(json.dumps(payload))
            return

        plan_text = "HA plan (awaiting approval):\n" + "\n".join(f"{i+1}. {a}" for i, a in enumerate(actions))
        self._publish_interpretation(plan_text)
        recorder.publish_event('start_user_confirmation')
        with self.state.lock:
            self.state.pending_plan = actions
            self.state.awaiting_approval = True

    # ---------- Execution ----------

    def _execute_actions(self, actions, label="plan"):
        if not actions:
            return False
        
        ok_all = True
        if self.dummy_mode:
            return ok_all
        
        with self._exec_lock:
            self._publish_feedback(f"[exec] Starting {label} ({len(actions)} step(s))")
            for i, action in enumerate(actions, 1):
                try:
                    self._publish_feedback(f"[exec] Step {i}/{len(actions)}: {action}")
                    # Determine event name only (word before params)
                    event = action.split()[0] if ' ' in action else action.split(',')[0]
                    # Dispatch to FSM; support either "name key=val ..." or bare "name"
                    if '=' in action or ' ' in action:
                        # Parse params in a robust way
                        words = [w for w in re.split(r'[,\s]+', action.strip()) if w]
                        action_name, parts = words[0], words[1:]
                        params = {}
                        for part in parts:
                            if '=' in part:
                                k, v = part.split('=', 1)
                                k = k.strip()
                                v = v.strip().strip('"\'')
                                # attempt number conversion
                                try:
                                    if re.match(r'^-?\d+\.\d+$', v):
                                        params[k] = float(v)
                                    elif re.match(r'^-?\d+$', v):
                                        params[k] = int(v)
                                    else:
                                        params[k] = v
                                except Exception:
                                    params[k] = v
                        self.spot_fsm.send(action_name, **params)
                    else:
                        self.spot_fsm.send(action.strip())

                    # Small pacing between steps
                    rospy.sleep(0.5)
                    self._publish_feedback(f"[exec] ✓ {action}")
                except Exception as e:
                    ok_all = False
                    self._publish_feedback(f"[exec] ✗ Failed '{action}': {e}")
                    rospy.logerr("Action failed '%s': %s", action, str(e))
                    break

            self._publish_feedback(f"[exec] Finished {label} — status: {'success' if ok_all else 'failed'}")
        return ok_all

    # ---------- Publish helpers ----------

    def _publish_interpretation(self, text):
        self.pub_interpretation.publish(String(data=text))

    def _publish_feedback(self, text):
        self.pub_feedback.publish(String(data=text))

    # ---------- Spin ----------

    def run(self):
        rospy.loginfo("Experiment Orchestrator is running. Speak to Spot on /hl/user_speech.")
        rospy.spin()

def main():
    node = ExperimentOrchestrator()
    node.run()

if __name__ == "__main__":
    main()
