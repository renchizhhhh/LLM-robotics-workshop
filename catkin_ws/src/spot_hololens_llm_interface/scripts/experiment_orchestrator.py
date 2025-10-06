#!/usr/bin/env python3
import os
import re
import json
import math
import threading
import time
from datetime import datetime

import rospy
from std_msgs.msg import String, Empty, Empty

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'spot_hololens_llm_interface'))
from finite_state_machine import SpotStateMachine 
from timing_utils import recorder          
import openai
from google import genai
from spot_hololens_llm_interface.srv import GetRobotPose, GetRobotPoseRequest, GetInitialPose, GetInitialPoseRequest 

class TokenTracker:
    """Track token usage for LLM calls and output to markdown file"""
    
    def __init__(self, output_file="token_usage.md"):
        self.output_file = output_file
        self.usage_log = []
        self.lock = threading.Lock()
        
    def log_usage(self, model, prompt_tokens, completion_tokens, total_tokens, cost=None):
        """Log token usage for a specific call"""
        with self.lock:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": cost
            }
            self.usage_log.append(entry)
            self._write_to_markdown()
            
    def _write_to_markdown(self):
        """Write token usage to markdown file"""
        try:
            with open(self.output_file, 'w') as f:
                f.write("# Token Usage Report\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Summary
                total_prompt_tokens = sum(entry['prompt_tokens'] for entry in self.usage_log)
                total_completion_tokens = sum(entry['completion_tokens'] for entry in self.usage_log)
                total_tokens = sum(entry['total_tokens'] for entry in self.usage_log)
                
                f.write("## Summary\n\n")
                f.write(f"- **Total Prompt Tokens**: {total_prompt_tokens:,}\n")
                f.write(f"- **Total Completion Tokens**: {total_completion_tokens:,}\n")
                f.write(f"- **Total Tokens**: {total_tokens:,}\n")
                f.write(f"- **Total API Calls**: {len(self.usage_log)}\n\n")
                
                # Detailed log
                f.write("## Detailed Usage Log\n\n")
                f.write("| Timestamp | Model | Prompt Tokens | Completion Tokens | Total Tokens | Cost |\n")
                f.write("|-----------|-------|---------------|------------------|--------------|------|\n")
                
                for entry in self.usage_log:
                    cost_str = f"${entry['cost']:.4f}" if entry['cost'] else "N/A"
                    f.write(f"| {entry['timestamp']} | {entry['model']} | {entry['prompt_tokens']:,} | {entry['completion_tokens']:,} | {entry['total_tokens']:,} | {cost_str} |\n")
                    
        except Exception as e:
            rospy.logwarn(f"Failed to write token usage to markdown: {e}")

# Global token tracker instance
token_tracker = TokenTracker()

LA_PROMPT = """\
You are converting a natural-language command into 1 Spot FSM actions.
Only include actions necessary to fulfill the command.

AVAILABLE ACTIONS:
- stand_up
- sit_down  
- start_moving, x=float, y=float, yaw=float, frame="body"
- start_automated_grasp, object_type="exact_object_name_from_user"
- start_move_arm_pose, x=float, y=float, z=float, qw=float, qx=float, qy=float, qz=float, duration=float, open_gripper=bool
- start_arm_command, command_type=open|close|stow|carry

Command: {command}
Current robot state: {current_state}

RULES:
1. One action per line, no quotes/brackets, no numbering
2. Robot must be in standing mode before moving and grasping
3. Only use "stand_up" if robot is not already standing, moving or grasping
4. In one move, the robot can either move in the x direction, the y direction, or the yaw direction, not two or three at once.
5. Use body frame coordinates for movements
6. Yaw in radians, movements and rotations should happen separately
7. When multiple actions are given, only execute the first action.
8. x>0 means forward, y>0 means left, yaw>0 means left.

OUTPUT FORMAT:
Return only the action list, one action per line.

Actions:"""

HA_PROMPT = """\
You are a path planning system for a Boston Dynamics Spot robot.

CURRENT STATE:
- Robot state: {current_state}

WORLD LAYOUT (vision frame coordinates):
- Pick-up location: (2.0, -1.0, 0.0) - Robot faces forward (yaw=0) when picking up
- Drop-off location: (2.0, 0.5, 0.0) - Robot faces left (yaw=1,57 radians) when dropping off

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
1. Visit destinations in EXACT order specified by user
2. One action per line, no quotes/brackets, no numbering
3. Robot must be in standing mode before moving and grasping
4. Only use "stand_up" if robot is not already standing, moving or grasping
5. In one move, the robot can either move in the x direction, the y direction, or the yaw direction, not two or three at once.
6. Use RELATIVE body frame coordinates for movements - each movement is relative to current position
7. For movements, calculate: move_x = target_x - current_x, move_y = target_y - current_y
8. Use EXACT object name from user command for object_type (e.g., "tomato can" not "tomato")
9. For drop-off procedure: start_move_arm_pose to (0.8, 0.0, 0.3) with quaternion (0.7071, 0.7071, 0.0, 0.0) for gripper pointing down (this is the arm pose for dropping off objects) in 1 second, then start_arm_command open, then start_arm_command close, then start_arm_command stow

PLANNING PROCESS:
1. Parse user command to identify destinations in order
2. Calculate relative body frame movements from current position to each destination
3. Plan path visiting each destination once in order
4. Ensure correct robot orientation at each destination (yaw=0 for pick-up, yaw=1.57 radians for drop-off)
5. For drop-off locations: add drop-off procedure (start_move_arm_pose, start_arm_command open, start_arm_command close, start_arm_command stow)
6. Generate action sequence with relative body frame movements

OUTPUT FORMAT:
Return only the action list, one action per line.

Task: {task}

Return action list:
"""

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
        self.plan_counter = 0
        self.pending_plan = [] 
        self.pending_plan_id = None
        self.active_plan_id = None
        self.stop_requested = False
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

        cur_state = None
        cur_state = getattr(self.spot_fsm.current_state, 'name', None) or str(self.spot_fsm.current_state)

        rospy.loginfo("Orchestrator: FSM reported initial state: %s", cur_state)

        if cur_state in ("disconnected", "unknown"):
            rospy.loginfo("Orchestrator: auto-connecting Spot ...")
            recorder.publish_event('start_connect')
            self.spot_fsm.send("connect")
            recorder.publish_event('stop_connect')

        if cur_state in ("connected", "disconnected", "powered_off", "unknown"):
            recorder.publish_event('start_power_on')
            self.spot_fsm.send("power_on")
            recorder.publish_event('stop_power_on')


        rospy.loginfo("Orchestrator: Spot initialization sequence finished.")

        # Publishers to HoloLens
        self.pub_plan = rospy.Publisher('/spot/plan', String, queue_size=10)
        self.pub_feedback = rospy.Publisher('/spot/execution_feedback', String, queue_size=20)
        # Publisher for GUI position updates
        self.pub_position = rospy.Publisher('/nl_control/robot_position', String, queue_size=1)

        # Subscribers
        self.sub_speech = rospy.Subscriber('/hl/user_speech', String, self._on_user_speech)
        self.sub_approval = rospy.Subscriber('/hl/approval', String, self._on_approval)
        self.sub_stop = rospy.Subscriber('/hl/stop', Empty, self._on_stop)
        # TODO: mode switch in the interface
        self.sub_mode = rospy.Subscriber('/experiment/mode', String, self._on_mode_switch)
        
        # Stop signal handling
        self.stop_requested = False

        # LLM
        self._setup_llm()

        # Thread safety
        self._exec_lock = threading.Lock()

        rospy.loginfo("Experiment Orchestrator started in %s mode.", self.state.mode)

    def _next_plan_id(self):
        with self.state.lock:
            self.state.plan_counter += 1
            return self.state.plan_counter

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
        
        # Setup plan interpreter (small model for natural language conversion)
        self._setup_plan_interpreter()
    
    def _setup_plan_interpreter(self):
        """Setup small LLM for plan interpretation"""
        # Use OpenAI GPT-4o-mini for plan interpretation (smaller, faster, cheaper)
        self.plan_interpreter_client = self.openai_client
        
        # Plan interpretation prompt
        self.PLAN_INTERPRETATION_PROMPT = """You are a robot plan interpreter. Convert a programmatic robot plan into natural language.

COORDINATE SYSTEM:
- X positive = forward movement
- Y positive = left movement  
- Y negative = right movement
- Yaw positive = rotation left

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

    def _interpret_plan(self, plan):
        """Convert programmatic plan to natural language"""
        try:
            if not plan:
                return "No plan generated"
            
            # Format plan as string
            plan_str = "\n".join([f"{i+1}. {action}" for i, action in enumerate(plan)])
            
            # Use small model for interpretation
            response = self.plan_interpreter_client.chat.completions.create(
                model="gpt-4o-mini",  # Small, fast model
                messages=[
                    {"role": "system", "content": "You are a helpful robot plan interpreter."},
                    {"role": "user", "content": self.PLAN_INTERPRETATION_PROMPT.format(plan=plan_str)}
                ],
                max_tokens=150,  # Keep it short
                temperature=0.3  # Consistent output
            )
            
            # Track token usage
            usage = response.usage
            token_tracker.log_usage(
                model="gpt-4o-mini",
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cost=usage.total_tokens * 0.00015 / 1000  # Approximate cost for gpt-4o-mini
            )
            
            interpretation = response.choices[0].message.content.strip()
            rospy.loginfo(f"Plan interpretation: {interpretation}")
            return interpretation
            
        except Exception as e:
            rospy.logwarn(f"Plan interpretation failed: {e}")
            return f"Robot will execute {len(plan)} actions: {', '.join(plan[:3])}{'...' if len(plan) > 3 else ''}"

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

    def _parse_commands(self, prompt_tmpl, use_gemini=False, **fmt):
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
                
                # Track token usage for Gemini (approximate)
                prompt_tokens = len(formatted.split()) * 1.3  # Rough estimation
                completion_tokens = len(response.text.split()) * 1.3
                total_tokens = prompt_tokens + completion_tokens
                
                token_tracker.log_usage(
                    model="gemini-2.5-pro",
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=int(completion_tokens),
                    total_tokens=int(total_tokens),
                    cost=total_tokens * 0.0005 / 1000  # Approximate cost for Gemini
                )
                
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
                
                # Track token usage for GPT-5 (approximate)
                prompt_tokens = len(formatted.split()) * 1.3  # Rough estimation
                completion_tokens = len(resp.output.split()) * 1.3
                total_tokens = prompt_tokens + completion_tokens
                
                token_tracker.log_usage(
                    model="gpt-5",
                    prompt_tokens=int(prompt_tokens),
                    completion_tokens=int(completion_tokens),
                    total_tokens=int(total_tokens),
                    cost=total_tokens * 0.002 / 1000  # Approximate cost for GPT-5
                )
                txt = (resp.output_text or "").strip()
                actions = [ln.strip() for ln in txt.splitlines() if ln.strip()]
                return actions or None
            except Exception as e:
                rospy.logerr("OpenAI LLM failure: %s", str(e))
                recorder.publish_event('stop_llm_processing')
                return None

    # ---------- Unified Command Parsing ----------
    def _parse_command_to_actions(self, command, mode='LA'):
        """Unified parser for both LA and HA commands. Always uses LLM, then parses output."""
        if mode == 'LA':
            return self._parse_commands(
                LA_PROMPT,
                use_gemini=False,
                command=command,
                current_state=self.spot_fsm.current_state.name,
            )
        else:  # HA mode
            return self._parse_commands(
                HA_PROMPT,
                use_gemini=True,
                task=command,
                current_state=self.spot_fsm.current_state.name,
                current_position=f"({self.state.current_position[0]:.2f}, {self.state.current_position[1]:.2f}, {self.state.current_position[2]:.2f})"
            )

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
            self.state.pending_plan_id = None
            self.state.active_plan_id = None
            self.state.stop_requested = False
        self._publish_feedback(f"[mode] Switched to {val}.")

    # ---------- HoloLens I/O ----------
    def _on_user_speech(self, msg):
        utterance = (msg.data or '').strip()
        if not utterance:
            return

        # Check for stop signal before processing new command
        if self.stop_requested:
            rospy.loginfo("Stop signal received - ignoring new command until stop is cleared")
            return
        
        # Reset stop flag when processing new command
        self.stop_requested = False

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


    def _on_stop(self, msg):
        """Handle stop signal from HoloLens"""
        rospy.loginfo("HoloLens stop signal received - will finish current task and stop plan execution")
        self.stop_requested = True

    # Handle approval/rejection of HA plan
    def _on_approval(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn("Invalid JSON on /hl/approval: %s", e)
            return
        with self.state.lock:
            awaiting = self.state.awaiting_approval
            pending_plan_id = self.state.pending_plan_id
            pending_plan = list(self.state.pending_plan)
        if not awaiting:
            rospy.logwarn("Plan not awaiting approval, ignoring /hl/approval message.")
            return
        plan_id = payload.get("plan_id")
        decision = payload.get("approved")
        # plan_id helps match approvals to a specific pending plan
        if plan_id is not None:
            try:
                plan_id = int(plan_id)
            except (TypeError, ValueError):
                rospy.logwarn(f"Invalid plan_id '{plan_id}' in approval message; falling back to pending plan id {pending_plan_id}.")
                plan_id = pending_plan_id
        if plan_id is None:
            plan_id = pending_plan_id
        if pending_plan_id is not None and plan_id != pending_plan_id:
            rospy.logwarn(f"Approval plan_id {plan_id} does not match pending plan {pending_plan_id}; ignoring message.")
            return
        if decision:
            self._publish_feedback(f"[approval] Plan {plan_id} approved. Executing plan...")
            with self.state.lock:
                self.state.pending_plan = []
                self.state.pending_plan_id = None
                self.state.awaiting_approval = False
                # self.state.num_interventions += 1
            self._execute_actions(pending_plan, label=f"HA plan {plan_id}", plan_id=plan_id)
            # End trial on successful plan execution
            # summary = self.state.complete_trial(outcome='success')
            # if summary:
            self._publish_feedback("[summary] Successful execution.")
        else:
            with self.state.lock:
                self.state.pending_plan = []
                self.state.pending_plan_id = None
                self.state.awaiting_approval = False
                # self.state.num_interventions += 1
            self._publish_feedback("[approval] Rejected. Describe a new plan.")

    def _on_stop(self, _msg):
        with self.state.lock:
            already_requested = self.state.stop_requested
            had_pending = self.state.awaiting_approval
            self.state.stop_requested = True
            self.state.awaiting_approval = False
            self.state.pending_plan = []
            self.state.pending_plan_id = None
        rospy.loginfo("Orchestrator: stop requested via /hl/stop")
        if already_requested:
            self._publish_feedback("[stop] Stop already active; waiting for current step to finish.")
        else:
            self._publish_feedback("[stop] Stop requested; halting after current action completes.")


    # ---------- LA / HA flows ----------
    def _handle_la(self, utterance):
        """Low Autonomy: map explicit command to immediate actions and run with no extra approval."""
        actions = self._parse_command_to_actions(utterance, mode='LA')
        
        if not actions:
            self._publish_feedback(f"[LA] Could not parse: '{utterance}'. Try a simple command.")
            return

        # Generate natural language interpretation for LA mode too
        natural_plan = self._interpret_plan(actions)
        rospy.loginfo(f"LA plan interpretation: {natural_plan}")
        
        payload = {
            "plan_id": self.state.current_plan_id,
            "action": actions,
            "natural_plan": natural_plan
        }
        self._publish_interpretation(json.dumps(payload))
        rospy.loginfo(f"Orchestrator: LA mode: LLM returned action: {actions}")
        self._execute_actions(actions, label=f"LA command {plan_id}", plan_id=plan_id)

    def _handle_ha(self, utterance):
        """High Autonomy: generate a plan, show it, and wait for approval to execute."""
        # Get actual robot position for HA planning
        if not self._get_actual_robot_position():
            rospy.logwarn("Using last known position for HA planning")
        
        actions = self._parse_command_to_actions(utterance, mode='HA')
        rospy.loginfo(f"Orchestrator: HA mode: LLM returned action: {actions}")

        if not actions:
            payload = {"plan_id": self.state.current_plan_id, "action": "Cannot generate a plan. Please rephrase the task."}
            self._publish_interpretation(json.dumps(payload))
            return

        # Generate natural language interpretation of the plan
        natural_plan = self._interpret_plan(actions)
        
        # Create plan text with both technical and natural language
        plan_text = f"HA plan (awaiting approval):\n\nNatural Language: {natural_plan}\n\nTechnical Steps:\n" + "\n".join(f"{i+1}. {a}" for i, a in enumerate(actions))
        self._publish_interpretation(plan_text)
        recorder.publish_event('start_user_confirmation')
        with self.state.lock:
            self.state.pending_plan = list(actions)
            self.state.pending_plan_id = plan_id
            self.state.awaiting_approval = True

    # ---------- Execution ----------

    def _execute_actions(self, actions, label="plan", plan_id=None):
        if not actions:
            return False
        if plan_id is None:
            plan_id = self._next_plan_id()

        ok_all = True
        stopped = False
        steps_completed = 0
        # if self.dummy_mode:
        #     return ok_all

        with self._exec_lock:
            with self.state.lock:
                if self.state.stop_requested:
                    self._publish_feedback(f"[stop] Stop active; skipping {label}.")
                    return False
                self.state.active_plan_id = plan_id

            self._publish_feedback(f"[exec] Starting {label} ({len(actions)} step(s))")
            for i, action in enumerate(actions, 1):
                # Check for stop signal before each action
                if self.stop_requested:
                    rospy.loginfo("Stop signal received - finishing current task and stopping plan execution")
                    self._publish_feedback("[exec] Stop signal received - plan execution stopped")
                    # Reset stop flag for next plan
                    self.stop_requested = False
                    return False
                
                try:
                    self._publish_feedback(f"[exec] Step {i}/{len(actions)}: {action}")
                    event = action.split()[0] if ' ' in action else action.split(',')[0]
                    if '=' in action or ' ' in action:
                        # Parse params in a robust way - handle both comma and space separated parameters
                        # First try to split on commas, if that doesn't work well, try spaces
                        action_stripped = action.strip()
                        
                        # Check if it's comma-separated (has commas and = signs)
                        if ',' in action_stripped and '=' in action_stripped:
                            # Parse comma-separated format
                            parts = []
                            current_part = ""
                            in_quotes = False
                            quote_char = None
                            
                            for char in action_stripped:
                                if char in ['"', "'"] and not in_quotes:
                                    in_quotes = True
                                    quote_char = char
                                    current_part += char
                                elif char == quote_char and in_quotes:
                                    in_quotes = False
                                    quote_char = None
                                    current_part += char
                                elif char == ',' and not in_quotes:
                                    if current_part.strip():
                                        parts.append(current_part.strip())
                                    current_part = ""
                                else:
                                    current_part += char
                            
                            if current_part.strip():
                                parts.append(current_part.strip())
                        else:
                            # Parse space-separated format (e.g., "start_moving, x=2.0 y=0.0 yaw=0.0 frame=body")
                            # Split on spaces but be careful with quoted strings
                            parts = []
                            current_part = ""
                            in_quotes = False
                            quote_char = None
                            
                            for char in action_stripped:
                                if char in ['"', "'"] and not in_quotes:
                                    in_quotes = True
                                    quote_char = char
                                    current_part += char
                                elif char == quote_char and in_quotes:
                                    in_quotes = False
                                    quote_char = None
                                    current_part += char
                                elif char == ' ' and not in_quotes:
                                    if current_part.strip():
                                        parts.append(current_part.strip())
                                    current_part = ""
                                else:
                                    current_part += char
                            
                            if current_part.strip():
                                parts.append(current_part.strip())
                        
                        action_name = parts[0] if parts else action
                        param_parts = parts[1:] if len(parts) > 1 else []
                        params = {}
                        for part in param_parts:
                            if '=' in part:
                                k, v = part.split('=', 1)
                                k = k.strip()
                                v = v.strip().strip('"\'')
                                # attempt number and boolean conversion
                                try:
                                    if re.match(r'^-?\d+\.\d+$', v):
                                        params[k] = float(v)
                                    elif re.match(r'^-?\d+$', v):
                                        params[k] = int(v)
                                    elif v.lower() in ['true', 'false']:
                                        params[k] = v.lower() == 'true'
                                    else:
                                        params[k] = v
                                except Exception:
                                    params[k] = v
                        
                        # Use body frame for movements
                        if action_name == 'start_moving' and 'x' in params and 'y' in params:
                            params['frame'] = 'body'  # Use body frame for movements
                            # Update position tracking for dummy mode
                            self._update_position_for_movement(params)
                        
                        rospy.loginfo(f"Orchestrator: executing action '{action_name}' with params {params}")
                        self.spot_fsm.send(action_name, **params)
                    else:
                        self.spot_fsm.send(action.strip())

                    # Small pacing between steps
                    rospy.sleep(0.2)
                    self._publish_feedback(f"[exec] ✓ {action}")
                    
                    # Check for stop signal after action completion
                    if self.stop_requested:
                        rospy.loginfo("Stop signal received after action completion - stopping plan execution")
                        self._publish_feedback("[exec] Stop signal received - plan execution stopped")
                        # Reset stop flag for next plan
                        self.stop_requested = False
                        return False
                        
                except Exception as e:
                    ok_all = False
                    self._publish_feedback(f"[exec] ✗ Failed '{action}': {e}")
                    rospy.logerr("Action failed '%s': %s", action, str(e))
                    break
                with self.state.lock:
                    if self.state.stop_requested:
                        stopped = True
                        self._publish_feedback(f"[stop] Stop request received; halting {label} after step {i}.")
                        ok_all = False
                        break

            status_text = 'success' if ok_all and not stopped else 'stopped' if stopped else 'failed'
            self._publish_feedback(f"[exec] Finished {label} — status: {status_text} (completed {steps_completed}/{len(actions)} steps)")

            with self.state.lock:
                if self.state.active_plan_id == plan_id:
                    self.state.active_plan_id = None
                if self.state.stop_requested:
                    self.state.stop_requested = False
        return ok_all

    # ---------- Publish helpers ----------

    def _publish_plan(self, plan_id, actions):
        self.pub_plan.publish(String(data=json.dumps({"plan_id": plan_id, "actions": actions})))

    def _publish_feedback(self, text):
        self.pub_feedback.publish(String(data=text))
    
    def _update_position_for_movement(self, params):
        """Update robot position for movement in dummy mode"""
        try:
            move_x = params.get('x', 0.0)
            move_y = params.get('y', 0.0) 
            move_yaw = params.get('yaw', 0.0)
            
            # Update position in body frame
            self.state.current_position[0] += move_x
            self.state.current_position[1] += move_y
            self.state.current_position[2] += move_yaw
            
            rospy.loginfo(f"Robot moved by ({move_x}, {move_y}, {move_yaw}) in body frame")
            rospy.loginfo(f"Updated position to: ({self.state.current_position[0]:.2f}, {self.state.current_position[1]:.2f}, {self.state.current_position[2]:.2f})")
            
            # Publish position update to GUI
            self._publish_position_update()
            
        except Exception as e:
            rospy.logwarn(f"Failed to update position for movement: {e}")
    
    def _publish_position_update(self):
        """Publish current robot position to GUI"""
        try:
            import json
            position_data = {
                'x': self.state.current_position[0],
                'y': self.state.current_position[1], 
                'yaw': self.state.current_position[2]
            }
            self.pub_position.publish(json.dumps(position_data))
            rospy.loginfo(f"Published position update: ({self.state.current_position[0]:.2f}, {self.state.current_position[1]:.2f}, {self.state.current_position[2]:.2f})")
        except Exception as e:
            rospy.logwarn(f"Failed to publish position update: {e}")

    # ---------- Spin ----------

    def run(self):
        rospy.loginfo("Experiment Orchestrator is running. Speak to Spot on /hl/user_speech.")
        
        # Publish initial position to GUI
        self._publish_position_update()
        rospy.spin()

def main():
    node = ExperimentOrchestrator()
    node.run()

if __name__ == "__main__":
    main()
