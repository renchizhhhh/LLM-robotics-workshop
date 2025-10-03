#!/usr/bin/env python3
import os
import re
import json
import math
import threading

import rospy
from std_msgs.msg import String, Empty

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'spot_hololens_llm_interface'))
from finite_state_machine import SpotStateMachine 
from timing_utils import recorder          
import openai 

LA_PROMPT = """\
You are converting a natural-language command into 1-2 Spot FSM actions.
Only include actions necessary to fulfill the command.
Return one action per line, format:
- "stand_up"
- "sit_down"
- "start_moving", x=<float> y=<float> yaw=<float> frame=body
- "arm_command", command_type=open|close|stow|carry
- "start_automated_grasp", object_type=user_specified

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

HA_PROMPT = """\
You are a path planning system for a Boston Dynamics Spot robot.

CURRENT STATE:
- Robot state: {current_state}

WORLD LAYOUT (vision frame coordinates):
- Pick-up location: (2.0, -1.0, 0.0) - Robot faces forward (yaw=0) when picking up
- Drop-off location: (2.0, 0.5, 0.0) - Robot faces left (yaw=3,14 radians) when dropping off

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

class ExperimentOrchestrator(object):
    def __init__(self):
        rospy.init_node('experiment_orchestrator', anonymous=True)
        self.state = ExperimentState(mode=rospy.get_param('~mode', 'LA').upper())

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

        # Subscribers
        self.sub_speech = rospy.Subscriber('/hl/user_speech', String, self._on_user_speech)
        self.sub_approval = rospy.Subscriber('/hl/approval', String, self._on_approval)
        self.sub_stop = rospy.Subscriber('/hl/stop', Empty, self._on_stop)
        # TODO: mode switch in the interface
        self.sub_mode = rospy.Subscriber('/experiment/mode', String, self._on_mode_switch)

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
        api_key = os.getenv('OPENAI_API_KEY')
        if openai is None or not api_key:
            raise Exception("No OpenAI service available. Please check the API key.")
        self.llm = openai.OpenAI(api_key=api_key)

    def _llm_actions(self, prompt_tmpl, **fmt):
        if not self.llm:
            return None
        try:
            formatted = prompt_tmpl.format(**fmt)
            recorder.publish_event('start_llm_processing')
            resp = self.llm.responses.create(
                model="gpt-5",
                input=formatted,
                reasoning={"effort": "minimal"},
            )
            recorder.publish_event('stop_llm_processing')
            txt = (resp.output_text or "").strip()
            actions = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            return actions or None
        except Exception as e:
            rospy.logerr("LLM failure: %s", str(e))
            recorder.publish_event('stop_llm_processing')
            return None

    # ---------- Parsing (LA rule-based) ----------
    def _parse_la_to_actions(self, command):
        """Parser for LA commands. Returns list[str] of FSM actions or None."""
        # fallback to LLM (single-command)
        actions = self._llm_actions(
            LA_PROMPT,
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
            self.state.pending_plan_id = None
            self.state.active_plan_id = None
            self.state.stop_requested = False
        self._publish_feedback(f"[mode] Switched to {val}.")

    # ---------- HoloLens I/O ----------
    def _on_user_speech(self, msg):
        utterance = (msg.data or '').strip()
        if not utterance:
            return

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
        actions = self._parse_la_to_actions(utterance)
        if not actions:
            self._publish_feedback(f"[LA] Could not parse: '{utterance}'. Try a simple command.")
            return

        plan_id = self._next_plan_id()
        rospy.loginfo(f"Orchestrator: LA mode: LLM returned action: {actions}")
        self._execute_actions(actions, label=f"LA command {plan_id}", plan_id=plan_id)

    def _handle_ha(self, utterance):
        """High Autonomy: generate a plan, show it, and wait for approval to execute."""
        actions = self._llm_actions(
            HA_PROMPT,
            task=utterance,
            current_state=self.spot_fsm.current_state.name,
        )
        rospy.loginfo(f"Orchestrator: HA mode: LLM returned action: {actions}")

        plan_id = self._next_plan_id()

        if not actions:
            self._publish_feedback("Cannot generate a plan. Please rephrase the task.")
            return

        # plan_text = f"HA plan {plan_id} (awaiting approval):\n" + "\n".join(f"{i+1}. {a}" for i, a in enumerate(actions))
        self._publish_plan(plan_id, actions)
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
                with self.state.lock:
                    if self.state.stop_requested:
                        stopped = True
                        ok_all = False
                        self._publish_feedback(f"[stop] Stop request already active; skipping remaining steps before step {i}.")
                        break
                try:
                    self._publish_feedback(f"[exec] Step {i}/{len(actions)}: {action}")
                    event = action.split()[0] if ' ' in action else action.split(',')[0]
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
                        rospy.loginfo(f"Orchestrator: executing action '{action_name}' with params {params}")
                        self.spot_fsm.send(action_name, **params)
                    else:
                        self.spot_fsm.send(action.strip())

                    # Small pacing between steps
                    rospy.sleep(0.5)
                    self._publish_feedback(f"[exec] ✓ {action}")
                    steps_completed = i
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

    # ---------- Spin ----------

    def run(self):
        rospy.loginfo("Experiment Orchestrator is running. Speak to Spot on /hl/user_speech.")
        rospy.spin()

def main():
    node = ExperimentOrchestrator()
    node.run()

if __name__ == "__main__":
    main()
