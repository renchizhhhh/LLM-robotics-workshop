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

LA_PROMPT = """\
You are converting a natural-language command into 1-2 Spot FSM actions.
Only include actions necessary to fulfill the command.
Return one action per line, format:
- "stand_up"
- "sit_down"
- "start_moving", x=<float> y=<float> yaw=<float> frame=body
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

HA_PROMPT = """\
You are converting a natural-language command into a list of Spot FSM actions.
Only include actions necessary to fulfill the command.
Return one action per line, format:
- "stand_up"
- "sit_down"
- "start_moving", x=<float> y=<float> yaw=<float> frame=body
- "arm_command", command_type="open|close|stow|carry

Task: {task}
Current robot state: {current_state}

Constraints:
- Only include "stand_up" if robot is not already standing or moving
- Movement and manipulation require standing or moving state
- Movement: x[-5,5], y[-3,3] meters, yaw in radians, movements and rotations should happen seperately
- Image sources: frontleft_fisheye_image, frontright_fisheye_image, left_fisheye_image, right_fisheye_image, back_fisheye_image
- Format: one action per line, no quotes/brackets, no numbering

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
        self.current_plan_id = 0
        self.pending_plan = [] 
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
        actions = self._llm_actions(
            HA_PROMPT,
            task=utterance,
            current_state=self.spot_fsm.current_state.name,
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
