import os
import json
import socket
import shutil
from flask import Flask, render_template, request, session, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit, join_room, leave_room
from functools import wraps
import qrcode
import qrcode.image.pil
from io import BytesIO
import base64
import threading
import uuid
from datetime import datetime

# Get the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')
PROMPT_DIR = os.path.join(BASE_DIR, 'prompt')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
USER_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'user_data'))
CONSENT_LOG_PATH = os.path.join(USER_DATA_DIR, 'consent_choices.jsonl')
PROMPT_TIMESTAMP_LOG_PATH = os.path.join(USER_DATA_DIR, 'prompt_received_timestamps.jsonl')
USER_DATA_BACKUP_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', 'user_data_saved'))

# Ensure expected data directories exist
os.makedirs(PROMPT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(USER_DATA_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES_DIR)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Dashboard authentication
DASHBOARD_PASSWORD = os.environ.get('DASHBOARD_PASSWORD', 'admin123')  # Default password, should be overridden

def dashboard_required(f):
    """Decorator to protect dashboard routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('dashboard_authenticated'):
            return redirect(url_for('dashboard_login'))
        return f(*args, **kwargs)
    return decorated_function

def check_dashboard_auth():
    """Check if current session is authenticated for dashboard"""
    return session.get('dashboard_authenticated', False)

# Global state management
class SurveyState:
    def __init__(self):
        self.current_step = 0  # 0: name entry, 1: combined step1&2 entry, 3: step3 entry, 4: step4 entry
        self.view_mode = 'entry'  # 'entry' | 'results' | 'final'
        self.current_result_step = None  # Which step's results are being shown (1, 2, 3, or 4)
        self.users = {}  # {session_id: {'name': str, 'step1': str, 'step2': str, 'step3': str, 'step4': str, 'consent': str}}
        self.name_to_session = {}  # {name: session_id} for lookup by name
        self.name_to_client = {}  # {name: client_id} to enforce per-client uniqueness
        self.lock = threading.Lock()
        self.active_timer = None  # {'step': int, 'duration': int, 'started_at': float}

    @staticmethod
    def _ensure_step_fields(record: dict):
        """Ensure all step fields exist on the user record."""
        for key in ('step1', 'step2', 'step3', 'step4'):
            record.setdefault(key, None)
        return record
    
    def add_user(self, session_id, name, consent=None, client_id=None):
        with self.lock:
            client_id = client_id or session_id
            existing_session = self.name_to_session.get(name)
            existing_client = self.name_to_client.get(name)

            # Clean up stale mappings if the stored session no longer exists
            if existing_session and existing_session not in self.users:
                self.name_to_session.pop(name, None)
                if self.name_to_client.get(name) == existing_client:
                    self.name_to_client.pop(name, None)
                existing_session = None
                existing_client = None

            # If another client already owns this name, block registration
            if existing_client and existing_client != client_id:
                return False, 'name_taken'
            if existing_session and existing_session in self.users and existing_session != session_id and not existing_client:
                # Legacy mapping without a client id still counts as "taken"
                return False, 'name_taken'

            # Check if name already exists (user reconnecting)
            if existing_session and existing_session in self.users:
                # Prevent consent change on reconnect
                existing_consent = self.users[existing_session].get('consent')
                if existing_consent and consent and consent != existing_consent:
                    return False, 'consent_conflict'
                # Update session_id if different
                if existing_session != session_id:
                    self.users[session_id] = self._ensure_step_fields(self.users.pop(existing_session))
                else:
                    self._ensure_step_fields(self.users[session_id])
                if consent is not None:
                    self.users[session_id]['consent'] = consent
                self.name_to_session[name] = session_id
                self.name_to_client[name] = client_id
                return True, None
            
            # Check if session already exists
            if session_id in self.users:
                self._ensure_step_fields(self.users[session_id])
                existing_consent = self.users[session_id].get('consent')
                existing_name = self.users[session_id].get('name')

                # If changing name, allow it and reset responses for a clean slate
                if existing_name != name:
                    if existing_name and self.name_to_session.get(existing_name) == session_id:
                        del self.name_to_session[existing_name]
                        self.name_to_client.pop(existing_name, None)
                    # If desired name is claimed by another client, block the change
                    claimed_client = self.name_to_client.get(name)
                    if claimed_client and claimed_client != client_id:
                        return False, 'name_taken'
                    claimed_session = self.name_to_session.get(name)
                    if claimed_session and claimed_session != session_id and not claimed_client:
                        return False, 'name_taken'
                    self.users[session_id] = self._ensure_step_fields({
                        'name': name,
                        'step1': None,
                        'step2': None,
                        'step3': None,
                        'step4': None,
                        'consent': consent if consent is not None else existing_consent
                    })
                    self.name_to_session[name] = session_id
                    self.name_to_client[name] = client_id
                    return True, None

                # Same name: allow consent update if not conflicting
                if existing_consent and consent and consent != existing_consent:
                    return False, 'consent_conflict'
                if consent is not None:
                    self.users[session_id]['consent'] = consent
                self.name_to_client[name] = client_id
                return True, None
            
            # New user
            self.users[session_id] = self._ensure_step_fields({
                'name': name,
                'step1': None,
                'step2': None,
                'step3': None,
                'step4': None,
                'consent': consent
            })
            self.name_to_session[name] = session_id
            self.name_to_client[name] = client_id
            return True, None
    
    def submit_response(self, session_id, step, response, user_name=None):
        with self.lock:
            # Try to find user by session_id first
            user_data = None
            if session_id in self.users:
                user_data = self.users[session_id]
            # If not found and name provided, try to find by name
            elif user_name and user_name in self.name_to_session:
                lookup_session = self.name_to_session[user_name]
                if lookup_session in self.users:
                    user_data = self.users[lookup_session]
                    # Update session_id to current one
                    if lookup_session != session_id:
                        self.users[session_id] = self.users.pop(lookup_session)
                        self.name_to_session[user_name] = session_id
                        user_data = self.users[session_id]
            
            if user_data:
                step_key = f'step{step}'
                if step_key in user_data:
                    if user_data[step_key] is None:
                        user_data[step_key] = response
                        return True
        return False
    
    def get_user_list(self):
        with self.lock:
            return [self.users[sid]['name'] for sid in self.users if 'name' in self.users[sid]]
    
    def get_users_with_status(self):
        """Get users with their completion status for each step"""
        with self.lock:
            users_status = []
            for sid in self.users:
                if 'name' in self.users[sid]:
                    user = self._ensure_step_fields(self.users[sid])
                    users_status.append({
                        'name': user['name'],
                        'step1': user.get('step1') is not None,
                        'step2': user.get('step2') is not None,
                        'step3': user.get('step3') is not None,
                        'step4': user.get('step4') is not None
                    })
            return users_status
    
    def get_responses_for_step(self, step):
        with self.lock:
            step_key = f'step{step}'
            return {self.users[sid]['name']: self.users[sid][step_key] 
                   for sid in self.users 
                   if step_key in self.users[sid] and self.users[sid][step_key] is not None}
    
    def get_user_data_for_request(self, session_id, user_name=None):
        """Return user data for current session or by name, rebinding session if needed."""
        with self.lock:
            if session_id in self.users:
                self._ensure_step_fields(self.users[session_id])
                return dict(self.users[session_id])
            if user_name and user_name in self.name_to_session:
                old_session = self.name_to_session[user_name]
                if old_session in self.users:
                    if old_session != session_id:
                        self.users[session_id] = self._ensure_step_fields(self.users.pop(old_session))
                        self.name_to_session[user_name] = session_id
                    return dict(self.users[session_id])
        return None

    def is_registered(self, session_id):
        with self.lock:
            return session_id in self.users

    def start_timer(self, step:int, duration:int):
        """Start a countdown timer for the given step."""
        with self.lock:
            self.active_timer = {
                'step': step,
                'duration': int(duration),
                'started_at': datetime.utcnow().timestamp()
            }

    def clear_timer(self):
        with self.lock:
            self.active_timer = None

    def get_remaining_seconds(self):
        with self.lock:
            if not self.active_timer:
                return None
            step = self.active_timer.get('step')
            duration = self.active_timer.get('duration', 0)
            started_at = self.active_timer.get('started_at', 0)
            elapsed = max(0, datetime.utcnow().timestamp() - started_at)
            remaining = max(0, duration - int(elapsed))
            return {'step': step, 'remaining': remaining}

survey_state = SurveyState()

DURATION_STEP1AND2 = 8 * 60  # seconds
DURATION_STEP3 = 4 * 60      # seconds
DURATION_STEP4 = 4 * 60      # seconds

def _get_mode_step():
    """Return the step associated with the current view mode."""
    if survey_state.view_mode == 'results':
        return survey_state.current_result_step or survey_state.current_step
    return survey_state.current_step

def emit_state_sync():
    """Emit the current step/mode to the requesting client."""
    mode_step = _get_mode_step()
    registered = survey_state.is_registered(request.sid)
    remaining = survey_state.get_remaining_seconds()
    emit('state_sync', {
        'step': survey_state.current_step,
        'mode': survey_state.view_mode,
        'mode_step': mode_step,
        'registered': registered,
        'remaining_seconds': remaining.get('remaining') if remaining and remaining.get('step') == survey_state.current_step else None
    })
    # Also mirror the existing step/mode events for clients that rely on them
    emit('step_update', {'step': survey_state.current_step})
    emit('mode_update', {
        'mode': survey_state.view_mode,
        'step': mode_step,
        'remaining_seconds': remaining.get('remaining') if remaining and remaining.get('step') == mode_step else None
    })

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def set_current_prompt_step(step):
    """Set or clear the current prompt step marker for the scoring GUI."""
    step_path = os.path.join(USER_DATA_DIR, 'current_prompt_step.json')
    try:
        if step is None:
            if os.path.exists(step_path):
                os.remove(step_path)
            return
        with open(step_path, 'w', encoding='utf-8') as sf:
            json.dump({'step': int(step)}, sf)
    except Exception:
        pass

def persist_consent_choice(name:str, consent_choice:str):
    """Append the group's consent choice to a local JSONL log."""
    try:
        record = {
            'name': name,
            'consent': consent_choice,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        with open(CONSENT_LOG_PATH, 'a', encoding='utf-8') as cf:
            cf.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception:
        # Do not block the flow if writing fails
        pass

def persist_prompt_timestamp(name: str, step, response: str):
    """Append a timestamped record for each prompt submission."""
    try:
        record = {
            'name': name,
            'step': step,
            'response': response,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        with open(PROMPT_TIMESTAMP_LOG_PATH, 'a', encoding='utf-8') as pf:
            pf.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception:
        # Keep submission flow resilient if logging fails
        pass

def persist_step_inputs(step:int, set_active_step: bool = False):
    """Persist prompt{step}.json and user_data inputs files for the given step.
    Safe no-op if there are no responses yet. When set_active_step=True, also update
    current_prompt_step.json so the scoring GUI writes to the correct file."""
    try:
        responses = survey_state.get_responses_for_step(step)
        if not responses:
            return
        inputs_step_path = os.path.join(USER_DATA_DIR, f'inputs{step}.json')
        names_step_path = os.path.join(USER_DATA_DIR, f'inputs{step}_names.json')

        # Start from any existing mapping to keep request_id indices stable, then
        # append/update new responses by name. This prevents reordering when late
        # submissions arrive after scoring has begun.
        existing_inputs = load_json(inputs_step_path) if os.path.exists(inputs_step_path) else {}
        existing_names = load_json(names_step_path) if os.path.exists(names_step_path) else {}
        inputs_map = {}
        names_map = {}

        # Preserve existing numeric order
        if isinstance(existing_inputs, dict):
            try:
                sorted_keys = sorted(existing_inputs.keys(), key=lambda k: int(k) if str(k).isdigit() else k)
            except Exception:
                sorted_keys = existing_inputs.keys()
            for key in sorted_keys:
                skey = str(key)
                inputs_map[skey] = existing_inputs.get(key)
                if isinstance(existing_names, dict) and skey in existing_names:
                    names_map[skey] = existing_names[skey]

        # Build quick lookup of existing indices by name
        name_to_idx = {}
        for k, v in (names_map or {}).items():
            try:
                idx_val = int(k)
            except Exception:
                continue
            if isinstance(v, str):
                name_to_idx[v] = idx_val

        next_index = len(inputs_map) + 1
        for name, prompt in responses.items():
            if not name:
                continue
            existing_idx = name_to_idx.get(name)
            if existing_idx:
                skey = str(existing_idx)
                inputs_map[skey] = prompt
                names_map[skey] = name
            else:
                skey = str(next_index)
                next_index += 1
                inputs_map[skey] = prompt
                names_map[skey] = name
        # Write per-step inputs file only
        with open(inputs_step_path, 'w', encoding='utf-8') as isf:
            json.dump(inputs_map, isf, indent=2, ensure_ascii=False)
        with open(names_step_path, 'w', encoding='utf-8') as nf:
            json.dump(names_map, nf, indent=2, ensure_ascii=False)
        if set_active_step:
            set_current_prompt_step(step)
    except Exception:
        # Silent fail - UI continues without crashing
        pass

def read_jsonl_latest_by_index(file_path):
    """Read a JSONL scored file and return latest record per input index (from request_id 'input-{i}')."""
    latest = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                req_id = obj.get('request_id', '')
                if req_id.startswith('input-'):
                    try:
                        idx = int(req_id.split('-')[1])
                    except Exception:
                        continue
                    ts = float(obj.get('timestamp', 0)) if isinstance(obj.get('timestamp', 0), (int, float)) else 0
                    prev = latest.get(idx)
                    if prev is None or ts >= prev.get('_ts', 0):
                        obj['_ts'] = ts
                        latest[idx] = obj
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return latest

def check_all_inputs_scored(step_to_show):
    """Check if all inputs for a step have been scored. Returns (all_scored: bool, missing_count: int)."""
    inputs_map = load_json(os.path.join(USER_DATA_DIR, f'inputs{step_to_show}.json'))
    if not inputs_map or not isinstance(inputs_map, dict):
        return False, 0
    
    # Try both filename formats for backward compatibility
    scored_path1 = os.path.join(USER_DATA_DIR, f'scored_inputs{step_to_show}.jsonl')
    scored_path2 = os.path.join(USER_DATA_DIR, f'scored_input_prompt{step_to_show}.jsonl')
    
    scored_path = scored_path1 if os.path.exists(scored_path1) else scored_path2
    if not os.path.exists(scored_path):
        return False, len(inputs_map)
    
    latest_records = read_jsonl_latest_by_index(scored_path)
    
    # Check if we have scores for all inputs (inputs are 1-indexed in the map)
    num_inputs = len(inputs_map)
    scored_indices = set(latest_records.keys())
    expected_indices = set(range(num_inputs))  # 0-indexed: 0, 1, 2, ...
    
    missing = expected_indices - scored_indices
    return len(missing) == 0, len(missing)

def get_scores_for_step(step_to_show, ensure_scored: bool = True):
    """Return (high_sorted, low_sorted, user_scores_dict, user_plans, user_prompts, user_breakdowns) for a step.
    When ensure_scored is True, raises ValueError if scoring is incomplete."""
    # Ensure per-step inputs file exists/updated
    persist_step_inputs(step_to_show)
    
    if ensure_scored:
        all_scored, missing_count = check_all_inputs_scored(step_to_show)
        if not all_scored:
            # If there is no data (after a reset/backup), return empty instead of throwing
            if missing_count == 0:
                return [], [], {}, {}
            raise ValueError(f"Not all inputs for step {step_to_show} have been scored yet. Missing {missing_count} score(s).")
    
    # Load scores from user_data scored_inputs{n}.jsonl, map indices to names
    inputs_map = load_json(os.path.join(USER_DATA_DIR, f'inputs{step_to_show}.json'))  # { '1': text }
    names_map = load_json(os.path.join(USER_DATA_DIR, f'inputs{step_to_show}_names.json'))  # { '1': name }
    if not inputs_map:
        return [], [], {}, {}
    if not isinstance(names_map, dict) or not names_map:
        raise ValueError(f"Missing names map for step {step_to_show}; cannot safely map scores to users")
    # Build prompt->name mapping from current survey_state responses
    responses_now = survey_state.get_responses_for_step(step_to_show)  # {name: prompt}
    value_to_name = {}
    for nm, txt in (responses_now or {}).items():
        if isinstance(txt, str) and txt not in value_to_name:
            value_to_name[txt] = nm
    
    # Try new filename format first, fall back to old format for backward compatibility
    scored_path1 = os.path.join(USER_DATA_DIR, f'scored_inputs{step_to_show}.jsonl')
    scored_path2 = os.path.join(USER_DATA_DIR, f'scored_input_prompt{step_to_show}.jsonl')
    scored_path = scored_path1 if os.path.exists(scored_path1) else scored_path2
    
    latest_records = read_jsonl_latest_by_index(scored_path)
    
    items = []
    present_names = set()
    user_scores = {}      # {name: score} for mobile users
    user_plans = {}       # {name: plan_snapshot}
    user_prompts = {}     # {name: prompt_text}
    user_breakdowns = {}  # {name: breakdown list}
    
    used_input_keys = set()  # keep mapping 1-idx string keys already assigned to a score

    for idx, rec in latest_records.items():
        # request_id input-{idx}; resolve name via inputs and current responses
        key = str(idx + 1)
        resolved_key = key
        prompt_from_inputs = inputs_map.get(key, '') if isinstance(inputs_map, dict) else ''
        primary_name = names_map.get(key)
        if not primary_name:
            raise ValueError(f"Missing name mapping for index {key} in step {step_to_show}")
        resolved_name = primary_name
        # Extract prompt text
        input_text = rec.get('input_text', '')
        if input_text:
            # Often formatted as: [n] "..."
            try:
                first_quote = input_text.find('"')
                last_quote = input_text.rfind('"')
                prompt_text = input_text[first_quote+1:last_quote] if first_quote != -1 and last_quote > first_quote else input_text
            except Exception:
                prompt_text = input_text
        else:
            # fallback to inputs file by index
            prompt_text = inputs_map.get(key, '') if isinstance(inputs_map, dict) else ''

        # Ensure scorer prompt matches stored prompt for this index
        if prompt_from_inputs and prompt_text and prompt_from_inputs.strip() != prompt_text.strip():
            # Try to remap to the correct name instead of blocking progression when order changed.
            try:
                candidate_keys = [
                    k for k, v in (inputs_map or {}).items()
                    if isinstance(v, str) and v.strip() == prompt_text.strip()
                ]
            except Exception:
                candidate_keys = []

            # Filter out already used slots to avoid double-assigning the same input index.
            candidate_keys = [k for k in candidate_keys if k not in used_input_keys]

            if len(candidate_keys) == 1 and names_map.get(candidate_keys[0]):
                resolved_key = candidate_keys[0]
                resolved_name = names_map.get(resolved_key)
            elif prompt_text in value_to_name:
                resolved_name = value_to_name[prompt_text]
            else:
                # Ambiguous mismatch: surface an explicit error instead of silently mis-mapping.
                raise ValueError(
                    f"Prompt mismatch for step {step_to_show} index {key}: "
                    f"inputs file has '{prompt_from_inputs}' but score record has '{prompt_text}'. "
                    "Could not uniquely remap this score to a user."
                )
        used_input_keys.add(resolved_key)
        
        # Prefer the authoritative names_map for display and aggregation
        display_name = resolved_name or value_to_name.get(prompt_text, f"User #{key}")
        present_names.add(display_name)
        
        score = rec.get('score')
        # If any action is a FAIL placeholder, treat the entire entry as a zero-score run
        actions_field = rec.get('actions') or []
        try:
            has_fail = any(isinstance(a, str) and a.strip().upper().startswith('FAIL') for a in actions_field)
        except Exception:
            has_fail = False

        if has_fail:
            numeric_score = 0.0
        else:
            try:
                numeric_score = float(score)
            except Exception:
                continue
        user_scores[display_name] = numeric_score
        user_prompts[display_name] = prompt_text

        plan_snapshot = rec.get('plan_snapshot')
        if isinstance(plan_snapshot, dict):
            user_plans[display_name] = plan_snapshot

        breakdown = rec.get('breakdown')
        if isinstance(breakdown, list):
            user_breakdowns[display_name] = breakdown
        
        items.append({
            'user_key': display_name,
            'display_name': display_name,
            'score': numeric_score,
            'prompt': prompt_text
        })
    
    # Add zero-score entries for registered users that did not submit/score
    combined_names = set(value_to_name.values()) if isinstance(value_to_name, dict) else set()
    combined_names.update(n for n in names_map.values() if n)
    all_registered = list(combined_names)  # only users included in this inputs file
    for name in all_registered:
        if not name:
            continue
        if name in present_names:
            continue
        user_scores[name] = 0.0
        user_plans.setdefault(name, None)
        user_prompts.setdefault(name, '')
        user_breakdowns.setdefault(name, [])
        items.append({
            'user_key': name,
            'display_name': name,
            'score': 0.0,
            'prompt': ''
        })
    
    high = [i for i in items if i['score'] >= 0]
    low = [i for i in items if i['score'] < 0]
    high_sorted = sorted(high, key=lambda x: x['score'], reverse=True)[:5]
    low_sorted = sorted(low, key=lambda x: x['score'])[:5]
    for idx, i in enumerate(low_sorted, start=1):
        i['display_name'] = f"Anonymous #{idx}"
    
    return high_sorted, low_sorted, user_scores, user_plans, user_prompts, user_breakdowns

def prepare_results_payload(step_to_show: int, allow_partial: bool = False):
    """Return payload dict for broadcasting results or an error descriptor.
    When allow_partial is True, return available scores even if some inputs are unscored."""
    all_scored, missing_count = check_all_inputs_scored(step_to_show)
    if not all_scored and not allow_partial:
        message = f'Waiting for scores: {missing_count} input(s) not yet scored for step {step_to_show}'
        return None, {'message': message, 'missing_count': missing_count}
    
    try:
        high_sorted, low_sorted, user_scores, user_plans, user_prompts, user_breakdowns = get_scores_for_step(
            step_to_show,
            ensure_scored=not allow_partial
        )
    except ValueError as exc:
        return None, {'message': str(exc), 'missing_count': None if all_scored else missing_count}
    
    payload = {
        'step': step_to_show,
        'top_high': high_sorted,
        'bottom_low': low_sorted,
        'user_scores': user_scores,
        'user_plans': user_plans,
        'user_prompts': user_prompts,
        'user_breakdowns': user_breakdowns
    }
    if allow_partial and not all_scored:
        payload['forced_missing_count'] = missing_count
    
    return payload, None

def notify_scores_pending(current_state_step: int, step_to_show: int, error_payload: dict):
    """Emit consistent error messaging when scores are not ready."""
    message = error_payload.get('message', 'Scores not ready')
    missing = error_payload.get('missing_count')
    emit('step_advanced', {
        'step': current_state_step,
        'error': message
    }, room='pc')
    socketio.emit('scores_not_ready', {
        'step': step_to_show,
        'missing_count': missing,
        'message': message
    }, room='pc')

def broadcast_results(payload: dict):
    """Send results payload to PC and mobile clients."""
    step_to_show = payload['step']
    forced_missing = payload.get('forced_missing_count')
    socketio.emit('results_ready', {
        'step': step_to_show,
        'top_high': payload['top_high'],
        'bottom_low': payload['bottom_low'],
        'forced_missing_count': forced_missing
    }, room='pc')
    socketio.emit('mode_update', {'mode': 'results', 'step': step_to_show})
    socketio.emit('results_ready_mobile', {
        'step': step_to_show,
        'user_scores': payload['user_scores'],
        'user_plans': payload.get('user_plans', {}),
        'user_prompts': payload.get('user_prompts', {}),
        'user_breakdowns': payload.get('user_breakdowns', {}),
        'forced_missing_count': forced_missing
    })
    if step_to_show < 4:
        set_current_prompt_step(step_to_show + 1)

def get_local_ip():
    """Get the local IP address of the machine, handling Docker containers"""
    # Check for environment variable override (useful for Docker/host networking)
    external_ip = os.environ.get('EXTERNAL_IP')
    if external_ip:
        return external_ip
    
    # Check if we're in Docker
    is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == 'true'
    
    if is_docker:
        # In Docker, try to get host IP from host.docker.internal
        try:
            # Try host.docker.internal (works on Docker Desktop)
            host_ip = socket.gethostbyname('host.docker.internal')
            return host_ip
        except (socket.gaierror, OSError):
            # Fallback: try to get host gateway (for Linux Docker)
            try:
                import subprocess
                result = subprocess.run(['ip', 'route', 'show', 'default'], 
                                      capture_output=True, text=True, timeout=2)
                for line in result.stdout.split('\n'):
                    if 'default via' in line:
                        parts = line.split()
                        if len(parts) >= 3:
                            return parts[2]  # Gateway IP
            except Exception:
                pass
            # Last resort: return localhost (for host access)
            return "127.0.0.1"
    else:
        # Not in Docker, get actual local IP
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

def generate_qr_code(url):
    """Generate QR code image as base64 string"""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

@app.route('/dashboard/login', methods=['GET', 'POST'])
def dashboard_login():
    """Login page for dashboard access"""
    if request.method == 'POST':
        password = request.form.get('password', '')
        if password == DASHBOARD_PASSWORD:
            session['dashboard_authenticated'] = True
            session.permanent = True  # Keep session alive
            return redirect(url_for('pc_interface'))
        else:
            return render_template('dashboard_login.html', error='Invalid password')
    return render_template('dashboard_login.html', error=None)

@app.route('/dashboard/logout')
def dashboard_logout():
    """Logout from dashboard"""
    session.pop('dashboard_authenticated', None)
    return redirect(url_for('dashboard_login'))

@app.route('/')
@dashboard_required
def pc_interface():
    """PC control interface with QR code"""
    local_ip = get_local_ip()
    port = request.environ.get('SERVER_PORT', '5000')
    
    # If accessed via localhost/127.0.0.1, prefer that for QR code (Docker host access)
    request_host = request.host.split(':')[0] if ':' in request.host else request.host
    if request_host in ['localhost', '127.0.0.1'] and local_ip != '127.0.0.1':
        # Check if EXTERNAL_IP is set (for mobile access)
        mobile_ip = os.environ.get('EXTERNAL_IP', local_ip)
        # For QR code, use external IP if set, otherwise try to get network IP
        url = f"http://{mobile_ip}:{port}/mobile"
    else:
        url = f"http://{local_ip}:{port}/mobile"
    
    qr_image = generate_qr_code(url)
    return render_template('pc_interface.html', qr_image=qr_image, url=url, host_ip=request_host)

@app.route('/mobile')
def mobile_entry():
    """Mobile user entry point - name input"""
    return render_template('mobile_entry.html')

@app.route('/mobile/step3')
def mobile_step3():
    """Mobile step 3 form"""
    return render_template('mobile_step3.html')

@app.route('/mobile/step4')
def mobile_step4():
    """Mobile step 4 form"""
    return render_template('mobile_step4.html')

@app.route('/mobile/thanks')
def mobile_thanks():
    """Thank you page"""
    return render_template('mobile_thanks.html')

@app.route('/mobile/step1and2')
def mobile_step1and2():
    """Mobile combined step 1 and 2 form"""
    return render_template('mobile_step1and2.html')

@app.route('/mobile/score/<int:step>')
def mobile_score(step:int):
    """Mobile score display for prompts 1-4"""
    if step not in (1, 2, 3, 4):
        return redirect(url_for('mobile_entry'))
    return render_template('mobile_score.html', step=step)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    session_id = request.sid
    emit('connected', {'session_id': session_id})
    emit_state_sync()

@socketio.on('register_name')
def handle_register_name(data):
    """Register a user's name"""
    session_id = request.sid
    name = data.get('name', '').strip()
    consent_choice = data.get('consent_choice', '').strip()
    client_id = data.get('client_id') or session.get('client_id')
    valid_consent = {'workshop_only', 'workshop_research'}

    if not client_id:
        client_id = str(uuid.uuid4())
    session['client_id'] = client_id
    
    if not name:
        emit('registration_error', {'error': 'Name cannot be empty', 'code': 'name_required'})
        return

    if consent_choice not in valid_consent:
        emit('registration_error', {'error': 'Please select how your prompts may be used.', 'code': 'consent_required'})
        return

    success, error_code = survey_state.add_user(session_id, name, consent_choice, client_id)
    if success:
        session['registered_name'] = name
        emit('registration_success', {'name': name, 'consent_choice': consent_choice})
        join_room('pc')  # Join PC room to receive updates
        socketio.emit('user_registered', {'name': name}, room='pc')
        persist_consent_choice(name, consent_choice)
        # Send current step to user
        emit('step_update', {'step': survey_state.current_step})
        emit('mode_update', {'mode': survey_state.view_mode, 'step': survey_state.current_step})
    else:
        error_message = 'Registration failed. Please try again.'
        if error_code == 'consent_conflict':
            error_message = 'Consent choice cannot be changed for this group. Please use the original selection.'
        elif error_code == 'name_taken':
            error_message = 'This group name is already registered. Please choose another name.'
        emit('registration_error', {'error': error_message, 'code': error_code or 'unknown'})

@socketio.on('join_pc')
def handle_join_pc():
    """PC interface joins the PC room"""
    # Check if session is authenticated
    if not check_dashboard_auth():
        emit('auth_error', {'error': 'Not authenticated'})
        return
    
    join_room('pc')
    emit('pc_joined', {
        'users': survey_state.get_users_with_status(), 
        'current_step': survey_state.current_step
    })

@socketio.on('clear_saved_info')
def handle_clear_saved_info():
    """Broadcast to clients to clear saved connection info."""
    # Only allow from authenticated dashboard sessions
    if not check_dashboard_auth():
        emit('auth_error', {'error': 'Not authenticated'})
        return
    socketio.emit('clear_saved_info')

@socketio.on('submit_response')
def handle_submit_response(data):
    """Handle response submission from mobile user"""
    session_id = request.sid
    step = data.get('step')
    response = data.get('response', '').strip()
    user_name = data.get('name', None)  # Optional: user can send their name for reconnection
    
    # Handle combined step 1 and 2 submission
    if step == '1and2':
        step1_response = data.get('step1', '').strip()
        step2_response = data.get('step2', '').strip()
        
        if not step1_response or not step2_response:
            emit('submission_error', {'error': 'Both responses are required'})
            return
        
        # Try to get name from session if not provided
        if not user_name and session_id in survey_state.users:
            user_name = survey_state.users[session_id].get('name')
        
        # Submit both responses
        success1 = survey_state.submit_response(session_id, 1, step1_response, user_name)
        success2 = survey_state.submit_response(session_id, 2, step2_response, user_name)
        
        if success1 and success2:
            # Get the user's name for status update
            submitter_name = user_name
            if not submitter_name and session_id in survey_state.users:
                submitter_name = survey_state.users[session_id].get('name')
            
            remaining = survey_state.get_remaining_seconds()
            remaining_seconds = remaining.get('remaining') if remaining and remaining.get('step') == survey_state.current_step else None
            
            emit('submission_success', {'step': '1and2'})
            # Persist both prompt and inputs immediately so external GUI can see them
            persist_step_inputs(1, set_active_step=True)
            persist_step_inputs(2)
            # Log submission timestamps for auditing
            persist_prompt_timestamp(submitter_name, 1, step1_response)
            persist_prompt_timestamp(submitter_name, 2, step2_response)
            # Send updated user status to PC
            socketio.emit('response_submitted', {
                'step': '1and2', 
                'users': survey_state.get_users_with_status(),
                'remaining_seconds': remaining_seconds
            }, room='pc')
        else:
            emit('submission_error', {'error': 'You are not registered. Please scan the QR code to join again.'})
        return
    
    # Handle single step submission (steps 1-4 or legacy step 1/2)
    if step not in [1, 2, 3, 4]:
        emit('submission_error', {'error': 'Invalid step'})
        return
    
    # Try to get name from session if not provided
    if not user_name and session_id in survey_state.users:
        user_name = survey_state.users[session_id].get('name')
    
    if survey_state.submit_response(session_id, step, response, user_name):
        # Get the user's name for status update
        submitter_name = user_name
        if not submitter_name and session_id in survey_state.users:
            submitter_name = survey_state.users[session_id].get('name')
        
        remaining = survey_state.get_remaining_seconds()
        remaining_seconds = remaining.get('remaining') if remaining and remaining.get('step') == survey_state.current_step else None
        
        emit('submission_success', {'step': step})
        # Persist prompt and inputs immediately so external GUI can see them
        persist_step_inputs(step, set_active_step=True)
        persist_prompt_timestamp(submitter_name, step, response)
        # Send updated user status to PC
        socketio.emit('response_submitted', {
            'step': step, 
            'users': survey_state.get_users_with_status(),
            'remaining_seconds': remaining_seconds
        }, room='pc')
    else:
        emit('submission_error', {'error': 'You are not registered. Please scan the QR code to join again.'})

@socketio.on('request_saved_responses')
def handle_request_saved_responses(data):
    """Provide previously saved responses so users can resume after reconnect."""
    session_id = request.sid
    step = data.get('step')
    user_name = (data.get('name') or '').strip()
    
    user_record = survey_state.get_user_data_for_request(session_id, user_name)
    payload = {'step': step, 'responses': {}, 'submitted': False}
    
    if not user_record:
        emit('saved_responses', payload)
        return
    
    normalized_step = step
    if isinstance(step, str) and step.isdigit():
        normalized_step = int(step)
    
    if step == '1and2':
        step1 = user_record.get('step1')
        step2 = user_record.get('step2')
        payload['responses'] = {'step1': step1, 'step2': step2}
        payload['submitted'] = step1 is not None and step2 is not None
    elif normalized_step in [1, 2, 3, 4]:
        key = f'step{normalized_step}'
        value = user_record.get(key)
        payload['responses'] = {key: value}
        payload['submitted'] = value is not None
    else:
        payload['responses'] = {}
        payload['submitted'] = False
    
    emit('saved_responses', payload)

@socketio.on('advance_step')
def handle_advance_step(data=None):
    """PC advances state machine: entry -> results -> entry ... -> final -> reset"""
    # Check if session is authenticated
    if not check_dashboard_auth():
        emit('auth_error', {'error': 'Not authenticated'})
        return

    # Allow an explicit force flag from the PC UI to bypass scoring completeness checks
    force = False
    if isinstance(data, dict):
        try:
            force = bool(data.get('force'))
        except Exception:
            force = False

    def build_forced_payload(step_to_show: int, error_payload: dict):
        """Return a minimal results payload when forcing past missing scores."""
        missing = None
        if isinstance(error_payload, dict):
            missing = error_payload.get('missing_count')
        return {
            'step': step_to_show,
            'top_high': [],
            'bottom_low': [],
            'user_scores': {},
            'user_plans': {},
            'user_prompts': {},
            'user_breakdowns': {},
            'forced_missing_count': missing
        }
    
    # ENTRY -> initialize first step (combined prompts 1&2)
    if survey_state.view_mode == 'entry' and survey_state.current_step == 0:
        survey_state.current_step = 1
        survey_state.current_result_step = None
        survey_state.start_timer(1, DURATION_STEP1AND2)
        socketio.emit('step_update', {'step': survey_state.current_step})
        socketio.emit('mode_update', {'mode': 'entry', 'step': survey_state.current_step, 'remaining_seconds': DURATION_STEP1AND2})
        set_current_prompt_step(1)
        emit('step_advanced', {'step': survey_state.current_step}, room='pc')
        return

    # ENTRY step 1 (combined prompts) -> RESULTS step 1
    if survey_state.view_mode == 'entry' and survey_state.current_step == 1:
        payload, error = prepare_results_payload(1)
        if error:
            if not force:
                notify_scores_pending(survey_state.current_step, 1, error)
                return
            payload, partial_error = prepare_results_payload(1, allow_partial=True)
            if partial_error or not payload:
                payload = build_forced_payload(1, partial_error or error)
        survey_state.view_mode = 'results'
        survey_state.current_result_step = 1
        survey_state.clear_timer()
        broadcast_results(payload)
        emit('step_advanced', {'step': survey_state.current_step}, room='pc')
        return

    # RESULTS step 1 -> RESULTS step 2
    if survey_state.view_mode == 'results' and survey_state.current_result_step == 1:
        payload, error = prepare_results_payload(2)
        if error:
            if not force:
                notify_scores_pending(survey_state.current_step, 2, error)
                return
            payload, partial_error = prepare_results_payload(2, allow_partial=True)
            if partial_error or not payload:
                payload = build_forced_payload(2, partial_error or error)
        survey_state.current_result_step = 2
        broadcast_results(payload)
        emit('step_advanced', {'step': survey_state.current_step}, room='pc')
        return

    # RESULTS step 2 -> ENTRY step 3
    if survey_state.view_mode == 'results' and survey_state.current_result_step == 2:
        survey_state.view_mode = 'entry'
        survey_state.current_step = 3
        survey_state.current_result_step = None
        survey_state.start_timer(3, DURATION_STEP3)
        socketio.emit('step_update', {'step': survey_state.current_step})
        socketio.emit('mode_update', {'mode': 'entry', 'step': survey_state.current_step, 'remaining_seconds': DURATION_STEP3})
        set_current_prompt_step(3)
        emit('step_advanced', {'step': survey_state.current_step}, room='pc')
        return

    # ENTRY step 3 -> RESULTS step 3
    if survey_state.view_mode == 'entry' and survey_state.current_step == 3:
        payload, error = prepare_results_payload(3)
        if error:
            if not force:
                notify_scores_pending(survey_state.current_step, 3, error)
                return
            payload, partial_error = prepare_results_payload(3, allow_partial=True)
            if partial_error or not payload:
                payload = build_forced_payload(3, partial_error or error)
        survey_state.view_mode = 'results'
        survey_state.current_result_step = 3
        survey_state.clear_timer()
        broadcast_results(payload)
        emit('step_advanced', {'step': survey_state.current_step}, room='pc')
        return

    # RESULTS step 3 -> ENTRY step 4
    if survey_state.view_mode == 'results' and survey_state.current_result_step == 3:
        survey_state.view_mode = 'entry'
        survey_state.current_step = 4
        survey_state.current_result_step = None
        survey_state.start_timer(4, DURATION_STEP4)
        socketio.emit('step_update', {'step': survey_state.current_step})
        socketio.emit('mode_update', {'mode': 'entry', 'step': survey_state.current_step, 'remaining_seconds': DURATION_STEP4})
        set_current_prompt_step(4)
        emit('step_advanced', {'step': survey_state.current_step}, room='pc')
        return

    # ENTRY step 4 -> RESULTS step 4
    if survey_state.view_mode == 'entry' and survey_state.current_step == 4:
        payload, error = prepare_results_payload(4)
        if error:
            if not force:
                notify_scores_pending(survey_state.current_step, 4, error)
                return
            payload, partial_error = prepare_results_payload(4, allow_partial=True)
            if partial_error or not payload:
                payload = build_forced_payload(4, partial_error or error)
        survey_state.view_mode = 'results'
        survey_state.current_result_step = 4
        survey_state.clear_timer()
        broadcast_results(payload)
        emit('step_advanced', {'step': survey_state.current_step}, room='pc')
        return

    # RESULTS after step 4 -> FINAL
    if survey_state.view_mode == 'results' and survey_state.current_result_step == 4:
        # Aggregate averages over up to 4 steps using cached inputs/scores per step
        # Use the same scoring/mapping logic as per-step displays to avoid mismatches
        per_user_scores: dict[str, list[float]] = {}
        all_names: set[str] = set()

        step_user_scores: dict[int, dict] = {}
        for s in [1, 2, 3, 4]:
            try:
                _high, _low, user_scores, _plans, _prompts, _breakdowns = get_scores_for_step(s, ensure_scored=False)
            except Exception:
                user_scores = {}
            step_user_scores[s] = user_scores or {}
            all_names.update(user_scores.keys())

        for name in all_names:
            scores_for_name: list[float] = []
            for s in [1, 2, 3, 4]:
                score_val = step_user_scores.get(s, {}).get(name, 0.0)
                try:
                    score_val = float(score_val)
                except Exception:
                    score_val = 0.0
                scores_for_name.append(score_val)
            per_user_scores[name] = scores_for_name

        leaderboard = []
        for name, arr in per_user_scores.items():
            if not arr:
                continue
            avg = sum(arr) / len(arr)
            leaderboard.append({'name': name, 'avg': avg})

        leaderboard.sort(key=lambda x: (-x['avg'], x['name']))
        top10 = []
        for idx, row in enumerate(leaderboard[:10], start=1):
            top10.append({'rank': idx, 'name': row['name'], 'avg': row['avg']})

        podium = []
        for place in [1, 2, 3]:
            if len(leaderboard) >= place:
                podium.append({'place': place, 'name': leaderboard[place-1]['name'], 'avg': leaderboard[place-1]['avg']})

        survey_state.view_mode = 'final'
        socketio.emit('final_leaderboard', {
            'podium': podium,
            'leaderboardTop10': top10
        }, room='pc')
        socketio.emit('mode_update', {'mode': 'final'})
        emit('step_advanced', {'step': survey_state.current_step}, room='pc')
        return

    # FINAL -> RESET to start
    if survey_state.view_mode == 'final':
        survey_state.current_step = 0
        survey_state.view_mode = 'entry'
        survey_state.current_result_step = None
        socketio.emit('reset_to_start', {})
        socketio.emit('step_update', {'step': survey_state.current_step})
        socketio.emit('mode_update', {'mode': 'entry', 'step': survey_state.current_step})
        set_current_prompt_step(None)
        emit('step_advanced', {'step': survey_state.current_step}, room='pc')
        return

@socketio.on('get_my_score')
def handle_get_my_score(data):
    """Handle request from mobile user to get their score for a specific step"""
    step = data.get('step')
    user_name = data.get('name')
    
    if step not in [1, 2, 3, 4] or not user_name:
        emit('my_score', {'step': step, 'name': user_name, 'score': None})
        return
    
    # Get scores for this step
    partial_data = False
    try:
        high_sorted, low_sorted, user_scores, user_plans, user_prompts, user_breakdowns = get_scores_for_step(step)
    except ValueError as exc:
        try:
            high_sorted, low_sorted, user_scores, user_plans, user_prompts, user_breakdowns = get_scores_for_step(step, ensure_scored=False)
            partial_data = True
        except Exception:
            # If scores are incomplete or mapping fails, return a neutral response instead of crashing
            emit('my_score', {
                'step': step,
                'name': user_name,
                'score': None,
                'plan_snapshot': None,
                'prompt': None,
                'breakdown': None,
                'error': str(exc)
            })
            return
    
    # Find user's score
    score = user_scores.get(user_name, None)
    plan_snapshot = user_plans.get(user_name)
    prompt_text = user_prompts.get(user_name)
    breakdown = user_breakdowns.get(user_name)
    
    emit('my_score', {
        'step': step,
        'name': user_name,
        'score': score,
        'plan_snapshot': plan_snapshot,
        'prompt': prompt_text,
        'breakdown': breakdown,
        'partial': partial_data
    })

@socketio.on('get_state')
def handle_get_state():
    """Get current survey state"""
    registered = survey_state.is_registered(request.sid)
    remaining = survey_state.get_remaining_seconds()
    emit('state_update', {
        'current_step': survey_state.current_step,
        'users': survey_state.get_users_with_status(),
        'view_mode': survey_state.view_mode,
        'current_result_step': survey_state.current_result_step,
        'mode_step': _get_mode_step(),
        'registered': registered,
        'remaining_seconds': remaining.get('remaining') if remaining and remaining.get('step') == survey_state.current_step else None
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    session_id = request.sid
    # Optionally clean up user data on disconnect
    # For now, we'll keep it to preserve responses


def backup_and_clear_user_data():
    """Backup current user_data contents then empty the directory."""
    try:
        if not os.path.exists(USER_DATA_DIR):
            return
        os.makedirs(USER_DATA_BACKUP_ROOT, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        backup_path = os.path.join(USER_DATA_BACKUP_ROOT, f"user_data_backup_{timestamp}")
        shutil.copytree(USER_DATA_DIR, backup_path)

        # Clear files and subdirectories after successful backup
        for entry in os.listdir(USER_DATA_DIR):
            entry_path = os.path.join(USER_DATA_DIR, entry)
            try:
                if os.path.isfile(entry_path) or os.path.islink(entry_path):
                    os.remove(entry_path)
                else:
                    shutil.rmtree(entry_path)
            except Exception:
                pass
        print(f"✓ Backed up user_data to: {backup_path}")
        print("✓ Cleared user_data directory")
    except Exception as exc:
        print(f"⚠️  Failed to back up user_data: {exc}")
        print("   user_data directory left untouched for safety")

if __name__ == '__main__':
    # Get local IP and port
    local_ip = get_local_ip()
    # Use port 10000 by default (commonly mapped in Docker), but allow override
    port = int(os.environ.get('PORT', 10000))
    
    # Check if in Docker
    is_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == 'true'
    
    print(f"\n{'='*60}")
    print(f"Starting Survey Server")
    print(f"{'='*60}")
    
    # Display dashboard password info
    if DASHBOARD_PASSWORD == 'admin123':
        print(f"⚠️  WARNING: Using default dashboard password!")
        print(f"   Set DASHBOARD_PASSWORD environment variable for security")
    else:
        print(f"✓ Custom dashboard password is set")
    print(f"   Dashboard password: {DASHBOARD_PASSWORD}")
    print()
    
    if is_docker:
        print(f"⚠️  Running in Docker container")
        print(f"\nFor HOST access (Windows PC):")
        print(f"   Dashboard: http://localhost:{port}")
        print(f"   Login with password: {DASHBOARD_PASSWORD}")
        print(f"\nFor MOBILE access (same network):")
        print(f"   Set EXTERNAL_IP environment variable to your host machine's IP")
        print(f"   Example: EXTERNAL_IP=192.168.1.100 python survey_server.py")
        print(f"\nCurrent QR code will use: http://{local_ip}:{port}/mobile")
        if local_ip == "127.0.0.1":
            print(f"   ⚠️  This is localhost - not accessible from mobile devices!")
            print(f"   To fix: Set EXTERNAL_IP=<your-host-ip> or use docker run with --net=host")
    else:
        print(f"Dashboard: http://{local_ip}:{port}")
        print(f"   Login with password: {DASHBOARD_PASSWORD}")
        print(f"\nMobile URL: http://{local_ip}:{port}/mobile")
        print(f"   (No authentication required for mobile users)")
    
    print(f"\nMake sure Docker port {port} is mapped: -p {port}:{port}")
    print(f"{'='*60}\n")

    backup_and_clear_user_data()
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
