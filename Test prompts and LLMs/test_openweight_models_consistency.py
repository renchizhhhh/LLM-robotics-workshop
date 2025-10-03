#!/usr/bin/env python3
"""
Test script for Deepseek model consistency testing
Tests the same model multiple times to check consistency and reliability.
"""

import os
import json
import time
import asyncio
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# LLM imports
import openai
try:
    from together import Together
except ImportError:
    Together = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class LLMProvider(Enum):
    # Groq models
    GROQ_DEEPSEEK_R1_DISTILL_LLAMA_70B = "deepseek-r1-distill-llama-70b"
    OPENAI_GPT_OSS_120B = "openai/gpt-oss-120b"
    OPENAI_GPT_OSS_20B = "openai/gpt-oss-20b"
    OPENAI_GPT_OSS_20B_LOCAL = "openai/gpt-oss-20b-local"
    GROQ_LLAMA_3_3_70B_VERSATILE = "llama-3.3-70b-versatile"
    GROQ_LLAMA_4_SCOUT_17B = "meta-llama/llama-4-scout-17b-16e-instruct"
    GROQ_QWEN3_32B = "qwen/qwen3-32b"
    
    # Together AI models (serverless)
    TOGETHER_LLAMA_3_3_70B_INSTRUCT = "meta-llama/Meta-Llama-3.3-70B-Instruct-Turbo"
    TOGETHER_LLAMA_3_3_70B_INSTRUCT_FREE = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    TOGETHER_LLAMA_3_1_70B_INSTRUCT = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    TOGETHER_LLAMA_3_70B_CHAT_HF = "meta-llama/Llama-3-70b-chat-hf"
    TOGETHER_DEEPSEEK_R1 = "deepseek-ai/DeepSeek-R1"
    TOGETHER_DEEPSEEK_R1_DISTILL_70B = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    TOGETHER_DEEPSEEK_R1_DISTILL_70B_FREE = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    TOGETHER_QWEN_2_5_72B_INSTRUCT = "Qwen/Qwen2.5-72B-Instruct-Turbo"
    TOGETHER_QWEN3_NEXT_80B_A3B_INSTRUCT = "Qwen/Qwen3-Next-80B-A3B-Instruct"
    TOGETHER_QWEN3_235B_A22B_INSTRUCT = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
    TOGETHER_MIXTRAL_8X22B_INSTRUCT = "mistralai/Mixtral-8x22B-Instruct-v0.1"
    TOGETHER_GPT_OSS_120B = "openai/gpt-oss-120b"

class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

@dataclass
class TestScenario:
    name: str
    difficulty: DifficultyLevel
    la_command: str
    ha_task: str
    description: str
    ground_truth_la: List[List[str]]  # Multiple valid ground truth options for LA mode
    ground_truth_ha: List[List[str]]  # Multiple valid ground truth options for HA mode

@dataclass
class TestResult:
    run_number: int
    llm_provider: LLMProvider
    scenario: TestScenario
    mode: str  # 'LA' or 'HA'
    prompt_used: str
    response: str
    parsed_actions: List[str]
    execution_time: float
    success: bool
    score: float  # Score based on ground truth comparison (0.0 to 1.0)
    error_message: Optional[str] = None

class DeepseekConsistencyTester:
    def __init__(self):
        self.setup_llms()
        self.test_scenarios = self.create_test_scenarios()
        
    def setup_llms(self):
        """Setup LLM clients"""
        # Groq via OpenAI compat endpoint
        groq_api_key = os.getenv('GROQ_API_KEY')
        groq_base_url = os.getenv('GROQ_BASE_URL', 'https://api.groq.com/openai/v1')
        self.groq_client = None
        if groq_api_key:
            self.groq_client = openai.OpenAI(api_key=groq_api_key, base_url=groq_base_url)
        
        # OpenAI client for GPT-OSS-120B
        openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = None
        if openai_api_key:
            self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Together AI client
        together_api_key = os.getenv('TOGETHER_API_KEY')
        self.together_client = None
        if together_api_key and Together:
            self.together_client = Together(api_key=together_api_key)
        
        # Local models setup
        self.local_tokenizer = None
        self.local_model = None
        if TRANSFORMERS_AVAILABLE:
            try:
                print("Loading local openai/gpt-oss-20b model...")
                self.local_tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
                self.local_model = AutoModelForCausalLM.from_pretrained(
                    "openai/gpt-oss-20b",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                print("Local model loaded successfully!")
            except Exception as e:
                print(f"Warning: Could not load local model: {e}")
                self.local_tokenizer = None
                self.local_model = None
        
        if not self.groq_client and not self.openai_client and not self.together_client and not self.local_model:
            raise Exception("At least one API key (GROQ_API_KEY, OPENAI_API_KEY, or TOGETHER_API_KEY) must be set or local model must be available")
    
    def _is_together_model(self, provider: LLMProvider) -> bool:
        """Check if provider is a Together AI model"""
        return provider in [
            LLMProvider.TOGETHER_LLAMA_3_3_70B_INSTRUCT,
            LLMProvider.TOGETHER_LLAMA_3_3_70B_INSTRUCT_FREE,
            LLMProvider.TOGETHER_LLAMA_3_1_70B_INSTRUCT,
            LLMProvider.TOGETHER_LLAMA_3_70B_CHAT_HF,
            LLMProvider.TOGETHER_DEEPSEEK_R1,
            LLMProvider.TOGETHER_DEEPSEEK_R1_DISTILL_70B,
            LLMProvider.TOGETHER_DEEPSEEK_R1_DISTILL_70B_FREE,
            LLMProvider.TOGETHER_QWEN_2_5_72B_INSTRUCT,
            LLMProvider.TOGETHER_QWEN3_NEXT_80B_A3B_INSTRUCT,
            LLMProvider.TOGETHER_QWEN3_235B_A22B_INSTRUCT,
            LLMProvider.TOGETHER_MIXTRAL_8X22B_INSTRUCT,
            LLMProvider.TOGETHER_GPT_OSS_120B
        ]
    
    def _is_local_model(self, provider: LLMProvider) -> bool:
        """Check if provider is a local model"""
        return provider == LLMProvider.OPENAI_GPT_OSS_20B_LOCAL
        
    def create_test_scenarios(self) -> List[TestScenario]:
        """Create test scenarios with different difficulty levels"""
        return [
            # HARD: Complex Multi-step Task
            TestScenario(
                name="Complex Multi-step Task",
                difficulty=DifficultyLevel.HARD,
                la_command="pick up the pringles can and place it on the table",
                ha_task="go to the pick-up location, pick up the pringles can, go to the drop-off location, place the pringles can down, then return to the starting position",
                description="Complex task involving multiple movements, grasping, placing, and returning",
                ground_truth_la=[
                    # Full sequence (ideal)
                    [
                        "start_automated_grasp, object_type=pringles can",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow"
                    ],
                    [
                        "start_automated_grasp, object_type=\"pringles can\"",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow"
                    ],
                    [
                        "start_automated_grasp object_type=pringles can",
                        "start_move_arm_pose x=0.8 y=0.0 z=0.3 qw=0.7071 qx=0.7071 qy=0.0 qz=0.0 duration=1.0 open_gripper=true",
                        "start_arm_command command_type=open",
                        "start_arm_command command_type=close",
                        "start_arm_command command_type=stow"
                    ],
                    # Just grasp (what LLMs actually do in LA mode)
                    [
                        "start_automated_grasp, object_type=pringles can"
                    ],
                    [
                        "start_automated_grasp, object_type=\"pringles can\""
                    ],
                    [
                        "start_automated_grasp object_type=pringles can"
                    ],
                    # With pringles_can (underscore) variations
                    [
                        "start_automated_grasp, object_type=pringles_can"
                    ],
                    [
                        "start_automated_grasp, object_type=\"pringles_can\""
                    ],
                    [
                        "start_automated_grasp object_type=pringles_can"
                    ]
                ],
                ground_truth_ha=[
                    # Correct step-by-step approach (following prompt rules)
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=pringles can",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow",
                        "start_moving, x=-2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-0.5, yaw=0.0, frame=body"
                    ],
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=\"body\"",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=\"body\"",
                        "start_automated_grasp, object_type=\"pringles can\"",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=\"body\"",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow",
                        "start_moving, x=-2.0, y=0.0, yaw=0.0, frame=\"body\"",
                        "start_moving, x=0.0, y=-0.5, yaw=0.0, frame=\"body\""
                    ],
                    [
                        "start_moving x=2.0 y=0.0 yaw=0.0 frame=body",
                        "start_moving x=0.0 y=-1.0 yaw=0.0 frame=body",
                        "start_automated_grasp object_type=pringles can",
                        "start_moving x=0.0 y=1.5 yaw=0.0 frame=body",
                        "start_move_arm_pose x=0.8 y=0.0 z=0.3 qw=0.7071 qx=0.7071 qy=0.0 qz=0.0 duration=1.0 open_gripper=true",
                        "start_arm_command command_type=open",
                        "start_arm_command command_type=close",
                        "start_arm_command command_type=stow",
                        "start_moving x=-2.0 y=0.0 yaw=0.0 frame=body",
                        "start_moving x=0.0 y=-0.5 yaw=0.0 frame=body"
                    ],
                    # With pringles_can (underscore) variations
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=pringles_can",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow",
                        "start_moving, x=-2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-0.5, yaw=0.0, frame=body"
                    ],
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=\"body\"",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=\"body\"",
                        "start_automated_grasp, object_type=\"pringles_can\"",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=\"body\"",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow",
                        "start_moving, x=-2.0, y=0.0, yaw=0.0, frame=\"body\"",
                        "start_moving, x=0.0, y=-0.5, yaw=0.0, frame=\"body\""
                    ],
                    [
                        "start_moving x=2.0 y=0.0 yaw=0.0 frame=body",
                        "start_moving x=0.0 y=-1.0 yaw=0.0 frame=body",
                        "start_automated_grasp object_type=pringles_can",
                        "start_moving x=0.0 y=1.5 yaw=0.0 frame=body",
                        "start_move_arm_pose x=0.8 y=0.0 z=0.3 qw=0.7071 qx=0.7071 qy=0.0 qz=0.0 duration=1.0 open_gripper=true",
                        "start_arm_command command_type=open",
                        "start_arm_command command_type=close",
                        "start_arm_command command_type=stow",
                        "start_moving x=-2.0 y=0.0 yaw=0.0 frame=body",
                        "start_moving x=0.0 y=-0.5 yaw=0.0 frame=body"
                    ]
                ]
            )
        ]
    
    def get_prompts(self):
        """Get the prompts from test_prompts_V2.py"""
        LA_PROMPT = """\
Convert command to Spot FSM action(s). Generate ONLY the action(s) needed.

Command: {command}
Current robot state: {current_state}

AVAILABLE ACTIONS:
- stand_up
- sit_down  
- start_moving, x=float, y=float, yaw=float, frame=body
- start_automated_grasp, object_type=exact_object_name_from_user
- start_move_arm_pose, x=float, y=float, z=float, qw=float, qx=float, qy=float, qz=float, duration=float, open_gripper=bool
- start_arm_command, command_type=open|close|stow|carry

STRICT FORMATTING RULES:
1. One action per line, no quotes/brackets, no numbering
2. ALL movement parameters are REQUIRED: x=value, y=value, yaw=value, frame=body
3. Use exact object names from command
4. FORBIDDEN: Do NOT include movements with all zero values (x=0.0, y=0.0, yaw=0.0) - these are invalid
5. MANDATORY: Use commas between ALL parameters: start_moving, x=value, y=value, yaw=value, frame=body
6. For drop-off tasks: use arm_pose to position gripper above surface, then open gripper to release object
7. For drop-off sequence: arm_pose + open + close + stow
8. For arm poses: use standard drop-off pose (x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0)
9. Use reasonable duration values (typically 1-2 seconds for arm movements)

CRITICAL: Every parameter must be complete. NEVER truncate parameters like "frame" without "=body".

EXAMPLES OF CORRECT FORMAT:
start_moving, x=2.0, y=0.0, yaw=0.0, frame=body
start_automated_grasp, object_type=pringles can
start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true
start_arm_command, command_type=open

MANDATORY: Return ONLY the action(s). No reasoning, no explanations, no <think> tags.

Actions:"""

        HA_PROMPT = """You are a path planning system for a Boston Dynamics Spot robot.

CURRENT STATE:
- Robot state: {current_state}
- Current position (vision frame): {current_position}

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
5. Move ONE axis at a time: ONLY x OR ONLY y OR ONLY yaw (NEVER combine axes)
   - WRONG: x=1.0, y=-2.0, yaw=0.0 (combines x and y)
   - WRONG: x=0.0, y=0.5, yaw=3.14 (combines y and yaw) - FORBIDDEN
   - CORRECT: x=3.0, y=0.0, yaw=0.0 (only x)
   - CORRECT: x=0.0, y=3.0, yaw=0.0 (only y)
   - CORRECT: x=0.0, y=0.0, yaw=3.14 (only yaw)
6. Use RELATIVE body frame coordinates for movements - each movement is relative to current position
7. For movements, calculate: move_x = target_x - current_x, move_y = target_y - current_y
8. For pick-up: walk to pick-up location and face forward (yaw=0)
9. For drop-off: walk to drop-off location and face left (yaw=1.57 radians)
10. Use EXACT object name from user command for object_type
11. For drop-off procedure: start_move_arm_pose to (0.8, 0.0, 0.3) with quaternion (0.7071, 0.7071, 0.0, 0.0) for gripper pointing down in 1 second, then start_arm_command open, then start_arm_command close, then start_arm_command stow
12. ALL movement parameters are REQUIRED: x=value, y=value, yaw=value, frame=body
13. Do NOT include movements with all zero values (x=0.0, y=0.0, yaw=0.0)
14. Use exact object names: "pringles can" not "pringles_can"
15. Rotations are allowed and valid - use yaw parameter for orientation changes
16. MANDATORY: Use commas between ALL parameters: start_moving, x=value, y=value, yaw=value, frame=body
17. FORBIDDEN: Do NOT write "start_moving x=1.0 y=0.0 yaw=0.0 frame=body" (missing commas)
18. CORRECT FORMAT: "start_moving, x=1.0, y=0.0, yaw=0.0, frame=body"

PLANNING PROCESS:
1. Parse user command to identify destinations in order
2. Calculate relative body frame movements from current position to each destination
3. Plan path visiting each destination once in order
4. Ensure correct robot orientation at each destination (yaw=0 for pick-up, yaw=1.57 radians for drop-off)
5. For drop-off locations: add drop-off procedure (start_move_arm_pose, start_arm_command open, start_arm_command close, start_arm_command stow)
6. Generate action sequence with relative body frame movements

OUTPUT FORMAT:
Return only the action list, one action per line.

VALIDATION: Before writing each movement, check: does it move only ONE axis (x OR y OR yaw)?
CRITICAL: NEVER write "x=0.0, y=1.5, yaw=1.57" - this combines y and yaw which is FORBIDDEN.

Task: {task}

Actions:"""
        
        return LA_PROMPT, HA_PROMPT
    
    async def test_llm(self, provider: LLMProvider, prompt: str, **kwargs) -> Tuple[str, float]:
        """Test a single LLM with given prompt and parameters"""
        start_time = time.time()
        
        try:
            # Groq modellen
            if provider == LLMProvider.GROQ_DEEPSEEK_R1_DISTILL_LLAMA_70B:
                if not self.groq_client:
                    raise Exception("GROQ_API_KEY niet gezet maar een Groq model is geselecteerd")
                response = self.groq_client.chat.completions.create(
                    model=provider.value,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.0
                )
                result = (response.choices[0].message.content or "").strip()
            elif provider == LLMProvider.OPENAI_GPT_OSS_120B:
                if not self.groq_client:
                    raise Exception("GROQ_API_KEY niet gezet maar een Groq model is geselecteerd")
                response = self.groq_client.chat.completions.create(
                    model=provider.value,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.0
                )
                result = (response.choices[0].message.content or "").strip()
            elif provider == LLMProvider.OPENAI_GPT_OSS_20B:
                if not self.groq_client:
                    raise Exception("GROQ_API_KEY niet gezet maar een Groq model is geselecteerd")
                response = self.groq_client.chat.completions.create(
                    model=provider.value,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.0
                )
                result = (response.choices[0].message.content or "").strip()
            elif provider == LLMProvider.OPENAI_GPT_OSS_20B_LOCAL:
                if not self.local_model or not self.local_tokenizer:
                    raise Exception("Local model not loaded but local model is selected")
                
                # Format prompt as messages for chat template
                messages = [{"role": "user", "content": prompt}]
                
                # Apply chat template
                inputs = self.local_tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.local_model.device)
                
                # Generate response
                with torch.no_grad():
                    outputs = self.local_model.generate(
                        **inputs,
                        max_new_tokens=2000,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=self.local_tokenizer.eos_token_id
                    )
                
                # Decode only the new tokens
                result = self.local_tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[-1]:],
                    skip_special_tokens=True
                ).strip()
            elif provider == LLMProvider.GROQ_LLAMA_3_3_70B_VERSATILE:
                if not self.groq_client:
                    raise Exception("GROQ_API_KEY niet gezet maar een Groq model is geselecteerd")
                response = self.groq_client.chat.completions.create(
                    model=provider.value,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.0
                )
                result = (response.choices[0].message.content or "").strip()
            elif provider == LLMProvider.GROQ_LLAMA_4_SCOUT_17B:
                if not self.groq_client:
                    raise Exception("GROQ_API_KEY niet gezet maar een Groq model is geselecteerd")
                response = self.groq_client.chat.completions.create(
                    model=provider.value,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.0
                )
                result = (response.choices[0].message.content or "").strip()
            elif provider == LLMProvider.GROQ_QWEN3_32B:
                if not self.groq_client:
                    raise Exception("GROQ_API_KEY niet gezet maar een Groq model is geselecteerd")
                response = self.groq_client.chat.completions.create(
                    model=provider.value,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.0
                )
                result = (response.choices[0].message.content or "").strip()
            # Together AI modellen
            elif self._is_together_model(provider):
                if not self.together_client:
                    raise Exception("TOGETHER_API_KEY niet gezet maar een Together AI model is geselecteerd")
                response = self.together_client.chat.completions.create(
                    model=provider.value,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.0
                )
                result = (response.choices[0].message.content or "").strip()
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
                
            execution_time = time.time() - start_time
            return result, execution_time
            
        except Exception as e:
            execution_time = time.time() - start_time
            raise Exception(f"LLM {provider.value} failed: {str(e)}")
    
    def parse_actions(self, response: str) -> List[str]:
        """Parse actions from LLM response"""
        actions = []
        in_think_block = False
        
        for line in response.splitlines():
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Handle <think> blocks
            if line.startswith('<think>'):
                in_think_block = True
                continue
            elif line.startswith('</think>'):
                in_think_block = False
                continue
            elif in_think_block:
                continue
                
            # Skip comments
            if line.startswith('#') or line.startswith('//'):
                continue
                
            # Remove numbering if present
            if line[0].isdigit() and ('.' in line or ')' in line):
                line = line.split('.', 1)[1] if '.' in line else line.split(')', 1)[1]
                line = line.strip()
            
            # Only add lines that look like actions (contain action keywords)
            if any(keyword in line.lower() for keyword in ['start_moving', 'start_automated_grasp', 'start_move_arm_pose', 'start_arm_command', 'stand_up', 'sit_down']):
                actions.append(line)
        
        return actions
    
    def parse_actions(self, response: str) -> List[str]:
        """Parse LLM response into action list"""
        actions = []
        in_think_block = False
        
        for line in response.splitlines():
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Handle <think> blocks
            if line.startswith('<think>'):
                in_think_block = True
                continue
            elif line.startswith('</think>'):
                in_think_block = False
                continue
            elif in_think_block:
                continue
                
            # Skip comments
            if line.startswith('#') or line.startswith('//'):
                continue
                
            # Remove numbering if present
            if line[0].isdigit() and ('.' in line or ')' in line):
                line = line.split('.', 1)[1] if '.' in line else line.split(')', 1)[1]
                line = line.strip()
            
            # Only add lines that look like actions (contain action keywords)
            if any(keyword in line.lower() for keyword in ['start_moving', 'start_automated_grasp', 'start_move_arm_pose', 'start_arm_command', 'stand_up', 'sit_down']):
                actions.append(line)
        
        return actions
    
    def calculate_score(self, predicted_actions: List[str], ground_truth_options: List[List[str]]) -> float:
        """Calculate score based on ground truth comparison with multiple valid options"""
        if not ground_truth_options:
            return 0.0
        
        # Normalize actions for comparison (remove extra spaces, convert to lowercase, remove quotes)
        def normalize_action(action: str) -> str:
            # Remove quotes and normalize spacing
            normalized = action.replace('"', '').replace("'", '')
            normalized = ' '.join(normalized.lower().split())
            return normalized
        
        pred_normalized = [normalize_action(action) for action in predicted_actions]
        
        # Find the best match among all ground truth options
        best_score = 0.0
        
        for ground_truth in ground_truth_options:
            if not ground_truth:
                continue
                
            gt_normalized = [normalize_action(action) for action in ground_truth]
            
            # Calculate exact match score
            exact_matches = 0
            for i, pred_action in enumerate(pred_normalized):
                if i < len(gt_normalized) and pred_action == gt_normalized[i]:
                    exact_matches += 1
            
            # Calculate sequence similarity
            sequence_score = exact_matches / len(gt_normalized) if gt_normalized else 0.0
            
            # Calculate action presence score (how many ground truth actions are present)
            presence_score = 0
            for gt_action in gt_normalized:
                if gt_action in pred_normalized:
                    presence_score += 1
            presence_score = presence_score / len(gt_normalized) if gt_normalized else 0.0
            
            # Calculate length penalty (more lenient for extra actions)
            length_penalty = 1.0
            if len(pred_normalized) != len(gt_normalized):
                # Much more lenient for extra actions, moderate penalty for missing actions
                if len(pred_normalized) > len(gt_normalized):
                    # Only 5% penalty for extra actions (LLMs often add extra steps)
                    length_penalty = 0.95
                else:
                    # 20% penalty for missing actions
                    length_penalty = 0.8
            
            # Calculate score for this ground truth option
            # More weight on presence since LLMs may reorder actions
            option_score = (sequence_score * 0.4 + presence_score * 0.6) * length_penalty
            best_score = max(best_score, option_score)
        
        return min(1.0, max(0.0, best_score))
    
    async def run_single_test(self, run_number: int, provider: LLMProvider, scenario: TestScenario, mode: str) -> TestResult:
        """Run a single test with given parameters"""
        la_prompt, ha_prompt = self.get_prompts()
        
        if mode == 'LA':
            prompt = la_prompt.format(
                command=scenario.la_command,
                current_state="standing"
            )
            ground_truth_options = scenario.ground_truth_la
        else:  # HA mode
            prompt = ha_prompt.format(
                task=scenario.ha_task,
                current_state="standing",
                current_position="(0.0, 0.0, 0.0)"
            )
            ground_truth_options = scenario.ground_truth_ha
        
        try:
            response, execution_time = await self.test_llm(provider, prompt)
            parsed_actions = self.parse_actions(response)
            score = self.calculate_score(parsed_actions, ground_truth_options)
            
            return TestResult(
                run_number=run_number,
                llm_provider=provider,
                scenario=scenario,
                mode=mode,
                prompt_used=prompt,
                response=response,
                parsed_actions=parsed_actions,
                execution_time=execution_time,
                success=True,
                score=score
            )
            
        except Exception as e:
            return TestResult(
                run_number=run_number,
                llm_provider=provider,
                scenario=scenario,
                mode=mode,
                prompt_used=prompt,
                response="",
                parsed_actions=[],
                execution_time=0.0,
                success=False,
                score=0.0,
                error_message=str(e)
            )
    
    async def run_consistency_tests(self, num_runs: int, selected_scenarios: List[str], selected_providers: List[LLMProvider]) -> List[TestResult]:
        """Run consistency tests with multiple runs"""
        results = []
        modes = ['HA']  # Only test HA mode
        
        # Filter scenarios based on selection
        filtered_scenarios = [s for s in self.test_scenarios if s.difficulty.value in selected_scenarios]
        
        total_tests = num_runs * len(filtered_scenarios) * len(modes) * len(selected_providers)
        print(f"Running {num_runs} runs × {len(filtered_scenarios)} scenarios × HA mode × {len(selected_providers)} providers = {total_tests} total tests")
        print(f"Models: {', '.join([p.value for p in selected_providers])}")
        print(f"Scenarios: {', '.join(selected_scenarios)}")
        print(f"Mode: HA only")
        
        for provider in selected_providers:
            print(f"\n=== TESTING {provider.value} ===")
            for run in range(1, num_runs + 1):
                print(f"\n--- RUN {run}/{num_runs} ---")
                for scenario in filtered_scenarios:
                    print(f"  Scenario: {scenario.name} ({scenario.difficulty.value})")
                    print(f"    Mode: HA")
                    result = await self.run_single_test(run, provider, scenario, 'HA')
                    results.append(result)
                    
                    if result.success:
                        print(f"      ✓ Success ({result.execution_time:.2f}s) - Score: {result.score:.2f} - {len(result.parsed_actions)} actions")
                    else:
                        print(f"      ✗ Failed: {result.error_message}")
        
        return results
    
    def save_results(self, results: List[TestResult], filename: str = "deepseek_consistency_results.json"):
        """Save test results to JSON file"""
        data = []
        for result in results:
            # Create final plan summary
            final_plan = {
                "total_actions": len(result.parsed_actions),
                "action_sequence": result.parsed_actions,
                "movement_actions": [action for action in result.parsed_actions if 'start_moving' in action],
                "grasp_actions": [action for action in result.parsed_actions if 'start_automated_grasp' in action],
                "arm_actions": [action for action in result.parsed_actions if 'start_move_arm_pose' in action or 'start_arm_command' in action]
            }
            
            data.append({
                "run_number": result.run_number,
                "llm_provider": result.llm_provider.value,
                "scenario_name": result.scenario.name,
                "scenario_difficulty": result.scenario.difficulty.value,
                "mode": result.mode,
                "success": result.success,
                "score": result.score,
                "execution_time": result.execution_time,
                "parsed_actions": result.parsed_actions,
                "final_plan": final_plan,
                "ground_truth_options": result.scenario.ground_truth_la if result.mode == 'LA' else result.scenario.ground_truth_ha,
                "response": result.response,
                "error_message": result.error_message
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved to {filename}")
    
    def print_consistency_analysis(self, results: List[TestResult]):
        """Print detailed consistency analysis"""
        print("\n" + "="*80)
        print("DEEPSEEK CONSISTENCY ANALYSIS")
        print("="*80)
        
        # Group by scenario and mode
        by_scenario_mode = {}
        for result in results:
            key = (result.scenario.name, result.mode)
            if key not in by_scenario_mode:
                by_scenario_mode[key] = []
            by_scenario_mode[key].append(result)
        
        for (scenario_name, mode), scenario_results in by_scenario_mode.items():
            print(f"\n{scenario_name} - HA Mode:")
            
            # Basic stats
            successful = [r for r in scenario_results if r.success]
            failed = [r for r in scenario_results if not r.success]
            
            if successful:
                scores = [r.score for r in successful]
                times = [r.execution_time for r in successful]
                action_counts = [len(r.parsed_actions) for r in successful]
                
                print(f"  Success Rate: {len(successful)}/{len(scenario_results)} ({len(successful)/len(scenario_results)*100:.1f}%)")
                print(f"  Score Stats: {min(scores):.3f} - {max(scores):.3f} (avg: {sum(scores)/len(scores):.3f})")
                print(f"  Time Stats: {min(times):.2f}s - {max(times):.2f}s (avg: {sum(times)/len(times):.2f}s)")
                print(f"  Action Count: {min(action_counts)} - {max(action_counts)} (avg: {sum(action_counts)/len(action_counts):.1f})")
                
                # Consistency analysis
                score_variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
                print(f"  Score Variance: {score_variance:.6f}")
                
                # Check for identical responses
                responses = [r.response for r in successful]
                unique_responses = len(set(responses))
                print(f"  Unique Responses: {unique_responses}/{len(responses)} ({unique_responses/len(responses)*100:.1f}%)")
                
                # Show score distribution
                high_scores = [s for s in scores if s >= 0.8]
                medium_scores = [s for s in scores if 0.5 <= s < 0.8]
                low_scores = [s for s in scores if s < 0.5]
                
                print(f"  Score Distribution:")
                print(f"    High (≥0.8): {len(high_scores)}/{len(scores)} ({len(high_scores)/len(scores)*100:.1f}%)")
                print(f"    Medium (0.5-0.8): {len(medium_scores)}/{len(scores)} ({len(medium_scores)/len(scores)*100:.1f}%)")
                print(f"    Low (<0.5): {len(low_scores)}/{len(scores)} ({len(low_scores)/len(scores)*100:.1f}%)")
                
                # Show examples of different responses
                if unique_responses > 1:
                    print(f"  Response Variations:")
                    for i, (run, result) in enumerate([(r.run_number, r) for r in successful]):
                        if i < 3:  # Show first 3 examples
                            print(f"    Run {run}: Score {result.score:.3f}, {len(result.parsed_actions)} actions")
                            for j, action in enumerate(result.parsed_actions[:3]):
                                print(f"      {j+1}. {action[:60]}...")
                            if len(result.parsed_actions) > 3:
                                print(f"      ... and {len(result.parsed_actions)-3} more")
                        elif i == 3:
                            print(f"    ... and {unique_responses-3} more variations")
                            break
            
            if failed:
                print(f"  Failures: {len(failed)}")
                error_types = {}
                for result in failed:
                    if result.error_message:
                        error_key = result.error_message[:50] + "..." if len(result.error_message) > 50 else result.error_message
                        error_types[error_key] = error_types.get(error_key, 0) + 1
                
                for error, count in error_types.items():
                    print(f"    - {error}: {count} times")
        
        # Overall consistency summary
        print(f"\n" + "="*60)
        print("OVERALL CONSISTENCY SUMMARY")
        print("="*60)
        
        all_successful = [r for r in results if r.success]
        if all_successful:
            all_scores = [r.score for r in all_successful]
            all_times = [r.execution_time for r in all_successful]
            
            print(f"Total Successful Tests: {len(all_successful)}/{len(results)}")
            print(f"Overall Score Range: {min(all_scores):.3f} - {max(all_scores):.3f}")
            print(f"Overall Score Average: {sum(all_scores)/len(all_scores):.3f}")
            print(f"Overall Score Std Dev: {(sum((s - sum(all_scores)/len(all_scores))**2 for s in all_scores) / len(all_scores))**0.5:.3f}")
            print(f"Overall Time Range: {min(all_times):.2f}s - {max(all_times):.2f}s")
            print(f"Overall Time Average: {sum(all_times)/len(all_times):.2f}s")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test Deepseek model consistency')
    
    parser.add_argument(
        '--runs',
        type=int,
        default=5,
        help='Number of runs per scenario (default: 5)'
    )
    
    parser.add_argument(
        '--scenarios',
        nargs='+',
        default=['hard'],
        choices=['easy', 'medium', 'hard'],
        help='Scenarios to test (default: hard)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        default=['deepseek-r1-distill-llama-70b'],
        choices=[
            # Groq models
            'deepseek-r1-distill-llama-70b', 'openai/gpt-oss-120b', 'openai/gpt-oss-20b', 'openai/gpt-oss-20b-local', 'llama-3.3-70b-versatile', 
            'meta-llama/llama-4-scout-17b-16e-instruct', 'qwen/qwen3-32b',
            # Together AI models (serverless)
            'meta-llama/Meta-Llama-3.3-70B-Instruct-Turbo', 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
            'meta-llama/Llama-3-70b-chat-hf', 'deepseek-ai/DeepSeek-R1', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free',
            'Qwen/Qwen2.5-72B-Instruct-Turbo', 'Qwen/Qwen3-Next-80B-A3B-Instruct', 'Qwen/Qwen3-235B-A22B-Instruct-2507-tput',
            'mistralai/Mixtral-8x22B-Instruct-v0.1'
        ],
        help='Models to test (default: deepseek-r1-distill-llama-70b)'
    )
    
    parser.add_argument(
        '--output',
        default='llm_consistency_results.json',
        help='Output file for results (default: llm_consistency_results.json)'
    )
    
    return parser.parse_args()

async def main():
    """Main function to run consistency tests"""
    args = parse_arguments()
    
    print("Starting LLM Consistency Testing")
    print("="*50)
    print(f"Number of runs: {args.runs}")
    print(f"Testing models: {', '.join(args.models)}")
    print(f"Testing scenarios: {', '.join(args.scenarios)}")
    print(f"Output file: {args.output}")
    print()
    
    # Check environment variables
    if 'deepseek-r1-distill-llama-70b' in args.models and not os.getenv('GROQ_API_KEY'):
        print("ERROR: GROQ_API_KEY environment variable not set for Deepseek model")
        return
    
    if 'openai/gpt-oss-120b' in args.models and not os.getenv('GROQ_API_KEY'):
        print("ERROR: GROQ_API_KEY environment variable not set for GPT-OSS-120B model")
        return
    
    if 'openai/gpt-oss-20b' in args.models and not os.getenv('GROQ_API_KEY'):
        print("ERROR: GROQ_API_KEY environment variable not set for GPT-OSS-20B model")
        return
    
    if 'openai/gpt-oss-20b-local' in args.models and not TRANSFORMERS_AVAILABLE:
        print("ERROR: transformers library not available for local GPT-OSS-20B model")
        return
    
    if 'llama-3.3-70b-versatile' in args.models and not os.getenv('GROQ_API_KEY'):
        print("ERROR: GROQ_API_KEY environment variable not set for Llama-3.3-70B-Versatile model")
        return
    if 'meta-llama/llama-4-scout-17b-16e-instruct' in args.models and not os.getenv('GROQ_API_KEY'):
        print("ERROR: GROQ_API_KEY environment variable not set for Llama-4-Scout-17B model")
        return
    if 'qwen/qwen3-32b' in args.models and not os.getenv('GROQ_API_KEY'):
        print("ERROR: GROQ_API_KEY environment variable not set for Qwen3-32B model")
        return
    
    # Check Together AI API key for Together models
    together_models = [
        'meta-llama/Meta-Llama-3.3-70B-Instruct-Turbo', 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
        'meta-llama/Llama-3-70b-chat-hf', 'deepseek-ai/DeepSeek-R1', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B', 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free',
        'Qwen/Qwen2.5-72B-Instruct-Turbo', 'Qwen/Qwen3-Next-80B-A3B-Instruct', 'Qwen/Qwen3-235B-A22B-Instruct-2507-tput',
        'mistralai/Mixtral-8x22B-Instruct-v0.1'
    ]
    if any(model in args.models for model in together_models) and not os.getenv('TOGETHER_API_KEY'):
        print("ERROR: TOGETHER_API_KEY environment variable not set for Together AI models")
        return
    
    # Convert model strings to LLMProvider enums
    selected_providers = []
    for model_str in args.models:
        try:
            provider = LLMProvider(model_str)
            selected_providers.append(provider)
        except ValueError:
            print(f"WARNING: Unknown model '{model_str}', skipping...")
    
    if not selected_providers:
        print("ERROR: No valid models selected")
        return
    
    # Create tester and run tests
    tester = DeepseekConsistencyTester()
    results = await tester.run_consistency_tests(args.runs, args.scenarios, selected_providers)
    
    # Save and display results
    tester.save_results(results, args.output)
    tester.print_consistency_analysis(results)
    
    print("\nConsistency testing completed!")

if __name__ == "__main__":
    asyncio.run(main())
