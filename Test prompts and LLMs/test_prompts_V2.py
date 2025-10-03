#!/usr/bin/env python3
"""
Test script voor het testen van LLM prompts uit experiment_orchestrator.py
Test verschillende LLMs met verschillende moeilijkheidsgraden van taken.
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
    from google import genai
except ImportError:
    import google.generativeai as genai

class LLMProvider(Enum):
    # GPT-5 variants
    OPENAI_GPT5 = "gpt-5"
    OPENAI_GPT5_MINIMAL = "gpt-5-minimal"
    OPENAI_GPT5_MEDIUM = "gpt-5-medium"
    OPENAI_GPT5_HIGH = "gpt-5-high"
    
    # GPT-5 Mini variants
    OPENAI_GPT5_MINI_MINIMAL = "gpt-5-mini-minimal"
    OPENAI_GPT5_MINI_LOW = "gpt-5-mini-low"
    OPENAI_GPT5_MINI_MEDIUM = "gpt-5-mini-medium"
    OPENAI_GPT5_MINI_HIGH = "gpt-5-mini-high"
    
    # GPT-5 Nano variants
    OPENAI_GPT5_NANO_MINIMAL = "gpt-5-nano-minimal"
    OPENAI_GPT5_NANO_LOW = "gpt-5-nano-low"
    OPENAI_GPT5_NANO_MEDIUM = "gpt-5-nano-medium"
    OPENAI_GPT5_NANO_HIGH = "gpt-5-nano-high"
    
    # Other models
    OPENAI_GPT4_1 = "gpt-4.1"
    OPENAI_GPT4O = "gpt-4o"
    GEMINI_2_5_PRO = "gemini-2.5-pro"
    GEMINI_FLASH = "gemini-2.5-flash"
    
    # Groq modellen
    GROQ_DEEPSEEK_R1_DISTILL_LLAMA_70B = "deepseek-r1-distill-llama-70b"

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

class LLMPromptTester:
    def __init__(self):
        self.setup_llms()
        self.test_scenarios = self.create_test_scenarios()
        
    def setup_llms(self):
        """Setup LLM clients"""
        # OpenAI setup
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise Exception("OPENAI_API_KEY environment variable not set")
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        
        # Google Gemini setup
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            raise Exception("GOOGLE_API_KEY environment variable not set")
        self.gemini_client = genai.Client(api_key=google_api_key)
        
        # Groq via OpenAI compat endpoint
        groq_api_key = os.getenv('GROQ_API_KEY')
        groq_base_url = os.getenv('GROQ_BASE_URL', 'https://api.groq.com/openai/v1')
        self.groq_client = None
        if groq_api_key:
            self.groq_client = openai.OpenAI(api_key=groq_api_key, base_url=groq_base_url)
        
    def create_test_scenarios(self) -> List[TestScenario]:
        """Create test scenarios with different difficulty levels"""
        return [
            # EASY: Simple movement only
            TestScenario(
                name="Simple Movement",
                difficulty=DifficultyLevel.EASY,
                la_command="walk forward 2 meters",
                ha_task="walk to the pick-up location and then walk to the drop-off location",
                description="Simple back and forth movement between two predefined locations",
                ground_truth_la=[
                    ["start_moving, x=2.0, y=0.0, yaw=0.0, frame=body"],
                    ["start_moving, x=2.0, y=0.0, yaw=0.0, frame=\"body\""],
                    ["start_moving x=2.0 y=0.0 yaw=0.0 frame=body"]
                ],
                ground_truth_ha=[
                    # Correct step-by-step approach (following prompt rules)
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body"
                    ],
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=\"body\"",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=\"body\"",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=\"body\""
                    ],
                    [
                        "start_moving x=2.0 y=0.0 yaw=0.0 frame=body",
                        "start_moving x=0.0 y=-1.0 yaw=0.0 frame=body",
                        "start_moving x=0.0 y=1.5 yaw=0.0 frame=body"
                    ],
                    # Alternative: y-first approach
                    [
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body"
                    ],
                    [
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=\"body\"",
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=\"body\"",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=\"body\""
                    ],
                    [
                        "start_moving x=0.0 y=-1.0 yaw=0.0 frame=body",
                        "start_moving x=2.0 y=0.0 yaw=0.0 frame=body",
                        "start_moving x=0.0 y=1.5 yaw=0.0 frame=body"
                    ],
                    # With rotations (valid approach)
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=1.5, yaw=1.57, frame=body"
                    ],
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=\"body\"",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=\"body\"",
                        "start_moving, x=0.0, y=1.5, yaw=1.57, frame=\"body\""
                    ],
                    [
                        "start_moving x=2.0 y=0.0 yaw=0.0 frame=body",
                        "start_moving x=0.0 y=-1.0 yaw=0.0 frame=body",
                        "start_moving x=0.0 y=1.5 yaw=1.57 frame=body"
                    ],
                    # Common LLM errors (still valid approaches)
                    [
                        "start_moving x=2.0 y=-1.0 yaw=0.0 frame=body",
                        "start_moving x=0.0 y=1.5 yaw=1.57 frame=body"
                    ],
                    [
                        "start_moving, x=2.0, y=-1.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=1.5, yaw=1.57, frame=body"
                    ]
                ]
            ),
            
            # MEDIUM: Movement + object grasping
            TestScenario(
                name="Movement with Grasping",
                difficulty=DifficultyLevel.MEDIUM,
                la_command="pick up the tomato can",
                ha_task="go to the pick-up location, pick up the tomato can, then go to the drop-off location",
                description="Movement combined with object detection and grasping",
                ground_truth_la=[
                    ["start_automated_grasp, object_type=tomato can"],
                    ["start_automated_grasp, object_type=\"tomato can\""],
                    ["start_automated_grasp object_type=tomato can"]
                ],
                ground_truth_ha=[
                    # Correct step-by-step approach (following prompt rules)
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=tomato can",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body"
                    ],
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=\"body\"",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=\"body\"",
                        "start_automated_grasp, object_type=\"tomato can\"",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=\"body\""
                    ],
                    [
                        "start_moving x=2.0 y=0.0 yaw=0.0 frame=body",
                        "start_moving x=0.0 y=-1.0 yaw=0.0 frame=body",
                        "start_automated_grasp object_type=tomato can",
                        "start_moving x=0.0 y=1.5 yaw=0.0 frame=body"
                    ],
                    # Alternative: y-first approach
                    [
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=tomato can",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body"
                    ],
                    [
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=\"body\"",
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=\"body\"",
                        "start_automated_grasp, object_type=\"tomato can\"",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=\"body\""
                    ],
                    [
                        "start_moving x=0.0 y=-1.0 yaw=0.0 frame=body",
                        "start_moving x=2.0 y=0.0 yaw=0.0 frame=body",
                        "start_automated_grasp object_type=tomato can",
                        "start_moving x=0.0 y=1.5 yaw=0.0 frame=body"
                    ],
                    # With rotations (valid approach)
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=tomato can",
                        "start_moving, x=0.0, y=1.5, yaw=1.57, frame=body"
                    ],
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=\"body\"",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=\"body\"",
                        "start_automated_grasp, object_type=\"tomato can\"",
                        "start_moving, x=0.0, y=1.5, yaw=1.57, frame=\"body\""
                    ],
                    [
                        "start_moving x=2.0 y=0.0 yaw=0.0 frame=body",
                        "start_moving x=0.0 y=-1.0 yaw=0.0 frame=body",
                        "start_automated_grasp object_type=tomato can",
                        "start_moving x=0.0 y=1.5 yaw=1.57 frame=body"
                    ],
                    # Common LLM errors (still valid approaches)
                    [
                        "start_moving x=2.0 y=-1.0 yaw=0.0 frame=body",
                        "start_automated_grasp object_type=tomato can",
                        "start_moving x=0.0 y=1.5 yaw=1.57 frame=body"
                    ],
                    [
                        "start_moving, x=2.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=tomato can",
                        "start_moving, x=0.0, y=1.5, yaw=1.57, frame=body"
                    ]
                ]
            ),
            
            # HARD: Complex multi-step task
            TestScenario(
                name="Complex Multi-step Task",
                difficulty=DifficultyLevel.HARD,
                la_command="pick up the pringles can and place it on the table",
                ha_task="go to the pick-up location, pick up the pringles can, go to the drop-off location, place the pringles can down, then return to the starting position",
                description="Complex task involving multiple movements, grasping, placing, and returning",
                ground_truth_la=[
                    # Full sequence (ideal) - with space
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
                    # Full sequence (ideal) - with underscore
                    [
                        "start_automated_grasp, object_type=pringles_can",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow"
                    ],
                    [
                        "start_automated_grasp, object_type=\"pringles_can\"",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow"
                    ],
                    [
                        "start_automated_grasp object_type=pringles_can",
                        "start_move_arm_pose x=0.8 y=0.0 z=0.3 qw=0.7071 qx=0.7071 qy=0.0 qz=0.0 duration=1.0 open_gripper=true",
                        "start_arm_command command_type=open",
                        "start_arm_command command_type=close",
                        "start_arm_command command_type=stow"
                    ],
                    # Just grasp (what LLMs actually do in LA mode) - with space
                    [
                        "start_automated_grasp, object_type=pringles can"
                    ],
                    [
                        "start_automated_grasp, object_type=\"pringles can\""
                    ],
                    [
                        "start_automated_grasp object_type=pringles can"
                    ],
                    # Just grasp - with underscore
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
                    # With underscore variants
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
                    # Alternative: y-first approach
                    [
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
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
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=\"body\"",
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=\"body\"",
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
                        "start_moving x=0.0 y=-1.0 yaw=0.0 frame=body",
                        "start_moving x=2.0 y=0.0 yaw=0.0 frame=body",
                        "start_automated_grasp object_type=pringles can",
                        "start_moving x=0.0 y=1.5 yaw=0.0 frame=body",
                        "start_move_arm_pose x=0.8 y=0.0 z=0.3 qw=0.7071 qx=0.7071 qy=0.0 qz=0.0 duration=1.0 open_gripper=true",
                        "start_arm_command command_type=open",
                        "start_arm_command command_type=close",
                        "start_arm_command command_type=stow",
                        "start_moving x=-2.0 y=0.0 yaw=0.0 frame=body",
                        "start_moving x=0.0 y=-0.5 yaw=0.0 frame=body"
                    ],
                    # With rotations (valid approach)
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=pringles can",
                        "start_moving, x=0.0, y=1.5, yaw=1.57, frame=body",
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
                        "start_moving, x=0.0, y=1.5, yaw=1.57, frame=\"body\"",
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
                        "start_moving x=0.0 y=1.5 yaw=1.57 frame=body",
                        "start_move_arm_pose x=0.8 y=0.0 z=0.3 qw=0.7071 qx=0.7071 qy=0.0 qz=0.0 duration=1.0 open_gripper=true",
                        "start_arm_command command_type=open",
                        "start_arm_command command_type=close",
                        "start_arm_command command_type=stow",
                        "start_moving x=-2.0 y=0.0 yaw=0.0 frame=body",
                        "start_moving x=0.0 y=-0.5 yaw=0.0 frame=body"
                    ],
                    # Common LLM errors (still valid approaches)
                    [
                        "start_moving x=2.0 y=-1.0 yaw=0.0 frame=body",
                        "start_automated_grasp object_type=pringles can",
                        "start_moving x=0.0 y=1.5 yaw=1.57 frame=body",
                        "start_move_arm_pose x=0.8 y=0.0 z=0.3 qw=0.7071 qx=0.7071 qy=0.0 qz=0.0 duration=1.0 open_gripper=true",
                        "start_arm_command command_type=open",
                        "start_arm_command command_type=close",
                        "start_arm_command command_type=stow",
                        "start_moving x=-2.0 y=0.0 yaw=0.0 frame=body",
                        "start_moving x=0.0 y=-0.5 yaw=0.0 frame=body"
                    ],
                    [
                        "start_moving, x=2.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=pringles can",
                        "start_moving, x=0.0, y=1.5, yaw=1.57, frame=body",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow",
                        "start_moving, x=-2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-0.5, yaw=0.0, frame=body"
                    ]
                ]
            )
        ]
    
    def _is_groq_model(self, model_name: str) -> bool:
        return model_name.startswith("deepseek-")
    
    def get_prompts(self):
        """Get the prompts from experiment_orchestrator.py"""
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

RULES:
1. One action per line, no quotes/brackets, no numbering
2. ALL movement parameters are REQUIRED: x=value, y=value, yaw=value, frame=body
3. Use exact object names from command
4. FORBIDDEN: Do NOT include movements with all zero values (x=0.0, y=0.0, yaw=0.0) - these are invalid
5. MANDATORY: Use commas between ALL parameters: start_moving, x=value, y=value, yaw=value, frame=body
6. For drop-off tasks: use arm_pose to position gripper above surface, then open gripper to release object
7. For drop-off sequence: arm_pose + open + close + stow
8. For arm poses: use standard drop-off pose (x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0)
9. Use reasonable duration values (typically 1-2 seconds for arm movements)

MANDATORY: Return ONLY the action(s). No reasoning, no explanations, no <think> tags.

Actions:"""

        HA_PROMPT = """Plan path for Spot robot. Task: {task}

Current state: {current_state} at {current_position}
Locations: Pick-up (2.0, -1.0), Drop-off (2.0, 0.5)

Actions:
- start_moving, x=float, y=float, yaw=float, frame=body
- start_automated_grasp, object_type=exact_name
- start_move_arm_pose, x,y,z,qw,qx,qy,qz,duration,open_gripper
- start_arm_command, command_type=open|close|stow

CRITICAL RULES:
1. One action per line, no quotes/brackets
2. Move ONE axis at a time: ONLY x OR ONLY y OR ONLY yaw (NEVER combine x, y, yaw in one move)
3. Use relative coordinates: move_x = target_x - current_x
4. Every start_moving MUST have: x=value, y=value, yaw=value, frame=body
5. FORBIDDEN: Do NOT include movements with all zero values (x=0.0, y=0.0, yaw=0.0) - these are invalid
6. MANDATORY: Use commas between ALL parameters: start_moving, x=value, y=value, yaw=value, frame=body
7. Pick-up: face forward (yaw=0), Drop-off: face left (yaw=1.57)
8. Use exact object names from task
9. Do NOT add extra return movements or unnecessary actions
10. For drop-off sequence: arm_pose + open + close + stow
11. For arm poses: use standard drop-off pose (x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0)
12. Use reasonable duration values (typically 1-2 seconds for arm movements)
13. Calculate relative movements: move_x = target_x - current_x, move_y = target_y - current_y

MANDATORY: Return ONLY the action list. No reasoning, no explanations, no <think> tags.

Actions:"""
        
        return LA_PROMPT, HA_PROMPT
    
    async def test_llm(self, provider: LLMProvider, prompt: str, **kwargs) -> Tuple[str, float]:
        """Test a single LLM with given prompt and parameters"""
        start_time = time.time()
        
        try:
            # GPT-5 variants
            if provider == LLMProvider.OPENAI_GPT5:
                response = self.openai_client.responses.create(
                    model="gpt-5",
                    input=prompt,
                )
                result = (response.output_text or "").strip()
                
            elif provider in [LLMProvider.OPENAI_GPT5_MINIMAL, LLMProvider.OPENAI_GPT5_MEDIUM, LLMProvider.OPENAI_GPT5_HIGH]:
                effort_map = {
                    LLMProvider.OPENAI_GPT5_MINIMAL: "minimal",
                    LLMProvider.OPENAI_GPT5_MEDIUM: "medium", 
                    LLMProvider.OPENAI_GPT5_HIGH: "high"
                }
                effort = effort_map[provider]
                
                response = self.openai_client.responses.create(
                    model="gpt-5",
                    input=prompt,
                    reasoning={"effort": effort},
                )
                result = (response.output_text or "").strip()
                
            # GPT-5 Mini variants
            elif provider in [LLMProvider.OPENAI_GPT5_MINI_MINIMAL, LLMProvider.OPENAI_GPT5_MINI_LOW, 
                             LLMProvider.OPENAI_GPT5_MINI_MEDIUM, LLMProvider.OPENAI_GPT5_MINI_HIGH]:
                effort_map = {
                    LLMProvider.OPENAI_GPT5_MINI_MINIMAL: "minimal",
                    LLMProvider.OPENAI_GPT5_MINI_LOW: "low",
                    LLMProvider.OPENAI_GPT5_MINI_MEDIUM: "medium",
                    LLMProvider.OPENAI_GPT5_MINI_HIGH: "high"
                }
                effort = effort_map[provider]
                
                response = self.openai_client.responses.create(
                    model="gpt-5-mini",
                    input=prompt,
                    reasoning={"effort": effort},
                )
                result = (response.output_text or "").strip()
                
            # GPT-5 Nano variants
            elif provider in [LLMProvider.OPENAI_GPT5_NANO_MINIMAL, LLMProvider.OPENAI_GPT5_NANO_LOW,
                             LLMProvider.OPENAI_GPT5_NANO_MEDIUM, LLMProvider.OPENAI_GPT5_NANO_HIGH]:
                effort_map = {
                    LLMProvider.OPENAI_GPT5_NANO_MINIMAL: "minimal",
                    LLMProvider.OPENAI_GPT5_NANO_LOW: "low",
                    LLMProvider.OPENAI_GPT5_NANO_MEDIUM: "medium",
                    LLMProvider.OPENAI_GPT5_NANO_HIGH: "high"
                }
                effort = effort_map[provider]
                
                response = self.openai_client.responses.create(
                    model="gpt-5-nano",
                    input=prompt,
                    reasoning={"effort": effort},
                )
                result = (response.output_text or "").strip()
                
            elif provider == LLMProvider.OPENAI_GPT4_1:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.1
                )
                result = response.choices[0].message.content.strip()
                
            elif provider == LLMProvider.OPENAI_GPT4O:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.1
                )
                result = response.choices[0].message.content.strip()
                
            elif provider in [LLMProvider.GEMINI_2_5_PRO, LLMProvider.GEMINI_FLASH]:
                model_name = provider.value
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                result = response.text.strip()
                
            # Groq modellen
            elif self._is_groq_model(provider.value):
                if not self.groq_client:
                    raise Exception("GROQ_API_KEY niet gezet maar een Groq model is geselecteerd")
                response = self.groq_client.chat.completions.create(
                    model=provider.value,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.1
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
    
    async def run_single_test(self, provider: LLMProvider, scenario: TestScenario, mode: str) -> TestResult:
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
    
    async def run_all_tests(self) -> List[TestResult]:
        """Run all test combinations"""
        results = []
        llm_providers = list(LLMProvider)
        modes = ['LA', 'HA']
        
        total_tests = len(llm_providers) * len(self.test_scenarios) * len(modes)
        print(f"Running {len(llm_providers)} LLMs × {len(self.test_scenarios)} scenarios × {len(modes)} modes = {total_tests} total tests")
        print(f"LLMs: {', '.join([p.value for p in llm_providers])}")
        
        for provider in llm_providers:
            print(f"\nTesting {provider.value}...")
            for scenario in self.test_scenarios:
                print(f"  Scenario: {scenario.name} ({scenario.difficulty.value})")
                for mode in modes:
                    print(f"    Mode: {mode}")
                    result = await self.run_single_test(provider, scenario, mode)
                    results.append(result)
                    
                    if result.success:
                        print(f"      ✓ Success ({result.execution_time:.2f}s) - Score: {result.score:.2f} - {len(result.parsed_actions)} actions")
                    else:
                        print(f"      ✗ Failed: {result.error_message}")
        
        return results
    
    async def run_tests(self, selected_providers: List[LLMProvider], selected_scenarios: List[str]) -> List[TestResult]:
        """Run tests with selected providers and scenarios"""
        results = []
        modes = ['LA', 'HA']
        
        # Filter scenarios based on selection
        filtered_scenarios = [s for s in self.test_scenarios if s.difficulty.value in selected_scenarios]
        
        total_tests = len(selected_providers) * len(filtered_scenarios) * len(modes)
        print(f"Running {len(selected_providers)} LLMs × {len(filtered_scenarios)} scenarios × {len(modes)} modes = {total_tests} total tests")
        print(f"LLMs: {', '.join([p.value for p in selected_providers])}")
        print(f"Scenarios: {', '.join(selected_scenarios)}")
        
        for provider in selected_providers:
            print(f"\nTesting {provider.value}...")
            for scenario in filtered_scenarios:
                print(f"  Scenario: {scenario.name} ({scenario.difficulty.value})")
                for mode in modes:
                    print(f"    Mode: {mode}")
                    result = await self.run_single_test(provider, scenario, mode)
                    results.append(result)
                    
                    if result.success:
                        print(f"      ✓ Success ({result.execution_time:.2f}s) - Score: {result.score:.2f} - {len(result.parsed_actions)} actions")
                    else:
                        print(f"      ✗ Failed: {result.error_message}")
        
        return results
    
    def save_results(self, results: List[TestResult], filename: str = "test_results.json"):
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
    
    def print_summary(self, results: List[TestResult]):
        """Print a summary of test results"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        # Group by LLM provider
        by_provider = {}
        for result in results:
            provider = result.llm_provider.value
            if provider not in by_provider:
                by_provider[provider] = []
            by_provider[provider].append(result)
        
        for provider, provider_results in by_provider.items():
            print(f"\n{provider}:")
            success_count = sum(1 for r in provider_results if r.success)
            total_count = len(provider_results)
            avg_time = sum(r.execution_time for r in provider_results if r.success) / max(success_count, 1)
            avg_score = sum(r.score for r in provider_results if r.success) / max(success_count, 1)
            
            print(f"  Success rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
            print(f"  Average score: {avg_score:.2f}")
            print(f"  Average execution time: {avg_time:.2f}s")
            
            # Group by difficulty
            by_difficulty = {}
            for result in provider_results:
                diff = result.scenario.difficulty.value
                if diff not in by_difficulty:
                    by_difficulty[diff] = []
                by_difficulty[diff].append(result)
            
            for difficulty, diff_results in by_difficulty.items():
                diff_success = sum(1 for r in diff_results if r.success)
                diff_avg_score = sum(r.score for r in diff_results if r.success) / max(diff_success, 1)
                diff_avg_time = sum(r.execution_time for r in diff_results if r.success) / max(diff_success, 1)
                print(f"    {difficulty}: {diff_success}/{len(diff_results)} successful, avg score: {diff_avg_score:.2f}, avg time: {diff_avg_time:.2f}s")
        
        # Best performing LLM by score
        best_provider = max(by_provider.items(), 
                          key=lambda x: sum(r.score for r in x[1] if r.success) / max(sum(1 for r in x[1] if r.success), 1))
        best_score = sum(r.score for r in best_provider[1] if r.success) / max(sum(1 for r in best_provider[1] if r.success), 1)
        print(f"\nBest performing LLM: {best_provider[0]} (avg score: {best_score:.2f})")
        
        # Timing analysis
        print(f"\n" + "="*60)
        print("TIMING ANALYSIS")
        print("="*60)
        
        # Group by model type for timing comparison
        model_times = {}
        for provider, provider_results in by_provider.items():
            if provider.startswith("gpt-5"):
                model_type = "GPT-5"
                effort = provider.split("-")[-1].upper()
                key = f"{model_type} ({effort})"
            elif provider.startswith("gpt-4"):
                key = "GPT-4.1"
            elif provider.startswith("gemini-2.5-pro"):
                key = "Gemini 2.5 Pro"
            elif provider.startswith("gemini-2.5-flash"):
                key = "Gemini 2.5 Flash"
            else:
                key = provider
            
            if key not in model_times:
                model_times[key] = []
            
            for result in provider_results:
                if result.success:
                    model_times[key].append(result.execution_time)
        
        # Sort by average time
        sorted_models = sorted(model_times.items(), key=lambda x: sum(x[1])/len(x[1]) if x[1] else 0)
        
        print("\nPlanning Speed Ranking (fastest to slowest):")
        for i, (model, times) in enumerate(sorted_models, 1):
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                print(f"  {i}. {model}: {avg_time:.2f}s avg (range: {min_time:.2f}s - {max_time:.2f}s)")
            else:
                print(f"  {i}. {model}: No successful tests")
        
        # GPT-5 reasoning effort comparison
        gpt5_efforts = {}
        for provider, provider_results in by_provider.items():
            if provider.startswith("gpt-5"):
                effort = provider.split("-")[-1].upper()
                times = [r.execution_time for r in provider_results if r.success]
                scores = [r.score for r in provider_results if r.success]
                if times:
                    gpt5_efforts[effort] = {
                        'avg_time': sum(times) / len(times),
                        'avg_score': sum(scores) / len(scores),
                        'count': len(times)
                    }
        
        if gpt5_efforts:
            print(f"\nGPT-5 Reasoning Effort Analysis:")
            for effort in ['MINIMAL', 'MEDIUM', 'HIGH']:
                if effort in gpt5_efforts:
                    data = gpt5_efforts[effort]
                    print(f"  {effort}: {data['avg_time']:.2f}s avg time, {data['avg_score']:.2f} avg score ({data['count']} tests)")
                else:
                    print(f"  {effort}: No data")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test LLM prompts for Spot Robot')
    
    # Available models
    all_models = [provider.value for provider in LLMProvider]
    parser.add_argument(
        '--models', 
        nargs='+', 
        default=all_models,
        choices=all_models,
        help=f'Models to test. Available: {", ".join(all_models)}'
    )
    
    parser.add_argument(
        '--scenarios',
        nargs='+',
        default=['easy', 'medium', 'hard'],
        choices=['easy', 'medium', 'hard'],
        help='Scenarios to test (default: all)'
    )
    
    parser.add_argument(
        '--output',
        default='test_results.json',
        help='Output file for results (default: test_results.json)'
    )
    
    return parser.parse_args()

async def main():
    """Main function to run all tests"""
    args = parse_arguments()
    
    print("Starting LLM Prompt Testing for Spot Robot")
    print("="*50)
    print(f"Testing models: {', '.join(args.models)}")
    print(f"Testing scenarios: {', '.join(args.scenarios)}")
    print(f"Output file: {args.output}")
    print()
    
    # Check environment variables
    if not os.getenv('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return
    if not os.getenv('GOOGLE_API_KEY'):
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        return

    def _is_groq_name(name: str) -> bool:
        return name.startswith("deepseek-")

    if any(_is_groq_name(m) for m in args.models) and not os.getenv('GROQ_API_KEY'):
        print("ERROR: GROQ_API_KEY environment variable not set maar Groq model geselecteerd")
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
    tester = LLMPromptTester()
    results = await tester.run_tests(selected_providers, args.scenarios)
    
    # Save and display results
    tester.save_results(results, args.output)
    tester.print_summary(results)
    
    print("\nTesting completed!")

if __name__ == "__main__":
    asyncio.run(main())
