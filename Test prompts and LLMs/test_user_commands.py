#!/usr/bin/env python3
"""
Test script for different user command styles with Gemini 2.5 Pro
Tests how different ways of phrasing commands affect robot performance.
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
import google.generativeai as genai

class CommandStyle(Enum):
    HIGH_LEVEL = "high_level"
    STEP_BY_STEP = "step_by_step"
    TECHNICAL = "technical"
    CASUAL = "casual"
    DETAILED = "detailed"

@dataclass
class UserCommandTest:
    style: CommandStyle
    la_command: str
    ha_task: str
    description: str

@dataclass
class TestResult:
    style: CommandStyle
    mode: str
    command: str
    response: str
    parsed_actions: List[str]
    execution_time: float
    success: bool
    score: float
    error_message: Optional[str] = None
    ground_truth_options: List[List[str]] = None

class UserCommandTester:
    def __init__(self):
        self.setup_llm()
        self.test_commands = self.create_test_commands()
        
        # Get prompts from experiment_orchestrator.py
        self.LA_PROMPT, self.HA_PROMPT = self.get_prompts()
        
        # Define ground truth options for each task type
        self.ground_truths = self.create_ground_truths()
    
    def setup_llm(self):
        """Setup Gemini 2.5 Pro"""
        google_api_key = os.getenv('GOOGLE_API_KEY')
        if not google_api_key:
            raise Exception("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=google_api_key)
        self.gemini_client = genai
    
    def get_prompts(self):
        """Get the prompts from experiment_orchestrator.py"""
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
9. ALL movement parameters are REQUIRED: x=value, y=value, yaw=value, frame=body
10. Do NOT include movements with all zero values (x=0.0, y=0.0, yaw=0.0)
11. Use exact object names: "pringles can" not "pringles_can"
12. Rotations are allowed and valid - use yaw parameter for orientation changes

OUTPUT FORMAT:
Return only the action list, one action per line.

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
5. In one move, the robot can either move in the x direction, the y direction, or the yaw direction, not two or three at once.
6. Use RELATIVE body frame coordinates for movements - each movement is relative to current position
7. For movements, calculate: move_x = target_x - current_x, move_y = target_y - current_y
8. For pick-up: walk to pick-up location and face forward (yaw=0)
9. For drop-off: walk to drop-off location and face left (yaw=1.57 radians)
10. Use EXACT object name from user command for object_type (e.g., "tomato can" not "tomato")
11. For drop-off procedure: start_move_arm_pose to (0.8, 0.0, 0.3) with quaternion (0.7071, 0.7071, 0.0, 0.0) for gripper pointing down (this is the arm pose for dropping off objects) in 1 second, then start_arm_command open, then start_arm_command close, then start_arm_command stow
12. ALL movement parameters are REQUIRED: x=value, y=value, yaw=value, frame=body
13. Do NOT include movements with all zero values (x=0.0, y=0.0, yaw=0.0)
14. Use exact object names: "pringles can" not "pringles_can"
15. Rotations are allowed and valid - use yaw parameter for orientation changes

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

Actions:"""
        
        return LA_PROMPT, HA_PROMPT
    
    def create_ground_truths(self) -> Dict[str, Dict[str, List[List[str]]]]:
        """Create ground truth options for each task type and mode"""
        return {
            "easy": {
                "LA": [
                    ["start_moving, x=2.0, y=0.0, yaw=0.0, frame=body"],
                    ["start_moving, x=2.0, y=0.0, yaw=0.0, frame=\"body\""],
                    ["start_moving x=2.0 y=0.0 yaw=0.0 frame=body"]
                ],
                "HA": [
                    # Minimal approach - just the basic movements
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
                    # With rotations (valid approach)
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=1.5, yaw=1.57, frame=body"
                    ],
                    # Common LLM diagonal approach
                    [
                        "start_moving x=2.0 y=-1.0 yaw=0.0 frame=body",
                        "start_moving x=0.0 y=1.5 yaw=1.57 frame=body"
                    ],
                    # Comprehensive approach - includes return to start
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body",
                        "start_moving, x=-2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-0.5, yaw=0.0, frame=body"
                    ],
                    # Comprehensive with rotations and extra positioning
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=0.0, yaw=1.57, frame=body",
                        "start_moving, x=0.0, y=0.0, yaw=-1.57, frame=body",
                        "start_moving, x=-2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-0.5, yaw=0.0, frame=body"
                    ],
                    # Thorough approach with arm movements (LLM sometimes adds these)
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=0.0, yaw=1.57, frame=body",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=false",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow",
                        "start_moving, x=0.0, y=0.0, yaw=-1.57, frame=body",
                        "start_moving, x=-2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-0.5, yaw=0.0, frame=body"
                    ],
                    # Alternative thorough approach with different arm parameters
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body", 
                        "start_moving, x=0.0, y=1.5, yaw=1.57, frame=body",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow",
                        "start_moving, x=-2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-0.5, yaw=0.0, frame=body"
                    ]
                ]
            },
            "medium": {
                "LA": [
                    ["start_automated_grasp, object_type=tomato can"],
                    ["start_automated_grasp, object_type=\"tomato can\""],
                    ["start_automated_grasp object_type=tomato can"],
                    ["start_automated_grasp, object_type=can"],
                    ["start_automated_grasp, object_type=\"can\""],
                    ["start_automated_grasp object_type=can"]
                ],
                "HA": [
                    # Minimal approach - basic pick and place
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
                    # With rotations for proper orientation
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=tomato can",
                        "start_moving, x=0.0, y=1.5, yaw=1.57, frame=body"
                    ],
                    # Common LLM diagonal approach
                    [
                        "start_moving x=2.0 y=-1.0 yaw=0.0 frame=body",
                        "start_automated_grasp object_type=tomato can",
                        "start_moving x=0.0 y=1.5 yaw=1.57 frame=body"
                    ],
                    # Comprehensive with drop-off sequence and return
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=tomato can",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow"
                    ],
                    # Comprehensive with return to start
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=tomato can",
                        "start_moving, x=0.0, y=1.5, yaw=1.57, frame=body",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close", 
                        "start_arm_command, command_type=stow",
                        "start_moving, x=-2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-0.5, yaw=0.0, frame=body"
                    ],
                    # Alternative object names (LLM might shorten "tomato can" to "can")
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=can",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body"
                    ],
                    # With standing command at start (sometimes LLM adds this)
                    [
                        "stand_up",
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=tomato can",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body"
                    ],
                    # Y-first movement approach
                    [
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=tomato can",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body"
                    ],
                    # Complete 9-step sequence that LLM consistently generates
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=tomato can",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=0.0, yaw=1.57, frame=body",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=false",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow"
                    ],
                    # With quotes variation of the 9-step sequence
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=\"body\"",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=\"body\"",
                        "start_automated_grasp, object_type=\"tomato can\"",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=\"body\"",
                        "start_moving, x=0.0, y=0.0, yaw=1.57, frame=\"body\"",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=false",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow"
                    ],
                    # Alternative with different rotation timing
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=tomato can",
                        "start_moving, x=0.0, y=1.5, yaw=1.57, frame=body",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=false",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow"
                    ],
                    # Alternative with "can" object name
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=can",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=0.0, yaw=1.57, frame=body",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=false",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow"
                    ]
                ]
            },
            "hard": {
                "LA": [
                    ["start_automated_grasp, object_type=pringles can"],
                    ["start_automated_grasp, object_type=\"pringles can\""],
                    ["start_automated_grasp object_type=pringles can"],
                    ["start_automated_grasp, object_type=can"],
                    ["start_automated_grasp, object_type=\"can\""],
                    ["start_automated_grasp object_type=can"],
                    ["start_automated_grasp, object_type=pringles"],
                    ["start_automated_grasp, object_type=\"pringles\""],
                    ["start_automated_grasp object_type=pringles"]
                ],
                "HA": [
                    # Standard full sequence - the canonical approach
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
                    # With quotes around frame parameter
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
                    # Without commas and with spaces
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
                    # With rotations for proper orientation
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
                    # Common LLM diagonal approach
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
                    # With alternative object names (LLM might use just "can" or "pringles")
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=can",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow",
                        "start_moving, x=-2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-0.5, yaw=0.0, frame=body"
                    ],
                    # With "pringles" instead of "pringles can"
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=pringles",
                        "start_moving, x=0.0, y=1.5, yaw=1.57, frame=body",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow",
                        "start_moving, x=-2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-0.5, yaw=0.0, frame=body"
                    ],
                    # With stand_up command at start
                    [
                        "stand_up",
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
                    # Y-first movement approach
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
                    # Different arm pose parameters (LLM might vary these)
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=pringles can",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=false",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow",
                        "start_moving, x=-2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-0.5, yaw=0.0, frame=body"
                    ],
                    # Extra rotations before/after drop (LLM sometimes adds these)
                    [
                        "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                        "start_automated_grasp, object_type=pringles can",
                        "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=0.0, yaw=1.57, frame=body",
                        "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=false",
                        "start_arm_command, command_type=open",
                        "start_arm_command, command_type=close",
                        "start_arm_command, command_type=stow",
                        "start_moving, x=0.0, y=0.0, yaw=-1.57, frame=body",
                        "start_moving, x=-2.0, y=0.0, yaw=0.0, frame=body",
                        "start_moving, x=0.0, y=-0.5, yaw=0.0, frame=body"
                    ]
                ]
            }
        }
    
    def create_test_commands(self) -> List[UserCommandTest]:
        """Create test commands with different user styles"""
        return [
            # EASY TASK: Simple Movement
            UserCommandTest(
                style=CommandStyle.HIGH_LEVEL,
                la_command="Go forward 2 meters",
                ha_task="Walk to the pick-up area, then to the drop-off area, then back to the starting position",
                description="High-level natural language command"
            ),
            UserCommandTest(
                style=CommandStyle.STEP_BY_STEP,
                la_command="Move forward 2 meters",
                ha_task="First, walk to the pick-up location. Then, walk to the drop-off location. Finally, return to the starting position",
                description="Step-by-step explicit instructions"
            ),
            UserCommandTest(
                style=CommandStyle.TECHNICAL,
                la_command="Execute movement: x=2.0, y=0.0, yaw=0.0",
                ha_task="Navigate to coordinates (2.0, -1.0), then to (2.0, 0.5), then return to origin",
                description="Technical precise commands"
            ),
            UserCommandTest(
                style=CommandStyle.CASUAL,
                la_command="Hey robot, can you go forward a bit?",
                ha_task="Go to the pick-up spot, then to the drop-off spot, and come back here",
                description="Casual conversational style"
            ),
            UserCommandTest(
                style=CommandStyle.DETAILED,
                la_command="Please move the robot forward by exactly 2 meters in the positive x direction",
                ha_task="I need you to: 1) Navigate to the designated pick-up area at coordinates (2.0, -1.0), 2) Proceed to the drop-off area at coordinates (2.0, 0.5), 3) Return to the original starting position",
                description="Detailed verbose instructions"
            ),
            
            # Additional EASY TASK variations
            UserCommandTest(
                style=CommandStyle.CASUAL,
                la_command="Move ahead please",
                ha_task="Can you go over there and then come back?",
                description="Very casual, vague commands"
            ),
            UserCommandTest(
                style=CommandStyle.HIGH_LEVEL,
                la_command="Drive forward",
                ha_task="Visit the pick-up zone and drop-off zone, then return home",
                description="Simple, direct commands"
            ),
            UserCommandTest(
                style=CommandStyle.TECHNICAL,
                la_command="Move x=2.0",
                ha_task="Execute waypoint navigation: (2.0, -1.0) -> (2.0, 0.5) -> (0.0, 0.0)",
                description="Minimal technical commands"
            ),
            UserCommandTest(
                style=CommandStyle.STEP_BY_STEP,
                la_command="Take 2 steps forward",
                ha_task="1) Go to pick-up area 2) Go to drop-off area 3) Return to start",
                description="Numbered step instructions"
            ),
            UserCommandTest(
                style=CommandStyle.CASUAL,
                la_command="Can you walk over there?",
                ha_task="Go check out the pick-up area and drop-off area, then come back",
                description="Conversational, exploratory language"
            ),
            
            # MEDIUM TASK: Movement + Grasping
            UserCommandTest(
                style=CommandStyle.HIGH_LEVEL,
                la_command="Pick up the tomato can",
                ha_task="Go get the tomato can from the pick-up area and bring it to the drop-off area",
                description="High-level natural language command"
            ),
            UserCommandTest(
                style=CommandStyle.STEP_BY_STEP,
                la_command="Grasp the tomato can object",
                ha_task="Step 1: Walk to the pick-up location. Step 2: Pick up the tomato can. Step 3: Walk to the drop-off location. Step 4: Place the object down",
                description="Step-by-step explicit instructions"
            ),
            UserCommandTest(
                style=CommandStyle.TECHNICAL,
                la_command="Execute automated grasp for object_type='tomato can'",
                ha_task="Navigate to (2.0, -1.0), execute start_automated_grasp for 'tomato can', then navigate to (2.0, 0.5) for drop-off",
                description="Technical precise commands"
            ),
            UserCommandTest(
                style=CommandStyle.CASUAL,
                la_command="Can you grab that tomato can for me?",
                ha_task="Hey, could you go pick up the tomato can and drop it off at the other spot?",
                description="Casual conversational style"
            ),
            UserCommandTest(
                style=CommandStyle.DETAILED,
                la_command="Please execute the automated grasping procedure for the tomato can object using the robot's arm",
                ha_task="I need you to perform the following sequence: Navigate to the pick-up area located at coordinates (2.0, -1.0), execute the automated grasping procedure for the 'tomato can' object, then transport it to the drop-off area at coordinates (2.0, 0.5) and place it down",
                description="Detailed verbose instructions"
            ),
            
            # Additional MEDIUM TASK variations
            UserCommandTest(
                style=CommandStyle.CASUAL,
                la_command="Grab that can",
                ha_task="Go get the tomato can and move it to the other area",
                description="Very casual, minimal commands"
            ),
            UserCommandTest(
                style=CommandStyle.HIGH_LEVEL,
                la_command="Collect the tomato can",
                ha_task="Retrieve the tomato can from the pick-up zone and deliver it to the drop-off zone",
                description="Professional, business-like language"
            ),
            UserCommandTest(
                style=CommandStyle.TECHNICAL,
                la_command="start_automated_grasp tomato can",
                ha_task="Waypoint 1: (2.0, -1.0) + grasp 'tomato can', Waypoint 2: (2.0, 0.5) + drop",
                description="Minimal technical syntax"
            ),
            UserCommandTest(
                style=CommandStyle.STEP_BY_STEP,
                la_command="Take the tomato can",
                ha_task="A) Go to pick-up B) Grab tomato can C) Go to drop-off D) Release object",
                description="Lettered step instructions"
            ),
            UserCommandTest(
                style=CommandStyle.CASUAL,
                la_command="Could you please pick up that tomato can?",
                ha_task="Would you mind going to get the tomato can and putting it in the other location?",
                description="Polite, conversational style"
            ),
            UserCommandTest(
                style=CommandStyle.HIGH_LEVEL,
                la_command="Move the tomato can",
                ha_task="Transport the tomato can from the pick-up location to the drop-off location",
                description="Simple, clear instructions"
            ),
            
            # HARD TASK: Complex Multi-step
            UserCommandTest(
                style=CommandStyle.HIGH_LEVEL,
                la_command="Pick up the pringles can",
                ha_task="Go get the pringles can, drop it off, and come back to the starting position",
                description="High-level natural language command"
            ),
            UserCommandTest(
                style=CommandStyle.STEP_BY_STEP,
                la_command="Grasp the pringles can object",
                ha_task="First, walk to the pick-up location. Second, pick up the pringles can. Third, walk to the drop-off location. Fourth, place the can down. Fifth, return to the starting position",
                description="Step-by-step explicit instructions"
            ),
            UserCommandTest(
                style=CommandStyle.TECHNICAL,
                la_command="Execute automated grasp for object_type='pringles can'",
                ha_task="Navigate to (2.0, -1.0), grasp 'pringles can', navigate to (2.0, 0.5), execute drop-off procedure, return to origin",
                description="Technical precise commands"
            ),
            UserCommandTest(
                style=CommandStyle.CASUAL,
                la_command="Hey, can you grab that pringles can and put it somewhere else?",
                ha_task="Could you go get the pringles can from over there, drop it off at the other place, and then come back here?",
                description="Casual conversational style"
            ),
            UserCommandTest(
                style=CommandStyle.DETAILED,
                la_command="Please execute the automated grasping procedure for the pringles can object",
                ha_task="I need you to complete this multi-step task: 1) Navigate to the pick-up area at (2.0, -1.0), 2) Execute automated grasping of the 'pringles can' object, 3) Transport it to the drop-off area at (2.0, 0.5), 4) Perform the complete drop-off procedure including arm positioning and gripper operations, 5) Return to the original starting position",
                description="Detailed verbose instructions"
            ),
            
            # Additional HARD TASK variations
            UserCommandTest(
                style=CommandStyle.CASUAL,
                la_command="Grab that pringles can",
                ha_task="Go get the pringles can, drop it off, and come back",
                description="Very casual, minimal commands"
            ),
            UserCommandTest(
                style=CommandStyle.HIGH_LEVEL,
                la_command="Complete the pringles can task",
                ha_task="Handle the pringles can: pick it up, move it to the drop-off area, and return to start",
                description="Task-oriented language"
            ),
            UserCommandTest(
                style=CommandStyle.TECHNICAL,
                la_command="start_automated_grasp pringles can",
                ha_task="Mission: (2.0, -1.0) + grasp 'pringles can' -> (2.0, 0.5) + drop -> (0.0, 0.0)",
                description="Mission-style technical commands"
            ),
            UserCommandTest(
                style=CommandStyle.STEP_BY_STEP,
                la_command="Take the pringles can",
                ha_task="1) Go to pick-up 2) Grab pringles can 3) Go to drop-off 4) Place it down 5) Come back",
                description="Simple numbered steps"
            ),
            UserCommandTest(
                style=CommandStyle.CASUAL,
                la_command="Could you handle the pringles can for me?",
                ha_task="I need you to pick up the pringles can, take it to the other area, and then come back to where you started",
                description="Polite, personal request"
            ),
            UserCommandTest(
                style=CommandStyle.HIGH_LEVEL,
                la_command="Move the pringles can",
                ha_task="Retrieve the pringles can from the pick-up location, deliver it to the drop-off location, and return to the starting position",
                description="Professional delivery language"
            ),
            UserCommandTest(
                style=CommandStyle.CASUAL,
                la_command="Can you do something with that pringles can?",
                ha_task="Go grab the pringles can, put it somewhere else, and come back",
                description="Vague, exploratory language"
            ),
            UserCommandTest(
                style=CommandStyle.TECHNICAL,
                la_command="Execute full pringles can workflow",
                ha_task="Workflow: Navigate(2.0,-1.0) -> Grasp('pringles can') -> Navigate(2.0,0.5) -> Drop() -> Navigate(0.0,0.0)",
                description="Workflow-style technical commands"
            )
        ]
    
    async def test_command(self, command_test: UserCommandTest, mode: str) -> TestResult:
        """Test a single command with Gemini 2.5 Pro"""
        start_time = time.time()
        
        try:
            if mode == 'LA':
                prompt = self.LA_PROMPT.format(
                    command=command_test.la_command,
                    current_state="standing"
                )
            else:  # HA mode
                prompt = self.HA_PROMPT.format(
                    task=command_test.ha_task,
                    current_state="standing",
                    current_position="(0.0, 0.0, 0.0)"
                )
            
            model = genai.GenerativeModel('gemini-2.5-pro')
            response = model.generate_content(prompt)
            result = response.text.strip()
            
            # Parse actions
            parsed_actions = self.parse_actions(result)
            
            # Get ground truth for this task type and mode
            task_type = self.get_task_type(command_test)
            ground_truth_options = self.ground_truths[task_type][mode]
            
            # Calculate score based on ground truth comparison
            score = self.calculate_score(parsed_actions, ground_truth_options)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                style=command_test.style,
                mode=mode,
                command=command_test.la_command if mode == 'LA' else command_test.ha_task,
                response=result,
                parsed_actions=parsed_actions,
                execution_time=execution_time,
                success=len(parsed_actions) > 0,
                score=score,
                ground_truth_options=ground_truth_options
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                style=command_test.style,
                mode=mode,
                command=command_test.la_command if mode == 'LA' else command_test.ha_task,
                response="",
                parsed_actions=[],
                execution_time=execution_time,
                success=False,
                score=0.0,
                error_message=str(e)
            )
    
    def parse_actions(self, response: str) -> List[str]:
        """Parse actions from LLM response"""
        actions = []
        for line in response.splitlines():
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('//'):
                # Remove numbering if present
                if line[0].isdigit() and ('.' in line or ')' in line):
                    line = line.split('.', 1)[1].strip() if '.' in line else line.split(')', 1)[1].strip()
                actions.append(line)
        return actions
    
    def normalize_action(self, action: str) -> str:
        """Normalize action for comparison"""
        # Remove quotes from parameters
        action = action.replace('"', '').replace("'", '')
        # Normalize spaces around commas and equals
        action = action.replace(' = ', '=').replace('= ', '=').replace(' =', '=')
        action = action.replace(' , ', ',').replace(', ', ',').replace(' ,', ',')
        return action.strip()
    
    def calculate_score(self, predicted_actions: List[str], ground_truth_options: List[List[str]]) -> float:
        """Calculate score based on ground truth comparison"""
        if not predicted_actions or not ground_truth_options:
            return 0.0
        
        # Normalize predicted actions
        predicted_normalized = [self.normalize_action(action) for action in predicted_actions]
        
        best_score = 0.0
        
        for ground_truth in ground_truth_options:
            # Normalize ground truth actions
            gt_normalized = [self.normalize_action(action) for action in ground_truth]
            
            # Calculate sequence similarity (order matters)
            sequence_matches = 0
            min_len = min(len(predicted_normalized), len(gt_normalized))
            for i in range(min_len):
                if predicted_normalized[i] == gt_normalized[i]:
                    sequence_matches += 1
            
            sequence_score = sequence_matches / max(len(predicted_normalized), len(gt_normalized)) if max(len(predicted_normalized), len(gt_normalized)) > 0 else 0
            
            # Calculate presence score (actions present, order doesn't matter)
            predicted_set = set(predicted_normalized)
            gt_set = set(gt_normalized)
            presence_score = len(predicted_set & gt_set) / len(gt_set) if len(gt_set) > 0 else 0
            
            # Length penalty for extra actions
            length_penalty = 1.0
            if len(predicted_normalized) > len(gt_normalized):
                length_penalty = 0.95  # Small penalty for extra actions
            elif len(predicted_normalized) < len(gt_normalized):
                length_penalty = 0.8   # Moderate penalty for missing actions
            
            # Combined score (weighted)
            combined_score = (sequence_score * 0.4 + presence_score * 0.6) * length_penalty
            best_score = max(best_score, combined_score)
        
        return best_score
    
    def get_task_type(self, command_test: UserCommandTest) -> str:
        """Determine task type based on command content"""
        if "pringles can" in command_test.ha_task.lower() or "pringles can" in command_test.la_command.lower():
            return "hard"
        elif "tomato can" in command_test.ha_task.lower() or "tomato can" in command_test.la_command.lower():
            return "medium"
        else:
            return "easy"
    
    async def run_all_tests(self) -> List[TestResult]:
        """Run all command style tests"""
        results = []
        modes = ['LA', 'HA']
        
        total_tests = len(self.test_commands) * len(modes)
        print(f"Running {len(self.test_commands)} command styles × {len(modes)} modes = {total_tests} total tests")
        print(f"Command styles: {', '.join([s.value for s in CommandStyle])}")
        
        for command_test in self.test_commands:
            print(f"\nTesting {command_test.style.value} style: {command_test.description}")
            for mode in modes:
                print(f"  Mode: {mode}")
                result = await self.test_command(command_test, mode)
                results.append(result)
                
                if result.success:
                    print(f"    ✓ Success ({result.execution_time:.2f}s) - {len(result.parsed_actions)} actions")
                    print(f"    Command: {result.command[:50]}...")
                else:
                    print(f"    ✗ Failed: {result.error_message}")
        
        return results
    
    async def run_tests(self, selected_styles: List[CommandStyle], selected_modes: List[str]) -> List[TestResult]:
        """Run tests with selected styles and modes"""
        results = []
        
        # Filter commands based on selected styles
        filtered_commands = [c for c in self.test_commands if c.style in selected_styles]
        
        total_tests = len(filtered_commands) * len(selected_modes)
        print(f"Running {len(filtered_commands)} command styles × {len(selected_modes)} modes = {total_tests} total tests")
        print(f"Command styles: {', '.join([s.value for s in selected_styles])}")
        print(f"Modes: {', '.join(selected_modes)}")
        
        for command_test in filtered_commands:
            print(f"\nTesting {command_test.style.value} style: {command_test.description}")
            for mode in selected_modes:
                print(f"  Mode: {mode}")
                result = await self.test_command(command_test, mode)
                results.append(result)
                
                if result.success:
                    print(f"    ✓ Success ({result.execution_time:.2f}s) - {len(result.parsed_actions)} actions - Score: {result.score:.2f}")
                    print(f"    Command: {result.command[:50]}...")
                else:
                    print(f"    ✗ Failed: {result.error_message}")
        
        return results
    
    def save_results(self, results: List[TestResult], filename: str = "user_command_results.json"):
        """Save test results to JSON file"""
        data = []
        for result in results:
            data.append({
                "command_style": result.style.value,
                "mode": result.mode,
                "command": result.command,
                "response": result.response,
                "parsed_actions": result.parsed_actions,
                "execution_time": result.execution_time,
                "success": result.success,
                "score": result.score,
                "error_message": result.error_message
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def print_summary(self, results: List[TestResult]):
        """Print summary of results"""
        print("\n" + "="*80)
        print("USER COMMAND STYLE TESTING SUMMARY")
        print("="*80)
        
        # Group by style
        by_style = {}
        for result in results:
            style = result.style.value
            if style not in by_style:
                by_style[style] = []
            by_style[style].append(result)
        
        for style, style_results in by_style.items():
            la_results = [r for r in style_results if r.mode == 'LA']
            ha_results = [r for r in style_results if r.mode == 'HA']
            
            print(f"\n{style.upper()} STYLE:")
            la_success = [r for r in la_results if r.success]
            ha_success = [r for r in ha_results if r.success]
            print(f"  LA Mode: {len(la_success)}/{len(la_results)} successful")
            print(f"  HA Mode: {len(ha_success)}/{len(ha_results)} successful")
            
            if la_results:
                avg_time_la = sum(r.execution_time for r in la_results) / len(la_results)
                avg_score_la = sum(r.score for r in la_results) / len(la_results)
                avg_steps_la = sum(len(r.parsed_actions) for r in la_results) / len(la_results)
                print(f"  LA Avg Time: {avg_time_la:.2f}s, Avg Score: {avg_score_la:.2f}, Avg Steps: {avg_steps_la:.1f}")
            
            if ha_results:
                avg_time_ha = sum(r.execution_time for r in ha_results) / len(ha_results)
                avg_score_ha = sum(r.score for r in ha_results) / len(ha_results)
                avg_steps_ha = sum(len(r.parsed_actions) for r in ha_results) / len(ha_results)
                print(f"  HA Avg Time: {avg_time_ha:.2f}s, Avg Score: {avg_score_ha:.2f}, Avg Steps: {avg_steps_ha:.1f}")
        
        # Task-type analysis
        self.print_task_type_analysis(results)
        
        # Overall stats
        total_success = len([r for r in results if r.success])
        total_tests = len(results)
        avg_time = sum(r.execution_time for r in results) / len(results)
        avg_score = sum(r.score for r in results) / len(results)
        avg_steps = sum(len(r.parsed_actions) for r in results) / len(results)
        
        print(f"\nOVERALL:")
        print(f"  Success Rate: {total_success}/{total_tests} ({total_success/total_tests*100:.1f}%)")
        print(f"  Average Time: {avg_time:.2f}s")
        print(f"  Average Score: {avg_score:.2f}")
        print(f"  Average Steps: {avg_steps:.1f}")
        
        # Best performing style
        style_scores = {}
        for style, style_results in by_style.items():
            avg_score = sum(r.score for r in style_results) / len(style_results)
            style_scores[style] = avg_score
        
        best_style = max(style_scores, key=style_scores.get)
        print(f"  Best Style: {best_style} (avg score: {style_scores[best_style]:.2f})")
    
    def print_task_type_analysis(self, results: List[TestResult]):
        """Print detailed analysis by task type"""
        print(f"\n{'='*80}")
        print("TASK TYPE ANALYSIS")
        print("="*80)
        
        # Group results by task type
        by_task_type = {"easy": [], "medium": [], "hard": []}
        
        for result in results:
            # Create a dummy command test to determine task type
            dummy_command = type('obj', (object,), {
                'la_command': result.command,
                'ha_task': result.command
            })()
            task_type = self.get_task_type(dummy_command)
            by_task_type[task_type].append(result)
        
        for task_type, task_results in by_task_type.items():
            if not task_results:
                continue
                
            print(f"\n{task_type.upper()} TASKS:")
            
            # Basic stats
            successful = [r for r in task_results if r.success]
            failed = [r for r in task_results if not r.success]
            
            if task_results:
                avg_score = sum(r.score for r in task_results) / len(task_results)
                avg_steps = sum(len(r.parsed_actions) for r in task_results) / len(task_results)
                avg_time = sum(r.execution_time for r in task_results) / len(task_results)
                
                print(f"  Success: {len(successful)}/{len(task_results)} ({len(successful)/len(task_results)*100:.1f}%)")
                print(f"  Avg Score: {avg_score:.3f}")
                print(f"  Avg Steps: {avg_steps:.1f}")
                print(f"  Avg Time: {avg_time:.2f}s")
            
            # Score distribution
            if successful:
                scores = [r.score for r in successful]
                high_scores = [s for s in scores if s >= 0.8]
                medium_scores = [s for s in scores if 0.5 <= s < 0.8]
                low_scores = [s for s in scores if s < 0.5]
                
                print(f"  Score Distribution:")
                print(f"    High (≥0.8): {len(high_scores)}/{len(scores)} ({len(high_scores)/len(scores)*100:.1f}%)")
                print(f"    Medium (0.5-0.8): {len(medium_scores)}/{len(scores)} ({len(medium_scores)/len(scores)*100:.1f}%)")
                print(f"    Low (<0.5): {len(low_scores)}/{len(scores)} ({len(low_scores)/len(scores)*100:.1f}%)")
            
            # Common issues analysis
            print(f"  Common Issues:")
            
            if failed:
                print(f"    • {len(failed)} complete failures")
                error_types = {}
                for result in failed:
                    if result.error_message:
                        error_key = result.error_message[:50] + "..." if len(result.error_message) > 50 else result.error_message
                        error_types[error_key] = error_types.get(error_key, 0) + 1
                
                for error, count in error_types.items():
                    print(f"      - {error}: {count} times")
            
            # Analyze low-scoring successful results
            low_scoring = [r for r in successful if r.score < 0.8]
            if low_scoring:
                print(f"    • {len(low_scoring)} low-scoring results (score < 0.8)")
                
                # Analyze common patterns in low scores
                step_counts = [len(r.parsed_actions) for r in low_scoring]
                if step_counts:
                    avg_low_steps = sum(step_counts) / len(step_counts)
                    print(f"      - Avg steps in low-scoring: {avg_low_steps:.1f}")
                
                # Check for common action patterns
                action_patterns = {}
                for result in low_scoring:
                    if result.parsed_actions:
                        first_action = result.parsed_actions[0].split(',')[0].strip()
                        action_patterns[first_action] = action_patterns.get(first_action, 0) + 1
                
                if action_patterns:
                    print(f"      - Common first actions in low-scoring:")
                    for action, count in sorted(action_patterns.items(), key=lambda x: x[1], reverse=True)[:3]:
                        print(f"        * {action}: {count} times")
                
                # Show examples of low-scoring responses
                if len(low_scoring) > 0:
                    example = low_scoring[0]
                    print(f"      - Example low-scoring response (score: {example.score:.3f}):")
                    print(f"        Command: {example.command[:60]}...")
                    print(f"        Actions: {len(example.parsed_actions)} steps")
                    for i, action in enumerate(example.parsed_actions[:3]):
                        print(f"          {i+1}. {action[:50]}...")
                    if len(example.parsed_actions) > 3:
                        print(f"          ... and {len(example.parsed_actions)-3} more")
            else:
                print(f"    • No significant issues found!")
                if successful:
                    print(f"      - All {len(successful)} responses scored ≥ 0.8")

def parse_instances(instances_str: str) -> List[int]:
    """Parse instances string to list of integers.
    
    Supports:
    - Ranges: "1-10" -> [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    - Comma-separated: "1,2,3" -> [1, 2, 3]
    - Mixed: "1,3-5,7" -> [1, 3, 4, 5, 7]
    - Single numbers: "5" -> [5]
    """
    if not instances_str:
        return []
    
    instances = []
    parts = instances_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part:
            # Handle range like "1-10"
            try:
                start, end = part.split('-', 1)
                start, end = int(start.strip()), int(end.strip())
                if start > end:
                    raise ValueError(f"Invalid range: {part} (start > end)")
                instances.extend(range(start, end + 1))
            except ValueError as e:
                raise ValueError(f"Invalid range format '{part}': {e}")
        else:
            # Handle single number
            try:
                instances.append(int(part))
            except ValueError:
                raise ValueError(f"Invalid number format: '{part}'")
    
    # Remove duplicates and sort
    return sorted(list(set(instances)))

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test different user command styles with Gemini 2.5 Pro')
    
    parser.add_argument(
        '--styles',
        nargs='+',
        default=[s.value for s in CommandStyle],
        choices=[s.value for s in CommandStyle],
        help='Command styles to test (default: all)'
    )
    
    parser.add_argument(
        '--modes',
        nargs='+',
        default=['LA', 'HA'],
        choices=['LA', 'HA'],
        help='Modes to test (default: both)'
    )
    
    parser.add_argument(
        '--instances',
        type=str,
        help='Test instances to run. Supports ranges (1-10), comma-separated (1,2,3), or mixed (1,3-5,7). Default: all instances'
    )
    
    parser.add_argument(
        '--output',
        default='user_command_results.json',
        help='Output file for results (default: user_command_results.json)'
    )
    
    return parser.parse_args()

async def main():
    """Main function to run all tests"""
    args = parse_arguments()
    
    print("Starting User Command Style Testing with Gemini 2.5 Pro")
    print("="*60)
    print(f"Testing styles: {', '.join(args.styles)}")
    print(f"Testing modes: {', '.join(args.modes)}")
    
    # Parse instances if provided
    selected_instances = None
    if args.instances:
        try:
            selected_instances = parse_instances(args.instances)
            print(f"Testing instances: {selected_instances}")
        except ValueError as e:
            print(f"ERROR: Invalid instances format: {e}")
            return
    else:
        print("Testing instances: all")
    
    print(f"Output file: {args.output}")
    print()
    
    # Check environment variables
    if not os.getenv('GOOGLE_API_KEY'):
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        return
    
    # Create tester and run tests
    tester = UserCommandTester()
    
    # Filter commands based on selected styles
    selected_styles = [CommandStyle(s) for s in args.styles]
    filtered_commands = [c for c in tester.test_commands if c.style in selected_styles]
    
    # Filter commands based on selected instances (1-indexed)
    if selected_instances is not None:
        # Convert to 0-indexed for list access
        instance_indices = [i - 1 for i in selected_instances if 1 <= i <= len(filtered_commands)]
        if not instance_indices:
            print(f"ERROR: No valid instances found. Available range: 1-{len(filtered_commands)}")
            return
        filtered_commands = [filtered_commands[i] for i in instance_indices]
        print(f"Running {len(filtered_commands)} selected instances")
    
    tester.test_commands = filtered_commands
    
    results = await tester.run_tests(selected_styles, args.modes)
    
    # Save and display results
    tester.save_results(results, args.output)
    tester.print_summary(results)
    
    print("\nTesting completed!")

if __name__ == "__main__":
    asyncio.run(main())
