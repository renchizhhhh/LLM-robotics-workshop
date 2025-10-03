#!/usr/bin/env python3
"""
Test script for HA prompt ablation study
Tests different combinations of HA prompt components to identify which parts are essential.
Uses GPT-OSS-120B model and runs 3 iterations for each combination.
"""

import os
import json
import time
import asyncio
import argparse
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# LLM imports
import openai
try:
    from together import Together
except ImportError:
    Together = None

class PromptComponent(Enum):
    # Core sections
    ROLE_DEFINITION = "role_definition"
    CURRENT_STATE = "current_state"
    WORLD_LAYOUT = "world_layout"
    ROBOT_SPECS = "robot_specs"
    AVAILABLE_ACTIONS = "available_actions"
    RULES = "rules"
    PLANNING_PROCESS = "planning_process"
    OUTPUT_FORMAT = "output_format"
    
    # Specific rule categories
    BASIC_RULES = "basic_rules"  # Rules 1-4
    MOVEMENT_RULES = "movement_rules"  # Rules 5-7
    ORIENTATION_RULES = "orientation_rules"  # Rules 8-9
    OBJECT_RULES = "object_rules"  # Rules 10, 14
    ARM_RULES = "arm_rules"  # Rule 11
    PARAMETER_RULES = "parameter_rules"  # Rules 12-13, 15

@dataclass
class PromptVariant:
    name: str
    components: List[PromptComponent]
    description: str

@dataclass
class TestResult:
    variant_name: str
    components_used: List[str]
    run_number: int
    mode: str
    task_type: str
    command: str
    response: str
    parsed_actions: List[str]
    execution_time: float
    success: bool
    score: float
    prompt: str = ""
    formatted_prompt: str = ""
    error_message: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

class HAPromptAblationStudy:
    def __init__(self):
        self.setup_llm()
        self.test_tasks = self.create_test_tasks()
        self.ground_truths = self.create_ground_truths()
        self.prompt_variants = self.create_prompt_variants()
        
    def setup_llm(self):
        """Setup LLM client - prefer Together AI, fallback to Groq"""
        # Try Together AI first (preferred for 120B model)
        together_api_key = os.getenv('TOGETHER_API_KEY')
        if together_api_key and Together:
            self.together_client = Together(api_key=together_api_key)
            self.llm_provider = "together"
            print("Using Together AI for GPT-OSS-120B")
            return
        
        # Fallback to Groq
        groq_api_key = os.getenv('GROQ_API_KEY')
        groq_base_url = os.getenv('GROQ_BASE_URL', 'https://api.groq.com/openai/v1')
        
        if groq_api_key:
            self.groq_client = openai.OpenAI(api_key=groq_api_key, base_url=groq_base_url)
            self.llm_provider = "groq"
            print("Using Groq for GPT-OSS-120B")
            return
        
        raise Exception("Neither TOGETHER_API_KEY nor GROQ_API_KEY environment variable set")
    
    def create_test_tasks(self) -> List[Dict[str, str]]:
        """Create test tasks for different difficulty levels"""
        return [
            {
                "type": "hard",
                "task": "go to the pick-up location, pick up the pringles can, go to the drop-off location, place the pringles can down, then return to the starting position",
                "description": "Complex multi-step task"
            }
        ]
    
    def create_ground_truths(self) -> Dict[str, List[List[str]]]:
        """Create ground truth options for each task type with more variations"""
        return {
            "hard": [
                # Full sequence
                [
                    "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                    "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                    "start_automated_grasp, object_type=pringles can",
                    "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body",
                    "start_moving, x=0.0, y=0.0, yaw=1.57, frame=body",
                    "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                    "start_arm_command, command_type=open",
                    "start_arm_command, command_type=close",
                    "start_arm_command, command_type=stow",
                    "start_moving x=0.0 y=0.0 yaw=-1.57 frame=body",
                    "start_moving x=-2.0 y=0.0 yaw=0.0 frame=body",
                    "start_moving x=0.0 y=-0.5 yaw=0.0 frame=body"
                ],
                # Without quotes
                [
                    "start_moving x=2.0 y=0.0 yaw=0.0 frame=body",
                    "start_moving x=0.0 y=-1.0 yaw=0.0 frame=body",
                    "start_automated_grasp object_type=pringles can",
                    "start_moving x=0.0 y=1.5 yaw=0.0 frame=body",
                    "start_moving x=0.0 y=0.0 yaw=1.57 frame=body",
                    "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                    "start_arm_command command_type=open",
                    "start_arm_command command_type=close",
                    "start_arm_command command_type=stow",
                    "start_moving x=0.0 y=2.0 yaw=0.0 frame=body",
                    "start_moving x=-0.5 y=0.0 yaw=0.0 frame=body",
                    "start_moving x=0.0 y=0.0 yaw=-1.57 frame=body"
                ],
                # Alternative object names
                [
                    "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                    "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                    "start_automated_grasp, object_type=can",
                    "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body",
                    "start_moving, x=0.0, y=0.0, yaw=1.57, frame=body",
                    "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                    "start_arm_command, command_type=open",
                    "start_arm_command, command_type=close",
                    "start_arm_command, command_type=stow",
                    "start_moving, x=-2.0, y=0.0, yaw=0.0, frame=body",
                    "start_moving, x=0.0, y=-0.5, yaw=0.0, frame=body"
                ],
                # Alternative object names
                [
                    "start_moving, x=0.0, y=-1.0, yaw=0.0, frame=body",
                    "start_moving, x=2.0, y=0.0, yaw=0.0, frame=body",
                    "start_automated_grasp, object_type=can",
                    "start_moving, x=0.0, y=1.5, yaw=0.0, frame=body",
                    "start_moving, x=0.0, y=0.0, yaw=1.57, frame=body",
                    "start_move_arm_pose, x=0.8, y=0.0, z=0.3, qw=0.7071, qx=0.7071, qy=0.0, qz=0.0, duration=1.0, open_gripper=true",
                    "start_arm_command, command_type=open",
                    "start_arm_command, command_type=close",
                    "start_arm_command, command_type=stow",
                    "start_moving, x=-2.0, y=0.0, yaw=0.0, frame=body",
                    "start_moving, x=0.0, y=-0.5, yaw=0.0, frame=body"
                ],
                # With stand_up and sit_down
                [
                    "stand_up",
                    "start_moving x=2.0 y=0 yaw=0 frame=body",
                    "start_moving x=0 y=-1.0 yaw=0 frame=body",
                    "start_automated_grasp object_type=pringles can",
                    "start_moving x=0 y=1.5 yaw=0 frame=body",
                    "start_moving x=0 y=0 yaw=1.57 frame=body",
                    "start_arm_command command_type=open",
                    "start_moving x=0 y=0 yaw=-1.57 frame=body",
                    "start_moving x=-2.0 y=0 yaw=0 frame=body",
                    "start_moving x=0 y=-0.5 yaw=0 frame=body",
                    "sit_down"
                ],
                # With get_initial_pose and stand_up
                [
                    "get_initial_pose",
                    "stand_up",
                    "start_moving x=2.0 y=0 yaw=0 frame=body",
                    "start_moving x=0 y=-1.0 yaw=0 frame=body",
                    "start_automated_grasp object_type=pringles can",
                    "start_moving x=0 y=1.5 yaw=0 frame=body",
                    "start_moving x=0 y=0 yaw=1.57 frame=body",
                    "start_arm_command command_type=open",
                    "start_moving x=0 y=0 yaw=-1.57 frame=body",
                    "start_moving x=-2.0 y=0 yaw=0 frame=body",
                    "start_moving x=0 y=-0.5 yaw=0 frame=body",
                    "sit_down"
                ],
                [
                    "stand_up",
                    "get_initial_pose",
                    "start_moving x=2.0 y=0 yaw=0 frame=body",
                    "start_moving x=0 y=-1.0 yaw=0 frame=body",
                    "start_automated_grasp object_type=pringles can",
                    "start_moving x=0 y=1.5 yaw=0 frame=body",
                    "start_moving x=0 y=0 yaw=1.57 frame=body",
                    "start_arm_command command_type=open",
                    "start_moving x=0 y=0 yaw=-1.57 frame=body",
                    "start_moving x=-2.0 y=0 yaw=0 frame=body",
                    "start_moving x=0 y=-0.5 yaw=0 frame=body",
                    "sit_down"
                ],

            ]
        }
    
    def create_prompt_variants(self) -> List[PromptVariant]:
        """Create systematic prompt variants for ablation study"""
        return [
            # 1. Minimal prompt (just role + actions + basic rules)
            PromptVariant(
                name="minimal",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.BASIC_RULES],
                description="Minimal prompt with just role, actions, and basic rules"
            ),
            
            # 2. Core components only
            PromptVariant(
                name="core_only",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.RULES, PromptComponent.OUTPUT_FORMAT],
                description="Core components without state/layout info"
            ),
            
            # 3. With state information
            PromptVariant(
                name="with_state",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.CURRENT_STATE, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.RULES, PromptComponent.OUTPUT_FORMAT],
                description="Core + current state information"
            ),
            
            # 4. With world layout
            PromptVariant(
                name="with_layout",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.WORLD_LAYOUT, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.RULES, PromptComponent.OUTPUT_FORMAT],
                description="Core + world layout information"
            ),
            
            # 5. With robot specs
            PromptVariant(
                name="with_specs",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.ROBOT_SPECS, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.RULES, PromptComponent.OUTPUT_FORMAT],
                description="Core + robot specifications"
            ),
            
            # 6. Without planning process
            PromptVariant(
                name="no_planning",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.CURRENT_STATE, PromptComponent.WORLD_LAYOUT, PromptComponent.ROBOT_SPECS, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.RULES, PromptComponent.OUTPUT_FORMAT],
                description="Full prompt without planning process section"
            ),
            
            # 7. Without movement rules
            PromptVariant(
                name="no_movement_rules",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.CURRENT_STATE, PromptComponent.WORLD_LAYOUT, PromptComponent.ROBOT_SPECS, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.BASIC_RULES, PromptComponent.ORIENTATION_RULES, PromptComponent.OBJECT_RULES, PromptComponent.ARM_RULES, PromptComponent.PARAMETER_RULES, PromptComponent.PLANNING_PROCESS, PromptComponent.OUTPUT_FORMAT],
                description="Full prompt without movement-specific rules (5-7)"
            ),
            
            # 8. Without orientation rules
            PromptVariant(
                name="no_orientation_rules",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.CURRENT_STATE, PromptComponent.WORLD_LAYOUT, PromptComponent.ROBOT_SPECS, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.BASIC_RULES, PromptComponent.MOVEMENT_RULES, PromptComponent.OBJECT_RULES, PromptComponent.ARM_RULES, PromptComponent.PARAMETER_RULES, PromptComponent.PLANNING_PROCESS, PromptComponent.OUTPUT_FORMAT],
                description="Full prompt without orientation rules (8-9)"
            ),
            
            # 9. Without object rules
            PromptVariant(
                name="no_object_rules",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.CURRENT_STATE, PromptComponent.WORLD_LAYOUT, PromptComponent.ROBOT_SPECS, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.BASIC_RULES, PromptComponent.MOVEMENT_RULES, PromptComponent.ORIENTATION_RULES, PromptComponent.ARM_RULES, PromptComponent.PARAMETER_RULES, PromptComponent.PLANNING_PROCESS, PromptComponent.OUTPUT_FORMAT],
                description="Full prompt without object-specific rules (10, 14)"
            ),
            
            # 10. Without arm rules
            PromptVariant(
                name="no_arm_rules",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.CURRENT_STATE, PromptComponent.WORLD_LAYOUT, PromptComponent.ROBOT_SPECS, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.BASIC_RULES, PromptComponent.MOVEMENT_RULES, PromptComponent.ORIENTATION_RULES, PromptComponent.OBJECT_RULES, PromptComponent.PARAMETER_RULES, PromptComponent.PLANNING_PROCESS, PromptComponent.OUTPUT_FORMAT],
                description="Full prompt without arm procedure rules (11)"
            ),
            
            # 11. Without parameter rules
            PromptVariant(
                name="no_parameter_rules",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.CURRENT_STATE, PromptComponent.WORLD_LAYOUT, PromptComponent.ROBOT_SPECS, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.BASIC_RULES, PromptComponent.MOVEMENT_RULES, PromptComponent.ORIENTATION_RULES, PromptComponent.OBJECT_RULES, PromptComponent.ARM_RULES, PromptComponent.PLANNING_PROCESS, PromptComponent.OUTPUT_FORMAT],
                description="Full prompt without parameter rules (12-13, 15)"
            ),
            
            # 12. Full prompt (baseline) - with all individual components
            PromptVariant(
                name="full_prompt",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.CURRENT_STATE, PromptComponent.WORLD_LAYOUT, PromptComponent.ROBOT_SPECS, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.BASIC_RULES, PromptComponent.MOVEMENT_RULES, PromptComponent.ORIENTATION_RULES, PromptComponent.OBJECT_RULES, PromptComponent.ARM_RULES, PromptComponent.PARAMETER_RULES, PromptComponent.PLANNING_PROCESS, PromptComponent.OUTPUT_FORMAT],
                description="Complete original prompt (baseline)"
            ),
            
            # 13. Essential components only
            PromptVariant(
                name="essential_only",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.WORLD_LAYOUT, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.MOVEMENT_RULES, PromptComponent.ORIENTATION_RULES, PromptComponent.OUTPUT_FORMAT],
                description="Only essential components for navigation"
            ),
            
            # 14. Without basic rules
            PromptVariant(
                name="no_basic_rules",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.CURRENT_STATE, PromptComponent.WORLD_LAYOUT, PromptComponent.ROBOT_SPECS, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.MOVEMENT_RULES, PromptComponent.ORIENTATION_RULES, PromptComponent.OBJECT_RULES, PromptComponent.ARM_RULES, PromptComponent.PARAMETER_RULES, PromptComponent.PLANNING_PROCESS, PromptComponent.OUTPUT_FORMAT],
                description="Full prompt without basic rules (1-4)"
            ),
            
            # 15. No output format
            PromptVariant(
                name="no_output_format",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.CURRENT_STATE, PromptComponent.WORLD_LAYOUT, PromptComponent.ROBOT_SPECS, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.BASIC_RULES, PromptComponent.MOVEMENT_RULES, PromptComponent.ORIENTATION_RULES, PromptComponent.OBJECT_RULES, PromptComponent.ARM_RULES, PromptComponent.PARAMETER_RULES, PromptComponent.PLANNING_PROCESS],
                description="Full prompt without output format specification"
            ),
            
            # 16. Only movement and orientation rules
            PromptVariant(
                name="movement_orientation_only",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.MOVEMENT_RULES, PromptComponent.ORIENTATION_RULES, PromptComponent.OUTPUT_FORMAT],
                description="Only movement and orientation rules"
            ),
            
            # 17. Only object and arm rules
            PromptVariant(
                name="object_arm_only",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.OBJECT_RULES, PromptComponent.ARM_RULES, PromptComponent.OUTPUT_FORMAT],
                description="Only object and arm rules"
            ),
            
            # 18. State + Layout + Movement rules
            PromptVariant(
                name="state_layout_movement",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.CURRENT_STATE, PromptComponent.WORLD_LAYOUT, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.MOVEMENT_RULES, PromptComponent.OUTPUT_FORMAT],
                description="State, layout, and movement rules"
            ),
            
            # 19. Specs + Planning + Parameter rules
            PromptVariant(
                name="specs_planning_parameter",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.ROBOT_SPECS, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.PARAMETER_RULES, PromptComponent.PLANNING_PROCESS, PromptComponent.OUTPUT_FORMAT],
                description="Specs, planning, and parameter rules"
            ),
            
            # 20. Basic + Movement + Orientation + Object rules
            PromptVariant(
                name="core_rules_only",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.BASIC_RULES, PromptComponent.MOVEMENT_RULES, PromptComponent.ORIENTATION_RULES, PromptComponent.OBJECT_RULES, PromptComponent.OUTPUT_FORMAT],
                description="Core rules without state/layout/specs"
            ),
            
            # 21. State + Layout + Specs + Basic rules
            PromptVariant(
                name="info_basic_only",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.CURRENT_STATE, PromptComponent.WORLD_LAYOUT, PromptComponent.ROBOT_SPECS, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.BASIC_RULES, PromptComponent.OUTPUT_FORMAT],
                description="Information and basic rules only"
            ),
            
            # 22. Movement + Orientation + Arm + Parameter rules
            PromptVariant(
                name="detailed_rules_only",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.MOVEMENT_RULES, PromptComponent.ORIENTATION_RULES, PromptComponent.ARM_RULES, PromptComponent.PARAMETER_RULES, PromptComponent.OUTPUT_FORMAT],
                description="Detailed rules without basic rules"
            ),
            
            # 23. State + Layout + Movement + Orientation rules
            PromptVariant(
                name="navigation_focused",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.CURRENT_STATE, PromptComponent.WORLD_LAYOUT, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.MOVEMENT_RULES, PromptComponent.ORIENTATION_RULES, PromptComponent.OUTPUT_FORMAT],
                description="Navigation-focused prompt"
            ),
            
            # 24. Object + Arm + Parameter + Planning rules
            PromptVariant(
                name="manipulation_focused",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.OBJECT_RULES, PromptComponent.ARM_RULES, PromptComponent.PARAMETER_RULES, PromptComponent.PLANNING_PROCESS, PromptComponent.OUTPUT_FORMAT],
                description="Manipulation-focused prompt"
            ),
            
            # 25. All rules without state/layout/specs
            PromptVariant(
                name="rules_only",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.BASIC_RULES, PromptComponent.MOVEMENT_RULES, PromptComponent.ORIENTATION_RULES, PromptComponent.OBJECT_RULES, PromptComponent.ARM_RULES, PromptComponent.PARAMETER_RULES, PromptComponent.OUTPUT_FORMAT],
                description="All rules without contextual information"
            ),
            
            # 26. State + Layout + Specs + Planning + Output
            PromptVariant(
                name="context_planning_only",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.CURRENT_STATE, PromptComponent.WORLD_LAYOUT, PromptComponent.ROBOT_SPECS, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.PLANNING_PROCESS, PromptComponent.OUTPUT_FORMAT],
                description="Context and planning without specific rules"
            ),
            
            # 27. Basic + Movement + Object + Arm rules
            PromptVariant(
                name="essential_manipulation",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.BASIC_RULES, PromptComponent.MOVEMENT_RULES, PromptComponent.OBJECT_RULES, PromptComponent.ARM_RULES, PromptComponent.OUTPUT_FORMAT],
                description="Essential manipulation rules"
            ),
            
            # 28. Layout + Orientation + Object + Parameter rules
            PromptVariant(
                name="precision_focused",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.WORLD_LAYOUT, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.ORIENTATION_RULES, PromptComponent.OBJECT_RULES, PromptComponent.PARAMETER_RULES, PromptComponent.OUTPUT_FORMAT],
                description="Precision-focused prompt"
            ),
            
            # 29. State + Specs + Movement + Arm + Planning rules
            PromptVariant(
                name="technical_focused",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.CURRENT_STATE, PromptComponent.ROBOT_SPECS, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.MOVEMENT_RULES, PromptComponent.ARM_RULES, PromptComponent.PLANNING_PROCESS, PromptComponent.OUTPUT_FORMAT],
                description="Technical-focused prompt"
            ),
            
            # 30. All components except parameter rules
            PromptVariant(
                name="no_parameter_rules_v2",
                components=[PromptComponent.ROLE_DEFINITION, PromptComponent.CURRENT_STATE, PromptComponent.WORLD_LAYOUT, PromptComponent.ROBOT_SPECS, PromptComponent.AVAILABLE_ACTIONS, PromptComponent.BASIC_RULES, PromptComponent.MOVEMENT_RULES, PromptComponent.ORIENTATION_RULES, PromptComponent.OBJECT_RULES, PromptComponent.ARM_RULES, PromptComponent.PLANNING_PROCESS, PromptComponent.OUTPUT_FORMAT],
                description="All components except parameter rules"
            )
        ]
    
    def get_component_text(self, component: PromptComponent) -> str:
        """Get the text for a specific prompt component"""
        components = {
            PromptComponent.ROLE_DEFINITION: "You are a path planning system for a Boston Dynamics Spot robot.",
            
            PromptComponent.CURRENT_STATE: """CURRENT STATE:
- Robot state: {current_state}
- Current position (vision frame): {current_position}""",
            
            PromptComponent.WORLD_LAYOUT: """WORLD LAYOUT (vision frame coordinates):
- Pick-up location: (2.0, -1.0, 0.0) - Robot faces forward (yaw=0) when picking up
- Drop-off location: (2.0, 0.5, 0.0) - Robot faces left (yaw=1.57 radians) when dropping off""",
            
            PromptComponent.ROBOT_SPECS: """ROBOT SPECS:
- Size: 0.7m wide × 1.4m long
- Movement: Uses body frame for positioning
- Coordinate system: x=forward, y=left, yaw=rotation (body frame)""",
            
            PromptComponent.AVAILABLE_ACTIONS: """AVAILABLE ACTIONS:
- stand_up
- sit_down  
- start_moving, x=float, y=float, yaw=float, frame="body"
- get_image, image_source="camera_name"
- get_initial_pose
- start_automated_grasp, object_type="exact_object_name_from_user" (arm will be stowed after grasping)
- start_move_arm_pose, x=float, y=float, z=float, qw=float, qx=float, qy=float, qz=float, duration=float, open_gripper=bool
- start_arm_command, command_type=open|close|stow|carry""",
            
            PromptComponent.BASIC_RULES: """RULES:
1. Visit destinations in EXACT order specified by user
2. One action per line, no quotes/brackets, no numbering
3. Robot must be in standing mode before moving and grasping
4. Only use "stand_up" if robot is not already standing, moving or grasping""",
            
            PromptComponent.MOVEMENT_RULES: """5. In one move, the robot can either move in the x direction, the y direction, or the yaw direction, not two or three at once.
6. Use RELATIVE body frame coordinates for movements - each movement is relative to current position
7. For movements, calculate: move_x = target_x - current_x, move_y = target_y - current_y""",
            
            PromptComponent.ORIENTATION_RULES: """8. For pick-up: walk to pick-up location and face forward (yaw=0)
9. For drop-off: walk to drop-off location and face left (yaw=1.57 radians)""",
            
            PromptComponent.OBJECT_RULES: """10. Use EXACT object name from user command for object_type (e.g., "tomato can" not "tomato")
14. Use exact object names: "pringles can" not "pringles_can" """,
            
            PromptComponent.ARM_RULES: """11. For drop-off procedure: start_move_arm_pose to (0.8, 0.0, 0.3) with quaternion (0.7071, 0.7071, 0.0, 0.0) for gripper pointing down (this is the arm pose for dropping off objects) in 1 second, then start_arm_command open, then start_arm_command close, then start_arm_command stow""",
            
            PromptComponent.PARAMETER_RULES: """12. ALL movement parameters are REQUIRED: x=value, y=value, yaw=value, frame=body
13. Do NOT include movements with all zero values (x=0.0, y=0.0, yaw=0.0)
15. Rotations are allowed and valid - use yaw parameter for orientation changes""",
            
            PromptComponent.PLANNING_PROCESS: """PLANNING PROCESS:
1. Parse user command to identify destinations in order
2. Calculate relative body frame movements from current position to each destination
3. Plan path visiting each destination once in order
4. Ensure correct robot orientation at each destination (yaw=0 for pick-up, yaw=1.57 radians for drop-off)
5. For drop-off locations: add drop-off procedure (start_move_arm_pose, start_arm_command open, start_arm_command close, start_arm_command stow)
6. Generate action sequence with relative body frame movements""",
            
            PromptComponent.OUTPUT_FORMAT: """OUTPUT FORMAT:
Return only the action list, one action per line.

Task: {task}

Actions:"""
        }
        return components.get(component, "")
    
    def build_prompt(self, variant: PromptVariant) -> str:
        """Build a prompt from the specified components"""
        prompt_parts = []
        
        for component in variant.components:
            text = self.get_component_text(component)
            if text:
                prompt_parts.append(text)
        
        return "\n\n".join(prompt_parts)
    
    async def test_prompt_variant(self, variant: PromptVariant, task: Dict[str, str], run_number: int) -> TestResult:
        """Test a single prompt variant with a task"""
        start_time = time.time()
        
        try:
            prompt = self.build_prompt(variant)
            formatted_prompt = prompt.format(
                current_state="standing",
                current_position="(0.0, 0.0, 0.0)",
                task=task["task"]
            )
            
            # Use Together AI if available, otherwise fallback to Groq
            if hasattr(self, 'together_client') and self.together_client:
                response = self.together_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": formatted_prompt}],
                    max_tokens=2000,
                    temperature=0.0
                )
            else:
                response = self.groq_client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[{"role": "user", "content": formatted_prompt}],
                    max_tokens=2000,
                    temperature=0.0
                )
            result = response.choices[0].message.content.strip()
            
            # Extract token usage
            input_tokens = getattr(response.usage, 'prompt_tokens', 0) if hasattr(response, 'usage') and response.usage else 0
            output_tokens = getattr(response.usage, 'completion_tokens', 0) if hasattr(response, 'usage') and response.usage else 0
            total_tokens = getattr(response.usage, 'total_tokens', 0) if hasattr(response, 'usage') and response.usage else 0
            
            # Parse actions
            parsed_actions = self.parse_actions(result)
            
            # Get ground truth for this task type
            ground_truth_options = self.ground_truths[task["type"]]
            
            # Calculate score
            score = self.calculate_score(parsed_actions, ground_truth_options)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                variant_name=variant.name,
                components_used=[c.value for c in variant.components],
                run_number=run_number,
                mode="HA",
                task_type=task["type"],
                command=task["task"],
                response=result,
                parsed_actions=parsed_actions,
                execution_time=execution_time,
                success=len(parsed_actions) > 0,
                score=score,
                prompt=prompt,
                formatted_prompt=formatted_prompt,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            # Build prompt even for error case
            prompt = self.build_prompt(variant)
            formatted_prompt = prompt.format(
                current_state="standing",
                current_position="(0.0, 0.0, 0.0)",
                task=task["task"]
            )
            return TestResult(
                variant_name=variant.name,
                components_used=[c.value for c in variant.components],
                run_number=run_number,
                mode="HA",
                task_type=task["type"],
                command=task["task"],
                response="",
                parsed_actions=[],
                execution_time=execution_time,
                success=False,
                score=0.0,
                prompt=prompt,
                formatted_prompt=formatted_prompt,
                error_message=str(e),
                input_tokens=0,
                output_tokens=0,
                total_tokens=0
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
        """Normalize action for comparison with semantic understanding"""
        action = action.replace('"', '').replace("'", '')
        action = action.replace(' = ', '=').replace('= ', '=').replace(' =', '=')
        action = action.replace(' , ', ',').replace(', ', ',').replace(' ,', ',')
        action = action.strip()
        
        # Semantic normalization for better matching
        # Normalize object names (pringles_can, pringles can, Pringles can -> pringles can)
        if 'pringles' in action.lower():
            action = action.replace('pringles_can', 'pringles can')
            action = action.replace('Pringles can', 'pringles can')
        
        # Normalize command_type parameter (command_type=open -> open)
        if 'command_type=' in action:
            action = action.replace('command_type=', '')
        
        # Normalize frame parameter (frame="body" -> frame=body)
        action = action.replace('frame="body"', 'frame=body')
        
        return action
    
    def calculate_score(self, predicted_actions: List[str], ground_truth_options: List[List[str]]) -> float:
        """Calculate semantic score based on functional correctness"""
        if not predicted_actions or not ground_truth_options:
            return 0.0
        
        predicted_normalized = [self.normalize_action(action) for action in predicted_actions]
        best_score = 0.0
        
        for ground_truth in ground_truth_options:
            gt_normalized = [self.normalize_action(action) for action in ground_truth]
            
            # Semantic scoring components
            semantic_score = self._calculate_semantic_score(predicted_normalized, gt_normalized)
            best_score = max(best_score, semantic_score)
        
        return best_score
    
    def _calculate_semantic_score(self, predicted: List[str], ground_truth: List[str]) -> float:
        """Calculate semantic similarity score focusing on functional correctness"""
        
        # 1. Action type matching (movement, grasp, arm commands)
        predicted_types = self._extract_action_types(predicted)
        gt_types = self._extract_action_types(ground_truth)
        type_score = self._calculate_type_similarity(predicted_types, gt_types)
        
        # 2. Movement pattern matching (coordinates and directions)
        movement_score = self._calculate_movement_similarity(predicted, ground_truth)
        
        # 3. Sequence structure matching (grasp -> move -> drop pattern)
        sequence_score = self._calculate_sequence_similarity(predicted, ground_truth)
        
        # 4. Object handling matching (grasp and drop actions)
        object_score = self._calculate_object_handling_similarity(predicted, ground_truth)
        
        # 5. Return to start matching
        return_score = self._calculate_return_similarity(predicted, ground_truth)
        
        # Weighted combination focusing on functional correctness
        semantic_score = (
            type_score * 0.25 +           # Action types are important
            movement_score * 0.30 +        # Movement patterns are crucial
            sequence_score * 0.25 +        # Sequence structure matters
            object_score * 0.15 +          # Object handling is important
            return_score * 0.05            # Return is nice but not critical
        )
        
        return min(semantic_score, 1.0)  # Cap at 1.0
    
    def _extract_action_types(self, actions: List[str]) -> List[str]:
        """Extract action types from action strings"""
        types = []
        for action in actions:
            if 'start_moving' in action:
                types.append('movement')
            elif 'start_automated_grasp' in action:
                types.append('grasp')
            elif 'start_move_arm_pose' in action:
                types.append('arm_pose')
            elif 'start_arm_command' in action:
                types.append('arm_command')
            elif 'stand_up' in action or 'sit_down' in action:
                types.append('posture')
            elif 'get_initial_pose' in action:
                types.append('initialization')
        return types
    
    def _calculate_type_similarity(self, predicted_types: List[str], gt_types: List[str]) -> float:
        """Calculate similarity of action types"""
        if not gt_types:
            return 1.0 if not predicted_types else 0.0
        
        # Count each type
        pred_counts = {t: predicted_types.count(t) for t in set(predicted_types)}
        gt_counts = {t: gt_types.count(t) for t in set(gt_types)}
        
        # Calculate similarity for each type
        total_similarity = 0.0
        all_types = set(predicted_types) | set(gt_types)
        
        for action_type in all_types:
            pred_count = pred_counts.get(action_type, 0)
            gt_count = gt_counts.get(action_type, 0)
            
            if gt_count > 0:
                similarity = min(pred_count, gt_count) / max(pred_count, gt_count)
                total_similarity += similarity * (gt_count / len(gt_types))
        
        return total_similarity
    
    def _calculate_movement_similarity(self, predicted: List[str], ground_truth: List[str]) -> float:
        """Calculate similarity of movement patterns"""
        pred_movements = [a for a in predicted if 'start_moving' in a]
        gt_movements = [a for a in ground_truth if 'start_moving' in a]
        
        if not gt_movements:
            return 1.0 if not pred_movements else 0.0
        
        # Extract movement coordinates
        pred_coords = self._extract_movement_coords(pred_movements)
        gt_coords = self._extract_movement_coords(gt_movements)
        
        if not gt_coords:
            return 1.0
        
        # Calculate coordinate similarity (tolerance for small differences)
        coord_similarity = 0.0
        min_len = min(len(pred_coords), len(gt_coords))
        
        for i in range(min_len):
            pred_x, pred_y, pred_yaw = pred_coords[i]
            gt_x, gt_y, gt_yaw = gt_coords[i]
            
            # Calculate distance similarity (tolerance of 0.5 units)
            x_sim = max(0, 1 - abs(pred_x - gt_x) / 2.0)
            y_sim = max(0, 1 - abs(pred_y - gt_y) / 2.0)
            yaw_sim = max(0, 1 - abs(pred_yaw - gt_yaw) / 3.14)
            
            coord_similarity += (x_sim + y_sim + yaw_sim) / 3.0
        
        return coord_similarity / min_len if min_len > 0 else 0.0
    
    def _extract_movement_coords(self, movements: List[str]) -> List[tuple]:
        """Extract x, y, yaw coordinates from movement actions"""
        coords = []
        for movement in movements:
            try:
                # Extract x, y, yaw values
                x_match = re.search(r'x=([-\d.]+)', movement)
                y_match = re.search(r'y=([-\d.]+)', movement)
                yaw_match = re.search(r'yaw=([-\d.]+)', movement)
                
                if x_match and y_match and yaw_match:
                    x = float(x_match.group(1))
                    y = float(y_match.group(1))
                    yaw = float(yaw_match.group(1))
                    coords.append((x, y, yaw))
            except:
                continue
        return coords
    
    def _calculate_sequence_similarity(self, predicted: List[str], ground_truth: List[str]) -> float:
        """Calculate similarity of action sequence structure"""
        # Look for key sequence patterns: move -> grasp -> move -> orient -> drop -> return
        pred_sequence = self._extract_sequence_pattern(predicted)
        gt_sequence = self._extract_sequence_pattern(ground_truth)
        
        if not gt_sequence:
            return 1.0
        
        # Calculate sequence similarity
        matches = 0
        min_len = min(len(pred_sequence), len(gt_sequence))
        
        for i in range(min_len):
            if pred_sequence[i] == gt_sequence[i]:
                matches += 1
        
        return matches / len(gt_sequence) if len(gt_sequence) > 0 else 0.0
    
    def _extract_sequence_pattern(self, actions: List[str]) -> List[str]:
        """Extract sequence pattern from actions"""
        pattern = []
        for action in actions:
            if 'start_moving' in action:
                pattern.append('move')
            elif 'start_automated_grasp' in action:
                pattern.append('grasp')
            elif 'start_move_arm_pose' in action:
                pattern.append('arm_pose')
            elif 'start_arm_command' in action and 'open' in action:
                pattern.append('arm_open')
            elif 'start_arm_command' in action and 'close' in action:
                pattern.append('arm_close')
            elif 'start_arm_command' in action and 'stow' in action:
                pattern.append('arm_stow')
        return pattern
    
    def _calculate_object_handling_similarity(self, predicted: List[str], ground_truth: List[str]) -> float:
        """Calculate similarity of object handling (grasp and drop)"""
        pred_grasps = [a for a in predicted if 'start_automated_grasp' in a]
        gt_grasps = [a for a in ground_truth if 'start_automated_grasp' in a]
        
        if not gt_grasps:
            return 1.0 if not pred_grasps else 0.0
        
        # Check if grasp actions are present and similar
        grasp_score = 1.0 if pred_grasps else 0.0
        
        # Check object name similarity (normalized)
        if pred_grasps and gt_grasps:
            pred_obj = self._extract_object_name(pred_grasps[0])
            gt_obj = self._extract_object_name(gt_grasps[0])
            object_similarity = 1.0 if pred_obj == gt_obj else 0.5  # Partial credit for different formatting
            grasp_score = object_similarity
        
        return grasp_score
    
    def _extract_object_name(self, grasp_action: str) -> str:
        """Extract object name from grasp action"""
        try:
            obj_match = re.search(r'object_type=([^,\s]+)', grasp_action)
            if obj_match:
                return obj_match.group(1).lower().replace('_', ' ')
        except:
            pass
        return ""
    
    def _calculate_return_similarity(self, predicted: List[str], ground_truth: List[str]) -> float:
        """Calculate similarity of return to start position"""
        # Look for return movements (negative x values or return patterns)
        pred_returns = [a for a in predicted if 'start_moving' in a and ('x=-' in a or 'x=-' in a)]
        gt_returns = [a for a in ground_truth if 'start_moving' in a and ('x=-' in a or 'x=-' in a)]
        
        if not gt_returns:
            return 1.0  # No return expected
        
        return 1.0 if pred_returns else 0.0
    
    async def run_ablation_study(self, num_runs: int = 3) -> List[TestResult]:
        """Run ablation study with multiple runs for each combination"""
        results = []
        
        total_tests = len(self.prompt_variants) * len(self.test_tasks) * num_runs
        print(f"Running ablation study: {len(self.prompt_variants)} variants × {len(self.test_tasks)} hard task × {num_runs} runs = {total_tests} total tests")
        print(f"Using GPT-OSS-120B model")
        
        for variant in self.prompt_variants:
            print(f"\nTesting variant: {variant.name}")
            print(f"  Description: {variant.description}")
            print(f"  Components: {', '.join([c.value for c in variant.components])}")
            
            for task in self.test_tasks:
                print(f"    Task: {task['type']} - {task['description']}")
                
                for run in range(1, num_runs + 1):
                    result = await self.test_prompt_variant(variant, task, run)
                    results.append(result)
                    
                    if result.success:
                        print(f"      Run {run}: ✓ Success ({result.execution_time:.2f}s) - Score: {result.score:.2f} - {len(result.parsed_actions)} actions")
                    else:
                        print(f"      Run {run}: ✗ Failed: {result.error_message}")
        
        return results
    
    def save_results(self, results: List[TestResult], filename: str = "ha_prompt_ablation_results.json"):
        """Save test results to JSON file with detailed plan information"""
        data = []
        for result in results:
            # Create detailed plan summary
            final_plan = {
                "total_actions": len(result.parsed_actions),
                "action_sequence": result.parsed_actions,
                "movement_actions": [action for action in result.parsed_actions if 'start_moving' in action],
                "grasp_actions": [action for action in result.parsed_actions if 'start_automated_grasp' in action],
                "arm_actions": [action for action in result.parsed_actions if 'start_move_arm_pose' in action or 'start_arm_command' in action],
                "action_types": {
                    "movement_count": len([action for action in result.parsed_actions if 'start_moving' in action]),
                    "grasp_count": len([action for action in result.parsed_actions if 'start_automated_grasp' in action]),
                    "arm_pose_count": len([action for action in result.parsed_actions if 'start_move_arm_pose' in action]),
                    "arm_command_count": len([action for action in result.parsed_actions if 'start_arm_command' in action])
                }
            }
            
            data.append({
                "variant_name": result.variant_name,
                "components_used": result.components_used,
                "run_number": result.run_number,
                "task_type": result.task_type,
                "command": result.command,
                "response": result.response,
                "parsed_actions": result.parsed_actions,
                "execution_time": result.execution_time,
                "success": result.success,
                "score": result.score,
                "error_message": result.error_message,
                "prompt": result.prompt,
                "formatted_prompt": result.formatted_prompt,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
                "final_plan": final_plan,
                "ground_truth_options": self.ground_truths.get(result.task_type, [])
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def save_detailed_plans(self, results: List[TestResult], filename: str = "detailed_plans.json"):
        """Save detailed plan information in a separate JSON file"""
        plans_data = []
        
        # Group results by variant and run
        by_variant = {}
        for result in results:
            variant = result.variant_name
            if variant not in by_variant:
                by_variant[variant] = []
            by_variant[variant].append(result)
        
        for variant, variant_results in by_variant.items():
            variant_plan = {
                "variant_name": variant,
                "components_used": variant_results[0].components_used,
                "runs": []
            }
            
            for result in variant_results:
                run_plan = {
                    "run_number": result.run_number,
                    "success": result.success,
                    "score": result.score,
                    "execution_time": result.execution_time,
                    "total_actions": len(result.parsed_actions),
                    "action_sequence": result.parsed_actions,
                    "movement_actions": [action for action in result.parsed_actions if 'start_moving' in action],
                    "grasp_actions": [action for action in result.parsed_actions if 'start_automated_grasp' in action],
                    "arm_actions": [action for action in result.parsed_actions if 'start_move_arm_pose' in action or 'start_arm_command' in action],
                    "raw_response": result.response,
                    "prompt": result.prompt,
                    "formatted_prompt": result.formatted_prompt,
                    "error_message": result.error_message,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "total_tokens": result.total_tokens
                }
                variant_plan["runs"].append(run_plan)
            
            plans_data.append(variant_plan)
        
        with open(filename, 'w') as f:
            json.dump(plans_data, f, indent=2)
        
        print(f"Detailed plans saved to {filename}")
    
    def generate_markdown_table(self, results: List[TestResult], filename: str = "ablation_study_results.md"):
        """Generate markdown table with ablation study results"""
        
        # Group results by variant
        by_variant = {}
        for result in results:
            variant = result.variant_name
            if variant not in by_variant:
                by_variant[variant] = []
            by_variant[variant].append(result)
        
        # Calculate statistics for each variant
        variant_stats = {}
        for variant, variant_results in by_variant.items():
            successful = [r for r in variant_results if r.success]
            if successful:
                avg_score = sum(r.score for r in successful) / len(successful)
                avg_time = sum(r.execution_time for r in variant_results) / len(variant_results)
                avg_steps = sum(len(r.parsed_actions) for r in successful) / len(successful)
                success_rate = len(successful) / len(variant_results)
                
                # Calculate token statistics
                avg_input_tokens = sum(r.input_tokens for r in variant_results) / len(variant_results)
                avg_output_tokens = sum(r.output_tokens for r in variant_results) / len(variant_results)
                avg_total_tokens = sum(r.total_tokens for r in variant_results) / len(variant_results)
                
                variant_stats[variant] = {
                    'avg_score': avg_score,
                    'avg_time': avg_time,
                    'avg_steps': avg_steps,
                    'success_rate': success_rate,
                    'avg_input_tokens': avg_input_tokens,
                    'avg_output_tokens': avg_output_tokens,
                    'avg_total_tokens': avg_total_tokens,
                    'components': variant_results[0].components_used
                }
        
        # Create markdown table
        markdown_content = "# HA Prompt Ablation Study Results\n\n"
        markdown_content += "## Overview\n\n"
        markdown_content += f"- **Model**: GPT-OSS-120B\n"
        markdown_content += f"- **Task**: Hard (Complex multi-step task)\n"
        markdown_content += f"- **Total Tests**: {len(results)}\n"
        markdown_content += f"- **Variants Tested**: {len(variant_stats)}\n"
        markdown_content += f"- **Runs per Combination**: 3\n\n"
        
        markdown_content += "## Results Table\n\n"
        markdown_content += "**Legend:** ✓ = Component used, ✗ = Component not used\n\n"
        
        # Get all unique components
        all_components = set()
        for variant, stats in variant_stats.items():
            all_components.update(stats['components'])
        all_components = sorted(list(all_components))
        
        # Create component name mapping for shorter names
        component_names = {
            'role_definition': 'Role',
            'current_state': 'State',
            'world_layout': 'Layout',
            'robot_specs': 'Specs',
            'available_actions': 'Actions',
            'rules': 'Rules',
            'planning_process': 'Planning',
            'output_format': 'Output',
            'basic_rules': 'Basic',
            'movement_rules': 'Movement',
            'orientation_rules': 'Orientation',
            'object_rules': 'Object',
            'arm_rules': 'Arm',
            'parameter_rules': 'Parameter'
        }
        
        # Create header row
        header = "| Variant |"
        for component in all_components:
            short_name = component_names.get(component, component)
            header += f" {short_name} |"
        header += " Avg Score | Avg Time (s) | Avg Steps | Success Rate | Input Tokens | Output Tokens | Total Tokens |\n"
        markdown_content += header
        
        # Create separator row
        separator = "|---------|"
        for component in all_components:
            separator += "---|"
        separator += "-----------|--------------|-----------|--------------|--------------|---------------|--------------|\n"
        markdown_content += separator
        
        # Sort by average score
        sorted_variants = sorted(variant_stats.items(), key=lambda x: x[1]['avg_score'], reverse=True)
        
        for variant, stats in sorted_variants:
            row = f"| {variant} |"
            
            # Add checkmarks/crosses for each component
            for component in all_components:
                if component in stats['components']:
                    row += " ✓ |"
                else:
                    row += " ✗ |"
            
            # Add performance metrics
            row += f" {stats['avg_score']:.3f} | {stats['avg_time']:.2f} | {stats['avg_steps']:.1f} | {stats['success_rate']:.1%} | {stats['avg_input_tokens']:.0f} | {stats['avg_output_tokens']:.0f} | {stats['avg_total_tokens']:.0f} |\n"
            markdown_content += row
        
        markdown_content += "\n## Component Definitions\n\n"
        markdown_content += "| Component | Full Name | Description |\n"
        markdown_content += "|-----------|-----------|-------------|\n"
        markdown_content += "| Role | Role Definition | Defines the robot's role as path planner |\n"
        markdown_content += "| State | Current State | Robot's current position and status |\n"
        markdown_content += "| Layout | World Layout | Pick-up and drop-off locations |\n"
        markdown_content += "| Specs | Robot Specs | Robot dimensions and coordinate system |\n"
        markdown_content += "| Actions | Available Actions | List of possible robot actions |\n"
        markdown_content += "| Rules | All Rules | Complete set of behavioral rules |\n"
        markdown_content += "| Planning | Planning Process | Step-by-step planning instructions |\n"
        markdown_content += "| Output | Output Format | How to format the response |\n"
        markdown_content += "| Basic | Basic Rules | Core rules (1-4) |\n"
        markdown_content += "| Movement | Movement Rules | Movement-specific rules (5-7) |\n"
        markdown_content += "| Orientation | Orientation Rules | Orientation rules (8-9) |\n"
        markdown_content += "| Object | Object Rules | Object handling rules (10, 14) |\n"
        markdown_content += "| Arm | Arm Rules | Arm procedure rules (11) |\n"
        markdown_content += "| Parameter | Parameter Rules | Parameter formatting rules (12-13, 15) |\n\n"
        
        markdown_content += "## Component Analysis\n\n"
        
        # Analyze component importance
        component_importance = {}
        for variant, stats in variant_stats.items():
            for component in stats['components']:
                if component not in component_importance:
                    component_importance[component] = []
                component_importance[component].append(stats['avg_score'])
        
        # Calculate average scores for each component
        component_avg_scores = {}
        for component, scores in component_importance.items():
            component_avg_scores[component] = sum(scores) / len(scores)
        
        # Sort components by importance
        sorted_components = sorted(component_avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        markdown_content += "### Component Importance (Average Score)\n\n"
        for component, avg_score in sorted_components:
            markdown_content += f"- **{component}**: {avg_score:.3f}\n"
        
        markdown_content += "\n## Best Performing Combinations\n\n"
        markdown_content += "### Top 5 Variants by Score\n\n"
        for i, (variant, stats) in enumerate(sorted_variants[:5], 1):
            markdown_content += f"{i}. **{variant}** (Score: {stats['avg_score']:.3f})\n"
            markdown_content += f"   - Components: {', '.join(stats['components'])}\n"
            markdown_content += f"   - Avg Time: {stats['avg_time']:.2f}s\n"
            markdown_content += f"   - Avg Steps: {stats['avg_steps']:.1f}\n"
            markdown_content += f"   - Success Rate: {stats['success_rate']:.1%}\n"
            markdown_content += f"   - Avg Input Tokens: {stats['avg_input_tokens']:.0f}\n"
            markdown_content += f"   - Avg Output Tokens: {stats['avg_output_tokens']:.0f}\n"
            markdown_content += f"   - Avg Total Tokens: {stats['avg_total_tokens']:.0f}\n\n"
        
        # Save markdown file
        with open(filename, 'w') as f:
            f.write(markdown_content)
        
        print(f"\nMarkdown table saved to {filename}")
    
    def print_summary(self, results: List[TestResult]):
        """Print comprehensive analysis of results"""
        print("\n" + "="*80)
        print("HA PROMPT ABLATION STUDY ANALYSIS")
        print("="*80)
        
        # Group by variant
        by_variant = {}
        for result in results:
            variant = result.variant_name
            if variant not in by_variant:
                by_variant[variant] = []
            by_variant[variant].append(result)
        
        # Overall performance by variant
        print("\nVARIANT PERFORMANCE:")
        variant_scores = {}
        for variant, variant_results in by_variant.items():
            successful = [r for r in variant_results if r.success]
            if successful:
                avg_score = sum(r.score for r in successful) / len(successful)
                success_rate = len(successful) / len(variant_results)
                avg_time = sum(r.execution_time for r in variant_results) / len(variant_results)
                avg_steps = sum(len(r.parsed_actions) for r in successful) / len(successful)
                variant_scores[variant] = avg_score
                
                # Calculate token statistics
                avg_input_tokens = sum(r.input_tokens for r in variant_results) / len(variant_results)
                avg_output_tokens = sum(r.output_tokens for r in variant_results) / len(variant_results)
                avg_total_tokens = sum(r.total_tokens for r in variant_results) / len(variant_results)
                
                print(f"  {variant}:")
                print(f"    Success Rate: {len(successful)}/{len(variant_results)} ({success_rate*100:.1f}%)")
                print(f"    Avg Score: {avg_score:.3f}")
                print(f"    Avg Time: {avg_time:.2f}s")
                print(f"    Avg Steps: {avg_steps:.1f}")
                print(f"    Avg Input Tokens: {avg_input_tokens:.0f}")
                print(f"    Avg Output Tokens: {avg_output_tokens:.0f}")
                print(f"    Avg Total Tokens: {avg_total_tokens:.0f}")
            else:
                print(f"  {variant}: No successful tests")
        
        # Component importance analysis
        print(f"\n{'='*60}")
        print("COMPONENT IMPORTANCE ANALYSIS")
        print("="*60)
        
        component_importance = {}
        for variant, variant_results in by_variant.items():
            if variant_results:
                avg_score = sum(r.score for r in variant_results if r.success) / max(len([r for r in variant_results if r.success]), 1)
                components = variant_results[0].components_used
                
                for component in components:
                    if component not in component_importance:
                        component_importance[component] = []
                    component_importance[component].append(avg_score)
        
        # Calculate average scores for each component
        component_avg_scores = {}
        for component, scores in component_importance.items():
            component_avg_scores[component] = sum(scores) / len(scores)
        
        # Sort components by importance
        sorted_components = sorted(component_avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\nComponent Importance (higher = better performance):")
        for component, avg_score in sorted_components:
            print(f"  {component}: {avg_score:.3f}")
        
        # Recommendations
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print("="*60)
        
        # Find the best performing variants
        best_variants = sorted(variant_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\nTop 5 Performing Variants:")
        for i, (variant, score) in enumerate(best_variants, 1):
            print(f"  {i}. {variant}: {score:.3f}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run HA prompt ablation study')
    
    parser.add_argument(
        '--runs',
        type=int,
        default=3,
        help='Number of runs per combination (default: 3)'
    )
    
    parser.add_argument(
        '--output',
        default='ha_prompt_ablation_results.json',
        help='Output file for results (default: ha_prompt_ablation_results.json)'
    )
    
    parser.add_argument(
        '--markdown',
        default='ablation_study_results.md',
        help='Output markdown file for results table (default: ablation_study_results.md)'
    )
    
    return parser.parse_args()

async def main():
    """Main function to run ablation study"""
    args = parse_arguments()
    
    print("Starting HA Prompt Ablation Study")
    print("="*50)
    print(f"Number of runs per combination: {args.runs}")
    print(f"Output file: {args.output}")
    print(f"Markdown file: {args.markdown}")
    print()
    
    # Check environment variables
    if not os.getenv('GROQ_API_KEY'):
        print("ERROR: GROQ_API_KEY environment variable not set")
        return
    
    # Create tester and run ablation study
    tester = HAPromptAblationStudy()
    results = await tester.run_ablation_study(args.runs)
    
    # Save and display results
    tester.save_results(results, args.output)
    tester.save_detailed_plans(results, "detailed_plans.json")
    tester.generate_markdown_table(results, args.markdown)
    tester.print_summary(results)
    
    print("\nAblation study completed!")

if __name__ == "__main__":
    asyncio.run(main())