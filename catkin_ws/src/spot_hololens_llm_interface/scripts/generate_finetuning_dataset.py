#!/usr/bin/env python3
"""
Dataset Generator for LLM Fine-tuning
Generates training and validation datasets with prompt-completion pairs
for robot path planning across 10 different world configurations.
"""

import json
import random
import math
import sys
import os
from pathlib import Path

# Add path to import world_manager
sys.path.append(os.path.dirname(__file__))
from world_manager import WorldManager

# The prompt template from nl_control_gemini_no_world.py
PROMPT_TEMPLATE = """You are a path planning and action sequencing system for a Boston Dynamics Spot robot.

ACTIONS (one per line, no quotes, no numbering):
- stand_up
- sit_down
- start_moving x=<float> y=<float> yaw=<float> frame=body
- start_automated_grasp object_type=<exact_object_name_from_user>
- start_drop_off

RULES:
1. Execute destinations in exact order. Output only actions, one per line, no comments.
2. CRITICAL: If robot state is "Stand", NEVER use stand_up. Only use stand_up if "Sit" or "Powered off".
3. Use relative body-frame moves. One degree of freedom per move (x OR y OR yaw, not multiple).
4. For navigation: resolve locations via WORLD_MODEL.names/synonyms, then plan axis-aligned moves in body frame.
5. For PICK tasks: navigate to pickup location, rotate to pickup's pick_yaw (absolute direction), then use start_automated_grasp with exact object name (e.g. "tomato can").
6. For DROP tasks: navigate to drop-off zone, rotate to zone's yaw_hint (absolute direction), then use start_drop_off.
7. IMPORTANT: pick_yaw and yaw_hint are ABSOLUTE directions in world frame, calculate relative rotation from current robot yaw.
8. Use exact user object names. If location/object unresolved: output "FAIL unresolved=<what>"

PLANNING:
A. Parse TASK into ordered subgoals (pick, place, visit, look).
B. Resolve locations via WORLD_MODEL. Use canonical names.
C. For each subgoal: 
   - Navigate with axis-aligned moves to target location
   - Calculate required rotation: target_yaw - current_yaw (use CURRENT_STATE.pose_vision.yaw)
   - For PICK: target_yaw = waypoint's pick_yaw (absolute)
   - For DROP: target_yaw = zone's yaw_hint (absolute)
   - Rotate using calculated relative yaw, then execute pick/drop action
D. Don't repeat satisfied locations unless order requires revisit.

CURRENT_STATE = {current_state_json}
WORLD_MODEL = {world_model_json}
TASK = {command}"""


class DatasetGenerator:
    """Generate training/validation datasets for robot path planning."""
    
    def __init__(self, seed=42):
        self.wm = WorldManager()
        random.seed(seed)
        
        # Task templates - mix of casual and technical language
        self.task_templates = {
            'casual_pickup': [
                "grab the {object}",
                "pick up the {object}",
                "get me the {object}",
                "fetch the {object}",
                "could you bring me the {object}",
                "I need the {object}",
            ],
            'casual_dropoff': [
                "drop it at {zone}",
                "bring it to {zone}",
                "take it to the {zone}",
                "put it at {zone}",
                "deliver it to {zone}",
            ],
            'casual_combined': [
                "pick up the {object} and drop it at {zone}",
                "get the {object} and bring it to {zone}",
                "fetch the {object} and take it to the {zone}",
                "grab the {object} and deliver it to {zone}",
                "bring the {object} to the {zone}",
            ],
            'technical_pickup': [
                "execute pickup of {object}",
                "navigate to and grasp {object}",
                "retrieve {object}",
                "acquire {object}",
            ],
            'technical_dropoff': [
                "navigate to {zone} and execute drop-off",
                "transport to {zone}",
                "deliver to {zone}",
            ],
            'technical_combined': [
                "execute pickup of {object} and deliver to {zone}",
                "retrieve {object} and transport to {zone}",
                "acquire {object} and navigate to {zone} for drop-off",
            ],
            'urgent': [
                "quickly get the {object} to {zone}!",
                "fast! bring the {object} to {zone}",
                "{object} to {zone} - urgent!",
            ],
            'sequential': [
                "first pick up the {object1}, then get the {object2}, and drop them at {zone}",
                "grab the {object1} and {object2} and bring them to {zone}",
            ]
        }
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def generate_starting_position(self):
        """Generate a random but reasonable starting position."""
        positions = [
            (0.0, 0.0, 0.0),  # Origin
            (1.0, 0.0, 0.0),  # 1m forward
            (2.0, 0.0, 0.0),  # 2m forward
            (0.0, 1.0, 0.0),  # 1m left
            (0.0, -1.0, 0.0), # 1m right
            (1.0, 1.0, 0.0),  # Forward-left
            (1.0, -1.0, 0.0), # Forward-right
            (2.0, 1.0, 0.0),  # Further forward-left
            (-1.0, 0.0, 0.0), # Backward
            (0.0, 0.0, 1.57), # Origin, facing left
            (0.0, 0.0, -1.57),# Origin, facing right
            (1.0, 0.5, 0.0),  # Slight forward-left
            (1.5, -0.5, 0.0), # Forward-right
        ]
        return random.choice(positions)
    
    def generate_robot_state(self, position=None):
        """Generate a robot state JSON."""
        if position is None:
            position = self.generate_starting_position()
        
        x, y, yaw = position
        
        return {
            "robot_state": "Stand",
            "standing": True,
            "pose_vision": {
                "x": x,
                "y": y,
                "yaw": yaw
            },
            "arm": "stowed",
            "held_object": None
        }
    
    def get_objects_for_waypoint(self, waypoint_name):
        """Get object names for a pickup waypoint."""
        objects_map = {
            "PICKUP_BEVERAGES": ["water_bottle", "soda_can", "juice_box"],
            "PICKUP_PRODUCE": ["apple", "banana", "orange"],
            "PICKUP_DAIRY": ["milk_carton", "cheddar_block", "yogurt_cup"],
            "PICKUP_INCOMING": ["pallet", "shipping_box", "barcode_label"],
            "PICKUP_ELECTRONICS": ["circuit_board", "power_supply", "hdmi_cable"],
            "PICKUP_TEXTILES": ["fabric_roll", "cotton_bale", "yarn_spool"],
            "PICKUP_TOOLS": ["wrench", "screwdriver", "pliers"],
            "PICKUP_STATIONERY": ["pen", "pencil", "marker"],
            "PICKUP_OFFICE_SUPPLIES": ["stapler", "paper_clips", "folder"],
            "PICKUP_COMPUTERS": ["laptop", "keyboard", "mouse"],
            "PICKUP_COFFEE": ["coffee_cup", "espresso_pod", "sugar_packet"],
            "PICKUP_MEDICATIONS": ["pill_bottle", "syringe", "bandage"],
            "PICKUP_PPE": ["gloves_box", "face_mask", "sanitizer_bottle"],
            "PICKUP_EMERGENCY_SUPPLIES": ["first_aid_kit", "defibrillator", "oxygen_tank"],
            "PICKUP_INGREDIENTS": ["flour_bag", "tomatoes", "spices_jar"],
            "PICKUP_KNIVES": ["chef_knife", "paring_knife", "honing_rod"],
            "PICKUP_UTENSILS": ["fork", "spoon", "tongs"],
            "PICKUP_GLASSWARE": ["test_tube", "beaker", "vial"],
            "PICKUP_MICROSCOPES": ["microscope", "glass_slide", "cover_slip"],
            "PICKUP_SAMPLES": ["blood_sample", "tissue_sample", "culture_dish"],
            "PICKUP_ELECTRONICS_PHONES": ["smartphone", "charger", "earbuds"],
            "PICKUP_APPAREL_TOPS": ["shirt", "t_shirt", "jacket"],
            "PICKUP_TOYS": ["toy_car", "puzzle_box", "plush_bear"],
            "PICKUP_SECURITY_ITEMS": ["security_bin", "tray", "belt_bucket"],
            "PICKUP_TICKETING": ["boarding_pass", "luggage_tag", "passport"],
            "PICKUP_MAPS_INFO": ["terminal_map", "brochure", "guide"],
            "PICKUP_HANDTOOLS": ["hammer", "tape_measure", "chisel"],
            "PICKUP_LUMBER": ["wood_plank", "timber_beam", "plywood_sheet"],
            "PICKUP_SAFETY_HELMETS": ["hard_hat", "safety_vest", "ear_protectors"],
        }
        return objects_map.get(waypoint_name, ["item"])
    
    def generate_action_sequence(self, current_state, world_model, task_info):
        """Generate the ground truth action sequence."""
        actions = []
        robot_x = current_state["pose_vision"]["x"]
        robot_y = current_state["pose_vision"]["y"]
        robot_yaw = current_state["pose_vision"]["yaw"]
        
        # Extract task type and parameters
        task_type = task_info["type"]
        
        if task_type == "pickup_dropoff":
            # Navigate to pickup location
            pickup_waypoint = task_info["pickup_waypoint"]
            waypoint_data = world_model["waypoints"][pickup_waypoint]
            target_x = waypoint_data["x"]
            target_y = waypoint_data["y"]
            pick_yaw = waypoint_data["pick_yaw"]
            
            # Generate movement to pickup
            dx = target_x - robot_x
            dy = target_y - robot_y
            
            # Move in body frame (axis-aligned)
            if abs(dx) > 0.1:
                actions.append(f"start_moving x={dx:.1f} y=0.0 yaw=0.0 frame=body")
                robot_x = target_x
            
            if abs(dy) > 0.1:
                actions.append(f"start_moving x=0.0 y={dy:.1f} yaw=0.0 frame=body")
                robot_y = target_y
            
            # Rotate to pick_yaw
            yaw_diff = self.normalize_angle(pick_yaw - robot_yaw)
            if abs(yaw_diff) > 0.05:
                actions.append(f"start_moving x=0.0 y=0.0 yaw={yaw_diff:.2f} frame=body")
                robot_yaw = pick_yaw
            
            # Grasp object
            object_name = task_info["object"]
            actions.append(f"start_automated_grasp object_type={object_name}")
            
            # Navigate to drop-off zone
            dropoff_zone = task_info["dropoff_zone"]
            zone_data = world_model["zones"][dropoff_zone]
            target_x = zone_data["centroid"]["x"]
            target_y = zone_data["centroid"]["y"]
            drop_yaw = zone_data["yaw_hint"]
            
            # Generate movement to dropoff
            dx = target_x - robot_x
            dy = target_y - robot_y
            
            if abs(dx) > 0.1:
                actions.append(f"start_moving x={dx:.1f} y=0.0 yaw=0.0 frame=body")
                robot_x = target_x
            
            if abs(dy) > 0.1:
                actions.append(f"start_moving x=0.0 y={dy:.1f} yaw=0.0 frame=body")
                robot_y = target_y
            
            # Rotate to drop_yaw
            yaw_diff = self.normalize_angle(drop_yaw - robot_yaw)
            if abs(yaw_diff) > 0.05:
                actions.append(f"start_moving x=0.0 y=0.0 yaw={yaw_diff:.2f} frame=body")
            
            # Drop object
            actions.append("start_drop_off")
        
        return actions
    
    def generate_task_description(self, world_id, world_config):
        """Generate a natural language task and its parameters."""
        waypoints = world_config["waypoints"]
        zones = world_config["zones"]
        
        if not waypoints or not zones:
            return None
        
        # Randomly choose task style
        style = random.choice(['casual_combined', 'technical_combined', 'urgent', 
                              'casual_pickup', 'casual_dropoff'])
        
        # Pick a random waypoint and zone
        pickup_waypoint = random.choice(list(waypoints.keys()))
        dropoff_zone = random.choice(list(zones.keys()))
        
        # Get a random object from this waypoint
        objects = self.get_objects_for_waypoint(pickup_waypoint)
        object_name = random.choice(objects)
        
        # Generate task description based on style
        if 'combined' in style:
            templates = self.task_templates[style]
            task = random.choice(templates).format(
                object=object_name.replace('_', ' '),
                zone=dropoff_zone.lower()
            )
        elif 'urgent' in style:
            templates = self.task_templates[style]
            task = random.choice(templates).format(
                object=object_name.replace('_', ' '),
                zone=dropoff_zone.lower()
            )
        else:
            # For pickup-only or dropoff-only, create combined task
            pickup_templates = self.task_templates.get('casual_pickup', [])
            dropoff_templates = self.task_templates.get('casual_dropoff', [])
            
            pickup_part = random.choice(pickup_templates).format(
                object=object_name.replace('_', ' ')
            )
            dropoff_part = random.choice(dropoff_templates).format(
                zone=dropoff_zone.lower()
            )
            task = f"{pickup_part} and {dropoff_part}"
        
        return {
            "task": task,
            "type": "pickup_dropoff",
            "pickup_waypoint": pickup_waypoint,
            "dropoff_zone": dropoff_zone,
            "object": object_name
        }
    
    def generate_example(self, world_id=None):
        """Generate a complete training example."""
        # Select world
        if world_id is None:
            world_id = random.choice(list(self.wm.worlds.keys()))
        
        world_config = self.wm.get_world_config(world_id)
        if not world_config:
            return None
        
        # Generate starting position
        position = self.generate_starting_position()
        current_state = self.generate_robot_state(position)
        
        # Generate task
        task_info = self.generate_task_description(world_id, world_config)
        if not task_info:
            return None
        
        # Generate action sequence (ground truth)
        actions = self.generate_action_sequence(current_state, world_config, task_info)
        
        # Format prompt
        prompt = PROMPT_TEMPLATE.format(
            current_state_json=json.dumps(current_state, indent=2),
            world_model_json=json.dumps(world_config, indent=2),
            command=task_info["task"]
        )
        
        # Format completion
        completion = "\n".join(actions)
        
        return {
            "prompt": prompt,
            "completion": completion,
            "metadata": {
                "world_id": world_id,
                "world_name": self.wm.worlds[world_id]["name"],
                "starting_position": position,
                "task": task_info["task"]
            }
        }
    
    def generate_dataset(self, num_train=900, num_val=100):
        """Generate complete training and validation datasets."""
        print(f"Generating {num_train} training examples and {num_val} validation examples...")
        
        train_examples = []
        val_examples = []
        
        # Calculate examples per world for training
        train_per_world = num_train // 10
        
        # Generate training examples (distribute across all worlds)
        for world_id in self.wm.worlds.keys():
            print(f"  Generating examples for world {world_id} ({self.wm.worlds[world_id]['name']})...")
            for i in range(train_per_world):
                example = self.generate_example(world_id)
                if example:
                    train_examples.append(example)
        
        # Generate additional training examples to reach target
        while len(train_examples) < num_train:
            example = self.generate_example()
            if example:
                train_examples.append(example)
        
        # Generate validation examples (random worlds)
        print("  Generating validation examples...")
        for i in range(num_val):
            example = self.generate_example()
            if example:
                val_examples.append(example)
        
        return train_examples, val_examples
    
    def save_jsonl(self, examples, filepath):
        """Save examples to JSONL file."""
        with open(filepath, 'w') as f:
            for example in examples:
                # Together AI format with messages
                entry = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a path planning system for a Boston Dynamics Spot robot."
                        },
                        {
                            "role": "user",
                            "content": example["prompt"]
                        },
                        {
                            "role": "assistant",
                            "content": example["completion"]
                        }
                    ]
                }
                f.write(json.dumps(entry) + '\n')
        
        print(f"  Saved {len(examples)} examples to {filepath}")
    
    def save_metadata(self, train_examples, val_examples, filepath):
        """Save dataset metadata."""
        metadata = {
            "num_train": len(train_examples),
            "num_val": len(val_examples),
            "num_worlds": len(self.wm.worlds),
            "worlds": {
                world_id: {
                    "name": world_info["name"],
                    "description": world_info["description"]
                }
                for world_id, world_info in self.wm.worlds.items()
            },
            "train_examples_preview": [ex["metadata"] for ex in train_examples[:10]],
            "val_examples_preview": [ex["metadata"] for ex in val_examples[:5]]
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Saved metadata to {filepath}")


def main():
    """Main function to generate dataset."""
    print("="*80)
    print("LLM Fine-tuning Dataset Generator")
    print("="*80)
    print()
    
    # Create output directory
    output_dir = Path(__file__).parent / "finetuning_data"
    output_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = DatasetGenerator(seed=42)
    
    # Generate datasets
    print("Generating datasets...")
    train_examples, val_examples = generator.generate_dataset(num_train=900, num_val=100)
    
    print()
    print("Saving datasets...")
    
    # Save JSONL files
    generator.save_jsonl(train_examples, output_dir / "train.jsonl")
    generator.save_jsonl(val_examples, output_dir / "validation.jsonl")
    
    # Save metadata
    generator.save_metadata(train_examples, val_examples, output_dir / "dataset_metadata.json")
    
    print()
    print("="*80)
    print("Dataset generation complete!")
    print("="*80)
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    print(f"Output directory: {output_dir}")
    print()
    print("Files created:")
    print(f"  - train.jsonl ({len(train_examples)} examples)")
    print(f"  - validation.jsonl ({len(val_examples)} examples)")
    print(f"  - dataset_metadata.json (dataset information)")
    print()
    print("You can now upload these files to Together AI for fine-tuning!")
    print("="*80)


if __name__ == "__main__":
    main()


