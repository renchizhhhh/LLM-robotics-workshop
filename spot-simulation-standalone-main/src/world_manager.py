#!/usr/bin/env python3
"""
World Manager for Spot Robot - Provides 10 different world configurations
"""

import json
import os
from pathlib import Path

class WorldManager:
    """Manages different world configurations for the Spot robot."""
    
    def __init__(self):
        self.worlds = self._create_world_configurations()
        self.custom_worlds = {}
        self.load_custom_worlds()
    
    
    def _create_world_configurations(self):
        """Create 3 different world configurations with grid-based coordinates."""
        return {
            "1": {
                "name": "Simple Pick & Drop",
                "description": "Basic two-location scenario for testing",
                "config": {
                    "waypoints": {
                        "PICKUP_BEVERAGES": {"row": 2, "col": 3, "pick_direction": "N"}
                    },
                    "zones": {
                        "DeliveryArea": {
                            "row": 7, "col": 8,
                            "direction": "S",
                            "tags": ["delivery", "drop-off", "destination"]
                        }
                    },
                    "obstacle_cells": [
                        # Outer walls (border)
                        [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
                        [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9],
                        [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0],
                        [1, 9], [2, 9], [3, 9], [4, 9], [5, 9], [6, 9], [7, 9], [8, 9],
                        
                        # Some internal walls for obstacles
                        [3, 1], [3, 2], [3, 4], [3, 5],
                        [6, 1], [6, 2], [6, 4], [6, 5]
                    ],
                    "synonyms": {
                        "beverages": "PICKUP_BEVERAGES",
                        "drinks": "PICKUP_BEVERAGES",
                        "drink station": "PICKUP_BEVERAGES",
                        "beverage station": "PICKUP_BEVERAGES",
                        "delivery area": "DeliveryArea",
                        "drop-off area": "DeliveryArea",
                        "destination": "DeliveryArea"
                    }
                }
            },
            "2": {
                "name": "Maze Challenge",
                "description": "Complex maze with walls requiring careful navigation to reach pickup and drop-off locations",
                "config": {
                    "waypoints": {
                        "PICKUP_MAZE_ITEM": {"row": 3, "col": 7, "pick_direction": "N"}
                    },
                    "zones": {
                        "MazeExit": {
                            "row": 8, "col": 1,
                            "direction": "S",
                            "tags": ["exit", "delivery", "destination"]
                        }
                    },
                        "obstacle_cells": [
                            # Cells that are walls - robot cannot enter these
                            # Format: [row, col] - this cell is a wall
                            
                            # Outer walls (border)
                            [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
                            [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9],
                            [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0],
                            [1, 9], [2, 9], [3, 9], [4, 9], [5, 9], [6, 9], [7, 9], [8, 9],
                            
                            # Inner maze walls - create accessible paths
                            # Top section
                            [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7], [1, 8],
                            
                            # Middle section - create maze pattern
                            [2, 1], [2, 3], [2, 5], [2, 7],
                            [3, 1], [3, 3], [3, 5], [3, 6],  # Leave (3,7) accessible for pickup
                            [4, 1], [4, 5], [4, 6], [4, 7],  # Leave (4,3) accessible for robot start
                            [5, 1], [5, 3], [5, 5], [5, 6], [5, 7],
                            [6, 1], [6, 3], [6, 5], [6, 6], [6, 7],
                            [7, 1], [7, 3], [7, 5], [7, 6], [7, 7],
                            
                            # Bottom section - leave (8,1) accessible for dropoff
                            # Note: (8,1) is NOT in obstacle_cells, so it's accessible
                            # Also leaving (8,2) to (8,8) accessible for easier navigation
                        ],
                    "synonyms": {
                        "maze item": "PICKUP_MAZE_ITEM",
                        "item": "PICKUP_MAZE_ITEM",
                        "object": "PICKUP_MAZE_ITEM",
                        "package": "PICKUP_MAZE_ITEM",
                        "delivery": "MazeExit",
                        "exit": "MazeExit",
                        "destination": "MazeExit",
                        "drop-off": "MazeExit"
                    }
                }
            },
            "3": {
                "name": "Multi-path Maze",
                "description": "A maze with multiple paths and dead ends, requiring exploration.",
                "config": {
                    "waypoints": {
                        "PICKUP_MAZE_ITEM": {"row": 2, "col": 7, "pick_direction": "N"}
                    },
                    "zones": {
                        "MazeExit": {
                            "row": 8, "col": 1,
                            "direction": "S",
                            "tags": ["exit", "delivery", "destination"]
                        }
                    },
                    "obstacle_cells": [
                        # Cells that are walls - robot cannot enter these
                        # Format: [row, col] - this cell is a wall

                        # Outer walls (border)
                        [0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9],
                        [9, 0], [9, 1], [9, 2], [9, 3], [9, 4], [9, 5], [9, 6], [9, 7], [9, 8], [9, 9],
                        [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0],
                        [1, 9], [2, 9], [3, 9], [4, 9], [5, 9], [6, 9], [7, 9], [8, 9],

                        # Add some obstacles to create a more interesting path
                        [1, 1], [1, 2], [1, 3], [1, 8], 
                        [2, 1], [2, 2], [2, 3], [2, 6], 
                        [3, 6], [3, 7],  
                        [4, 2], [4, 3],
                        [5, 2], 
                        [6, 5], [6, 7], [6, 8],  
                        [7, 2], [7, 4], [7, 5], [7, 7], [7, 8], 
                        [8, 7], [8, 8], 
                    ],
                    "synonyms": {
                        "maze item": "PICKUP_MAZE_ITEM",
                        "item": "PICKUP_MAZE_ITEM",
                        "object": "PICKUP_MAZE_ITEM",
                        "package": "PICKUP_MAZE_ITEM",
                        "delivery": "MazeExit",
                        "exit": "MazeExit",
                        "destination": "MazeExit",
                        "drop-off": "MazeExit"
                    }
                }
            }
        }
    
    def get_world_list(self):
        """Get list of available worlds."""
        # Combine built-in and custom worlds
        all_worlds = {}
        all_worlds.update(self.worlds)
        all_worlds.update(self.custom_worlds)
        
        return [(key, world["name"], world["description"]) for key, world in all_worlds.items()]

    def get_world_config(self, world_id):
        """Get configuration for a specific world."""
        # Check built-in worlds first
        if world_id in self.worlds:
            return self.worlds[world_id]["config"]
        
        # Check custom worlds
        if world_id in self.custom_worlds:
            return self.custom_worlds[world_id]["config"]
        
        return {}
    
    def get_world_data(self, world_id):
        """Get full world data including robot_start_position."""
        # Check built-in worlds first
        if world_id in self.worlds:
            world = self.worlds[world_id]
            # For built-in worlds, add robot_start_position if not present
            if "robot_start_position" not in world:
                return {
                    **world,
                    "robot_start_position": {"row": 4, "col": 4, "facing": "N"}
                }
            return world
        
        # Check custom worlds
        if world_id in self.custom_worlds:
            return self.custom_worlds[world_id]
        
        return {}
    
    def get_world_info(self, world_id):
        """Get name and description for a specific world."""
        # Check built-in worlds first
        if world_id in self.worlds:
            return self.worlds[world_id]["name"], self.worlds[world_id]["description"]
        
        # Check custom worlds
        if world_id in self.custom_worlds:
            return self.custom_worlds[world_id]["name"], self.custom_worlds[world_id]["description"]
        
        return None, None

    def get_world_dimensions(self, world_id):
        """Return (rows, cols) grid dimensions for the given world_id.

        Defaults to (10, 10) if dimensions are not specified.
        """
        rows, cols = 10, 10
        try:
            if world_id in self.worlds:
                data = self.worlds[world_id]
            elif world_id in self.custom_worlds:
                data = self.custom_worlds[world_id]
            else:
                return rows, cols

            dims = data.get("grid_dimensions") or {}
            rows = int(dims.get("rows", rows))
            cols = int(dims.get("cols", cols))
        except Exception:
            rows, cols = 10, 10
        return rows, cols
    
    def get_initial_state(self, world_id):
        """Get initial state (robot position, objects) for a specific world."""
        # Check built-in worlds first
        if world_id in self.worlds:
            return self._get_builtin_initial_state(world_id)
        
        # Check custom worlds
        if world_id in self.custom_worlds:
            return self._get_custom_initial_state(world_id)
        
        return None
    
    def _get_builtin_initial_state(self, world_id):
        """Get initial state for built-in world."""
        if world_id not in self.worlds:
            return None
            
        # Default initial state for all worlds
        initial_state = {
            "robot_row": 4,
            "robot_col": 4, 
            "robot_facing": "N",
            "robot_state": "stand",
            "current_action": "idle",
            "has_object": False,
            "carried_object_name": None,
            "arm_status": "stowed",
            "gripper_status": "closed",
            "objects": {}
        }
        
        # World-specific object configurations
        if world_id == "1":  # Simple Pick & Drop
            initial_state["objects"] = {
                "tomato_can": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "apple": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "bottle": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "2":  # Grocery Store
            initial_state["objects"] = {
                "water_bottle": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "soda_can": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "juice_box": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "3":  # Warehouse
            initial_state["objects"] = {
                "package": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "box": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "crate": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "4":  # Office Building
            initial_state["objects"] = {
                "document": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "folder": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "file": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "5":  # Hospital
            initial_state["objects"] = {
                "medicine": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "supplies": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "equipment": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "6":  # Restaurant Kitchen
            initial_state["objects"] = {
                "ingredient": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "utensil": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "container": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "7":  # Laboratory
            initial_state["objects"] = {
                "sample": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "specimen": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "vial": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "8":  # Retail Store
            initial_state["objects"] = {
                "product": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "item": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "merchandise": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "9":  # Airport Terminal
            initial_state["objects"] = {
                "luggage": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "bag": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "suitcase": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "10":  # Construction Site
            initial_state["objects"] = {
                "tool": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "material": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "equipment": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        elif world_id == "11":  # Maze Challenge
            initial_state["objects"] = {
                "maze_item": {"row": 2, "col": 7, "present": True, "color": "#f85149"}
            }
        elif world_id == "12":  # Advanced Maze Challenge
            initial_state["objects"] = {
                "maze_item": {"row": 1, "col": 8, "present": True, "color": "#f85149"}
            }
        elif world_id == "13":  # Advanced Maze Challenge
            initial_state["objects"] = {
                "maze_item": {"row": 1, "col": 8, "present": True, "color": "#f85149"}
            }
        elif world_id == "14":  # Multi-path Maze
            initial_state["objects"] = {
                "maze_item": {"row": 2, "col": 7, "present": True, "color": "#f85149"}
            }
        else:
            # Default objects for unknown worlds
            initial_state["objects"] = {
                "object1": {"row": 2, "col": 3, "present": True, "color": "#f85149"},
                "object2": {"row": 1, "col": 3, "present": True, "color": "#39d353"},
                "object3": {"row": 2, "col": 4, "present": True, "color": "#58a6ff"}
            }
        
        return initial_state
    
    def _get_custom_initial_state(self, world_id):
        """Get initial state for custom world."""
        if world_id not in self.custom_worlds:
            return None
        
        world_data = self.custom_worlds[world_id]
        
        # Get robot start position from world data, or use defaults
        robot_start = world_data.get("robot_start_position", {})
        robot_row = robot_start.get("row", 4)
        robot_col = robot_start.get("col", 4)
        robot_facing = robot_start.get("facing", "N")
        
        # Default initial state for custom worlds
        initial_state = {
            "robot_row": robot_row,
            "robot_col": robot_col, 
            "robot_facing": robot_facing,
            "robot_state": "stand",
            "current_action": "idle",
            "has_object": False,
            "carried_object_name": None,
            "arm_status": "stowed",
            "gripper_status": "closed",
            "objects": {}
        }
        
        # Add objects for pickup locations
        config = world_data["config"]
        waypoints = config.get("waypoints", {})
        
        for name, data in waypoints.items():
            objects = data.get("objects", [])
            for i, obj_name in enumerate(objects):
                # Place objects near pickup location
                obj_row = data["row"] + (i % 2)
                obj_col = data["col"] + (i // 2)
                initial_state["objects"][obj_name] = {
                    "row": obj_row,
                    "col": obj_col,
                    "present": True,
                    "color": f"#{hash(obj_name) % 0xffffff:06x}"
                }
        
        return initial_state
    
    def save_world_to_file(self, world_id, filepath):
        """Save a world configuration to a JSON file."""
        if world_id in self.worlds:
            with open(filepath, 'w') as f:
                json.dump(self.worlds[world_id]["config"], f, indent=2)
            return True
        return False
    
    def load_world_from_file(self, filepath):
        """Load a world configuration from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading world from {filepath}: {e}")
            return None
    
    def load_custom_worlds(self):
        """Load custom worlds from the worlds/ directory."""
        self.custom_worlds = {}
        worlds_dir = Path(__file__).parent.parent / "worlds"
        
        if not worlds_dir.exists():
            return
        
        for file_path in worlds_dir.glob("custom_*.json"):
            try:
                with open(file_path, 'r') as f:
                    world_data = json.load(f)
                    if world_data.get('is_custom', False):
                        # Use filename without extension as world ID
                        world_id = file_path.stem
                        self.custom_worlds[world_id] = world_data
            except Exception as e:
                print(f"Error loading custom world from {file_path}: {e}")
    
    def save_custom_world(self, world_data, filename):
        """Save a custom world to the worlds/ directory."""
        try:
            worlds_dir = Path(__file__).parent.parent / "worlds"
            worlds_dir.mkdir(exist_ok=True)
            
            filepath = worlds_dir / filename
            with open(filepath, 'w') as f:
                json.dump(world_data, f, indent=2)
            
            # Reload custom worlds to include the new one
            self.load_custom_worlds()
            return True
        except Exception as e:
            print(f"Error saving custom world: {e}")
            return False
    
    def delete_custom_world(self, world_id):
        """Delete a custom world file."""
        try:
            # Find the world data to get filename
            if world_id not in self.custom_worlds:
                return False
            
            world_data = self.custom_worlds[world_id]
            world_name = world_data.get('name', 'unknown')
            
            # Generate filename
            import re
            sanitized_name = re.sub(r'[^\w\s-]', '', world_name)
            sanitized_name = re.sub(r'[-\s]+', '_', sanitized_name)
            filename = f"custom_{sanitized_name.lower()}.json"
            
            # Delete file
            worlds_dir = Path(__file__).parent.parent / "worlds"
            filepath = worlds_dir / filename
            
            if filepath.exists():
                filepath.unlink()
            
            # Reload custom worlds
            self.load_custom_worlds()
            return True
        except Exception as e:
            print(f"Error deleting custom world: {e}")
            return False

    @staticmethod
    def resolve_pickup_area(item_query: str, config: dict):
        """Return the PICKUP_* key for a natural-language item request."""
        q = (item_query or "").strip().lower()
        syn = config.get("synonyms", {}) or {}
        # Exact hit
        if q in syn:
            return syn[q]
        # Token-wise fallback
        tokens = q.replace("-", " ").replace("_", " ").split()
        for i in range(len(tokens), 0, -1):
            k = " ".join(tokens[:i])
            if k in syn:
                return syn[k]
        return None

def main():
    """Demo the world manager."""
    wm = WorldManager()
    
    print("Available Worlds:")
    print("=" * 50)
    
    for world_id, name, description in wm.get_world_list():
        print(f"{world_id}. {name}")
        print(f"   {description}")
        print()
    
    # Example usage
    print("Example: Loading Grocery Store world")
    config = wm.get_world_config("2")
    if config:
        print(f"Waypoints: {list(config['waypoints'].keys())}")
        print(f"Zones: {list(config['zones'].keys())}")
        print(f"Synonyms: {len(config['synonyms'])}")

if __name__ == "__main__":
    main()
