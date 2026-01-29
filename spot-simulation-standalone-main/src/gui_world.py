#!/usr/bin/env python3
"""
GUI World Module - World Management
Handles world loading, object creation, and world configuration.
"""

import math
from world_manager import WorldManager


class GUIWorldManager:
    """Manages world configuration, objects, and world switching."""
    
    def __init__(self, world_manager, world_config):
        self.world_manager = world_manager
        self.world_config = world_config
        self.current_world_id = None
        self.world_name = None
        self.world_description = None
        
        # Initialize world data from config
        self.waypoints = self.world_config.get("waypoints", {})
        self.zones = self.world_config.get("zones", {})
        self.objects = self.create_dynamic_objects()
    
    def create_dynamic_objects(self):
        """Create dynamic objects based on world configuration without generic placeholders."""
        objects = {}
        
        waypoints = self.world_config.get("waypoints", {})
        colors = ["#f85149", "#39d353", "#58a6ff", "#ffa657", "#ff7b72", "#00d4aa"]
        
        color_idx = 0
        
        for waypoint_name, waypoint_data in waypoints.items():
            if waypoint_data.get("pick_direction") is None:
                continue
            row = waypoint_data["row"]
            col = waypoint_data["col"]
            
            # Get specific object names based on waypoint category (PICKUP_* keys)
            object_names = self.get_objects_for_category(waypoint_name)
            
            # Only place real items; if none are defined for this category, skip
            for i in range(min(3, len(object_names))):
                # Place all objects in the same cell as the waypoint
                obj_name = object_names[i]
                obj_color = colors[color_idx % len(colors)]
                
                objects[obj_name] = {
                    "row": row,  # Same row as waypoint
                    "col": col,  # Same col as waypoint
                    "present": True,
                    "color": obj_color,
                    "waypoint": waypoint_name
                }
                color_idx += 1
        
        return objects
    
    def get_objects_for_category(self, category_name):
        """Get specific object names for each PICKUP_* category. Never return generic items."""
        name = str(category_name or "").upper()
        objects_by_category = {
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
            "PICKUP_MAZE_ITEM": ["maze_package"],
            "WHITE_BOX": ["tomato_can"],
        }
        return list(objects_by_category.get(name, []))
    
    def adjust_zones_for_collisions(self, min_distance=1.2):
        """Return a copy of zones with centroids nudged to avoid overlap with pickup waypoints."""
        zones = self.world_config.get("zones", {})
        waypoints = self.world_config.get("waypoints", {})
        adjusted = {}
        for zn, zd in zones.items():
            c = dict(zd.get("centroid", {}))
            zx, zy = float(c.get("x", 0.0)), float(c.get("y", 0.0))
            # push away if close to any pickup waypoint
            for wp_name, wp in waypoints.items():
                if wp.get("pick_yaw") is None:
                    continue
                dx = zx - float(wp.get("x", 0.0))
                dy = zy - float(wp.get("y", 0.0))
                dist = (math.hypot(dx, dy) or 1e-6)
                if dist < min_distance:
                    # Nudge outward along the vector from waypoint to zone
                    scale = (min_distance - dist) + 0.0
                    zx += (dx / dist) * scale
                    zy += (dy / dist) * scale
            new_zone = dict(zd)
            new_zone["centroid"] = {"x": zx, "y": zy, "z": zd.get("centroid", {}).get("z", 0.0)}
            adjusted[zn] = new_zone
        return adjusted
    
    def load_world(self, world_id):
        """Load a new world configuration."""
        from message_bus import rospy, String
        
        rospy.loginfo(f"Loading world: {world_id}")
        if world_id in self.world_manager.worlds or world_id in self.world_manager.custom_worlds:
            # Update current world ID
            self.current_world_id = world_id
            rospy.loginfo(f"Set current_world_id to: {self.current_world_id}")
            # Update world configuration
            self.world_config = self.world_manager.get_world_config(world_id)
            self.world_name, self.world_description = self.world_manager.get_world_info(world_id)
            
            # Update waypoints and zones
            self.waypoints = self.world_config.get("waypoints", {})
            self.zones = self.world_config.get("zones", {})
            
            # Update objects
            self.objects = self.create_dynamic_objects()
            
            rospy.loginfo(f"Loaded world: {self.world_name} - {self.world_description}")
            return True
        return False
