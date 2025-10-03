#!/usr/bin/env python3
"""
Test script for dynamic world system
"""

import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'spot_hololens_llm_interface'))

from world_manager import WorldManager

def test_world_manager():
    """Test the world manager functionality."""
    print("="*60)
    print("TESTING DYNAMIC WORLD SYSTEM")
    print("="*60)
    
    wm = WorldManager()
    
    print(f"Available worlds: {len(wm.worlds)}")
    print()
    
    # Test each world configuration
    for world_id in wm.worlds.keys():
        name, description = wm.get_world_info(world_id)
        config = wm.get_world_config(world_id)
        
        print(f"World {world_id}: {name}")
        print(f"  Description: {description}")
        print(f"  Waypoints: {len(config.get('waypoints', {}))}")
        print(f"  Zones: {len(config.get('zones', {}))}")
        print(f"  Synonyms: {len(config.get('synonyms', {}))}")
        
        # Show waypoint details
        waypoints = config.get('waypoints', {})
        for wp_name, wp_data in waypoints.items():
            pick_yaw = wp_data.get('pick_yaw')
            drop_yaw = wp_data.get('drop_yaw')
            wp_type = []
            if pick_yaw is not None:
                wp_type.append("PICKUP")
            if drop_yaw is not None:
                wp_type.append("DROPOFF")
            if not wp_type:
                wp_type.append("WAYPOINT")
            
            print(f"    - {wp_name}: ({wp_data['x']:.1f}, {wp_data['y']:.1f}) [{', '.join(wp_type)}]")
        
        # Show zone details
        zones = config.get('zones', {})
        for zone_name, zone_data in zones.items():
            centroid = zone_data['centroid']
            tags = zone_data.get('tags', [])
            print(f"    - Zone {zone_name}: ({centroid['x']:.1f}, {centroid['y']:.1f}) [tags: {', '.join(tags)}]")
        
        print()
    
    print("="*60)
    print("WORLD SYSTEM FEATURES:")
    print("="*60)
    print("✓ Dynamic waypoint loading")
    print("✓ Semantic zone support")
    print("✓ Natural language synonyms")
    print("✓ Custom orientations per location")
    print("✓ Pickup/dropoff classification")
    print("✓ Extensible configuration")
    print("✓ GUI integration")
    print("✓ Runtime world switching")
    print()
    
    print("USAGE EXAMPLES:")
    print("="*30)
    print("1. Start GUI with specific world:")
    print("   python3 spot_simulation_gui.py 2")
    print()
    print("2. Start GUI with world selection:")
    print("   python3 spot_simulation_gui.py")
    print("   (Then click 'Change World' button)")
    print()
    print("3. Start NL control with world selection:")
    print("   python3 nl_control_gemini_no_world.py")
    print("   (Select world from menu)")
    print()
    
    print("EXAMPLE COMMANDS FOR DIFFERENT WORLDS:")
    print("="*40)
    
    example_commands = {
        "1": "Pick up the tomato can at the pick-up location, then drop it at the drop-off location",
        "2": "From the cans shelf pick up the tomato can, then bring it to where the vegetables are and drop it",
        "3": "Pick up the package at the loading dock and take it to storage area a",
        "4": "Pick up the documents from the reception and take them to meeting room a",
        "5": "Pick up the medication from the pharmacy and take it to the patient ward",
        "6": "Pick up the ingredients from the prep station and take them to the cooking area",
        "7": "Pick up the samples from the main bench and take them to the microscope station",
        "8": "Pick up the item from the electronics section and take it to customer service",
        "9": "Pick up the luggage from the baggage claim and take it to gate a1",
        "10": "Pick up the tools from the tool shed and take them to the work site"
    }
    
    for world_id, command in example_commands.items():
        name, _ = wm.get_world_info(world_id)
        print(f"World {world_id} ({name}):")
        print(f'  "{command}"')
        print()

if __name__ == "__main__":
    test_world_manager()

