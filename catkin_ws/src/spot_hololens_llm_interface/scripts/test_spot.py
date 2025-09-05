#!/usr/bin/env python3
"""Simple test script that sends commands to the FSM.

Usage: 
1. Start services: roslaunch spot_hololens_llm_interface spot_launch_test_grasp.launch
2. Start FSM: python3 finite_state_machine.py
3. Run this script: python3 test_spot.py

This script sends commands to the FSM with numbered steps:
1. Connect -> 2. Power on -> 3. Stand -> 4. Get image -> 5. Move forward -> 
6. Rotate 90° -> 7. Rotate 90° -> 8. Get image -> 9. Sit -> 10. Power off -> 11. Disconnect
"""

import rospy
import time
import sys
import os

# Add the FSM to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'spot_hololens_llm_interface'))
from finite_state_machine import SpotStateMachine


def main():
    rospy.init_node('test_real_spot_manager_client', anonymous=True)
    
    # Check for dummy mode parameter
    dummy_mode = rospy.get_param('~dummy_mode', False)
    
    if dummy_mode:
        rospy.loginfo("Starting test with FSM in DUMMY MODE...")
        rospy.loginfo("No real robot - all actions simulated!")
    else:
        rospy.loginfo("Starting test with FSM in REAL MODE...")
        rospy.loginfo("Real robot control - be careful!")
    
    # Create FSM instance (will auto-detect dummy mode from parameter)
    spot_fsm = SpotStateMachine()
    
    rospy.loginfo("FSM created, starting test sequence...")
    
    try:
        # Step 1: Connect
        rospy.loginfo("Step 1: Connecting to robot...")
        spot_fsm.send("connect")
        
        # Step 2: Power on
        rospy.loginfo("Step 2: Powering on robot...")
        spot_fsm.send("power_on")
        
        # Step 3: Stand up
        rospy.loginfo("Step 3: Standing up...")
        spot_fsm.send("stand_up")
        
        # Step 4: Get initial image
        rospy.loginfo("Step 4: Getting initial image...")
        spot_fsm.send("get_image", image_source="frontleft_fisheye_image")
        
        # Step 5: Move forward 0.2 meters
        rospy.loginfo("Step 5: Moving forward 0.2 meters...")
        spot_fsm.send("start_moving", x=0.2, y=0.0, yaw=0.0, frame="body")
        
        # Step 6: Rotate half pi (90 degrees) to the left
        rospy.loginfo("Step 6: Rotating half pi (90 degrees) to the left...")
        spot_fsm.send("start_moving", x=0.0, y=0.0, yaw=1.57, frame="body")  # 1.57 radians = 90 degrees
        
        # Step 7: Rotate another half pi (90 degrees) to the left (total 180 degrees)
        rospy.loginfo("Step 7: Rotating another half pi (90 degrees) to the left...")
        spot_fsm.send("start_moving", x=0.0, y=0.0, yaw=1.57, frame="body")  # Another 90 degrees
        
        # Step 8: Get image after rotations
        rospy.loginfo("Step 8: Getting image after rotations...")
        spot_fsm.send("get_image", image_source="frontleft_fisheye_image")

        # Step 9: Sit down
        rospy.loginfo("Step 9: Sitting down...")
        spot_fsm.send("sit_down")
        
        # Step 10: Power off
        rospy.loginfo("Step 10: Powering off...")
        spot_fsm.send("power_off_from_sit")
        
        # Step 11: Disconnect
        rospy.loginfo("Step 11: Disconnecting...")
        spot_fsm.send("disconnect")
        
        rospy.loginfo("Test sequence completed successfully!")
        
    except Exception as e:
        rospy.logerr(f"Test failed with error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted by user")
        sys.exit(0)
