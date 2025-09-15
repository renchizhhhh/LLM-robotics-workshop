#!/usr/bin/env python3

import rospy
import sys
import time
import actionlib
import os

# Add the FSM to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'spot_hololens_llm_interface'))
from finite_state_machine import SpotStateMachine

def test_state_machine_grasp():
    """
    Test script for the Spot finite state machine with automated grasp.
    
    Flow:
    1. Initialize state machine
    2. Connect to robot
    3. Power on robot
    4. Stand up
    5. Execute automated grasp
    6. Sit down
    7. Power off
    8. Disconnect
    
    Usage: python3 test_state_machine_automated_grasp.py [dummy]
    """
    rospy.init_node('test_state_machine_automated_grasp')
    
    # Check for dummy mode flag
    dummy_mode = 'dummy' in sys.argv
    if dummy_mode:
        rospy.loginfo("Running in DUMMY mode")
    
    # Create FSM instance
    spot = SpotStateMachine(dummy_mode=dummy_mode)
    
    # Connect and prepare the robot
    rospy.loginfo("Connecting to robot...")
    spot.connect()
    rospy.sleep(1.0)
    
    rospy.loginfo("Powering on robot...")
    spot.power_on()
    rospy.sleep(1.0)
    
    rospy.loginfo("Standing up...")
    spot.stand_up()
    rospy.sleep(3.0)  # Give the robot time to stand
    
    # Check if Google API key is set for object detection
    if not os.getenv('GOOGLE_API_KEY') and not dummy_mode:
        rospy.logwarn("GOOGLE_API_KEY not set. This is required for real (non-dummy) object detection.")
    
    # Define grasp parameters
    image_source = "hand_color_image"
    object_type = "tomato can"  # Can be changed to any object type that the detection can handle
    
    rospy.loginfo(f"Starting automated grasp for {object_type}...")
    
    # Execute automated grasp with parameters
    spot.send('start_automated_grasp',
              image_source=image_source,
              object_type=object_type,
              force_top_down_grasp=False,
              force_horizontal_grasp=True,
              force_45_angle_grasp=False,
              force_squeeze_grasp=False,
              return_to_initial_pose=True)
    
    wait_timeout = 60.0  # seconds - max time to wait for grasp to complete
    poll_interval = 1.0  # seconds
    start_time = time.time()
    rospy.loginfo("Waiting for automated grasp to complete (polling FSM state)...")
    while not rospy.is_shutdown():
        try:
            cur_state_id = getattr(spot.current_state, 'id', None)
        except Exception:
            cur_state_id = None

        # If FSM isn't in 'grasping', assume the automated grasp finished (success or fail)
        if cur_state_id is None or cur_state_id != 'grasping':
            rospy.loginfo(f"FSM state is '{cur_state_id}' - assuming grasp completed")
            break

        elapsed = time.time() - start_time
        if elapsed >= wait_timeout:
            rospy.logerr(f"Timed out waiting for automated grasp after {wait_timeout}s; current state still '{cur_state_id}'")
            break

        rospy.loginfo(f"Grasp in progress... elapsed={int(elapsed)}s, state={cur_state_id}")
        rospy.sleep(poll_interval)
    
    # Sit down
    rospy.loginfo("Sitting down...")
    spot.sit_down()
    rospy.sleep(3.0)  # Give the robot time to sit
    
    # Power off and disconnect
    rospy.loginfo("Powering off...")
    spot.power_off_from_sit()
    rospy.sleep(1.0)
    
    rospy.loginfo("Disconnecting...")
    spot.disconnect()
    
    rospy.loginfo("Test completed!")
    
if __name__ == '__main__':
    try:
        test_state_machine_grasp()
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted")
    except Exception as e:
        rospy.logerr(f"Test failed: {e}")
