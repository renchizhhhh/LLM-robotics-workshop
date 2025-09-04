#!/usr/bin/env python3

import rospy
import actionlib
from spot_hololens_llm_interface_msgs.msg import (
    GraspObjectAction, 
    GraspObjectGoal,
    InteractiveGraspAction,
    InteractiveGraspGoal
)

def test_direct_grasp():
    """Test direct grasp at specific pixel coordinates"""
    rospy.loginfo("Testing direct grasp...")
    
    client = actionlib.SimpleActionClient('grasp_object', GraspObjectAction)
    client.wait_for_server()
    
    goal = GraspObjectGoal()
    goal.image_source = 'frontright_fisheye_image'
    goal.pixel_x = 320  # Example pixel coordinates
    goal.pixel_y = 240
    goal.force_top_down_grasp = True
    goal.return_to_initial_pose = True
    
    client.send_goal(goal)
    result = client.wait_for_result()
    
    if result:
        print(f"Direct grasp result: {client.get_result()}")
    else:
        print("Direct grasp failed or was cancelled")

def test_interactive_grasp():
    """Test interactive grasp with user click"""
    rospy.loginfo("Testing interactive grasp...")
    
    client = actionlib.SimpleActionClient('interactive_grasp', InteractiveGraspAction)
    client.wait_for_server()
    
    goal = InteractiveGraspGoal()
    goal.image_source = 'frontright_fisheye_image'
    goal.show_preview = True
    goal.window_title = 'Click on object to grasp'
    goal.force_top_down_grasp = True
    goal.return_to_initial_pose = True
    
    client.send_goal(goal)
    result = client.wait_for_result()
    
    if result:
        print(f"Interactive grasp result: {client.get_result()}")
    else:
        print("Interactive grasp failed or was cancelled")

if __name__ == '__main__':
    rospy.init_node('spot_grasp_test_client')
    
    # Wait for action servers to be available
    rospy.loginfo("Waiting for action servers...")
    
    try:
        # Test interactive grasp (comment out if you want to test direct grasp)
        test_interactive_grasp()
        
        # Uncomment to test direct grasp instead
        # test_direct_grasp()
        
    except rospy.ROSInterruptException:
        pass
