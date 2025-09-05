#!/usr/bin/env python3

import rospy
import actionlib
from spot_hololens_llm_interface.msg import (
    InteractiveGraspAction,
    InteractiveGraspGoal,
    InteractiveGraspFeedback,
    InteractiveGraspResult,
)

def feedback_cb(feedback: InteractiveGraspFeedback):
    rospy.loginfo(f"[feedback] state={feedback.current_state} progress={feedback.progress} msg='{feedback.status_message}'")

def test_interactive_grasp():
    """Test interactive grasp with user click"""
    client = actionlib.SimpleActionClient('interactive_grasp', InteractiveGraspAction)
    rospy.loginfo("Waiting for interactive_grasp action server...")
    if not client.wait_for_server(rospy.Duration(10.0)):
        rospy.logerr("interactive_grasp action server not available")
        return

    goal = InteractiveGraspGoal()
    # choose an image source the server supports, e.g. "hand_camera" or "front_fisheye"
    goal.image_source = "hand_color_image"
    # show_preview=True opens an OpenCV window on the server side for user click
    goal.show_preview = True
    goal.window_title = "Click object to grasp"
    goal.return_to_initial_pose = True
    # optional forcing flags
    goal.force_top_down_grasp = False
    goal.force_horizontal_grasp = True
    goal.force_45_angle_grasp = False
    goal.force_squeeze_grasp = False

    rospy.loginfo("Sending interactive grasp goal (click in server window)...")
    client.send_goal(goal, feedback_cb=feedback_cb)

    # wait for result (increase timeout for long operations)
    finished = client.wait_for_result(rospy.Duration(180.0))
    if not finished:
        rospy.logerr("Interactive grasp did not finish in time, cancelling")
        client.cancel_goal()
        return

    result: InteractiveGraspResult = client.get_result()
    if result is None:
        rospy.logerr("No result received")
        return

    if result.success:
        rospy.loginfo(f"Grasp succeeded state={result.grasp_state} selected_pixel=({result.selected_pixel_x},{result.selected_pixel_y})")
    else:
        rospy.logwarn(f"Grasp failed: {result.message}")


if __name__ == '__main__':
    rospy.init_node('spot_grasp_test_client')
    
    # Wait for action servers to be available
    rospy.loginfo("Waiting for action servers...")
    
    try:
        # Test interactive grasp (comment out if you want to test direct grasp)
        test_interactive_grasp()
        
    except rospy.ROSInterruptException:
        pass
