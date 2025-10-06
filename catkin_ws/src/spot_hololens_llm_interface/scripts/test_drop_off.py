#!/usr/bin/env python3

import rospy
import actionlib
import time
from spot_hololens_llm_interface.msg import MoveArmPoseAction, MoveArmPoseGoal, MoveArmPoseFeedback
from std_srvs.srv import Trigger

def robot_stand():
    """Make robot stand up"""
    try:
        rospy.wait_for_service('/spot_entrance/stand', timeout=5.0)
        stand_srv = rospy.ServiceProxy('/spot_entrance/stand', Trigger)
        resp = stand_srv()
        if resp.success:
            rospy.loginfo("Robot standing up")
            rospy.sleep(3.0)  # Wait for robot to stand
            return True
        else:
            rospy.logerr(f"Failed to stand: {resp.message}")
            return False
    except Exception as e:
        rospy.logerr(f"Stand service call failed: {e}")
        return False

def robot_sit():
    """Make robot sit down"""
    try:
        rospy.wait_for_service('/spot_entrance/sit', timeout=5.0)
        sit_srv = rospy.ServiceProxy('/spot_entrance/sit', Trigger)
        resp = sit_srv()
        if resp.success:
            rospy.loginfo("Robot sitting down")
            rospy.sleep(3.0)  # Wait for robot to sit
            return True
        else:
            rospy.logerr(f"Failed to sit: {resp.message}")
            return False
    except Exception as e:
        rospy.logerr(f"Sit service call failed: {e}")
        return False

def feedback_callback(feedback):
    """Callback to print feedback during drop-off action"""
    try:
        # Some feedback message types may not include all fields; access safely.
        progress = getattr(feedback, 'progress', 0.0)
        current_state = getattr(feedback, 'current_state', getattr(feedback, 'state', 'UNKNOWN'))
        status_message = getattr(feedback, 'status_message', None)

        if status_message is None:
            rospy.loginfo(f"Drop-off progress: {progress*100:.1f}% - State: {current_state}")
        else:
            rospy.loginfo(f"Drop-off progress: {progress*100:.1f}% - State: {current_state} - {status_message}")
    except Exception as e:
        # Protect action client from crashing due to unexpected feedback shape
        rospy.logwarn(f"Exception in feedback_callback: {e}")

def test_drop_off():
    """Test drop_off_object action: stand -> grasp (simulated) -> drop off -> sit"""
    # First, make robot stand
    rospy.loginfo("Making robot stand...")
    if not robot_stand():
        rospy.logerr("Failed to make robot stand")
        return False
    
    # Simulate grasping an object first (close gripper)
    rospy.loginfo("Simulating grasp - closing gripper to hold an object...")
    try:
        from spot_hololens_llm_interface.srv import ArmCommand, ArmCommandRequest
        rospy.wait_for_service('/spot_entrance/arm_command', timeout=5.0)
        arm_srv = rospy.ServiceProxy('/spot_entrance/arm_command', ArmCommand)
        
        # Close gripper to simulate holding an object
        req = ArmCommandRequest()
        req.command_type = "close"
        resp = arm_srv(req)
        
        if resp.success:
            rospy.loginfo("Gripper closed - simulating object hold")
        else:
            rospy.logwarn(f"Gripper close had issues: {resp.message}")
        
        rospy.sleep(2.0)
            
    except Exception as e:
        rospy.logwarn(f"Failed to close gripper: {e}")
    
    # Create action client for drop_off_object
    client = actionlib.SimpleActionClient('drop_off_object', MoveArmPoseAction)
    
    if not client.wait_for_server(rospy.Duration(10.0)):
        rospy.logerr("Drop off action server not available")
        robot_sit()  # Try to sit even if action server fails
        return False

    # Create goal for drop-off
    # The drop_off action uses the goal to specify the drop-off position
    goal = MoveArmPoseGoal()
    goal.x = 0.8      # 80cm forward along x-axis (body frame)
    goal.y = 0.0      # No lateral movement
    goal.z = 0.3      # 30cm above body
    # Quaternion for gripper pointing down (90 degree rotation around y-axis)
    goal.qw = 0.7071  # cos(45°)
    goal.qx = 0.7071  # sin(45°) around x-axis
    goal.qy = 0.0
    goal.qz = 0.0
    goal.duration = 1.0  # 1 second to complete movement
    goal.open_gripper = False  # This field is ignored in drop_off, but set for completeness

    rospy.loginfo("Executing drop-off sequence...")
    rospy.loginfo("  1. Moving arm to drop-off position (0.8, 0.0, 0.3)")
    rospy.loginfo("  2. Opening gripper to release object")
    rospy.loginfo("  3. Closing gripper")
    rospy.loginfo("  4. Stowing arm")
    
    # Send goal with feedback callback
    client.send_goal(goal, feedback_cb=feedback_callback)

    # Wait for result with timeout (allow enough time for full sequence)
    finished = client.wait_for_result(rospy.Duration(20.0))
    if not finished:
        rospy.logwarn("Drop-off action timed out")
        client.cancel_goal()
        robot_sit()  # Make sure to sit even if drop-off times out
        return False

    result = client.get_result()
    drop_off_success = False
    if result and result.success:
        rospy.loginfo(f"Drop-off successful! {result.message}")
        drop_off_success = True
    else:
        rospy.logwarn(f"Drop-off failed: {result.message if result else 'No result'}")
    
    # Wait a bit before sitting
    rospy.sleep(2.0)
    
    # Finally, make robot sit
    rospy.loginfo("Making robot sit...")
    if not robot_sit():
        rospy.logerr("Failed to make robot sit")
        return False
    
    return drop_off_success

def main():
    rospy.init_node('drop_off_test')
    rospy.loginfo("=" * 60)
    rospy.loginfo("Drop-Off Action Test")
    rospy.loginfo("=" * 60)
    rospy.loginfo("This test will:")
    rospy.loginfo("  1. Stand the robot up")
    rospy.loginfo("  2. Close gripper (simulate holding object)")
    rospy.loginfo("  3. Execute drop-off sequence:")
    rospy.loginfo("     - Move arm to drop-off position")
    rospy.loginfo("     - Open gripper (release object)")
    rospy.loginfo("     - Close gripper")
    rospy.loginfo("     - Stow arm")
    rospy.loginfo("  4. Sit the robot down")
    rospy.loginfo("=" * 60)
    
    try:
        rospy.sleep(2.0)  # Wait for services to be available
        success = test_drop_off()
        rospy.loginfo("=" * 60)
        if success:
            rospy.loginfo("TEST COMPLETED SUCCESSFULLY!")
        else:
            rospy.loginfo("TEST FAILED")
        rospy.loginfo("=" * 60)
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted")
    except Exception as e:
        rospy.logerr(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
