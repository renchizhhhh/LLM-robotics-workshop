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

def test_move_arm_pose():
    """Test move_arm_pose: stand -> move arm -> sit"""
    # First, make robot stand
    rospy.loginfo("Making robot stand...")
    if not robot_stand():
        rospy.logerr("Failed to make robot stand")
        return False
    
    # Create action client for move_arm_pose
    client = actionlib.SimpleActionClient('move_arm_pose', MoveArmPoseAction)
    
    if not client.wait_for_server(rospy.Duration(10.0)):
        rospy.logerr("Move arm pose action server not available")
        robot_sit()  # Try to sit even if action server fails
        return False

    # Create goal for arm pose - arm along x-axis of body frame with gripper pointing down
    goal = MoveArmPoseGoal()
    goal.x = 0.6      # 80cm forward along x-axis (body frame)
    goal.y = 0.0      # No lateral movement
    goal.z = 0.3      # 30cm above body
    # Quaternion for no rotation (identity quaternion)
    goal.qw = 1.0     # cos(0°) for no rotation
    goal.qx = 0.0     # sin(0°) for no rotation around x-axis
    goal.qy = 0.0
    goal.qz = 0.0
    goal.duration = 1  # 3 seconds to complete movement
    goal.open_gripper = False  # Don't open gripper during movement

    rospy.loginfo("Moving arm to position (0.6, 0.0, 0.3) along x-axis...")
    client.send_goal(goal)

    # Wait for result with feedback
    finished = client.wait_for_result(rospy.Duration(10.0))
    if not finished:
        rospy.logwarn("Move arm pose timed out")
        client.cancel_goal()
        robot_sit()  # Make sure to sit even if arm movement times out
        return False

    result = client.get_result()
    arm_success = False
    if result and result.success:
        rospy.loginfo(f"Arm movement successful! {result.message}")
        arm_success = True
        
        # Open gripper after successful arm movement
        rospy.loginfo("Opening gripper...")
        try:
            from spot_hololens_llm_interface.srv import ArmCommand, ArmCommandRequest
            rospy.wait_for_service('/spot_entrance/arm_command', timeout=5.0)
            arm_srv = rospy.ServiceProxy('/spot_entrance/arm_command', ArmCommand)
            
            # Open gripper (with retry if timeout)
            req = ArmCommandRequest()
            req.command_type = "open"
            resp = arm_srv(req)
            
            if resp.success:
                rospy.loginfo("Gripper opened successfully")
            else:
                rospy.logwarn(f"Gripper open had issues: {resp.message}, but continuing...")
            
            rospy.sleep(3.0)  # Wait longer for gripper to settle
            
            # Close gripper
            rospy.loginfo("Closing gripper...")
            req.command_type = "close"
            resp = arm_srv(req)
            
            if resp.success:
                rospy.loginfo("Gripper closed successfully")
            else:
                rospy.logwarn(f"Gripper close had issues: {resp.message}, but continuing...")
            
            rospy.sleep(3.0)  # Wait longer for gripper to settle
            
            # Stow arm
            rospy.loginfo("Stowing arm...")
            req.command_type = "stow"
            resp = arm_srv(req)
            
            if resp.success:
                rospy.loginfo("Arm stowed successfully")
            else:
                rospy.logwarn(f"Arm stow had issues: {resp.message}, but continuing...")
                
        except Exception as e:
            rospy.logwarn(f"Failed to control gripper/arm: {e}")
    else:
        rospy.logwarn(f"Arm movement failed: {result.message if result else 'No result'}")
    
    # Finally, make robot sit
    rospy.loginfo("Making robot sit...")
    if not robot_sit():
        rospy.logerr("Failed to make robot sit")
        return False
    
    return arm_success

def main():
    rospy.init_node('move_arm_pose_test')
    rospy.loginfo("Move Arm Pose Test - Testing arm positioning along x-axis")
    
    try:
        rospy.sleep(2.0)  # Wait for services
        success = test_move_arm_pose()
        rospy.loginfo("Test completed successfully!" if success else "Test failed")
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted")
    except Exception as e:
        rospy.logerr(f"Test failed: {e}")

if __name__ == '__main__':
    main()