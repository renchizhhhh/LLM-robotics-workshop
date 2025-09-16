#!/usr/bin/env python3
"""Execute a predefined Spot movement sequence using the existing services and action servers.

Sequence:
 - Check power on
 - Stand up
 - Rotate left 30 degrees
 - Grasp from pixel (horizontal grasp)
 - Move to specified vision pose (0.7,0.0,0.4, 0,0,-0.3,0.95)
 - Move arm to specified pose and open gripper
 - Move arm up and stow
 - Return to initial position (0,0,0 in vision frame)
 - Sit down

This script reuses the services and action servers started in the workspace.
"""
import rospy
import actionlib
import math
from tf.transformations import quaternion_from_euler

from std_srvs.srv import Trigger
from spot_hololens_llm_interface.msg import (
    InteractiveGraspAction, InteractiveGraspGoal,
    MoveArmPoseAction, MoveArmPoseGoal
)


def wait_service(name, timeout=10.0):
    rospy.loginfo(f"Waiting for service {name}...")
    try:
        rospy.wait_for_service(name, timeout=timeout)
        return True
    except rospy.ROSException:
        rospy.logerr(f"Service {name} not available")
        return False


def call_trigger(service_name):
    if not wait_service(service_name):
        return None
    try:
        proxy = rospy.ServiceProxy(service_name, Trigger)
        return proxy()
    except Exception as e:
        rospy.logerr(f"Call to {service_name} failed: {e}")
        return None


def send_move_arm_goal(client, x, y, z, qx, qy, qz, qw, duration=2.0, open_gripper=False):
    goal = MoveArmPoseGoal()
    goal.x = x
    goal.y = y
    goal.z = z
    goal.qx = qx
    goal.qy = qy
    goal.qz = qz
    goal.qw = qw
    goal.duration = duration
    goal.open_gripper = open_gripper

    rospy.loginfo(f"Sending move_arm_pose: pos=({x},{y},{z}) open_gripper={open_gripper}")
    client.send_goal(goal)
    finished = client.wait_for_result(rospy.Duration(duration + 10.0))
    if not finished:
        rospy.logerr("move_arm_pose did not finish in time; cancelling")
        client.cancel_goal()
        return None
    return client.get_result()


def send_interactive_grasp(client, image_source='hand_color_image'):
    goal = InteractiveGraspGoal()
    goal.image_source = image_source
    goal.window_title = "Click object to grasp"
    goal.show_preview = True
    goal.return_to_initial_pose = False
    goal.force_horizontal_grasp = True
    client.send_goal(goal)

    rospy.loginfo("Sending interactive grasp goal (click in server window)...")
    finished = client.wait_for_result(rospy.Duration(60.0))
    if not finished:
        rospy.logerr("interactive_grasp did not finish in time; cancelling")
        client.cancel_goal()
        return None
    return client.get_result()


def main():
    rospy.init_node('execute_spot_sequence')

    ns = '/spot_entrance'
    srv_power_on = ns + '/power_on'
    srv_power_off = ns + '/power_off'
    srv_stand = ns + '/stand'
    srv_sit = ns + '/sit'
    srv_move_to_position = ns + '/move_to_position'
    srv_arm_stow = ns + '/arm_command'

    # 1) Check power on (call power_on if not powered)
    resp = call_trigger(srv_power_on)
    if not resp or not getattr(resp, 'success', False):
        rospy.logerr('Power on failed or not acknowledged; aborting')
        return
    rospy.loginfo('Power on acknowledged')

    # 2) Stand up
    resp = call_trigger(srv_stand)
    if not resp or not getattr(resp, 'success', False):
        rospy.logwarn('Stand reported failure or uncertain')
    else:
        rospy.loginfo('Robot standing')

    # 3) Rotate left 30 degrees using move_to_position in body frame
    from spot_hololens_llm_interface.srv import MoveToPosition, MoveToPositionRequest, ArmCommand, ArmCommandRequest
    from geometry_msgs.msg import Pose, Point, Quaternion

    # Build rotate request: small rotation in body frame (yaw +30 deg)
    rotate_req = MoveToPositionRequest()
    p = Pose()
    p.position = Point(0.0, 0.0, 0.0)
    # quaternion for +30 deg yaw
    yaw = math.radians(30.0)
    q = quaternion_from_euler(0.0, 0.0, yaw)
    p.orientation = Quaternion(*q)
    rotate_req.target_pose = p
    rotate_req.frame_name = 'body'

    if not wait_service(srv_move_to_position):
        rospy.logwarn('move_to_position service not available; skipping rotation')
    else:
        try:
            proxy = rospy.ServiceProxy(srv_move_to_position, MoveToPosition)
            rospy.loginfo('Rotating left 30 degrees')
            resp = proxy(rotate_req)
            rospy.loginfo(f"Rotate response: success={getattr(resp, 'success', False)} message='{getattr(resp, 'message', '')}'")
        except Exception as e:
            rospy.logerr(f"Rotation call failed: {e}")

    # 4) Grasp from pixel (set horizontal grasp)
    grasp_client = actionlib.SimpleActionClient('interactive_grasp', InteractiveGraspAction)
    rospy.loginfo('Waiting for interactive_grasp action server...')
    if not grasp_client.wait_for_server(rospy.Duration(10.0)):
        rospy.logerr('interactive_grasp action server not available; aborting')
        return

    rospy.sleep(2.0)  # wait a bit for robot to stabilize
    send_interactive_grasp(grasp_client)

    # 5) Move to release pose
    move_req = MoveToPositionRequest()
    p = Pose()
    p.position = Point(0.7, 0.0, 0.4)
    p.orientation = Quaternion(0.0, 0.0, -0.3, 0.95)
    move_req.target_pose = p
    move_req.frame_name = 'vision'

    try:
        proxy = rospy.ServiceProxy(srv_move_to_position, MoveToPosition)
        rospy.loginfo('Moving to release vision pose')
        resp = proxy(move_req)
        rospy.loginfo(f"Move response: success={getattr(resp,'success',False)} message='{getattr(resp,'message','')}'")
    except Exception as e:
        rospy.logerr(f"Move to vision pose failed: {e}")

    # 6) Move arm to pose and open gripper
    arm_client = actionlib.SimpleActionClient('move_arm_pose', MoveArmPoseAction)
    rospy.loginfo('Waiting for move_arm_pose action server...')
    if not arm_client.wait_for_server(rospy.Duration(10.0)):
        rospy.logerr('move_arm_pose action server not available; aborting')
        return

    res = send_move_arm_goal(arm_client, 0.9, -0.20, 0.35, 0.67, 0.15, 0.14, 0.70, duration=3.0, open_gripper=False)
    if res is None:
        rospy.logerr('Arm goal failed or timed out')

    # 6.1) Open gripper to release object
    res = send_move_arm_goal(arm_client, 0.9, -0.20, 0.35, 0.67, 0.15, 0.14, 0.70, duration=3.0, open_gripper=True)

    # 7) Move arm up and stow (close gripper as stow)
    res = send_move_arm_goal(arm_client, 0.80, 0.0, 0.45, 0.0, 0.0, 0.0, 1.0, duration=2.0, open_gripper=False)
    if res is None:
        rospy.logerr('Arm stow goal failed or timed out')

    # 8) Stow arm using arm_command service
    # Build rotate request: small rotation in body frame (yaw +30 deg)
    arm_req = ArmCommandRequest()
    arm_req.command_type = 'stow'

    if not wait_service(srv_arm_stow):
        rospy.logwarn('arm_command service not available; skipping stow')
    else:
        try:
            proxy = rospy.ServiceProxy(srv_arm_stow, ArmCommand)
            rospy.loginfo('Sending arm stow command')
            resp = proxy(arm_req)
            rospy.loginfo(f"Stow response: success={getattr(resp, 'success', False)} message='{getattr(resp, 'message', '')}'")
        except Exception as e:
            rospy.logerr(f"Stow call failed: {e}")

    # 8) Move back to initial pose (vision frame 0,0,0)
    home_req = MoveToPositionRequest()
    p = Pose()
    p.position = Point(0.0, 0.0, 0.0)
    p.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    home_req.target_pose = p
    home_req.frame_name = 'vision'
    try:
        proxy = rospy.ServiceProxy(srv_move_to_position, MoveToPosition)
        rospy.loginfo('Returning to initial vision pose')
        resp = proxy(home_req)
        rospy.loginfo(f"Return response: success={getattr(resp,'success',False)} message='{getattr(resp,'message','')}'")
    except Exception as e:
        rospy.logerr(f"Return to initial pose failed: {e}")

    # 9) Sit down
    resp = call_trigger(srv_sit)
    if not resp or not getattr(resp, 'success', False):
        rospy.logwarn('Sit reported failure or uncertain')
    else:
        rospy.loginfo('Robot sitting')


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
