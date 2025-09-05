#!/usr/bin/env python3
"""Simple ROS client to exercise SpotRobotManager services safely.

Usage: run the SpotRobotManager node first (rosrun/roslaunch). Then run this
script from the package or with the python interpreter in the ROS environment.

This script calls: connect -> power_on -> stand -> move_to_position(body) -> move_to_initial_position(odom) -> sit -> power_off -> disconnect
It uses short durations and verifies service responses.
"""

import rospy
import time
from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose, Point, Quaternion

try:
    from spot_hololens_llm_interface.srv import MoveToPosition, MoveToPositionRequest, GetInitialPose, GetInitialPoseRequest
except Exception:
    MoveToPosition = None
    MoveToPositionRequest = None
    GetInitialPose = None
    GetInitialPoseRequest = None


def wait_and_call(service_name, service_type, req=None, timeout=10.0):
    rospy.loginfo(f"Waiting for service {service_name}...")
    try:
        rospy.wait_for_service(service_name, timeout=timeout)
    except rospy.ROSException as e:
        rospy.logerr(f"Service {service_name} not available: {e}")
        return None

    try:
        proxy = rospy.ServiceProxy(service_name, service_type)
        if req is None:
            return proxy()
        return proxy(req)
    except Exception as e:
        rospy.logerr(f"Call to {service_name} failed: {e}")
        return None

def main():
    rospy.init_node('test_real_spot_manager_client', anonymous=True)

    ns = '/spot_entrance'
    # Service names (private '~' in the node become '/spot_entrance/...')
    srv_connect = ns + '/connect'
    srv_power_on = ns + '/power_on'
    srv_power_off = ns + '/power_off'
    srv_stand = ns + '/stand'
    srv_sit = ns + '/sit'
    srv_move = ns + '/move_to_position'
    srv_get_initial_pose = ns + '/get_initial_pose'

    # # 1) Connect
    # resp = wait_and_call(srv_connect, Trigger)
    # if not resp or not getattr(resp, 'success', False):
    #     rospy.logerr(f"Connect failed or not acknowledged: {resp}")
    #     return
    # rospy.loginfo("Connected to robot manager")

    # # 2) Power on
    # resp = wait_and_call(srv_power_on, Trigger)
    # if not resp or not getattr(resp, 'success', False):
    #     rospy.logerr(f"Power on failed or not acknowledged: {resp}")
    #     # try to disconnect then exit
    #     wait_and_call(srv_connect.replace('/connect', '/disconnect'), Trigger)
    #     return
    # rospy.loginfo("Robot powered on")

    # 3) Stand
    # resp = wait_and_call(srv_stand, Trigger)
    # if not resp or not getattr(resp, 'success', False):
    #     rospy.logwarn(f"Stand reported failure or uncertain: {resp}")
    # else:
    #     rospy.loginfo("Robot standing")

    # 4) Move: send a small forward body-frame step (0.5 m)
    if MoveToPosition is None:
        rospy.logwarn("MoveToPosition service type not importable; skipping move test")
    else:
        req = MoveToPositionRequest()
        # Build a Pose: x forward, y left in body frame
        p = Pose()
        p.position = Point(0.0, 0.0, 0.0)
        # identity orientation (no rotation)

        p.orientation = Quaternion(0.0, 0.0, 0.258819, 0.9659258) # 30 degrees yaw
        req.target_pose = p
        req.frame_name = 'body'

        rospy.loginfo("Calling move_to_position (body frame) -> forward 0.5 m")
        resp = wait_and_call(srv_move, MoveToPosition, req=req)
        if not resp:
            rospy.logerr("Move service call failed")
        else:
            rospy.loginfo(f"Move response: success={getattr(resp, 'success', False)} message='{getattr(resp, 'message', '')}'")

    # 5) Get initial pose and move back to it
    # if GetInitialPose is not None:
    #     req = GetInitialPoseRequest()
    #     resp = wait_and_call(srv_get_initial_pose, GetInitialPose, req=req)
    #     if resp and getattr(resp, 'success', False):
    #         initial_pose = resp.initial_pose
    #         move_req = MoveToPositionRequest()
    #         move_req.target_pose = initial_pose.pose
    #         move_req.frame_name = 'odom'
    #         rospy.loginfo("Moving back to initial position in odom frame")
    #         move_resp = wait_and_call(srv_move, MoveToPosition, req=move_req)
    #         if not move_resp:
    #             rospy.logerr("Move back to initial position failed")
    #         else:
    #             rospy.loginfo(f"Move back response: success={getattr(move_resp, 'success', False)} message='{getattr(move_resp, 'message', '')}'")
    #     else:
    #         rospy.logwarn("Failed to get initial pose")
    # else:
    #     rospy.logwarn("GetInitialPose service not available")

    # 6) Sit
    # resp = wait_and_call(srv_sit, Trigger)
    # if not resp or not getattr(resp, 'success', False):
    #     rospy.logwarn(f"Sit reported failure or uncertain: {resp}")
    # else:
    #     rospy.loginfo("Robot sitting")

    # # 8) Power off
    # resp = wait_and_call(srv_power_off, Trigger)
    # if not resp or not getattr(resp, 'success', False):
    #     rospy.logwarn(f"Power off reported failure or uncertain: {resp}")
    # else:
    #     rospy.loginfo("Robot powered off")

    # # 9) Disconnect
    # resp = wait_and_call(ns + '/disconnect', Trigger)
    # rospy.loginfo(f"Disconnect response: {resp}")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
