#!/usr/bin/env python3
"""Simple ROS client to exercise SpotRobotManager services safely.

Usage: run the SpotRobotManager node first (rosrun/roslaunch). Then run this
script from the package or with the python interpreter in the ROS environment.

This script calls: connect -> power_on -> stand -> move_to_position(body) -> sit -> power_off -> disconnect
It uses short durations and verifies service responses.
"""

import rospy
import time
from std_srvs.srv import Trigger
from geometry_msgs.msg import Pose, Point, Quaternion

try:
    from spot_hololens_llm_interface.srv import MoveToPosition, MoveToPositionRequest
except Exception:
    MoveToPosition = None
    MoveToPositionRequest = None


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


def _wait_for_feedback(cmd_client, command_id, est_duration):
    """Wait for feedback from the robot after sending a command.

    Args:
        cmd_client: The command client to use.
        command_id: The ID of the command to wait for.
        est_duration: The estimated duration for the command to complete.

    Returns:
        bool: True if the command was successful, False otherwise.
        str: The status message from the robot.
    """
    # Initialize the last status and start time
    last_status = ""
    last_status_name = ""
    start_time = time.time()

    # Wait for a status update or until the estimated duration has passed
    while (time.time() - start_time) < est_duration:
        # Get the latest status from the command client
        try:
            status = cmd_client.get_status(command_id)
        except Exception as e:
            rospy.logerr(f"Error getting status: {e}")
            return False, str(e)

        # If we have a new status, update the last status
        if status != last_status:
            last_status = status
            last_status_name = status.upper()  # Convert to uppercase for comparison

            # Log the status update
            rospy.loginfo(f"Status update: {last_status_name}")

            # Only return true for explicit success conditions
            if ("SUCCESS" in last_status_name) or ("REACHED" in last_status_name) or ("COMPLETE" in last_status_name):
                return True, last_status_name

        # Sleep briefly before checking the status again
        time.sleep(0.1)

    # If we get NO_STATUS, consider it a failure
    if last_status_name == "NO_STATUS":
        return False, "NO_STATUS_FROM_ROBOT"

    # If we timed out, log a warning and return False
    rospy.logwarn("Timeout waiting for command feedback")
    return False, "TIMEOUT"


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
    resp = wait_and_call(srv_stand, Trigger)
    if not resp or not getattr(resp, 'success', False):
        rospy.logwarn(f"Stand reported failure or uncertain: {resp}")
    else:
        rospy.loginfo("Robot standing")

    # 4) Move: send a small forward body-frame step (0.5 m)
    if MoveToPosition is None:
        rospy.logwarn("MoveToPosition service type not importable; skipping move test")
    else:
        req = MoveToPositionRequest()
        # Build a Pose: x forward, y left in body frame
        p = Pose()
        p.position = Point(0.5, 0.0, 0.0)
        # identity orientation (no rotation)
        p.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
        req.target_pose = p
        req.frame_name = 'body'

        rospy.loginfo("Calling move_to_position (body frame) -> forward 0.5 m")
        resp = wait_and_call(srv_move, MoveToPosition, req=req)
        if not resp:
            rospy.logerr("Move service call failed")
        else:
            rospy.loginfo(f"Move response: success={getattr(resp, 'success', False)} message='{getattr(resp, 'message', '')}'")

    # 5) Sit
    resp = wait_and_call(srv_sit, Trigger)
    if not resp or not getattr(resp, 'success', False):
        rospy.logwarn(f"Sit reported failure or uncertain: {resp}")
    else:
        rospy.loginfo("Robot sitting")

    # 6) Power off
    resp = wait_and_call(srv_power_off, Trigger)
    if not resp or not getattr(resp, 'success', False):
        rospy.logwarn(f"Power off reported failure or uncertain: {resp}")
    else:
        rospy.loginfo("Robot powered off")

    # 7) Disconnect
    resp = wait_and_call(ns + '/disconnect', Trigger)
    rospy.loginfo(f"Disconnect response: {resp}")


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
