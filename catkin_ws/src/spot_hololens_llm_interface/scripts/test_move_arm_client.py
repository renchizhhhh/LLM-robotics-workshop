#!/usr/bin/env python3
import rospy
import actionlib

from spot_hololens_llm_interface.msg import MoveArmPoseAction, MoveArmPoseGoal


def send_goal(client, x, y, z, qw, qx, qy, qz, duration=2.0, open_gripper=False, timeout_extra=10.0):
    goal = MoveArmPoseGoal()
    goal.x = x
    goal.y = y
    goal.z = z
    goal.qw = qw
    goal.qx = qx
    goal.qy = qy
    goal.qz = qz
    goal.duration = duration
    goal.open_gripper = open_gripper

    rospy.loginfo(f"Sending MoveArmPose goal: pos=({x:.2f},{y:.2f},{z:.2f}) dur={duration} open_gripper={open_gripper}")
    client.send_goal(goal)

    # Wait for result a bit longer than duration
    finished = client.wait_for_result(rospy.Duration(duration + timeout_extra))
    if not finished:
        rospy.logerr("move_arm_pose did not finish in time; canceling goal")
        client.cancel_goal()
        return None
    result = client.get_result()
    return result


def main():
    rospy.init_node('test_move_arm_client')

    client = actionlib.SimpleActionClient('move_arm_pose', MoveArmPoseAction)
    rospy.loginfo("Waiting for move_arm_pose action server...")
    if not client.wait_for_server(rospy.Duration(10.0)):
        rospy.logerr('move_arm_pose action server not available')
        return

    # Pose parameters copied from pose_grasp_node: hold pose then releasing pose
    # Hold pose (approximate): x=0.75, y=0, z=0.25, identity quaternion
    res1 = send_goal(client, 0.75, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0, duration=2.0, open_gripper=True)
    if res1 is None:
        rospy.logerr('Hold pose goal failed or timed out')
    else:
        rospy.loginfo(f'Hold pose result: success={getattr(res1, "success", None)} message="{getattr(res1, "message", "")}"')

    input("Press Enter to continue...")

    # Releasing pose: move further out in x (1.0)
    res2 = send_goal(client, 0.85, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0, duration=2.0, open_gripper=False)
    if res2 is None:
        rospy.logerr('Releasing pose goal failed or timed out')
    else:
        rospy.loginfo(f'Releasing pose result: success={getattr(res2, "success", None)} message="{getattr(res2, "message", "")}"')


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
