#!/usr/bin/env python

import argparse
import sys
import cv2
import time
import logging
import math
from google.protobuf import duration_pb2

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand, blocking_sit, block_until_arm_arrives
from bosdyn.choreography.client.choreography import ChoreographyClient

from bosdyn.api import (arm_command_pb2, geometry_pb2, manipulation_api_pb2, robot_command_pb2, synchronized_command_pb2, trajectory_pb2)
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2

from bosdyn.client import math_helpers
from bosdyn.client.math_helpers import quat_to_eulerZYX
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b, get_vision_tform_body
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient

from spot_fsm_control.manipulator import ManipulatorFunctions
from bosdyn.util import seconds_to_duration

from spot_fsm_control.arm_impedance_control_helpers import (apply_force_at_current_position,
                                           get_impedance_mobility_params, get_root_T_ground_body)


class SpotControlInterface(ManipulatorFunctions):

    def __init__(self, direct_control_frequency):
        # super(SpotControlInterface, self).__init__()
        self.hostname = "192.168.80.3"
        self.command_client = None
        self.forward, self.strafe, self.rotate = 0, 0, 0
        self.prev_close_or_open = "close"
        self.direct_control_trajectory_list = []
        self.current_state_direct_control = False
        self.direct_control_frequency = direct_control_frequency
        self.init_pos_empty = False
        

    def stop(self):
        cmd = RobotCommandBuilder.stop_command()
        self.command_client.robot_command(cmd)
        cmd = RobotCommandBuilder.arm_stow_command()
        self.command_client.robot_command(cmd)

    def stand(self, height):
        cmd = RobotCommandBuilder.synchro_stand_command(body_height=height)
        self.command_client.robot_command(cmd)

    def move_command(self, duration=2):
        print("Move:", self.forward, self.strafe, self.rotate)
        cmd = RobotCommandBuilder.synchro_velocity_command(self.forward, self.strafe, self.rotate)
        self.command_client.robot_command(cmd, end_time_secs=time.time() + duration) # robot_command_async
        self.forward, self.strafe, self.rotate = 0, 0, 0

    def sit_down(self):
        print("sitting")
        cmd = RobotCommandBuilder.synchro_sit_command()
        self.command_client.robot_command(cmd)

    def stop(self):
        print("Stopping")
        cmd = RobotCommandBuilder.stop_command()
        self.command_client.robot_command(cmd)
        
    def two_d_location_body_frame_command(self, x, y, yaw, locomotion_hint=spot_command_pb2.HINT_CRAWL):
        trajectory_command = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(x, y, yaw, self.robot_sdk.get_frame_tree_snapshot(), locomotion_hint=locomotion_hint)
        cmd_id = self.command_client.robot_command(trajectory_command, end_time_secs=time.time()+10)
        max_time = 10
        start_time = time.time()
        while True:
            feedback = self.command_client.robot_command_feedback(cmd_id)
            mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
            if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
                print('Failed to reach the goal')
                break
            traj_feedback = mobility_feedback.se2_trajectory_feedback
            if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                    traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
                print('Arrived at the goal.')
                break
            time.sleep(0.2)
            print(".", end="")
            
            if time.time() - start_time > max_time:
                break
            
                
    def calibration_movement_in_robot_frame(self, x=1.1, y=0.0, yaw=0.0, locomotion_hint=spot_command_pb2.HINT_AUTO):
        odom_positions = []
        vision_odom_positions = []
        trajectory_command = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(x,y,yaw, self.robot_sdk.get_frame_tree_snapshot(), locomotion_hint=locomotion_hint) # Standard locomotion_hint == 0spot_command_pb2.HINT_AUTO
        cmd_id = self.command_client.robot_command(trajectory_command, end_time_secs=time.time()+10)
        
        while True:
            feedback = self.command_client.robot_command_feedback(cmd_id)
            mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
            odom_T_body = get_a_tform_b(self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,"odom","body")
            vision_T_body = get_a_tform_b(self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,"vision","body") 
            odomBodyEuler = quat_to_eulerZYX(odom_T_body.rot)
            visionBodyEuler = quat_to_eulerZYX(vision_T_body.rot)
            odom_positions.append([odom_T_body.x, odom_T_body.y, odom_T_body.z, odomBodyEuler[0], odomBodyEuler[1], odomBodyEuler[2]])
            vision_odom_positions.append([vision_T_body.x, vision_T_body.y, vision_T_body.y, visionBodyEuler[0], visionBodyEuler[1], visionBodyEuler[2]])
            if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
                print('Failed to reach the goal')
                break
            traj_feedback = mobility_feedback.se2_trajectory_feedback
            if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                    traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
                print('Arrived at the goal.')
                break
            time.sleep(0.1)
            
        return odom_positions, vision_odom_positions
        
    
    def pick_up_object_in_front_of_robot(self):
        image_responses = self.image_client.get_image_from_sources(['frontleft_fisheye_image'])
        image = image_responses[0]
        x, y = 525, 380

        pick_vec = geometry_pb2.Vec2(x=x, y=y)

        # Build the proto
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole)

        # Optionally add a grasp constraint.  This lets you tell the robot you only want top-down grasps or side-on grasps.
        # add_grasp_constraint(grasp)
        
        axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

        # The axis in the vision frame is the negative z-axis
        axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)
        
        # Add the vector constraint to our proto.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper_ewrt_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vo)

        # We'll take anything within about 10 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

        # Ask the robot to pick up the object
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

        # Send the request
        cmd_response = self.manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request)

        # Get feedback from the robot
        start_time = time.time()
        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            # Send the request
            response = self.manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print(
                f'Current state: {manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state)}'
            )

            if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                break

            if (time.time() - start_time) > 20:
                break
            time.sleep(0.25)
            
        self.ready_or_stow_arm(stow=True)
            
    def drop_off_object_in_front_of_robot(self):
        self.ready_or_stow_arm()
        
        task_T_tool_desired = math_helpers.SE3Pose(0.75, 0, 0.45, math_helpers.Quat(1, 0, 0, 0))
        odom_T_task = get_root_T_ground_body(robot_state=self.robot_sdk.robot_state_client.get_robot_state(),
                                             root_frame_name=GRAV_ALIGNED_BODY_FRAME_NAME)
        wr1_T_tool = math_helpers.SE3Pose(0, 0, 0, math_helpers.Quat.from_pitch(-math.pi / 2))
        
        self.robot_sdk.move_to_cartesian_pose_rt_task(task_T_tool_desired, odom_T_task, wr1_T_tool)
        
        time.sleep(2)
        
        self.gripper("open")
        time.sleep(0.5)
        self.gripper("close")
        
        self.ready_or_stow_arm(stow=True)

    def gripper(self, close_or_open):
        if close_or_open == "open":
            cmd = RobotCommandBuilder.claw_gripper_open_command()
            self.command_client.robot_command(cmd)
        elif close_or_open == "close":
            # rubber tomato 0.375
            cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(0.375)
            self.command_client.robot_command(cmd)
        self.prev_close_or_open = close_or_open
            
    def cartesian_hand_position(self, pos, quat, time=1):
        hand_ewrt_flat_body = geometry_pb2.Vec3(x=pos[0], y=pos[1], z=pos[2])
        flat_body_Q_hand = geometry_pb2.Quaternion(w=quat[0], x=quat[1], y=quat[2], z=quat[3])

        flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                                rotation=flat_body_Q_hand)

        robot_state = self.robot_state_client.get_robot_state()
        target_frame = "hand"
        source_frame = "flat_body"
        odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                         source_frame, target_frame)

        odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_proto(flat_body_T_hand)

        # duration in seconds
        seconds = time

        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
            odom_T_hand.rot.y, odom_T_hand.rot.z, source_frame, seconds)

        # Send the request
        cmd_id = self.command_client.robot_command(arm_command)

        # Wait until the arm arrives at the goal.
        block_until_arm_arrives(self.command_client, cmd_id)
        
    def move_to_cartesian_pose_rt_task(self, task_T_desired, root_T_task, wr1_T_tool):
        robot_cmd = RobotCommandBuilder.synchro_stand_command(params=get_impedance_mobility_params())
        arm_cart_cmd = robot_cmd.synchronized_command.arm_command.arm_cartesian_command

        # Set up our root frame, task frame, and tool frame.
        arm_cart_cmd.root_frame_name = GRAV_ALIGNED_BODY_FRAME_NAME
        arm_cart_cmd.root_tform_task.CopyFrom(root_T_task.to_proto())
        arm_cart_cmd.wrist_tform_tool.CopyFrom(wr1_T_tool.to_proto())

        # Do a single point goto to a desired pose in the task frame.
        cartesian_traj = arm_cart_cmd.pose_trajectory_in_task
        traj_pt = cartesian_traj.points.add()
        traj_pt.time_since_reference.CopyFrom(seconds_to_duration(1.0))
        traj_pt.pose.CopyFrom(task_T_desired.to_proto())

        # Execute the Cartesian command.
        cmd_id = self.command_client.robot_command(robot_cmd)
        
        # block_until_arm_arrives(self.command_client, cmd_id, 1/self.direct_control_frequency)
        
    def stop_direct_control(self):
        self.current_state_direct_control = False
        print("Stopping Direct Control")
                
    def ready_or_stow_arm(self, stow=False):
        if stow:
            cmd = RobotCommandBuilder.arm_stow_command()
        else:
            cmd = RobotCommandBuilder.arm_ready_command()
        cmd_id = self.command_client.robot_command(cmd) 
        block_until_arm_arrives(self.command_client, cmd_id)

    def keyboard_movement_control(self):
        wait_time = 20
        while True:
            self.forward, self.strafe, self.rotate = 0, 0, 0

            if cv2.waitKey(wait_time) & 0xFF == ord('p'):
                break
            if cv2.waitKey(wait_time) == ord('a'):
                self.strafe += 0.4
            if cv2.waitKey(wait_time) == ord('d'):
                self.strafe -= 0.4
            if cv2.waitKey(wait_time) == ord('i'): # high
                self.stand(0.5)
            if cv2.waitKey(wait_time) == ord('k'): # low
                self.stand(0.0)
            if cv2.waitKey(wait_time) == ord('q'):
                self.stop()
            if cv2.waitKey(wait_time) == ord('j'):
                self.rotate += 0.2
            if cv2.waitKey(wait_time) == ord('l'):
                self.rotate -= 0.2
            if cv2.waitKey(wait_time) == ord('w'):
                self.forward += 0.4
            if cv2.waitKey(wait_time) == ord('s'):
                self.forward -= 0.4
            if cv2.waitKey(wait_time) == ord("b"):
                self.sit_down()

            if (self.strafe != 0) or (self.forward != 0) or (self.rotate != 0):
                self.move_command()

            cv2.waitKey(1000)

        self.stand(0.4)
        return

def add_grasp_constraint(grasp):
    axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

    # The axis in the vision frame is the negative z-axis
    axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)
    
    # Add the vector constraint to our proto.
    constraint = grasp.grasp_params.allowable_orientation.add()
    constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
        axis_on_gripper_ewrt_gripper)
    constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
        axis_to_align_with_ewrt_vo)

    # We'll take anything within about 10 degrees for top-down or horizontal grasps.
    constraint.vector_alignment_with_tolerance.threshold_radians = 0.17
    
    return grasp
            

if __name__ == "__main__":
    spot = SpotControlInterface()