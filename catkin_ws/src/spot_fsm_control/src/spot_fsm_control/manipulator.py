#!/usr/bin/env python

import argparse
import sys
import logging
import time
import cv2
import numpy as np

from google.protobuf import duration_pb2

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import (arm_command_pb2, geometry_pb2, robot_command_pb2, synchronized_command_pb2,
                        trajectory_pb2, estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2)
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_vision_tform_body, math_helpers

from bosdyn.util import seconds_to_duration
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)

class ManipulatorFunctions:

    def __init__(self):
        pass

    def manual_manipulator_control(self):
        print("Manual manipulator control")
        pass

    def gaze_control(self):
        print("Gaze control")
        # Unstow the arm
        unstow = RobotCommandBuilder.arm_ready_command()
        unstow_command_id = self.command_client.robot_command(unstow)
        block_until_arm_arrives(self.command_client, unstow_command_id, 3.0)

        # Convert the location from the moving base frame to the world frame.
        robot_state = self.robot_state_client.get_robot_state()
        odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                        ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        # Look at a point 3 meters in front and 4 meters to the left.
        # We are not specifying a hand location, the robot will pick one.
        gaze_target_in_odom = odom_T_flat_body.transform_point(x=3.0, y=4.0, z=0)

        gaze_command = RobotCommandBuilder.arm_gaze_command(gaze_target_in_odom[0],
                                                            gaze_target_in_odom[1],
                                                            gaze_target_in_odom[2], ODOM_FRAME_NAME)
        # Make the open gripper RobotCommand
        gripper_command = RobotCommandBuilder.claw_gripper_open_command()

        # Combine the arm and gripper commands into one RobotCommand
        synchro_command = RobotCommandBuilder.build_synchro_command(gripper_command, gaze_command)
        gaze_command_id = self.command_client.robot_command(synchro_command)

        block_until_arm_arrives(self.command_client, gaze_command_id, 4.0)

        # Look to the left and the right with the hand.
        # Robot's frame is X+ forward, Z+ up, so left and right is +/- in Y.
        x = 4.0  # look 4 meters ahead
        start_y = 4.0
        end_y = -4.0
        z = 0.1  # Look ahead, not up or down

        traj_time = 10  # take 5.5 seconds to look from left to right.

        start_pos_in_odom_tuple = odom_T_flat_body.transform_point(x=x, y=start_y, z=z)
        start_pos_in_odom = geometry_pb2.Vec3(x=start_pos_in_odom_tuple[0],
                                            y=start_pos_in_odom_tuple[1],
                                            z=start_pos_in_odom_tuple[2])

        end_pos_in_odom_tuple = odom_T_flat_body.transform_point(x=x, y=end_y, z=z)
        end_pos_in_odom = geometry_pb2.Vec3(x=end_pos_in_odom_tuple[0], y=end_pos_in_odom_tuple[1],
                                            z=end_pos_in_odom_tuple[2])

        # Create the trajectory points
        point1 = trajectory_pb2.Vec3TrajectoryPoint(point=start_pos_in_odom)

        duration_seconds = int(traj_time)
        duration_nanos = int((traj_time - duration_seconds) * 1e9)

        point2 = trajectory_pb2.Vec3TrajectoryPoint(
            point=end_pos_in_odom,
            time_since_reference=duration_pb2.Duration(seconds=duration_seconds,
                                                    nanos=duration_nanos))

        # Build the trajectory proto
        traj_proto = trajectory_pb2.Vec3Trajectory(points=[point1, point2])

        # Build the proto
        gaze_cmd = arm_command_pb2.GazeCommand.Request(target_trajectory_in_frame1=traj_proto,
                                                    frame1_name=ODOM_FRAME_NAME,
                                                    frame2_name=ODOM_FRAME_NAME)
        arm_command = arm_command_pb2.ArmCommand.Request(arm_gaze_command=gaze_cmd)
        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_command)
        command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)

        # Make the open gripper RobotCommand
        gripper_command = RobotCommandBuilder.claw_gripper_open_command()

        # Combine the arm and gripper commands into one RobotCommand
        synchro_command = RobotCommandBuilder.build_synchro_command(gripper_command, command)
        gaze_command_id = self.command_client.robot_command(command)
        logging.info('Sending gaze trajectory.')

        # Wait until the robot completes the gaze before issuing the next command.
        block_until_arm_arrives(self.command_client, gaze_command_id, timeout_sec=traj_time + 3.0)

        # ------------- #

        # Now make a gaze trajectory that moves the hand around while maintaining the gaze.
        # We'll use the same trajectory as before, but add a trajectory for the hand to move to.

        # Hand will start to the left (while looking left) and move to the right.
        hand_x = 0.75  # in front of the robot.
        hand_y = 0  # centered
        hand_z_start = 0  # body height
        hand_z_end = 0.25  # above body height

        hand_vec3_start = geometry_pb2.Vec3(x=hand_x, y=hand_y, z=hand_z_start)
        hand_vec3_end = geometry_pb2.Vec3(x=hand_x, y=hand_y, z=hand_z_end)

        # We specify an orientation for the hand, which the robot will use its remaining degree
        # of freedom to achieve.  Most of it will be ignored in favor of the gaze direction.
        qw = 1
        qx = 0
        qy = 0
        qz = 0
        quat = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

        # Build a trajectory
        hand_pose1_in_flat_body = geometry_pb2.SE3Pose(position=hand_vec3_start, rotation=quat)
        hand_pose2_in_flat_body = geometry_pb2.SE3Pose(position=hand_vec3_end, rotation=quat)

        hand_pose1_in_odom = odom_T_flat_body * math_helpers.SE3Pose.from_obj(
            hand_pose1_in_flat_body)
        hand_pose2_in_odom = odom_T_flat_body * math_helpers.SE3Pose.from_obj(
            hand_pose2_in_flat_body)

        traj_point1 = trajectory_pb2.SE3TrajectoryPoint(pose=hand_pose1_in_odom.to_proto())

        # We'll make this trajectory the same length as the one above.
        traj_point2 = trajectory_pb2.SE3TrajectoryPoint(
            pose=hand_pose2_in_odom.to_proto(),
            time_since_reference=duration_pb2.Duration(seconds=duration_seconds,
                                                    nanos=duration_nanos))

        hand_traj = trajectory_pb2.SE3Trajectory(points=[traj_point1, traj_point2])

        # Build the proto
        gaze_cmd = arm_command_pb2.GazeCommand.Request(target_trajectory_in_frame1=traj_proto,
                                                    frame1_name=ODOM_FRAME_NAME,
                                                    tool_trajectory_in_frame2=hand_traj,
                                                    frame2_name=ODOM_FRAME_NAME)

        arm_command = arm_command_pb2.ArmCommand.Request(arm_gaze_command=gaze_cmd)
        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_command)
        command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)

        # Make the open gripper RobotCommand
        gripper_command = RobotCommandBuilder.claw_gripper_open_command()

        # Combine the arm and gripper commands into one RobotCommand
        synchro_command = RobotCommandBuilder.build_synchro_command(gripper_command, command)

        # Send the request
        gaze_command_id = self.command_client.robot_command(synchro_command)
        logging.info('Sending gaze trajectory with hand movement.')

        # Wait until the robot completes the gaze before powering off.
        block_until_arm_arrives(self.command_client, gaze_command_id, timeout_sec=traj_time + 3.0)

        stow = RobotCommandBuilder.arm_stow_command()
        stow_command_id = self.command_client.robot_command(stow)
        block_until_arm_arrives(self.command_client, stow_command_id, 3.0)

        
    def arm_object_grasp(self):
        print("Grasp an object")
        # Take a picture with a camera
        image_source = "right_fisheye_image"
        logging.info('Getting an image from: ' + image_source)
        image_responses = self.image_client.get_image_from_sources([image_source])

        if len(image_responses) != 1:
            print('Got invalid number of images: ' + str(len(image_responses)))
            print(image_responses)
            assert False

        image = image_responses[0]
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
        else:
            dtype = np.uint8
        img = np.fromstring(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(image.shot.image.rows, image.shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)

        # Show the image to the user and wait for them to click on a pixel
        logging.info('Click on an object to start grasping...')
        image_title = 'Click to grasp'
        cv2.namedWindow(image_title)
        cv2.setMouseCallback(image_title, cv_mouse_callback)

        global g_image_click, g_image_display
        g_image_click = None
        g_image_display = img
        cv2.imshow(image_title, g_image_display)
        while g_image_click is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                # Quit
                print('"q" pressed, exiting.')
                exit(0)

        logging.info('Picking object at image location (' + str(g_image_click[0]) + ', ' +
                        str(g_image_click[1]) + ')')

        pick_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])

        # Build the proto
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole)

        # Optionally add a grasp constraint.  This lets you tell the robot you only want top-down grasps or side-on grasps.
        add_grasp_constraint(grasp, self.robot_state_client)

        # Ask the robot to pick up the object
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

        # Send the request
        cmd_response = self.manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request)

        # Get feedback from the robot
        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            # Send the request
            response = self.manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print('Current state: ',
                manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

            if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                break

            time.sleep(0.25)

        logging.info('Finished grasp.')

    def walk_to_object(self):
        print("Walking to object")
        # Take a picture with a camera
        image_source = "right_fisheye_image"
        logging.info('Getting an image from: ' + image_source)
        image_responses = self.image_client.get_image_from_sources([image_source])

        if len(image_responses) != 1:
            print('Got invalid number of images: ' + str(len(image_responses)))
            print(image_responses)
            assert False

        image = image_responses[0]
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
        else:
            dtype = np.uint8
        img = np.fromstring(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(image.shot.image.rows, image.shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)

        # Show the image to the user and wait for them to click on a pixel
        logging.info('Click on an object to walk up to...')
        image_title = 'Click to walk up to something'
        cv2.namedWindow(image_title)
        cv2.setMouseCallback(image_title, cv_mouse_callback)

        global g_image_click, g_image_display
        g_image_display = img
        cv2.imshow(image_title, g_image_display)
        while g_image_click is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                # Quit
                print('"q" pressed, exiting.')
                exit(0)

        logging.info('Walking to object at image location (' + str(g_image_click[0]) + ', ' +
                          str(g_image_click[1]) + ')')

        walk_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])

        # Optionally populate the offset distance parameter.
        offset_distance = None

        # Build the proto
        walk_to = manipulation_api_pb2.WalkToObjectInImage(
            pixel_xy=walk_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole, offset_distance=offset_distance)

        # Ask the robot to pick up the object
        walk_to_request = manipulation_api_pb2.ManipulationApiRequest(
            walk_to_object_in_image=walk_to)

        # Send the request
        cmd_response = self.manipulation_api_client.manipulation_api_command(
            manipulation_api_request=walk_to_request)

        # Get feedback from the robot
        while True:
            time.sleep(0.25)
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            # Send the request
            response = self.manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print('Current state: ',
                  manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

            if response.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
                break

        logging.info('Finished.')


    def arm_trajectory(self):

        x = 0.75  # a reasonable position in front of the robot
        y1 = 0  # centered
        y2 = 0.4  # 0.4 meters to the robot's left
        y3 = -0.4  # 0.4 meters to the robot's right
        z = 0  # at the body's height

        # Use the same rotation as the robot's body.
        rotation = math_helpers.Quat()

        # Define times (in seconds) for each point in the trajectory.
        t_first_point = 0  # first point starts at t = 0 for the trajectory.
        t_second_point = 4.0
        t_third_point = 8.0

        # Build the points in the trajectory.
        hand_pose1 = math_helpers.SE3Pose(x=x, y=y1, z=z, rot=rotation)
        hand_pose2 = math_helpers.SE3Pose(x=x, y=y2, z=z, rot=rotation)
        hand_pose3 = math_helpers.SE3Pose(x=x, y=y3, z=z, rot=rotation)

        # Build the points by combining the pose and times into protos.
        traj_point1 = trajectory_pb2.SE3TrajectoryPoint(
            pose=hand_pose1.to_proto(), time_since_reference=seconds_to_duration(t_first_point))
        traj_point2 = trajectory_pb2.SE3TrajectoryPoint(
            pose=hand_pose2.to_proto(), time_since_reference=seconds_to_duration(t_second_point))
        traj_point3 = trajectory_pb2.SE3TrajectoryPoint(
            pose=hand_pose3.to_proto(), time_since_reference=seconds_to_duration(t_third_point))

        # Build the trajectory proto by combining the points.
        hand_traj = trajectory_pb2.SE3Trajectory(points=[traj_point1, traj_point2, traj_point3])

        # Build the command by taking the trajectory and specifying the frame it is expressed
        # in.
        #
        # In this case, we want to specify the trajectory in the body's frame, so we set the
        # root frame name to the flat body frame.
        arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(
            pose_trajectory_in_task=hand_traj, root_frame_name=GRAV_ALIGNED_BODY_FRAME_NAME)

        # Pack everything up in protos.
        arm_command = arm_command_pb2.ArmCommand.Request(
            arm_cartesian_command=arm_cartesian_command)

        synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
            arm_command=arm_command)

        robot_command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)

        # Keep the gripper closed the whole time.
        robot_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
            0, build_on_command=robot_command)

        logging.info("Sending trajectory command...")

        # Send the trajectory to the robot.
        cmd_id = self.command_client.robot_command(robot_command)

        # Wait until the arm arrives at the goal.
        while True:
            feedback_resp = self.command_client.robot_command_feedback(cmd_id)
            logging.info('Distance to final point: ' + '{:.2f} meters'.format(
                feedback_resp.feedback.synchronized_feedback.arm_command_feedback.
                arm_cartesian_feedback.measured_pos_distance_to_goal) + ', {:.2f} radians'.format(
                    feedback_resp.feedback.synchronized_feedback.arm_command_feedback.
                    arm_cartesian_feedback.measured_rot_distance_to_goal))

            if feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
                logging.info('Move complete.')
                break
            time.sleep(0.1)

        stow = RobotCommandBuilder.arm_stow_command()
        stow_command_id = self.command_client.robot_command(stow)
        block_until_arm_arrives(self.command_client, stow_command_id, 3.0)



def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # Draw some lines on the image.
        #print('mouse', x, y)
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click to grasp'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)


def add_grasp_constraint(grasp, robot_state_client):
    # There are 3 types of constraints:
    #   1. Vector alignment
    #   2. Full rotation
    #   3. Squeeze grasp
    #
    # You can specify more than one if you want and they will be OR'ed together.

    # For these options, we'll use a vector alignment constraint.
    force_top_down_grasp = False
    force_horizontal_grasp = False
    force_45_angle_grasp = False
    force_squeeze_grasp = False
    use_vector_constraint = force_top_down_grasp or force_horizontal_grasp

    # Specify the frame we're using.
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector_constraint:
        if force_top_down_grasp:
            # Add a constraint that requests that the x-axis of the gripper is pointing in the
            # negative-z direction in the vision frame.

            # The axis on the gripper is the x-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

            # The axis in the vision frame is the negative z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

        if force_horizontal_grasp:
            # Add a constraint that requests that the y-axis of the gripper is pointing in the
            # positive-z direction in the vision frame.  That means that the gripper is constrained to be rolled 90 degrees and pointed at the horizon.

            # The axis on the gripper is the y-axis.
            axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

            # The axis in the vision frame is the positive z-axis
            axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

        # Add the vector constraint to our proto.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
            axis_on_gripper_ewrt_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
            axis_to_align_with_ewrt_vo)

        # We'll take anything within about 10 degrees for top-down or horizontal grasps.
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.17

    elif force_45_angle_grasp:
        # Demonstration of a RotationWithTolerance constraint.  This constraint allows you to
        # specify a full orientation you want the hand to be in, along with a threshold.
        #
        # You might want this feature when grasping an object with known geometry and you want to
        # make sure you grasp a specific part of it.
        #
        # Here, since we don't have anything in particular we want to grasp,  we'll specify an
        # orientation that will have the hand aligned with robot and rotated down 45 degrees as an
        # example.

        # First, get the robot's position in the world.
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)

        # Rotation from the body to our desired grasp.
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45 degrees
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp

        # Turn into a proto
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(vision_Q_grasp.to_proto())

        # We'll accept anything within +/- 10 degrees
        constraint.rotation_with_tolerance.threshold_radians = 0.17

    elif force_squeeze_grasp:
        # Tell the robot to just squeeze on the ground at the given point.
        constraint = grasp.grasp_params.allowable_orientation.add()
        constraint.squeeze_grasp.SetInParent()