#!/usr/bin/env python
import os
import time
import sys
import threading
sys.path.append("./hagrid/")
sys.path.append("./src/")

import math
import rospy
from std_msgs.msg import String, Float32MultiArray
from geometry_msgs.msg import Pose
import ast
import csv
import numpy as np
import logging
from datetime import datetime

# for parsing protobuf messages like robot_state
from google.protobuf.json_format import MessageToDict
from pathlib import Path

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand, blocking_sit, block_until_arm_arrives
from bosdyn.choreography.client.choreography import ChoreographyClient

from bosdyn.api import (image_pb2, arm_command_pb2, geometry_pb2, robot_command_pb2, synchronized_command_pb2, trajectory_pb2)
from bosdyn.client import math_helpers
from bosdyn.client.math_helpers import quat_to_eulerZYX
from bosdyn.client.math_helpers import Quat, SE3Pose

from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b, get_odom_tform_body
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient

from spot_hololens_llm_interface.spot_control_interface import SpotControlInterface
from spot_hololens_llm_interface.finite_state_machine import SpotStateMachine
from spot_hololens_llm_interface.arm_impedance_control_helpers import get_root_T_ground_body
from spot_hololens_llm_interface.video_stream_saver import VideoStreamSaver

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)

DIRECT_CONTROL_FREQUENCY = 15  # Hz Max 60


def try_state_send(state_machine, action):
    """
    Send an action to the state machine with error handling and retry logic.
    
    Args:
        state_machine: The FSM instance
        action: The action command to send
    """
    print("Current state:", state_machine.current_state)
    try:
        state_machine.send(action)
    except Exception as e:
        print(e)
        # Never auto-retry pick commands to avoid double execution
        if action in ("pick_up_object", "pick_up_green", "pick_up_red", "pick_up_blue", "pick_up_qr", "pick_up_label"):
            print("Not retrying pick command after error")
            return
        try:
            state_machine.send("stop_action")
            print("Stop action first")
            state_machine.send(action)
        except Exception as e:
            print(e)
            try:
                state_machine.send("stand_up")
                print("Stand up first")
                state_machine.send(action)
            except Exception as e:
                print(f"{action} not possible")
                print(e)


class FsmNode:

    def __init__(self, robot: SpotControlInterface, robot_sdk, timestamp, participant=-1, condition="null"):
        ## data collection
        self.node_start_time = time.time()
        self.columns_hololens_data = ["timestamp", "gaze_origin [x,y,z]", "gaze_direction unit vector [x,y,z]", "gaze_direction screen position [x,y,?]", "camera position [x,y,z]", "camera orientation [w,x,y,z]"]
        self.columns_spot_data = ["timestamp", "position_odom_spot [x,y,z]", "orientation_odom_spot [yaw,pitch,roll]", "position_vision_spot [x,y,z]", "orientation_vision_spot [yaw,pitch,roll]"]
        self.columns_action_scripts = ["timestamp", "action_executed", "battery_percentage"]
        
        self.participant_number = participant
        participant_dir = Path(__file__).parent.parent.parent.parent.parent.joinpath(f"/data/experiments/P{participant:03d}/")

        self.pub_status = rospy.Publisher('/robot_status', String, queue_size=2)
        
        # Command deduplication settings
        self.last_command = None
        self.last_command_time = 0
        self.command_deduplication_window = 5.0  # seconds

        participant_dir.mkdir(parents=True, exist_ok=True)    
        self.filename_odom = f"{participant_dir}/{condition}_odom_{timestamp}.csv"
        self.filename_hololens = f"{participant_dir}/{condition}_hololens_{timestamp}.csv"
        self.filename_actions = f"{participant_dir}/{condition}_actions_{timestamp}.csv"
        
        self._initialize_csv_files()
        
        self.robot = robot
        self.robot_sdk = robot_sdk
        self.sm = SpotStateMachine(robot=robot)
        self.arm_pos_init = [0, 0, 0]
        self.arm_ori_init = [1, 0, 0, 0]
        zero_pose = math_helpers.SE3Pose(x=0, y=0, z=0, rot=math_helpers.Quat())
        zero_trajectory_pose = trajectory_pb2.SE3TrajectoryPoint(pose=zero_pose.to_proto())
        self.direct_control_trajectory_list = [zero_trajectory_pose]
        
        # Pick a task frame that is beneath the robot body center, on the ground.
        if robot:
            self.odom_T_task = get_root_T_ground_body(robot_state=self.robot.robot_state_client.get_robot_state(),
                                                root_frame_name=GRAV_ALIGNED_BODY_FRAME_NAME)

        # Set our tool frame to be the tip of the robot's bottom jaw. Flip the orientation so that
        # when the hand is pointed downwards, the tool's z-axis is pointed upward.
        self.wr1_T_tool = SE3Pose(0, 0, 0, Quat.from_pitch(-math.pi / 2))
        
        self.frequency_pose_count = int(60 // DIRECT_CONTROL_FREQUENCY)
        self.pose_receive_count = 0
        
        ## Odometry pose init
        self.pose = np.array([0, 0, 0, 0, 0, 0])
        
        self.initial_position_vision_odom = np.array([0, 0, 0, 0, 0, 0])
        self.initial_position_odom = np.array([0, 0, 0, 0, 0, 0])
        self.start_position_offset = np.array([1.2, 0, 0, 0, 0, 0])
        
        self.correction_yaw_odom = 0
        self.correction_yaw_vision = 0
        self.current_yaw_state = 0  # in radians

    def _initialize_csv_files(self):
        """Initialize CSV files with headers."""
        with open(self.filename_hololens, mode="w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.columns_hololens_data)
            
        with open(self.filename_odom, mode="w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.columns_spot_data)

        with open(self.filename_actions, mode="w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.columns_action_scripts)
            
    def _is_pick_command(self, command):
        """Check if command is a pick command."""
        return command in ("pick_up_object", "pick_up_green", "pick_up_red", "pick_up_blue", "pick_up_qr", "pick_up_label")

    def _is_in_pick_state(self):
        """Check if robot is currently in a pick state."""
        curr = str(self.sm.current_state).lower().replace(" ", "_")
        return curr in ("pick_object", "pick_green", "pick_red", "pick_blue", "pick_qr", "pick_label") or getattr(self.sm, "_pick_executing", False)

    def _is_duplicate_command(self, command):
        """Check if command is a duplicate within the time window."""
        current_time = time.time()
        window = 10.0 if self._is_pick_command(command) else self.command_deduplication_window
        return (self.last_command == command and 
                current_time - self.last_command_time < window)
            
    def callback_action(self, data):
        timestamp = time.time() - self.node_start_time
        robot_state = self.robot.robot_state_client.get_robot_state()
        battery_charge_percentage = MessageToDict(robot_state)
        percentage = battery_charge_percentage["batteryStates"][0]["chargePercentage"]

        with open(self.filename_actions, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, data.data, percentage])
        
        # Handle configuration messages
        if data.data.startswith("CONFIG_TARGET_LABEL:"):
            target_label = data.data.split(":", 1)[1]
            if self.robot:
                self.robot.target_label = target_label
                print(f"✅ Target label configured: {target_label}")
            return
        
        # Prevent duplicate pick operations
        if self._is_pick_command(data.data) and self._is_in_pick_state():
            print(f"⚠️  Skipping '{data.data}' because a pick is already in progress")
            return
        
        # Command deduplication with extended window for pick commands
        if self._is_duplicate_command(data.data):
            current_time = time.time()
            print(f"⚠️  Skipping duplicate command '{data.data}' (received {current_time - self.last_command_time:.1f}s ago)")
            return
            
        self.last_command = data.data
        self.last_command_time = time.time()
            
        self.pub_status.publish("running")
        try_state_send(self.sm, data.data)
        print(f"\nI heard: {data.data}")
        
    def triangulate_position(self, data):
        pose = self.get_robot_vision_pose()
        
        x_person = data.data[0]
        y_person = data.data[1]
        
        x_obj = data.data[3]
        y_obj = data.data[4]
        
        vector_person_robot = np.array([pose[0] - x_person, pose[1] - y_person])
        vector_person_object = np.array([x_obj - x_person, y_obj - y_person])
        
        # vector_robot_object = vector_person_object - vector_person_robot ## vector calculation  
        vector_robot_object = np.array([x_obj - pose[0], y_obj - pose[1]])  # determining relative movement directly from goal position to starting position
        
        print("Vector person --> Robot:", vector_person_robot)
        print("Vector person --> Object:", vector_person_object)
        
        x, y = self.rotation_matrix_calc(vector_robot_object[0], vector_robot_object[1], -1*pose[3])
        
        if vector_robot_object[0] >= 0:
            rotation = np.arctan(y/x) 
        if vector_robot_object[0] < 0 and vector_robot_object[1] >= 0:
            rotation = np.arctan(y/x) + np.pi    
        elif vector_robot_object[0] < 0 and vector_robot_object[1] < 0:
            rotation = np.arctan(y/x) - np.pi      
        
        data_to_save = [x_person, y_person, x_obj, y_obj, rotation, pose[0], pose[1]]
        
        return x, y, rotation, data_to_save
    
    def get_robot_vision_pose(self):
        vision_T_body = get_a_tform_b(self.robot.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,"vision","body") 
        visionBodyEuler = quat_to_eulerZYX(vision_T_body.rot)
        pose_raw = [vision_T_body.x, vision_T_body.y, vision_T_body.z, visionBodyEuler[0], visionBodyEuler[1], visionBodyEuler[2]]
        
        pose = self.pose_transformation_vision(pose_raw)
        return pose

    def pose_transformation_vision(self, data):
        data_zero = np.array(data) - self.initial_position_vision_odom
        x, y = self.rotation_matrix_calc(data_zero[0], data_zero[1], self.correction_yaw_vision)
        z = data_zero[2]
        
        data_transformed = [x, y, z, data_zero[3], data_zero[4], data_zero[5]]

        return np.array(data_transformed) + self.start_position_offset
    
    def get_robot_odom_pose(self):
        odom_T_body = get_a_tform_b(self.robot.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,"odom","body") 
        visionBodyEuler = quat_to_eulerZYX(odom_T_body.rot)
        pose_raw = [odom_T_body.x, odom_T_body.y, odom_T_body.z, visionBodyEuler[0], visionBodyEuler[1], visionBodyEuler[2]]
        
        pose = self.pose_transformation_odom(pose_raw)
        return pose

    def pose_transformation_odom(self, data):
        data_zero = np.array(data) - self.initial_position_odom
        x, y = self.rotation_matrix_calc(data_zero[0], data_zero[1], self.correction_yaw_odom)
        z = data_zero[2]
        
        data_transformed = [x, y, z, data_zero[3], data_zero[4], data_zero[5]]

        return np.array(data_transformed) + self.start_position_offset
        

    def rotation_matrix_calc(self, x, y, rotation):
        x = np.cos(rotation) * x - np.sin(rotation) * y
        y = np.sin(rotation) * x + np.cos(rotation) * y
        return x, y
        
    def calibrate_odometry_rotations(self, calibration_poses, frame="odom"):
        try:
            yaw_per_pose = []  # yaw in degrees
            for pose in calibration_poses[5:]:
                x = pose[0]
                y = pose[1]
                if x >= 0:
                    yaw_radian = np.arctan(y/x)
                if x < 0 and y >= 0:
                    yaw_radian = np.arctan(y/x) + np.pi
                elif x < 0 and y < 0:
                    yaw_radian = np.arctan(y/x) - np.pi
                    
                yaw_per_pose.append(yaw_radian)
                
            if frame == "odom":
                self.correction_yaw_odom = np.average(yaw_per_pose) * -1
                print("Correction YAW ODOM degrees: ", np.rad2deg(self.correction_yaw_odom))
            elif frame == "vision":
                self.correction_yaw_vision = np.average(yaw_per_pose) * -1
                print("Correction YAW VISION degrees: ", np.rad2deg(self.correction_yaw_vision)) 
            
        except Exception as e:
            print(e)
        
        return
        
    def calibration_movement(self):
        odom_positions, vision_odom_positions = self.robot.calibration_movement_in_robot_frame()
        
        self.initial_position_odom = np.array(odom_positions[0])
        self.initial_position_vision_odom = np.array(vision_odom_positions[0])
        
        return odom_positions, vision_odom_positions
    
    def callback_deictic_walk(self, data):
        x, y, yaw, data_to_save = self.triangulate_position(data)
        print(f"Walking to x:{x}, y:{y}, angle:{yaw}")
        
        self.robot.two_d_location_body_frame_command(x, y, yaw)
        time.sleep(1)
         
        new_pose = self.get_robot_vision_pose()
        print("New robot pose: ", new_pose)
        
        data_to_save.append(new_pose[0])
        data_to_save.append(new_pose[1])
        data_to_save.append(new_pose[3])
        with open(self.filename_odom, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(data_to_save)
            file.close()

    def callback_gripper(self, data):
        if self.robot.current_state_direct_control:
            print(data)
            close_or_open = data.data
            self.robot.gripper(close_or_open)
        else:
            pass
    
    def callback_hand_pose(self, data):
        if self.robot.current_state_direct_control:
            if self.robot.init_pos_empty:
                self.arm_pos_init = [data.position.x, data.position.y, data.position.z]
        
            pos = 1.0*(np.array([data.position.x, data.position.y, data.position.z] - np.array(self.arm_pos_init)))
            orientation = math_helpers.Quat(1, 0, 0, 0)
            hand_pose = math_helpers.SE3Pose(x=0.75+pos[0], y=pos[1], z=0.45+pos[2], rot=orientation)
            
            self.pose_receive_count += 1
            if self.pose_receive_count >= self.frequency_pose_count:
                print(hand_pose)
                self.pose_receive_count = 0
                self.robot.init_pos_empty = False
                self.robot.move_to_cartesian_pose_rt_task(hand_pose, self.odom_T_task, self.wr1_T_tool)
    
        
    def callback_data_collection(self, data):
        array = data.data
        timestamp = time.time() - self.node_start_time
        entry = [timestamp]
        entry.extend([array[:3], array[3:6], array[6:9], array[9:12], array[12:16]])
        with open(self.filename_hololens, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(entry)
            
        odometry_vision_data = self.get_robot_vision_pose()
        odometry_data = self.get_robot_odom_pose()
        row = [timestamp]
        row.extend([list(odometry_data[:3])])
        row.extend([list(odometry_data[3:6])])  
        row.extend([list(odometry_vision_data[:3])])
        row.extend([list(odometry_vision_data[3:6])])
        
        with open(self.filename_odom, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
    def run(self, video_stream_saver):
        rospy.init_node('listener', anonymous=True)
        vss_thread = threading.Thread(target=video_stream_saver.run)
        vss_thread.start()
        rospy.Subscriber("fsm_commands", String, self.callback_action)
        rospy.Subscriber("gripper", String, self.callback_gripper)
        rospy.Subscriber("hand_pose", Pose, self.callback_hand_pose)
        rospy.Subscriber("data_collection", Float32MultiArray, self.callback_data_collection)
        
        pub = rospy.Publisher('spot_odom', Float32MultiArray, queue_size=10)
        
        while not rospy.is_shutdown():
            msg = Float32MultiArray()
            row = []
            odometry_vision_data = self.get_robot_vision_pose()
            odometry_data = self.get_robot_odom_pose()
            row.extend(list(odometry_data[:3]))  # position odom [x,y,z]
            row.extend(list(odometry_data[3:6]))  # orientation odom [yaw,pitch,roll]
            row.extend(list(odometry_vision_data[:3]))  # position vision [x,y,z]
            row.extend(list(odometry_vision_data[3:6]))  # orientation vision [yaw,pitch,roll]
            msg.data = row 
            pub.publish(msg)
        
    ##############################################
    ############ Dummy with callbacks ############
    ##############################################
        
    
    def callback_action_dummy(self, data):
        print(f"\nChatter heard: {data.data}")
            
    def callback_gripper_dummy(self, data):
        print("Gripper:", data.data)

    def callback_hand_pose_dummy(self, data):
        print("Hand pose:", [data.position.x, data.position.y, data.position.z])
        
    def callback_data_collection_dummy(self, data):
        print("Data Collection:", data.data)
    
    def run_dummy(self):
        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber("fsm_commands", String, self.callback_action_dummy)
        rospy.Subscriber("gripper", String, self.callback_gripper_dummy)
        rospy.Subscriber("hand_pose", Pose, self.callback_hand_pose_dummy)
        rospy.spin()

if __name__ == "__main__":
    participant = 999
    conditions = [
            "speech_freewalking",
            "speech_stationary",
            "gestures_freewalking",
            "gestures_stationary",
            "controller_freewalking",
            "controller_stationary",
    ]
    condition = conditions[5]
     
    robotInterface = SpotControlInterface(DIRECT_CONTROL_FREQUENCY)  # Controlling the robot
    # robotInterface = None # Looking at message from HoloLens
    
    if robotInterface:
        sdk = bosdyn.client.create_standard_sdk('SpotControlInterface')
        # robot = sdk.create_robot("192.168.31.214")
        robot = sdk.create_robot("192.168.1.109")
        # robot = sdk.create_robot("172.20.10.3")
        robotInterface.robot_sdk = robot
        bosdyn.client.util.authenticate(robot)
        robot.time_sync.wait_for_sync()
        assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                        "such as the estop SDK example, to configure E-Stop."
        
        lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        lease_client.take()
        with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            # Now, we are ready to power on the robot. This call will block until the power
            # is on. Commands would fail if this did not happen. We can also check that the robot is
            # powered at any point.
            robot.logger.info("Powering on robot... This may take several seconds.")
            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), "Robot power on failed."
            robot.logger.info("Robot powered on.")

            robotInterface.image_client = robot.ensure_client(ImageClient.default_service_name)

            # Create a command client to be able to command the robot
            robotInterface.command_client = robot.ensure_client(RobotCommandClient.default_service_name)
            robotInterface.robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
            robotInterface.manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)
            
            time.sleep(1)

            timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
            fsm = FsmNode(robot=robotInterface, robot_sdk=robot, timestamp=timestamp, participant=participant, condition=condition)
            vss = VideoStreamSaver(robotInterface.image_client, participant, condition, timestamp)
            fsm.run(vss)
            
            robotInterface.sit_down()

        for out in vss.video_writers:
            out.release()
            
        vss.video_writers[0].release()
        vss.video_writers[1].release()

    else:
        timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
        fsm = FsmNode(robot=robotInterface, robot_sdk=None, timestamp=timestamp, participant=participant, condition=condition) 
        fsm.run_dummy()
        