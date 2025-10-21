#!/usr/bin/env python3

import rospy
import time
import numpy as np
import cv2
import math

from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives, block_for_trajectory_cmd
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b
from bosdyn.api import geometry_pb2, image_pb2, robot_command_pb2

from spot_hololens_llm_interface.srv import (
    GetImage, GetImageResponse,
    GetInitialPose, GetInitialPoseResponse, 
    GetRobotPose, GetRobotPoseResponse,
    
    MoveToPosition, MoveToPositionResponse,
    ArmCommand, ArmCommandResponse
)
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from tf.transformations import euler_from_quaternion

# Helper to poll feedback and decide success/failure
def _wait_for_feedback(cmd_client, command_id, est_duration, use_block=None):
    """Wait for command feedback."""
    # If caller requests SDK blocking helpers, try those first and fall back to polling.
    timeout_slack = 5.0
    success = False  # Default to False
    
    if use_block == 'trajectory':
        success = block_for_trajectory_cmd(cmd_client, command_id, timeout_sec=est_duration + timeout_slack)
    elif use_block == 'arm':
        success = block_until_arm_arrives(cmd_client, command_id, timeout_sec=est_duration + timeout_slack)
    elif use_block == 'gripper':
        # For gripper commands, just wait a short time and assume success
        import time
        time.sleep(1.0)  # Give gripper time to complete
        success = True  # Assume gripper commands complete successfully
    
    try:
        fb = cmd_client.robot_command_feedback(command_id)
        status = getattr(fb, 'status', None)
        if status is not None:
            try:
                status_name = robot_command_pb2.RobotCommandFeedbackStatus.Name(status)
            except Exception:
                status_name = str(status)
        else:
            status_name = 'NO_STATUS'
    except Exception:
        status_name = 'NO_STATUS'

    return bool(success), status_name


class SpotSharedServices:
    def __init__(self, robot_manager):
        """Initialize shared services with reference to robot manager"""
        self.robot_manager = robot_manager
        self.dummy_mode = getattr(robot_manager, 'dummy_mode', False)
        self.cv_bridge = CvBridge()
        
        # Grid position tracking for dummy mode (30cm per cell)
        self.current_cell = [4, 4]  # [row, col] - start in middle of 10x10 grid
        self.facing = "N"  # Current facing direction (N/S/E/W)
        
        # Setup robot operation services
        self.srv_get_image = rospy.Service('~get_image', GetImage, self.handle_get_image)
        self.srv_get_initial_pose = rospy.Service('~get_initial_pose', GetInitialPose, self.handle_get_initial_pose)
        self.srv_get_robot_pose = rospy.Service('~get_robot_pose', GetRobotPose, self.handle_get_robot_pose)
        self.srv_move_to_position = rospy.Service('~move_to_position', MoveToPosition, self.handle_move_to_position)
        self.srv_arm_command = rospy.Service('~arm_command', ArmCommand, self.handle_arm_command)
        
        if self.dummy_mode:
            rospy.loginfo("Spot shared services initialized in DUMMY MODE with grid tracking")
        else:
            rospy.loginfo("Spot shared services initialized")
    
    def handle_get_image(self, req):
        """Get image from specified camera source"""
        response = GetImageResponse()
        
        try:
            clients = self.robot_manager.get_clients()
            if not clients or not clients['image']:
                response.success = False
                response.message = "Robot not connected or image client not available"
                return response
                
            # Get image from robot
            image_responses = clients['image'].get_image_from_sources([req.image_source])
            if len(image_responses) != 1:
                response.success = False
                response.message = f"Invalid number of images: {len(image_responses)}"
                return response
                
            image = image_responses[0]
            
            # Convert image to ROS format
            if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                dtype = np.uint16
            else:
                dtype = np.uint8
                
            img = np.fromstring(image.shot.image.data, dtype=dtype)
            if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                img = img.reshape(image.shot.image.rows, image.shot.image.cols)
            else:
                img = cv2.imdecode(img, -1)
            
            # Convert to ROS Image message
            if len(img.shape) == 3:
                encoding = "bgr8"
            else:
                encoding = "mono8"
            response.image = self.cv_bridge.cv2_to_imgmsg(img, encoding)
            
            # Convert transforms (simplified for now)
            response.transforms = []  # TODO: Convert transforms properly
            response.frame_name_image_sensor = image.shot.frame_name_image_sensor
            
            # Convert camera info (simplified)
            response.camera_info = CameraInfo()
            if hasattr(image.source, 'pinhole'):
                response.camera_info.width = image.shot.image.cols
                response.camera_info.height = image.shot.image.rows
                # TODO: Fill in camera intrinsics from pinhole model
            
            response.success = True
            response.message = "Image retrieved successfully"
            return response
            
        except Exception as e:
            response.success = False
            response.message = f"Failed to get image: {str(e)}"
            return response
    
    def handle_get_initial_pose(self, req):
        """Get the stored initial pose"""
        response = GetInitialPoseResponse()
        
        if self.robot_manager.initial_pose is not None:
            # Convert SE2 pose to PoseStamped
            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = "vision"
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.pose.position.x = self.robot_manager.initial_pose.x
            pose_stamped.pose.position.y = self.robot_manager.initial_pose.y
            pose_stamped.pose.position.z = 0.0
            
            # Convert angle to quaternion
            import tf.transformations
            quat = tf.transformations.quaternion_from_euler(0, 0, self.robot_manager.initial_pose.angle)
            pose_stamped.pose.orientation.x = quat[0]
            pose_stamped.pose.orientation.y = quat[1]
            pose_stamped.pose.orientation.z = quat[2]
            pose_stamped.pose.orientation.w = quat[3]
            
            response.initial_pose = pose_stamped
            response.success = True
            response.message = "Initial pose retrieved"
        else:
            response.success = False
            response.message = "No initial pose stored. Robot didn't stand or move after boot?"
            
        return response

    def handle_get_robot_pose(self, req):
        """Return current robot pose in ODOM frame and end-effector pose in body frame."""
        response = GetRobotPoseResponse()

        # Handle dummy mode with grid coordinates
        if self.dummy_mode:
            # Convert grid cell to vision frame coordinates
            x = self.current_cell[1] * 0.3  # col * 30cm
            y = self.current_cell[0] * 0.3  # row * 30cm
            
            # Convert facing direction to yaw
            direction_to_yaw = {
                'N': 1.57,   # 90 degrees
                'S': -1.57,  # -90 degrees
                'E': 0.0,    # 0 degrees
                'W': 3.14    # 180 degrees
            }
            yaw = direction_to_yaw.get(self.facing, 0.0)
            
            # Create pose from grid position
            robot_pose = PoseStamped()
            robot_pose.header.frame_id = VISION_FRAME_NAME
            robot_pose.header.stamp = rospy.Time.now()
            robot_pose.pose.position.x = x
            robot_pose.pose.position.y = y
            robot_pose.pose.position.z = 0.0
            robot_pose.pose.orientation.x = 0.0
            robot_pose.pose.orientation.y = 0.0
            robot_pose.pose.orientation.z = yaw
            robot_pose.pose.orientation.w = 1.0
            
            response.success = True
            response.robot_pose = robot_pose
            response.message = f"Robot at grid cell {self.current_cell}, facing {self.facing} (dummy)"
            return response

        try:
            clients = self.robot_manager.get_clients()
            if not clients or not clients.get('robot_state'):
                response.success = False
                response.message = "Robot not connected or robot_state client not available"
                return response

            robot_state = clients['robot_state'].get_robot_state()
            ts = robot_state.kinematic_state.transforms_snapshot

            # Robot pose in ODOM frame
            vision_T_body = get_a_tform_b(ts, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

            robot_pose = PoseStamped()
            robot_pose.header.frame_id = VISION_FRAME_NAME
            robot_pose.header.stamp = rospy.Time.now()
            robot_pose.pose.position.x = vision_T_body.x
            robot_pose.pose.position.y = vision_T_body.y
            robot_pose.pose.position.z = vision_T_body.z
            robot_pose.pose.orientation.x = vision_T_body.rot.x
            robot_pose.pose.orientation.y = vision_T_body.rot.y
            robot_pose.pose.orientation.z = vision_T_body.rot.z
            robot_pose.pose.orientation.w = vision_T_body.rot.w

            # End-effector pose expressed in body frame
            # TODO: is this the same as body?
            body_T_ee = get_a_tform_b(ts, GRAV_ALIGNED_BODY_FRAME_NAME, 'hand')

            ee_pose = PoseStamped()
            ee_pose.header.frame_id = 'body'
            ee_pose.header.stamp = rospy.Time.now()
            ee_pose.pose.position.x = body_T_ee.x
            ee_pose.pose.position.y = body_T_ee.y
            ee_pose.pose.position.z = body_T_ee.z
            ee_pose.pose.orientation.x = body_T_ee.rot.x
            ee_pose.pose.orientation.y = body_T_ee.rot.y
            ee_pose.pose.orientation.z = body_T_ee.rot.z
            ee_pose.pose.orientation.w = body_T_ee.rot.w

            response.robot_pose = robot_pose
            response.end_effector_pose = ee_pose
            response.success = True
            response.message = 'Robot and end-effector poses retrieved'
            return response

        except Exception as e:
            response.success = False
            response.message = f"Failed to get robot pose: {e}"
            return response
    
    def handle_move_to_position(self, req):
        """Move robot to specified position (uses trajectory commands)

        Supported frames:
         - "body" / "flat_body" : target_pose is expressed in the robot's body frame (x forward, y left)
         - "vision" / "odom"       : target_pose is expressed in vision frame

        This implementation sends the trajectory command and then polls robot_command_feedback to
        determine whether the command actually completed successfully.
        """

        def _apply_move_result(resp, ok, status_name, frame_name, x, y, yaw):
            resp.success = ok
            if ok:
                resp.message = (f"{frame_name}-frame trajectory completed ({frame_name}-target: x={x:.2f}, "
                                f"y={y:.2f}, yaw={yaw:.2f}, status={status_name})")
            else:
                resp.message = (f"{frame_name}-frame trajectory failed or timed out ({frame_name}-target: x={x:.2f}, "
                                f"y={y:.2f}, yaw={yaw:.2f}, status={status_name})")
            return resp
        
        response = MoveToPositionResponse()
        
        # Handle dummy mode
        if self.dummy_mode:
            # Extract movement parameters
            x = req.target_pose.position.x
            y = req.target_pose.position.y
            yaw = req.target_pose.orientation.z  # Extract yaw from quaternion
            frame = req.frame_name or "body"
            
            # Check if this is a grid-based movement (vision frame with 30cm increments)
            if frame == "vision" and abs(x % 0.3) < 0.01 and abs(y % 0.3) < 0.01:
                # Convert meters back to grid cells for tracking
                new_col = round(x / 0.3)
                new_row = round(y / 0.3)
                
                # Update grid position
                old_cell = self.current_cell.copy()
                self.current_cell = [new_row, new_col]
                
                # Update facing direction based on yaw
                if abs(yaw - 1.57) < 0.1:  # ~90 degrees
                    self.facing = "N"
                elif abs(yaw + 1.57) < 0.1:  # ~-90 degrees
                    self.facing = "S"
                elif abs(yaw) < 0.1:  # ~0 degrees
                    self.facing = "E"
                elif abs(yaw - 3.14) < 0.1:  # ~180 degrees
                    self.facing = "W"
                
                rospy.loginfo(f"[DUMMY] Grid movement from cell {old_cell} to cell {self.current_cell}, now facing {self.facing}")
            else:
                rospy.loginfo(f"[DUMMY] Continuous movement to position x={x:.2f}, y={y:.2f} in {frame} frame...")
            
            # Simulate movement time based on distance
            distance = (x**2 + y**2)**0.5
            move_time = max(distance * 2.0, 1.0)  # 2 seconds per meter, minimum 1 second
            time.sleep(move_time)
            
            rospy.loginfo(f"[DUMMY] Move completed")
            
            response.success = True
            response.message = f"Move completed (dummy): x={x:.2f}, y={y:.2f}, frame={frame}"
            return response
        
        rotate_speed = 0.6  # rad/s
        forward_speed = 0.4  # m/s

        try:
            clients = self.robot_manager.get_clients()
            if not clients or not clients.get('command'):
                response.success = False
                response.message = "Robot not connected or command client not available"
                return response

            # Extract requested pose
            target = req.target_pose
            tx = getattr(target.position, "x", 0.0)
            ty = getattr(target.position, "y", 0.0)
            quat = (
                getattr(target.orientation, "x", 0.0),
                getattr(target.orientation, "y", 0.0),
                getattr(target.orientation, "z", 0.0),
                getattr(target.orientation, "w", 1.0),
            )
            _, _, tyaw = euler_from_quaternion(quat)

            frame = (req.frame_name or "").lower()
            cmd_client = clients['command']

            # Choose command builder based on frame
            if frame == "body" or frame == "flat_body":
                # For body-frame trajectory builder we need a frame tree snapshot.
                rstate_client = clients.get('robot_state')
                if not rstate_client:
                    response.success = False
                    response.message = "robot_state_client required to build body-frame trajectory"
                    return response

                bx, by, byaw = tx, ty, tyaw
                snapshot = rstate_client.get_robot_state().kinematic_state.transforms_snapshot

                distance = math.hypot(bx, by)
                estimate_duration = max(distance / forward_speed if distance > 0.02 else 0.0,
                                        abs(byaw) / rotate_speed if abs(byaw) > 0.02 else 0.0,
                                        0.5)

                cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                    bx, by, byaw, snapshot)
                # send command and capture command id
                try:
                    cmd_id = cmd_client.robot_command(cmd, end_time_secs=time.time() + estimate_duration + 5.0)
                    
                    # Let helper block for the trajectory and fall back to polling if needed
                    ok, status_name = _wait_for_feedback(cmd_client, cmd_id, estimate_duration, use_block='trajectory')
                    if not ok:
                        response.success = False
                        response.message = f"Body-frame trajectory command timed out or failed (status={status_name})"
                        return response
                except Exception as e:
                    response.success = False
                    response.message = f"Failed to send body-frame command: {e}"
                    return response

                return _apply_move_result(response, True, status_name, frame, bx, by, byaw)

            # TODO: check how close vision is to odom
            elif frame in ("vision", "odom"):
                # Build SE2 pose in odom/vision frame
                se2 = geometry_pb2.SE2Pose(
                    position=geometry_pb2.Vec2(x=tx, y=ty),
                    angle=tyaw
                )

                distance = math.hypot(tx, ty)
                estimate_duration = max(distance / forward_speed if distance > 0.02 else 0.0,
                                        abs(tyaw) / rotate_speed if abs(tyaw) > 0.02 else 0.0,
                                        0.5)

                cmd = RobotCommandBuilder.synchro_se2_trajectory_command(se2, VISION_FRAME_NAME)
                try:
                    cmd_id = cmd_client.robot_command(cmd, end_time_secs=time.time() + estimate_duration + 5.0)
                    
                    ok, status_name = _wait_for_feedback(cmd_client, cmd_id, estimate_duration, use_block='trajectory')
                    if not ok:
                        response.success = False
                        response.message = f"Odom-frame trajectory command timed out or failed (status={status_name})"
                        return response
                except Exception as e:
                    response.success = False
                    response.message = f"Failed to send odom-frame command: {e}"
                    return response

                return _apply_move_result(response, True, status_name or "COMPLETED", frame, tx, ty, tyaw)

            else:
                response.success = False
                response.message = f"Unsupported frame: {req.frame_name}"
                return response

        except Exception as e:
            response.success = False
            response.message = f"Failed to move to position: {str(e)}"
            return response
    
    def handle_arm_command(self, req):
        """Execute arm commands (open, close, stow, carry)"""
        from bosdyn.client.robot_command import RobotCommandBuilder
        response = ArmCommandResponse()
        
        try:
            # Handle dummy mode
            if self.dummy_mode:
                rospy.loginfo(f"DUMMY MODE - Simulating arm command: {req.command_type}")
                rospy.sleep(1.0)  # Simulate arm movement time
                response.success = True
                response.message = f"Arm command '{req.command_type}' simulated successfully (dummy)"
                return response
            
            clients = self.robot_manager.get_clients()
            if not clients or not clients['command']:
                response.success = False
                response.message = "Robot not connected or command client not available"
                return response
            
            if req.command_type == "open":
                cmd = RobotCommandBuilder.claw_gripper_open_command()
                use_block = 'gripper'
            elif req.command_type == "close":
                cmd = RobotCommandBuilder.claw_gripper_close_command()
                use_block = 'gripper'
            elif req.command_type == "stow":
                cmd = RobotCommandBuilder.arm_stow_command()
                use_block = 'arm'
            elif req.command_type == "carry":
                cmd = RobotCommandBuilder.arm_carry_command()
                use_block = 'arm'
            else:
                response.success = False
                response.message = f"Unknown arm command: {req.command_type}"
                return response
            
            cmd_id = clients['command'].robot_command(cmd, end_time_secs=time.time() + 3)
            ok, status_name = _wait_for_feedback(clients['command'], cmd_id, 3.0, use_block=use_block)
            if not ok:
                response.success = False
                response.message = f"Arm did not reach target position in time (status={status_name})"
                return response

            response.success = True
            response.message = f"Arm command '{req.command_type}' sent successfully"
            return response
            
        except Exception as e:
            response.success = False
            response.message = f"Failed to execute arm command: {str(e)}"
            return response
