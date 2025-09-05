#!/usr/bin/env python3

import rospy
import time
import numpy as np
import cv2
import math

from bosdyn.client.robot_command import RobotCommandBuilder, block_until_arm_arrives, block_for_trajectory_cmd
from bosdyn.client.frame_helpers import get_se2_a_tform_b, ODOM_FRAME_NAME, BODY_FRAME_NAME, VISION_FRAME_NAME
from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2, robot_command_pb2

from spot_hololens_llm_interface.srv import (
    GetImage, GetImageResponse,
    GetInitialPose, GetInitialPoseResponse, 
    ExecuteGrasp, ExecuteGraspResponse,
    GetGraspFeedback, GetGraspFeedbackResponse,
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
    if use_block == 'trajectory':
        success = block_for_trajectory_cmd(cmd_client, command_id, timeout_sec=est_duration + timeout_slack)
    if use_block == 'arm':
        success = block_until_arm_arrives(cmd_client, command_id, timeout_sec=est_duration + timeout_slack)
    
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
        self.cv_bridge = CvBridge()
        
        # Setup robot operation services
        self.srv_get_image = rospy.Service('~get_image', GetImage, self.handle_get_image)
        self.srv_get_initial_pose = rospy.Service('~get_initial_pose', GetInitialPose, self.handle_get_initial_pose)
        self.srv_execute_grasp = rospy.Service('~execute_grasp', ExecuteGrasp, self.handle_execute_grasp)
        self.srv_get_grasp_feedback = rospy.Service('~get_grasp_feedback', GetGraspFeedback, self.handle_get_grasp_feedback)
        self.srv_move_to_position = rospy.Service('~move_to_position', MoveToPosition, self.handle_move_to_position)
        self.srv_arm_command = rospy.Service('~arm_command', ArmCommand, self.handle_arm_command)
        
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
            pose_stamped.header.frame_id = "odom"
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
    
    def handle_execute_grasp(self, req):
        """Execute grasp command using robot's manipulation API"""
        response = ExecuteGraspResponse()

        response.success = True
        response.message = f"Skipping the grasp for debug. The request is: {req}."
        return response

        try:
            clients = self.robot_manager.get_clients()
            if not clients or not clients['manipulation_api']:
                response.success = False
                response.message = "Robot not connected or manipulation client not available"
                return response
            
            pick_vec = geometry_pb2.Vec2(x=req.x, y=req.y)
            # Build grasp request
            grasp = manipulation_api_pb2.PickObjectInImage(
                pixel_xy=pick_vec,
                # transforms_snapshot_for_camera=req.transforms_snapshot,  # TODO: Convert from ROS
                frame_name_image_sensor=req.frame_name_image_sensor,
                # camera_model=req.camera_model  # TODO: Convert from ROS CameraInfo
            )
            
            # Add grasp constraints
            grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME
            
            if req.force_top_down_grasp:
                axis_on_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)
                axis_to_align_with = geometry_pb2.Vec3(x=0, y=0, z=-1)
                constraint = grasp.grasp_params.allowable_orientation.add()
                constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(axis_on_gripper)
                constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(axis_to_align_with)
                constraint.vector_alignment_with_tolerance.threshold_radians = 0.17
            # TODO: Add other constraint types
            
            # Send grasp request
            grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
            cmd_response = clients['manipulation_api'].manipulation_api_command(
                manipulation_api_request=grasp_request
            )
            
            response.success = True
            response.message = "Grasp command sent successfully"
            response.manipulation_cmd_id = cmd_response.manipulation_cmd_id
            return response
            
        except Exception as e:
            response.success = False
            response.message = f"Failed to execute grasp: {str(e)}"
            return response
    
    def handle_get_grasp_feedback(self, req):
        """Get feedback for ongoing grasp operation"""
        response = GetGraspFeedbackResponse()
        
        try:
            clients = self.robot_manager.get_clients()
            if not clients or not clients['manipulation_api']:
                response.success = False
                response.message = "Robot not connected or manipulation client not available"
                return response
                
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=req.manipulation_cmd_id
            )
            
            feedback_response = clients['manipulation_api'].manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request
            )
            
            response.success = True
            response.message = "Feedback retrieved successfully"
            response.current_state = feedback_response.current_state
            response.state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(feedback_response.current_state)
            return response
            
        except Exception as e:
            response.success = False
            response.message = f"Failed to get grasp feedback: {str(e)}"
            return response
    
    def handle_move_to_position(self, req):
        """Move robot to specified position (uses trajectory commands)

        Supported frames:
         - "body" / "flat_body" : target_pose is expressed in the robot's body frame (x forward, y left)
         - "odom" / "map"       : target_pose is expressed in odom frame

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


            elif frame in ("odom", "map"):
                # Build SE2 pose in odom/map frame
                se2 = geometry_pb2.SE2Pose(
                    position=geometry_pb2.Vec2(x=tx, y=ty),
                    angle=tyaw
                )

                distance = math.hypot(tx, ty)
                estimate_duration = max(distance / forward_speed if distance > 0.02 else 0.0,
                                        abs(tyaw) / rotate_speed if abs(tyaw) > 0.02 else 0.0,
                                        0.5)

                cmd = RobotCommandBuilder.synchro_se2_trajectory_command(se2, ODOM_FRAME_NAME)
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
            clients = self.robot_manager.get_clients()
            if not clients or not clients['command']:
                response.success = False
                response.message = "Robot not connected or command client not available"
                return response
            
            if req.command_type == "open":
                cmd = RobotCommandBuilder.claw_gripper_open_command()
            elif req.command_type == "close":
                cmd = RobotCommandBuilder.claw_gripper_close_command()
            elif req.command_type == "stow":
                cmd = RobotCommandBuilder.arm_stow_command()
            elif req.command_type == "carry":
                cmd = RobotCommandBuilder.arm_carry_command()
            else:
                response.success = False
                response.message = f"Unknown arm command: {req.command_type}"
                return response
            
            cmd_id = clients['command'].robot_command(cmd, end_time_secs=time.time() + 3)
            ok, status_name = _wait_for_feedback(clients['command'], cmd_id, 3.0, use_block='arm')
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
