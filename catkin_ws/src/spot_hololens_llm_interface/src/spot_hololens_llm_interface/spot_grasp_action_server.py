#!/usr/bin/env python3

import rospy
import time
import cv2
from cv_bridge import CvBridge

import numpy as np
import actionlib
import threading

from spot_hololens_llm_interface.msg import (
    InteractiveGraspAction,
    InteractiveGraspGoal,
    InteractiveGraspResult,
    InteractiveGraspFeedback
    ,MoveArmPoseAction, MoveArmPoseGoal, MoveArmPoseResult, MoveArmPoseFeedback
)

from bosdyn.api import geometry_pb2, manipulation_api_pb2, image_pb2
from bosdyn.api import arm_command_pb2
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client import math_helpers


class SpotGraspActionServer:
    def __init__(self, robot_manager):
        """
        Initialize the grasp action server with direct access to robot manager.
        
        Args:
            robot_manager: Instance of SpotRobotManager for direct robot access
        """
        # Store robot manager for direct access
        self.robot_manager = robot_manager
        
        # CV bridge for image processing
        self.cv_bridge = CvBridge()
        
        # Mouse click handling for interactive grasp
        self.image_click = None
        self.image_display = None
        self.click_event = threading.Event()
        self.window_name = None
        
        # Action server for interactive grasping
        self.interactive_grasp_server = actionlib.SimpleActionServer(
            'interactive_grasp', 
            InteractiveGraspAction, 
            execute_cb=self.execute_interactive_grasp_cb, 
            auto_start=False
        )
        self.interactive_grasp_server.start()
        
        # Action server for moving arm to a specified pose
        self.move_arm_server = actionlib.SimpleActionServer(
            'move_arm_pose',
            MoveArmPoseAction,
            execute_cb=self.execute_move_arm_cb,
            auto_start=False
        )
        self.move_arm_server.start()
        rospy.loginfo("Spot grasp action server started with direct robot manager access")
    
    def execute_interactive_grasp_cb(self, goal):
        """Execute interactive grasp with user clicking on image"""
        rospy.loginfo("Executing interactive grasp")
        
        result = InteractiveGraspResult()
        feedback = InteractiveGraspFeedback()
        
        try:
            # Verify we have necessary clients
            clients = self.robot_manager.get_clients()
            if not clients or not clients['image'] or not clients['manipulation_api']:
                result.success = False
                result.message = "Robot not connected or required clients not available"
                self.interactive_grasp_server.set_aborted(result)
                return
            
            # Store initial pose if requested
            initial_pose = None
            if goal.return_to_initial_pose:
                if self.robot_manager.initial_pose is not None:
                    from geometry_msgs.msg import PoseStamped
                    import tf.transformations
                    
                    # Convert SE2 pose to PoseStamped
                    initial_pose = PoseStamped()
                    initial_pose.header.frame_id = "odom"
                    initial_pose.header.stamp = rospy.Time.now()
                    initial_pose.pose.position.x = self.robot_manager.initial_pose.x
                    initial_pose.pose.position.y = self.robot_manager.initial_pose.y
                    initial_pose.pose.position.z = 0.0
                    
                    # Convert angle to quaternion
                    quat = tf.transformations.quaternion_from_euler(0, 0, self.robot_manager.initial_pose.angle)
                    initial_pose.pose.orientation.x = quat[0]
                    initial_pose.pose.orientation.y = quat[1]
                    initial_pose.pose.orientation.z = quat[2]
                    initial_pose.pose.orientation.w = quat[3]
                else:
                    rospy.logwarn("No initial pose available for return to initial pose")
            
            # Get image directly from the robot
            feedback.current_state = "ACQUIRING_IMAGE"
            feedback.progress = 0.1
            feedback.status_message = f"Getting image from {goal.image_source}"
            self.interactive_grasp_server.publish_feedback(feedback)
            
            image_responses = clients['image'].get_image_from_sources([goal.image_source])
            if len(image_responses) != 1:
                result.success = False
                result.message = f"Invalid number of images: {len(image_responses)}"
                self.interactive_grasp_server.set_aborted(result)
                return
            
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
            
            # Process image for display if requested
            if goal.show_preview:
                # Show image and wait for user click
                feedback.current_state = "WAITING_FOR_USER_INPUT"
                feedback.progress = 0.2
                feedback.waiting_for_user_input = True
                feedback.status_message = "Click on an object to grasp"
                self.interactive_grasp_server.publish_feedback(feedback)
                
                window_title = goal.window_title if goal.window_title else 'Click to grasp'
                self.window_name = window_title
                cv2.namedWindow(self.window_name)
                cv2.setMouseCallback(self.window_name, self._mouse_callback)
                
                self.image_display = img
                self.image_click = None
                self.click_event.clear()
                
                cv2.imshow(self.window_name, self.image_display)
                
                # Wait for click or action cancellation
                while self.image_click is None and not rospy.is_shutdown():
                    if self.interactive_grasp_server.is_preempt_requested():
                        cv2.destroyAllWindows()
                        self.window_name = None
                        result.success = False
                        result.message = "Action was cancelled"
                        self.interactive_grasp_server.set_preempted(result)
                        return
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        cv2.destroyAllWindows()
                        self.window_name = None
                        result.success = False
                        result.message = "User cancelled with 'q' key"
                        self.interactive_grasp_server.set_aborted(result)
                        return
                    
                    time.sleep(0.1)
                
                cv2.destroyAllWindows()
                self.window_name = None
                
                if self.image_click is None:
                    result.success = False
                    result.message = "No click received"
                    self.interactive_grasp_server.set_aborted(result)
                    return
                
                pixel_x, pixel_y = self.image_click
                result.selected_pixel_x = pixel_x
                result.selected_pixel_y = pixel_y
            else:
                result.success = False
                result.message = "Interactive mode disabled but no alternative pixel selection method provided"
                self.interactive_grasp_server.set_aborted(result)
                return
            
            # Execute grasp
            feedback.current_state = "EXECUTING_GRASP"
            feedback.progress = 0.5
            feedback.waiting_for_user_input = False
            feedback.status_message = f"Executing grasp at pixel ({pixel_x}, {pixel_y})"
            self.interactive_grasp_server.publish_feedback(feedback)
            
            # Execute the grasp using direct access to manipulation API
            success, grasp_state = self._execute_grasp_at_pixel_direct(
                pixel_x, pixel_y, image, goal, feedback
            )
            
            # Return to initial pose if requested and successful
            if success and goal.return_to_initial_pose and initial_pose:
                feedback.current_state = "RETURNING_TO_INITIAL_POSE"
                feedback.progress = 0.9
                feedback.status_message = "Returning to initial position"
                self.interactive_grasp_server.publish_feedback(feedback)
                self._return_to_initial_pose_direct(initial_pose)
            
            result.success = success
            result.grasp_state = grasp_state
            result.message = "Interactive grasp completed successfully" if success else "Interactive grasp failed"
            
            if success:
                self.interactive_grasp_server.set_succeeded(result)
            else:
                self.interactive_grasp_server.set_aborted(result)
                
        except Exception as e:
            rospy.logerr(f"Interactive grasp action failed: {e}")
            result.success = False
            result.message = f"Exception during interactive grasp: {str(e)}"
            self.interactive_grasp_server.set_aborted(result)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for interactive grasp"""
        if event == cv2.EVENT_LBUTTONUP:
            self.image_click = (x, y)
            self.click_event.set()
        else:
            # Draw crosshairs
            if self.image_display is not None:
                clone = self.image_display.copy()
                color = (30, 30, 30)
                thickness = 2
                height, width = clone.shape[:2]
                cv2.line(clone, (0, y), (width, y), color, thickness)
                cv2.line(clone, (x, 0), (x, height), color, thickness)
                win = self.window_name if hasattr(self, 'window_name') and self.window_name else 'Click to grasp'
                cv2.imshow(win, clone)
    
    def _execute_grasp_at_pixel_direct(self, pixel_x, pixel_y, image, goal, feedback):
        """Execute grasp directly using manipulation API client"""
        try:
            clients = self.robot_manager.get_clients()
            if not clients or not clients['manipulation_api']:
                rospy.logerr("Manipulation API client not available")
                return False, manipulation_api_pb2.MANIP_STATE_GRASP_FAILED
            
            # Create the grasp vector from pixel coordinates
            pick_vec = geometry_pb2.Vec2(x=pixel_x, y=pixel_y)
            
            # Build grasp request
            grasp = manipulation_api_pb2.PickObjectInImage(
                pixel_xy=pick_vec,
                transforms_snapshot_for_camera=image.shot.transforms_snapshot,
                frame_name_image_sensor=image.shot.frame_name_image_sensor,
                camera_model=image.source.pinhole
            )
            
            # Add grasp constraints based on goal parameters
            grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME
            
            if goal.force_top_down_grasp:
                axis_on_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)
                axis_to_align_with = geometry_pb2.Vec3(x=0, y=0, z=-1)
                constraint = grasp.grasp_params.allowable_orientation.add()
                constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(axis_on_gripper)
                constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(axis_to_align_with)
                constraint.vector_alignment_with_tolerance.threshold_radians = 0.17
                feedback.current_state = "ADDING_TOP_DOWN_CONSTRAINT"
                feedback.progress = 0.6
                feedback.status_message = "Returning to initial position"
                
            elif goal.force_horizontal_grasp:
                axis_on_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)
                axis_to_align_with = geometry_pb2.Vec3(x=0, y=0, z=1)
                constraint = grasp.grasp_params.allowable_orientation.add()
                constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(axis_on_gripper)
                constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(axis_to_align_with)
                constraint.vector_alignment_with_tolerance.threshold_radians = 0.17
                
            elif goal.force_squeeze_grasp:
                constraint = grasp.grasp_params.allowable_orientation.add()
                constraint.squeeze_grasp.SetInParent()
            
            # Send grasp request
            grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
            cmd_response = clients['manipulation_api'].manipulation_api_command(
                manipulation_api_request=grasp_request
            )
            
            manipulation_cmd_id = cmd_response.manipulation_cmd_id
            
            # Monitor grasp progress
            while True:
                # TODO: check the set_preempted() method? Maybe it's needed to cancel the action properly
                if self.interactive_grasp_server.is_preempt_requested():
                    return False, manipulation_api_pb2.MANIP_STATE_GRASP_FAILED
                
                feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                    manipulation_cmd_id=manipulation_cmd_id
                )
                
                api_feedback_response = clients['manipulation_api'].manipulation_api_feedback_command(
                    manipulation_api_feedback_request=feedback_request
                )
                
                feedback.current_state = f"GRASP_{manipulation_api_pb2.ManipulationFeedbackState.Name(api_feedback_response.current_state)}"
                feedback.status_message = f"Manipulation state: {manipulation_api_pb2.ManipulationFeedbackState.Name(api_feedback_response.current_state)}"
                self.interactive_grasp_server.publish_feedback(feedback)
                
                if (api_feedback_response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or 
                    api_feedback_response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED):
                    break
                
                time.sleep(0.25)
            
            # Return success status and grasp state
            success = api_feedback_response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED
            
            # Handle post-grasp actions
            if success:
                self._handle_successful_grasp_direct(release_after_grasp=goal.return_to_initial_pose)
            
            return success, api_feedback_response.current_state
            
        except Exception as e:
            rospy.logerr(f"Grasp execution failed: {e}")
            return False, manipulation_api_pb2.MANIP_STATE_GRASP_FAILED
    
    def _handle_successful_grasp_direct(self, release_after_grasp=False):
        """Handle post-grasp actions using direct robot commands"""
        try:
            from bosdyn.client.robot_command import RobotCommandBuilder
            
            clients = self.robot_manager.get_clients()
            cmd_client = clients['command']
            
            if release_after_grasp:
                # Open gripper to release object
                open_cmd = RobotCommandBuilder.claw_gripper_open_command()
                cmd_client.robot_command(open_cmd, end_time_secs=time.time() + 2)
                time.sleep(1.5)
                
                # Raise arm to carry position
                carry_cmd = RobotCommandBuilder.arm_carry_command()
                cmd_client.robot_command(carry_cmd, end_time_secs=time.time() + 3)
                time.sleep(2.0)
                
                # Close gripper
                close_cmd = RobotCommandBuilder.claw_gripper_close_command()
                cmd_client.robot_command(close_cmd, end_time_secs=time.time() + 2)
                time.sleep(1.5)
            else:
                # Just carry the object
                carry_cmd = RobotCommandBuilder.arm_carry_command()
                cmd_client.robot_command(carry_cmd, end_time_secs=time.time() + 3)
                # Just stow the arm
                # stow_cmd = RobotCommandBuilder.arm_stow_command()
                # cmd_client.robot_command(stow_cmd, end_time_secs=time.time() + 3)
                time.sleep(2.0)
            
        except Exception as e:
            rospy.logerr(f"Failed to handle successful grasp: {e}")
    
    def _return_to_initial_pose_direct(self, initial_pose):
        """Return robot to initial pose using direct commands"""
        try:
            from bosdyn.client.robot_command import RobotCommandBuilder
            from tf.transformations import euler_from_quaternion
            
            clients = self.robot_manager.get_clients()
            cmd_client = clients['command']
            
            # Stow the arm
            stow_cmd = RobotCommandBuilder.arm_stow_command()
            cmd_client.robot_command(stow_cmd, end_time_secs=time.time() + 3)
            time.sleep(2.0)
            
            # Extract pose parameters
            tx = initial_pose.pose.position.x
            ty = initial_pose.pose.position.y
            quat = (
                initial_pose.pose.orientation.x,
                initial_pose.pose.orientation.y,
                initial_pose.pose.orientation.z,
                initial_pose.pose.orientation.w
            )
            _, _, tyaw = euler_from_quaternion(quat)
            
            # Build SE2 pose
            
            se2 = geometry_pb2.SE2Pose(
                position=geometry_pb2.Vec2(x=tx, y=ty),
                angle=tyaw
            )
            
            # Move back to initial position
            move_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(se2, VISION_FRAME_NAME)
            cmd_client.robot_command(move_cmd, end_time_secs=time.time() + 10)
                    
        except Exception as e:
            rospy.logerr(f"Failed to return to initial pose: {e}")

    def execute_move_arm_cb(self, goal):
        """Action server callback to move the arm to a desired odom pose."""
        rospy.loginfo("move_arm_pose requested")
        result = MoveArmPoseResult()
        feedback = MoveArmPoseFeedback()

        try:
            clients = self.robot_manager.get_clients()
            if not clients or 'command' not in clients or clients['command'] is None:
                result.success = False
                result.message = "Command client not available"
                self.move_arm_server.set_aborted(result)
                return

            cmd_client = clients['command']
            state_client = clients['robot_state']

            # Move to the <hold pose>
            x = goal.x
            y = goal.y
            z = goal.z
            hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)
            qw = goal.qw
            qx = goal.qx
            qy = goal.qy
            qz = goal.qz
            flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)
            flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                                    rotation=flat_body_Q_hand)
            
            robot_state = state_client.get_robot_state()
            odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                            ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
            odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_proto(flat_body_T_hand)

            # Build arm pose command in odom frame
            seconds = max(0.1, goal.duration) if hasattr(goal, 'duration') else 2.0

            arm_cmd = RobotCommandBuilder.arm_pose_command(
                odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
                odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, seconds)

            # Optionally set gripper state
            grip_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(1 if goal.open_gripper else 0)
            cmd = RobotCommandBuilder.build_synchro_command(grip_cmd, arm_cmd)

            cmd_id = cmd_client.robot_command(cmd)

            # Monitor feedback until finished or preempted
            while not rospy.is_shutdown():
                if self.move_arm_server.is_preempt_requested():
                    result.success = False
                    result.message = 'Preempted'
                    self.move_arm_server.set_preempted(result)
                    return

                try:
                    fb = cmd_client.robot_command_feedback(cmd_id)
                    arm_fb = fb.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback
                    # Compute crude progress metric based on measured_pos_distance_to_goal
                    pos_dist = arm_fb.measured_pos_distance_to_goal
                    rot_dist = arm_fb.measured_rot_distance_to_goal
                    feedback.current_state = 'MOVING'
                    # Simple mapping: large distance -> 0.0, small -> 1.0
                    progress = float(max(0.0, min(1.0, 1.0 - (pos_dist / (pos_dist + 0.001)))))
                    feedback.progress = progress
                    self.move_arm_server.publish_feedback(feedback)

                    if arm_fb.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
                        break
                except Exception:
                    # Ignore transient errors while polling
                    pass

                time.sleep(0.1)

            result.success = True
            result.message = 'Move completed'
            self.move_arm_server.set_succeeded(result)

        except Exception as e:
            rospy.logerr(f"move_arm_pose failed: {e}")
            result.success = False
            result.message = str(e)
            self.move_arm_server.set_aborted(result)


if __name__ == '__main__':
    rospy.init_node('spot_grasp_action_server')
    
    # Import robot manager
    from spot_hololens_llm_interface.spot_entrance import SpotRobotManager
    
    # Create the robot manager first
    robot_manager = SpotRobotManager(ready_for_command=True, start_services=False)
    
    # Pass the robot manager to the action server
    server = SpotGraspActionServer(robot_manager)
    
    rospy.loginfo("Spot grasp action server running with direct robot manager access...")
    rospy.spin()
