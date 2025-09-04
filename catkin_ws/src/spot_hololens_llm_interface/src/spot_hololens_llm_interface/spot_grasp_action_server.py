#!/usr/bin/env python3

import rospy
import time
import cv2
import numpy as np
import actionlib
import threading

from spot_hololens_llm_interface.msg import (
    GraspObjectAction, 
    GraspObjectGoal, 
    GraspObjectResult, 
    GraspObjectFeedback,
    InteractiveGraspAction,
    InteractiveGraspGoal,
    InteractiveGraspResult,
    InteractiveGraspFeedback
)

from spot_hololens_llm_interface.srv import (
    GetImage, GetImageRequest,
    GetInitialPose, GetInitialPoseRequest,
    ExecuteGrasp, ExecuteGraspRequest,
    GetGraspFeedback, GetGraspFeedbackRequest,
    MoveToPosition, MoveToPositionRequest,
    ArmCommand, ArmCommandRequest
)

from bosdyn.api import geometry_pb2, manipulation_api_pb2
from cv_bridge import CvBridge


class SpotGraspActionServer:
    def __init__(self):
        # Wait for spot_entrance node to be available
        rospy.loginfo("Waiting for spot_entrance services...")
        try:
            rospy.wait_for_service('/spot_entrance/connect', timeout=30)
            rospy.wait_for_service('/spot_entrance/get_image', timeout=30)
            rospy.wait_for_service('/spot_entrance/execute_grasp', timeout=30)
        except rospy.ROSException:
            rospy.logerr("Spot entrance services not available!")
            raise
        
        # Create service proxies
        self.get_image_srv = rospy.ServiceProxy('/spot_entrance/get_image', GetImage)
        self.get_initial_pose_srv = rospy.ServiceProxy('/spot_entrance/get_initial_pose', GetInitialPose)
        self.execute_grasp_srv = rospy.ServiceProxy('/spot_entrance/execute_grasp', ExecuteGrasp)
        self.get_grasp_feedback_srv = rospy.ServiceProxy('/spot_entrance/get_grasp_feedback', GetGraspFeedback)
        self.move_to_position_srv = rospy.ServiceProxy('/spot_entrance/move_to_position', MoveToPosition)
        self.arm_command_srv = rospy.ServiceProxy('/spot_entrance/arm_command', ArmCommand)
        
        # CV bridge for image processing
        self.cv_bridge = CvBridge()
        
        # Mouse click handling for interactive grasp
        self.image_click = None
        self.image_display = None
        self.click_event = threading.Event()
        
        # Action servers
        self.grasp_server = actionlib.SimpleActionServer(
            'grasp_object', 
            GraspObjectAction, 
            execute_cb=self.execute_grasp_cb, 
            auto_start=False
        )
        
        self.interactive_grasp_server = actionlib.SimpleActionServer(
            'interactive_grasp', 
            InteractiveGraspAction, 
            execute_cb=self.execute_interactive_grasp_cb, 
            auto_start=False
        )
        
        self.grasp_server.start()
        self.interactive_grasp_server.start()
        
        rospy.loginfo("Spot grasp action servers started")
    
    def execute_grasp_cb(self, goal):
        """Execute grasp with specified pixel coordinates"""
        rospy.loginfo(f"Executing grasp at pixel ({goal.pixel_x}, {goal.pixel_y})")
        
        result = GraspObjectResult()
        feedback = GraspObjectFeedback()
        
        try:
            # Store initial pose if requested
            initial_pose = None
            if goal.return_to_initial_pose:
                pose_resp = self.get_initial_pose_srv(GetInitialPoseRequest())
                if pose_resp.success:
                    initial_pose = pose_resp.initial_pose
                else:
                    rospy.logwarn(f"Failed to get initial pose: {pose_resp.message}")
            
            # Get image from specified source
            feedback.current_state = "ACQUIRING_IMAGE"
            feedback.progress = 0.1
            feedback.status_message = f"Getting image from {goal.image_source}"
            self.grasp_server.publish_feedback(feedback)
            
            image_req = GetImageRequest()
            image_req.image_source = goal.image_source
            image_resp = self.get_image_srv(image_req)
            
            if not image_resp.success:
                result.success = False
                result.message = f"Failed to get image: {image_resp.message}"
                self.grasp_server.set_aborted(result)
                return
            
            # Execute grasp
            feedback.current_state = "EXECUTING_GRASP"
            feedback.progress = 0.3
            feedback.status_message = "Executing grasp command"
            self.grasp_server.publish_feedback(feedback)
            
            success, grasp_state = self._execute_grasp_at_pixel(
                goal.pixel_x, goal.pixel_y, image_resp, goal, feedback
            )
            
            # Return to initial pose if requested and successful
            if success and goal.return_to_initial_pose and initial_pose:
                feedback.current_state = "RETURNING_TO_INITIAL_POSE"
                feedback.progress = 0.9
                feedback.status_message = "Returning to initial position"
                self.grasp_server.publish_feedback(feedback)
                self._return_to_initial_pose(initial_pose)
            
            result.success = success
            result.grasp_state = grasp_state
            result.message = "Grasp completed successfully" if success else "Grasp failed"
            
            if success:
                self.grasp_server.set_succeeded(result)
            else:
                self.grasp_server.set_aborted(result)
                
        except Exception as e:
            rospy.logerr(f"Grasp action failed: {e}")
            result.success = False
            result.message = f"Exception during grasp: {str(e)}"
            self.grasp_server.set_aborted(result)
    
    def execute_interactive_grasp_cb(self, goal):
        """Execute interactive grasp with user clicking on image"""
        rospy.loginfo("Executing interactive grasp")
        
        result = InteractiveGraspResult()
        feedback = InteractiveGraspFeedback()
        
        try:
            # Store initial pose if requested
            initial_pose = None
            if goal.return_to_initial_pose:
                pose_resp = self.get_initial_pose_srv(GetInitialPoseRequest())
                if pose_resp.success:
                    initial_pose = pose_resp.initial_pose
                else:
                    rospy.logwarn(f"Failed to get initial pose: {pose_resp.message}")
            
            # Get image from specified source
            feedback.current_state = "ACQUIRING_IMAGE"
            feedback.progress = 0.1
            feedback.status_message = f"Getting image from {goal.image_source}"
            self.interactive_grasp_server.publish_feedback(feedback)
            
            image_req = GetImageRequest()
            image_req.image_source = goal.image_source
            image_resp = self.get_image_srv(image_req)
            
            if not image_resp.success:
                result.success = False
                result.message = f"Failed to get image: {image_resp.message}"
                self.interactive_grasp_server.set_aborted(result)
                return
            
            # Process image for display
            if goal.show_preview:
                # Convert ROS image to OpenCV
                cv_image = self.cv_bridge.imgmsg_to_cv2(image_resp.image, "bgr8")
                
                # Show image and wait for user click
                feedback.current_state = "WAITING_FOR_USER_INPUT"
                feedback.progress = 0.2
                feedback.waiting_for_user_input = True
                feedback.status_message = "Click on an object to grasp"
                self.interactive_grasp_server.publish_feedback(feedback)
                
                window_title = goal.window_title if goal.window_title else 'Click to grasp'
                cv2.namedWindow(window_title)
                cv2.setMouseCallback(window_title, self._mouse_callback)
                
                self.image_display = cv_image
                self.image_click = None
                self.click_event.clear()
                
                cv2.imshow(window_title, self.image_display)
                
                # Wait for click or action cancellation
                while self.image_click is None and not rospy.is_shutdown():
                    if self.interactive_grasp_server.is_preempt_requested():
                        cv2.destroyAllWindows()
                        result.success = False
                        result.message = "Action was cancelled"
                        self.interactive_grasp_server.set_preempted(result)
                        return
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        cv2.destroyAllWindows()
                        result.success = False
                        result.message = "User cancelled with 'q' key"
                        self.interactive_grasp_server.set_aborted(result)
                        return
                    
                    time.sleep(0.1)
                
                cv2.destroyAllWindows()
                
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
            
            success, grasp_state = self._execute_grasp_at_pixel(
                pixel_x, pixel_y, image_resp, goal, feedback
            )
            
            # Return to initial pose if requested and successful
            if success and goal.return_to_initial_pose and initial_pose:
                feedback.current_state = "RETURNING_TO_INITIAL_POSE"
                feedback.progress = 0.9
                feedback.status_message = "Returning to initial position"
                self.interactive_grasp_server.publish_feedback(feedback)
                self._return_to_initial_pose(initial_pose)
            
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
                cv2.imshow('Click to grasp', clone)
    
    def _execute_grasp_at_pixel(self, pixel_x, pixel_y, image_response, goal, feedback):
        """Execute the actual grasp at specified pixel coordinates"""
        try:
            # Build the grasp request
            grasp_req = ExecuteGraspRequest()
            grasp_req.pixel_xy = geometry_pb2.Vec2(x=pixel_x, y=pixel_y)
            grasp_req.force_top_down_grasp = goal.force_top_down_grasp
            grasp_req.force_horizontal_grasp = goal.force_horizontal_grasp
            grasp_req.force_45_angle_grasp = goal.force_45_angle_grasp
            grasp_req.force_squeeze_grasp = goal.force_squeeze_grasp
            grasp_req.transforms_snapshot = image_response.transforms
            grasp_req.frame_name_image_sensor = image_response.frame_name_image_sensor
            grasp_req.camera_model = image_response.camera_info
            
            # Send grasp request
            grasp_resp = self.execute_grasp_srv(grasp_req)
            if not grasp_resp.success:
                return False, manipulation_api_pb2.MANIP_STATE_GRASP_FAILED
            
            manipulation_cmd_id = grasp_resp.manipulation_cmd_id
            
            # Monitor grasp progress
            while True:
                if self.grasp_server and self.grasp_server.is_preempt_requested():
                    return False, manipulation_api_pb2.MANIP_STATE_GRASP_FAILED
                if self.interactive_grasp_server and self.interactive_grasp_server.is_preempt_requested():
                    return False, manipulation_api_pb2.MANIP_STATE_GRASP_FAILED
                
                feedback_req = GetGraspFeedbackRequest()
                feedback_req.manipulation_cmd_id = manipulation_cmd_id
                feedback_resp = self.get_grasp_feedback_srv(feedback_req)
                
                if not feedback_resp.success:
                    rospy.logerr(f"Failed to get grasp feedback: {feedback_resp.message}")
                    break
                
                feedback.current_state = f"GRASP_{feedback_resp.state_name}"
                feedback.status_message = f"Manipulation state: {feedback_resp.state_name}"
                
                if hasattr(self, 'grasp_server') and self.grasp_server.is_active():
                    self.grasp_server.publish_feedback(feedback)
                elif hasattr(self, 'interactive_grasp_server') and self.interactive_grasp_server.is_active():
                    self.interactive_grasp_server.publish_feedback(feedback)
                
                if (feedback_resp.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or 
                    feedback_resp.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED):
                    break
                
                time.sleep(0.25)
            
            success = feedback_resp.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED
            
            # Handle post-grasp actions
            if success:
                self._handle_successful_grasp()
            
            return success, feedback_resp.current_state
            
        except Exception as e:
            rospy.logerr(f"Grasp execution failed: {e}")
            return False, manipulation_api_pb2.MANIP_STATE_GRASP_FAILED
    
    def _handle_successful_grasp(self):
        """Handle actions after a successful grasp (like opening gripper)"""
        try:
            # Open gripper to release object
            arm_req = ArmCommandRequest()
            arm_req.command_type = "open"
            self.arm_command_srv(arm_req)
            time.sleep(1.5)
            
            # Raise arm to carry position
            arm_req.command_type = "carry"
            self.arm_command_srv(arm_req)
            time.sleep(2.0)
            
            # Close gripper
            arm_req.command_type = "close"
            self.arm_command_srv(arm_req)
            time.sleep(1.5)
            
        except Exception as e:
            rospy.logerr(f"Failed to handle successful grasp: {e}")
    
    def _return_to_initial_pose(self, initial_pose):
        """Return robot to initial pose and stow arm"""
        try:
            # Stow the arm
            arm_req = ArmCommandRequest()
            arm_req.command_type = "stow"
            self.arm_command_srv(arm_req)
            time.sleep(2.0)
            
            # Move back to initial pose
            move_req = MoveToPositionRequest()
            move_req.target_pose = initial_pose.pose
            move_req.frame_name = initial_pose.header.frame_id
            move_resp = self.move_to_position_srv(move_req)
            if not move_resp.success:
                rospy.logwarn(f"Failed to return to initial pose: {move_resp.message}")
                    
        except Exception as e:
            rospy.logerr(f"Failed to return to initial pose: {e}")


if __name__ == '__main__':
    rospy.init_node('spot_grasp_action_server')
    server = SpotGraspActionServer()
    rospy.loginfo("Spot grasp action server running...")
    rospy.spin()
