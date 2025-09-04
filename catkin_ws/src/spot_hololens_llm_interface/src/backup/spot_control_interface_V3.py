#!/usr/bin/env python

import argparse
import sys
import cv2
import time
import logging
import math
import numpy as np
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
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, BODY_FRAME_NAME, get_a_tform_b, get_vision_tform_body
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.api import image_pb2
from bosdyn.client.image import build_image_request
from bosdyn.client.math_helpers import SE3Pose, Quat

from spot_hololens_llm_interface.manipulator import ManipulatorFunctions
from bosdyn.util import seconds_to_duration

from spot_hololens_llm_interface.arm_impedance_control_helpers import (apply_force_at_current_position,
                                           get_impedance_mobility_params, get_root_T_ground_body)


class SpotControlInterface(ManipulatorFunctions):
    """
    Clean V2 interface for Spot robot control.
    Each method represents exactly one API call with parameters.
    All methods are blocking where logical for deterministic FSM behavior.
    """

    def __init__(self, hostname="192.168.80.3"):
        self.hostname = hostname
        self.sdk = None
        self.robot = None
        self.command_client = None
        self.lease_client = None
        self.lease = None
        self.robot_state_client = None
        self.image_client = None
        self.manipulation_api_client = None
        self.world_object_client = None
        
    def connect_and_power_on(self):
        """Connect to robot and power on."""
        print("Connecting to Spot...")
        
        # Create SDK instance
        self.sdk = bosdyn.client.create_standard_sdk('SpotFSMControl')
        
        # Create robot instance
        self.robot = self.sdk.create_robot(self.hostname)
        
        # Authenticate
        bosdyn.client.util.authenticate(self.robot)
        
        # Time sync
        self.robot.time_sync.wait_for_sync()
        
        # Get clients
        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.lease_client = self.robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.manipulation_api_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)
        self.world_object_client = self.robot.ensure_client(WorldObjectClient.default_service_name)
        
        # Get lease
        self.lease = self.lease_client.acquire()
        
        # Power on
        self.robot.power_on(timeout_sec=20)
        
        print("Spot connected and powered on")
        
    def shutdown(self, power_off=False):
        """Shutdown robot gracefully."""
        print("Shutting down Spot...")
        
        try:
            # Stow arm if possible
            self.arm_stow()
        except:
            pass
            
        try:
            # Sit down
            self.sit()
        except:
            pass
            
        # Return lease
        if self.lease:
            self.lease_client.return_lease(self.lease)
            
        # Power off if requested
        if power_off:
            self.robot.power_off(cut_immediately=False, timeout_sec=20)
            
        print("Spot shutdown complete")
        
    def stand(self, height=0.0):
        """Stand with specified height."""
        print(f"Standing with height: {height}")
        cmd = RobotCommandBuilder.synchro_stand_command(body_height=height)
        self.command_client.robot_command(cmd)
        return True
        
    def sit(self):
        """Sit down."""
        print("Sitting down")
        cmd = RobotCommandBuilder.synchro_sit_command()
        self.command_client.robot_command(cmd)
        return True

    def stop(self):
        """Stop all movement."""
        print("Stopping")
        cmd = RobotCommandBuilder.stop_command()
        self.command_client.robot_command(cmd)
        return True
        
    def move_relative(self, x, y, yaw, timeout=10.0):
        """Move relative in body frame."""
        print(f"Moving relative: x={x}, y={y}, yaw={yaw}")
        
        # Create trajectory command
        trajectory_command = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
            x, y, yaw, self.robot.get_frame_tree_snapshot()
        )
        
        # Send command
        cmd_id = self.command_client.robot_command(trajectory_command, end_time_secs=time.time() + timeout)
        
        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < timeout:
            feedback = self.command_client.robot_command_feedback(cmd_id)
            mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
            
            if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
                print('Failed to reach the goal')
                return False
                
            traj_feedback = mobility_feedback.se2_trajectory_feedback
            if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                    traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
                print('Arrived at the goal.')
                return True
                
            time.sleep(0.2)
            
        print('Timeout reached')
        return False
        
    def arm_ready(self):
        """Move arm to ready position."""
        print("Moving arm to ready position")
        cmd = RobotCommandBuilder.arm_ready_command()
        self.command_client.robot_command(cmd)
        return True
        
    def arm_stow(self):
        """Stow the arm."""
        print("Stowing arm")
        cmd = RobotCommandBuilder.arm_stow_command()
        self.command_client.robot_command(cmd)
        return True
        
    def aim_hand(self, x, y, z):
        """Aim hand camera at specified position with downward pointing camera."""
        print(f"Aiming hand at: x={x}, y={y}, z={z}")
        
        try:
            # Use the V1 approach that actually works
            print("Using V1 arm positioning approach")
            
            # First, move arm to ready position (this works)
            print("Moving arm to ready position")
            self.arm_ready()
            
            # Then, try to use arm_pose_command like V1 does
            print("Attempting to position hand downward using arm_pose_command")
            
            from bosdyn.client.math_helpers import SE3Pose, Quat
            import math
            
            # Create downward-pointing rotation (90 degrees pitch = pointing down)
            # Let's try the opposite direction since -90 degrees pointed up
            downward_rotation = Quat.from_pitch(math.pi/2)  # π/2 radians = 90 degrees (should be down)
            
            # Position the hand at a more conservative height for detection
            # Use the original coordinates but with a safer height
            hand_position = SE3Pose(
                x=x, y=y, z=z + 0.05,  # Only 5cm above target (safer height)
                rot=downward_rotation
            )
            
            print(f"Target hand position: x={x}, y={y}, z={z+0.05}")
            print(f"Hand will point down with 90-degree pitch (should be down)")
            
            # Try using arm_pose_command like V1 (this should work)
            try:
                from bosdyn.client.frame_helpers import BODY_FRAME_NAME
                
                # Use the same approach as V1's ready_or_stow_arm
                # But let's try using the original coordinates more directly
                arm_command = RobotCommandBuilder.arm_pose_command(
                    x, y, z + 0.05,  # Use original coordinates with minimal offset
                    hand_position.rot.w, hand_position.rot.x,
                    hand_position.rot.y, hand_position.rot.z,
                    BODY_FRAME_NAME, 1.0  # 1 second duration
                )
                
                # Send the request
                cmd_id = self.command_client.robot_command(arm_command)
                
                # Wait until the arm arrives at the goal
                block_until_arm_arrives(self.command_client, cmd_id, timeout_sec=10.0)
                
                print("Hand successfully positioned downward using arm_pose_command")
                return True
                
            except Exception as pose_error:
                print(f"arm_pose_command failed: {pose_error}")
                print("Falling back to arm_ready (no downward pointing)")
                return self.arm_ready()
            
            # If we get here, the command succeeded but let's verify the position
            # If the arm went too high, fall back to a simpler approach
            print("Checking if arm position is reasonable...")
            if z + 0.05 > 0.8:  # If height is more than 80cm, it's probably too high
                print("Arm position seems too high, falling back to simple arm_ready")
                return self.arm_ready()
            
        except Exception as e:
            print(f"Error in aim_hand: {e}")
            # Fallback to arm_ready if everything fails
            print("Falling back to arm_ready")
            return self.arm_ready()
        

        
    def detect_label(self, label, provider='ultralytics', conf=0.01, image_source='hand_color_image'):
        """Detect a label in the image."""
        print(f"Detecting label: {label} with {provider}")
        
        try:
            # Get image
            image = self.get_bgr_image(image_source)
            if image is None:
                print("Failed to get image")
                return False
                
            # Use Ultralytics if available
            if provider == 'ultralytics':
                return self._detect_with_ultralytics(image, label, conf, image_source)
            else:
                print(f"Provider {provider} not supported")
                return False
                
        except Exception as e:
            print(f"Detection error: {e}")
            return False
    
    def _detect_with_ultralytics(self, bgr, target_label="bottle", conf=0.01, image_source="hand_color_image"):
        """Detect objects using Ultralytics YOLO (based on V1 implementation)."""
        try:
            from ultralytics import YOLO
        except Exception as e:
            print("Ultralytics not available:", e)
            return False

        # Load model (cache it)
        if not hasattr(self, "_yolo_model"):
            try:
                self._yolo_model = YOLO('yolov8n.pt')
                print(f"YOLO model loaded: yolov8n.pt")
            except Exception as e:
                print(f"Failed to load YOLO model: {e}")
                return False

        # Get model info
        names = self._yolo_model.model.names
        print(f"\n=== YOLO DETECTION DEBUG ===")
        print(f"Looking for: '{target_label}'")
        print(f"Available classes: {list(names.values())}")
        
        # Save debug image
        import os
        import cv2
        import time
        
        # Check if image is valid before saving
        if bgr is None:
            print("ERROR: Cannot save debug image - image data is None")
            return False
        
        if not hasattr(bgr, 'shape') or len(bgr.shape) != 3:
            print(f"ERROR: Cannot save debug image - invalid image shape: {bgr.shape if hasattr(bgr, 'shape') else 'No shape'}")
            return False
        
        # Use absolute path for debug images so they're easy to find
        debug_dir = "/catkin_ws/debug_images"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        # Debug: Check image data
        print(f"Debug: Image shape: {bgr.shape}")
        print(f"Debug: Image type: {type(bgr)}")
        print(f"Debug: Image data type: {bgr.dtype}")
        print(f"Debug: Image min/max values: {bgr.min()}/{bgr.max()}")
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        debug_filename = f"{debug_dir}/debug_detection_{timestamp}.jpg"
        
        # Try to save with error handling
        try:
            success = cv2.imwrite(debug_filename, bgr)
            if success:
                print(f"Debug image saved successfully: {debug_filename}")
                print(f"Debug image absolute path: {os.path.abspath(debug_filename)}")
                print(f"Debug image file size: {os.path.getsize(debug_filename)} bytes")
            else:
                print(f"ERROR: cv2.imwrite failed to save image")
                print(f"Attempted path: {debug_filename}")
        except Exception as e:
            print(f"ERROR saving debug image: {e}")
            print(f"Attempted path: {debug_filename}")
        
        # Check if target_label exists in the model
        original_label = target_label
        print(f"DEBUG: Original target label: '{original_label}'")
        
        if target_label not in names.values():
            print(f"WARNING: '{target_label}' not found in model classes!")
            print(f"Available classes: {list(names.values())}")
            # Try to find similar classes with better matching
            similar = []
            
            # First try exact substring match
            for name in names.values():
                if target_label.lower() in name.lower() or name.lower() in target_label.lower():
                    similar.append(name)
            
            # If no exact match, try fuzzy matching for common cases
            if not similar and target_label.lower() == 'ball':
                similar = [name for name in names.values() if 'sport' in name.lower() or 'baseball' in name.lower()]
            
            if similar:
                print(f"Similar classes found: {similar}")
                target_label = similar[0]  # Use first similar class
                print(f"DEBUG: Mapped '{original_label}' -> '{target_label}'")
                print(f"Now searching for: '{target_label}'")
        else:
            print(f"DEBUG: Target label '{target_label}' found in model classes")
        
        # Run detection with very low confidence to see ALL objects (like V1)
        results = self._yolo_model.predict(bgr, conf=0.01, verbose=False)
        if not results:
            print("No objects detected at all (even with very low confidence)")
            return False

        # Collect all detections
        all_detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                label = names.get(cls_id, str(cls_id))
                score = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                
                all_detections.append({
                    'label': label,
                    'score': score,
                    'center': (cx, cy),
                    'bbox': (x1, y1, x2, y2)
                })
        
        # Sort by confidence
        all_detections.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n=== YOLO DETECTION DEBUG ===")
        print(f"Looking for: '{target_label}'")
        print(f"Confidence threshold: {conf}")
        print(f"Total objects detected: {len(all_detections)}")
        
        if all_detections:
            print("All detected objects (sorted by confidence):")
            for i, det in enumerate(all_detections):
                print(f"  {i+1}. {det['label']} (confidence: {det['score']:.3f}) at pixel ({det['center'][0]}, {det['center'][1]})")
        else:
            print("No objects detected above confidence threshold")
        
        print("=" * 30)
        
        # Find best match for target label
        best = None
        best_score = -1.0
        
        for det in all_detections:
            if det['label'] == target_label:
                if det['score'] >= conf:
                    if det['score'] > best_score:
                        best = det
                        best_score = det['score']
        
        if best:
            print(f"✅ Found target '{target_label}' with confidence {best['score']:.3f}")
            # Store detection for pick_from_pixel
            if hasattr(self, 'last_detection'):
                self.last_detection = {
                    'label': best['label'],
                    'confidence': best['score'],
                    'bbox': best['bbox'],
                    'center': best['center']
                }
            return True
        else:
            print(f"❌ Target '{target_label}' not found or below confidence threshold {conf}")
            if all_detections:
                print(f"Available objects: {[det['label'] for det in all_detections]}")
            return False
            
    def pick_from_pixel(self, image_source, x, y, top_down=True, auto_walk=False, timeout=25.0):
        """Pick object at pixel coordinates using V1 approach."""
        print(f"Picking from pixel: x={x}, y={y}, top_down={top_down}")
        
        try:
            # Get image to get the transforms snapshot (like V1)
            image_responses = self.image_client.get_image_from_sources([image_source])
            if not image_responses:
                print(f"Failed to get image from {image_source}")
                return False
                
            image = image_responses[0]
            
            # Create pick request using V1 approach
            pick = manipulation_api_pb2.PickObjectInImage(
                pixel_xy=geometry_pb2.Vec2(x=x, y=y),
                transforms_snapshot_for_camera=image.shot.transforms_snapshot,
                frame_name_image_sensor=image.shot.frame_name_image_sensor
            )
            
            # Set top_down and auto_walk
            if top_down:
                pick.grasp_params.allowable_orientation.add()
                pick.grasp_params.allowable_orientation[0].vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(
                    geometry_pb2.Vec3(x=1, y=0, z=0))
                pick.grasp_params.allowable_orientation[0].vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(
                    geometry_pb2.Vec3(x=0, y=0, z=-1))
                pick.grasp_params.allowable_orientation[0].vector_alignment_with_tolerance.threshold_radians = 0.17
                
                # Set the frame name for grasp parameters (like V1 does)
                if hasattr(pick.grasp_params, 'grasp_params_frame_name'):
                    pick.grasp_params.grasp_params_frame_name = "vision"
                    print("Set grasp_params_frame_name to 'vision'")
                else:
                    print("grasp_params_frame_name field not available")
            
            # Create the manipulation request
            req = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=pick)
            
            # Send request
            cmd_response = self.manipulation_api_client.manipulation_api_command(manipulation_api_request=req)
            cmd_id = cmd_response.manipulation_cmd_id
            
            print(f"Pick command sent with ID: {cmd_id}")
            
            # Wait for completion
            start_time = time.time()
            
            # Resolve enum container (supports older/newer SDKs) - like V1 does
            EnumNew = getattr(manipulation_api_pb2, "ManipulationApiFeedbackState", None)
            EnumOld = getattr(manipulation_api_pb2, "ManipulationFeedbackState", None)
            if EnumNew:
                ENUM = EnumNew
                SUCCESS = getattr(ENUM, "STATE_GRASP_SUCCEEDED", None)
                FAILED = getattr(ENUM, "STATE_FAILED", None)
                UNKNOWN = getattr(ENUM, "STATE_UNKNOWN", None)
                name_fn = ENUM.Name
            else:
                ENUM = EnumOld
                SUCCESS = getattr(ENUM, "MANIP_STATE_GRASP_SUCCEEDED", None)
                FAILED = getattr(ENUM, "MANIP_STATE_FAILED", None)
                UNKNOWN = getattr(ENUM, "MANIP_STATE_UNKNOWN", None)
                name_fn = ENUM.Name if ENUM else (lambda v: str(v))
            
            # Also check for intermediate states that indicate progress
            SEARCHING = getattr(ENUM, "MANIP_STATE_SEARCHING_FOR_GRASP", None)
            GRASPING = getattr(ENUM, "MANIP_STATE_GRASPING_OBJECT", None)
            
            print(f"Using feedback enum: {ENUM}")
            print(f"Success state: {SUCCESS}, Failed state: {FAILED}, Unknown state: {UNKNOWN}")
            print(f"Searching state: {SEARCHING}, Grasping state: {GRASPING}")
            
            # Track time spent in searching state
            searching_start_time = None
            
            while time.time() - start_time < timeout:
                try:
                    feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(manipulation_cmd_id=cmd_id)
                    feedback = self.manipulation_api_client.manipulation_api_feedback_command(manipulation_api_feedback_request=feedback_request)
                    
                    # Field name differs by SDK - like V1 does
                    state_val = getattr(feedback, "current_state", None)
                    if state_val is None:
                        state_val = getattr(feedback, "state", None)
                    
                    try:
                        print(f"Manipulation state: {name_fn(state_val)}")
                    except Exception:
                        print(f"Manipulation state (raw): {state_val}")
                    
                    # Handle intermediate states
                    if state_val == SEARCHING:
                        if searching_start_time is None:
                            searching_start_time = time.time()
                            print("Started searching for grasp...")
                        else:
                            search_duration = time.time() - searching_start_time
                            print(f"Still searching for grasp... ({search_duration:.1f}s)")
                            
                        # Timeout if searching for too long (10 seconds)
                        if time.time() - searching_start_time > 10.0:
                            print("Searching timeout - robot may be stuck")
                            return False
                            
                    elif state_val == GRASPING:
                        print("Grasping object...")
                        searching_start_time = None  # Reset searching timer
                        
                    elif state_val in tuple(v for v in (SUCCESS, FAILED, UNKNOWN) if v is not None):
                        success = state_val == SUCCESS
                        print(f"Pick completed with success: {success}")
                        return success
                        
                except Exception as e:
                    print(f"Error getting feedback: {e}")
                    
                time.sleep(0.2)  # Use 0.2 like V1
                
            print("Pick timeout")
            return False
            
        except Exception as e:
            print(f"Pick error: {e}")
            return False
            
    def grasp_body_point(self, x, y, z, approach=0.12, lift=0.15):
        """Grasp object at body frame coordinates."""
        print(f"Grasping body point: x={x}, y={y}, z={z}")
        
        # Create grasp request
        grasp_request = manipulation_api_pb2.GraspObjectRequest(
            grasp_params=manipulation_api_pb2.GraspParams(
                grasp_params_frame_name=BODY_FRAME_NAME,
                grasp_params_target_in_frame=geometry_pb2.Vec3(x=x, y=y, z=z),
                approach_angle=geometry_pb2.EulerZXY(yaw=0, roll=0, pitch=0),
                approach_distance=approach,
                lift_distance=lift
            )
        )
        
        # Send request
        cmd_id = self.manipulation_api_client.grasp_object(grasp_request)
        
        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < 30.0:  # 30 second timeout
            feedback = self.manipulation_api_client.manipulation_api_feedback_command(cmd_id)
            
            if feedback.current_state == manipulation_api_pb2.MANIP_STATE_GRASPING_OBJECT:
                print("Grasping object...")
            elif feedback.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                print("Grasp succeeded!")
                return True
            elif feedback.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                print("Grasp failed")
                return False
                
            time.sleep(0.1)
            
        print("Grasp timeout")
        return False
        
    def get_bgr_image(self, source):
        """Get BGR image from specified source."""
        try:
            image_request = build_image_request(source, pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8)
            image_response = self.image_client.get_image([image_request])
            
            if len(image_response) > 0:
                image = image_response[0]
                print(f"Image format: {image.shot.image.pixel_format}")
                print(f"Image dimensions: {image.shot.image.rows}x{image.shot.image.cols}")
                print(f"Image data size: {len(image.shot.image.data)}")
                
                # Handle different pixel formats
                img = np.frombuffer(image.shot.image.data, dtype=np.uint8)
                total_pixels = image.shot.image.rows * image.shot.image.cols
                
                # First, try JPEG decoding for any format (common for compressed images)
                try:
                    img_decoded = cv2.imdecode(img, cv2.IMREAD_COLOR)
                    if img_decoded is not None:
                        print(f"Successfully decoded format {image.shot.image.pixel_format} as JPEG")
                        return img_decoded
                except Exception as e:
                    print(f"JPEG decode failed: {e}")
                
                # Handle specific formats
                if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                    # Standard RGB format
                    expected_size = total_pixels * 3
                    if len(img) == expected_size:
                        img = img.reshape(image.shot.image.rows, image.shot.image.cols, 3)
                        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    else:
                        print(f"RGB format size mismatch: got {len(img)}, expected {expected_size}")
                        
                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                    # Grayscale format
                    expected_size = total_pixels
                    if len(img) == expected_size:
                        img = img.reshape(image.shot.image.rows, image.shot.image.cols)
                        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    else:
                        print(f"Grayscale format size mismatch: got {len(img)}, expected {expected_size}")
                        
                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                    # RGBA format
                    expected_size = total_pixels * 4
                    if len(img) == expected_size:
                        img = img.reshape(image.shot.image.rows, image.shot.image.cols, 4)
                        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                    else:
                        print(f"RGBA format size mismatch: got {len(img)}, expected {expected_size}")
                        
                elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_JPEG:
                    # JPEG compressed format - already tried above, but fallback
                    print("JPEG format detected but decode failed")
                    
                else:
                    # Unknown format - try size-based interpretation
                    print(f"Unknown pixel format {image.shot.image.pixel_format}, attempting size-based interpretation...")
                    
                    if len(img) == total_pixels:
                        # Grayscale
                        print("Interpreting as grayscale")
                        img = img.reshape(image.shot.image.rows, image.shot.image.cols)
                        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif len(img) == total_pixels * 3:
                        # RGB
                        print("Interpreting as RGB")
                        img = img.reshape(image.shot.image.rows, image.shot.image.cols, 3)
                        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    elif len(img) == total_pixels * 4:
                        # RGBA
                        print("Interpreting as RGBA")
                        img = img.reshape(image.shot.image.rows, image.shot.image.cols, 4)
                        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                    else:
                        print(f"Unknown image format with {len(img)} bytes")
                        print(f"Expected sizes: {total_pixels} (grayscale), {total_pixels * 3} (RGB), {total_pixels * 4} (RGBA)")
                
                # If we get here, all attempts failed
                print(f"All image format attempts failed for format {image.shot.image.pixel_format}")
                return None
                    
            return None
            
        except Exception as e:
            print(f"Image error: {e}")
            return None
            
    def move_to_cartesian_pose_rt_task(self, task_T_tool_desired, root_T_task, wr1_T_tool):
        """Move arm to cartesian pose using RT task."""
        print("Moving arm to cartesian pose")
        
        # Create arm command
        arm_command = arm_command_pb2.ArmCommand.Request(
            cartesian_command=arm_command_pb2.ArmCartesianCommand.Request(
                pose_trajectory_in_task=trajectory_pb2.SE3Trajectory(
                    reference_time=self.robot.time_sync.robot_timestamp_from_local_secs(time.time()),
                    poses=[geometry_pb2.SE3Pose(
                        position=geometry_pb2.Vec3(
                            x=task_T_tool_desired.x,
                            y=task_T_tool_desired.y,
                            z=task_T_tool_desired.z
                        ),
                        rotation=geometry_pb2.Quaternion(
                            w=task_T_tool_desired.rot.w,
                            x=task_T_tool_desired.rot.x,
                            y=task_T_tool_desired.rot.y,
                            z=task_T_tool_desired.rot.z
                        )
                    )]
                ),
                root_frame_name=root_T_task.frame_name,
                root_tform_task=geometry_pb2.SE3Pose(
                    position=geometry_pb2.Vec3(
                        x=root_T_task.x,
                        y=root_T_task.y,
                        z=root_T_task.z
                    ),
                    rotation=geometry_pb2.Quaternion(
                        w=root_T_task.rot.w,
                        x=root_T_task.rot.x,
                        y=root_T_task.rot.y,
                        z=root_T_task.rot.z
                    )
                ),
                wr1_tform_tool=geometry_pb2.SE3Pose(
                    position=geometry_pb2.Vec3(
                        x=wr1_T_tool.x,
                        y=wr1_T_tool.y,
                        z=wr1_T_tool.z
                    ),
                    rotation=geometry_pb2.Quaternion(
                        w=wr1_T_tool.rot.w,
                        x=wr1_T_tool.rot.x,
                        y=wr1_T_tool.rot.y,
                        z=wr1_T_tool.rot.z
                    )
                )
            )
        )
        
        # Send command
        cmd = RobotCommandBuilder.build_arm_command(arm_command)
        cmd_id = self.command_client.robot_command(cmd)
        
        # Wait for completion
        block_until_arm_arrives(self.command_client, cmd_id, timeout_sec=10.0)
        return True
