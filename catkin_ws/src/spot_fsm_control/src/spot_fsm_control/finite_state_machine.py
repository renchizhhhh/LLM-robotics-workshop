#!/usr/bin/env python

import time
import math
import json
import logging
from statemachine import StateMachine, State
import numpy as np
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b, BODY_FRAME_NAME
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import geometry_pb2, manipulation_api_pb2, world_object_pb2, image_pb2
from bosdyn.util import seconds_to_duration


class SpotStateMachine(StateMachine):
    """
    A clean V2 state machine for Spot robot control.
    Each robot action is exactly one API call with parameters.
    Complex tasks are built by the LLM as a plan of small actions.
    The FSM executes 1 action at a time and returns to neutral state after each action.
    """
    
    # States
    sit = State(initial=True)
    stand = State()
    action = State()
    
    # Transitions
    sit_to_stand = sit.to(stand)
    sit_to_action = sit.to(action)
    stand_to_sit = stand.to(sit)
    stand_to_action = stand.to(action)
    action_to_stand = action.to(stand)
    action_to_sit = action.to(sit)
    
    def __init__(self, spot_interface=None, dummy_mode=False):
        super().__init__()
        self.spot_interface = spot_interface
        self.dummy_mode = dummy_mode
        self.pending_action = None
        self.action_result = None
        self.last_detection = None
        
        # Command deduplication and pick protection
        self.PICK_DUPLICATE_WINDOW = 10.0
        self._last_pick_command = None
        self._pick_command_time = 0
        self._pick_executing = False
        
        # Data logging
        self.data_logger = None
        
        if dummy_mode:
            logging.info("FSM running in DUMMY MODE - no real robot commands will be sent")
        else:
            logging.info("FSM running in REAL ROBOT MODE")
        
    def enqueue_action(self, action_dict):
        """
        Queue an action for execution.
        
        Args:
            action_dict: Dictionary with 'op' key and action parameters
            
        Returns:
            bool: True if action was executed successfully, False otherwise
        """
        op = action_dict.get('op')
        
        # Check for duplicate pick commands
        if op in ['pick_from_pixel', 'pick_apriltag', 'grasp_body_point']:
            if self._is_pick_command_duplicate(op):
                logging.warning(f"Skipping duplicate {op} command within time window")
                return False
        
        logging.info(f"Enqueueing action: {action_dict}")
        self.pending_action = action_dict
        self.action_result = None
        
        # Auto-stand if currently sitting and action requires standing
        standing_required = [
            'move_relative', 'arm_ready', 'arm_stow', 'aim_hand', 
            'gripper', 'open_gripper', 'close_gripper', 'detect_label', 'list_objects', 'pick_from_pixel', 'grasp_body_point', 'pick_apriltag'
        ]
        
        if self.current_state == self.sit and op in standing_required:
            logging.info("Auto-standing before action that requires standing")
            if not self.dummy_mode and self.spot_interface:
                self.spot_interface.stand(height=0.0)
            self.send('sit_to_stand')
        
        # Start action state
        if self.current_state == self.stand:
            self.send('stand_to_action')
        elif self.current_state == self.sit and op not in standing_required:
            self.send('sit_to_action')
        
        # Wait for action to complete and return result
        start_time = time.time()
        while self.pending_action and time.time() - start_time < 30.0:
            time.sleep(0.1)
        
        return self.action_result is True
    
    def on_enter_action(self):
        """Execute the pending action when entering action state."""
        if not self.pending_action:
            logging.info("No pending action, returning to stand")
            self.send('action_to_stand')
            return
            
        action = self.pending_action
        op = action.get('op')
        
        logging.info(f"Executing action: {op} with params: {action}")
        
        try:
            result = self._execute_action(action)
            self.action_result = result
            
            if result:
                logging.info(f"Action {op} completed successfully")
            else:
                logging.warning(f"Action {op} failed")
            
            self.pending_action = None
            
            # Return to stand state (unless action was sit)
            print(f"Current state before transition: {self.current_state}")
            if op == 'sit':
                print(f"Action {op} completed, transitioning to sit state")
                if self.current_state != self.sit:
                    self.send('action_to_sit')
                else:
                    print("Already in sit state, no transition needed")
            else:
                print(f"Action {op} completed, transitioning to stand state")
                if self.current_state != self.stand:
                    self.send('action_to_stand')
                else:
                    print("Already in stand state, no transition needed")
            print(f"State transition logic completed for: {'sit' if op == 'sit' else 'stand'}")
                
        except Exception as e:
            logging.error(f"Action {op} failed: {e}")
            self.action_result = False
            self.pending_action = None
            
            # Return to stand state on error (unless we were trying to sit)
            print(f"Current state before error transition: {self.current_state}")
            if op == 'sit':
                print(f"Action {op} failed, transitioning to sit state")
                if self.current_state != self.sit:
                    self.send('action_to_sit')
                else:
                    print("Already in sit state, no transition needed")
            else:
                print(f"Action {op} failed, transitioning to stand state")
                if self.current_state != self.stand:
                    self.send('action_to_stand')
                else:
                    print("Already in stand state, no transition needed")
    
    def _execute_action(self, action):
        """Execute a single action based on the operation type with safety checks."""
        op = action.get('op')
        
        # Check if robot is powered on and ready
        if not self._check_robot_ready():
            logging.warning("Robot not ready for action")
            return False
        
        # Check state-based preconditions
        if not self._check_action_preconditions(op):
            logging.warning(f"Action {op} not allowed in current state {self.current_state}")
            return False
        
        if op == 'stand':
            height = action.get('height', 0.0)
            height = max(-0.2, min(0.5, height))
            logging.info(f"Standing with height: {height}")
            if self.dummy_mode:
                return True
            return self.spot_interface.stand(height)
            
        elif op == 'sit':
            logging.info("Sitting down")
            if self.dummy_mode:
                return True
            return self.spot_interface.sit()
            
        elif op == 'move_relative':
            x = action.get('x', 0.0)
            y = action.get('y', 0.0)
            yaw = action.get('yaw', 0.0)
            timeout = action.get('timeout', 10.0)
            
            x = max(-5.0, min(5.0, x))
            y = max(-3.0, min(3.0, y))
            timeout = max(1.0, min(60.0, timeout))
            
            # Handle full rotations by breaking them into smaller segments
            # This ensures the robot actually performs the full rotation
            if abs(yaw) > math.pi:
                logging.info(f"Large yaw detected ({yaw:.3f} rad), breaking into segments for full rotation")
                
                # Calculate number of segments needed (max π/2 per segment)
                segment_size = math.pi / 2  # 90 degrees per segment
                num_segments = int(math.ceil(abs(yaw) / segment_size))
                segment_yaw = yaw / num_segments
                
                logging.info(f"Breaking {yaw:.3f} rad rotation into {num_segments} segments of {segment_yaw:.3f} rad each")
                
                # Execute segments sequentially
                for i in range(num_segments):
                    logging.info(f"Executing rotation segment {i+1}/{num_segments} of {num_segments}")
                    logging.info(f"Segment {i+1}: rotating {segment_yaw:.3f} rad ({math.degrees(segment_yaw):.1f} degrees)")
                    
                    # Use full timeout for each segment to ensure completion
                    segment_timeout = timeout
                    result = self.spot_interface.move_relative(x, y, segment_yaw, segment_timeout)
                    
                    if not result:
                        logging.warning(f"Rotation segment {i+1} failed")
                        return False
                    else:
                        logging.info(f"Segment {i+1} completed successfully")
                    
                    # Longer pause between segments for stability
                    if i < num_segments - 1:  # Don't sleep after the last segment
                        logging.info(f"Pausing 1 second before next segment...")
                        time.sleep(1.0)
                        
                        # Check if robot is still ready for next segment
                        if not self._check_robot_ready():
                            logging.warning("Robot not ready for next rotation segment")
                            return False
                
                logging.info(f"Full rotation completed: {yaw:.3f} rad ({math.degrees(yaw):.1f} degrees) in {num_segments} segments")
                return True
            else:
                # Small rotations can be done directly
                logging.info(f"Moving relative: x={x}, y={y}, yaw={yaw:.3f}")
                if self.dummy_mode:
                    return True
                return self.spot_interface.move_relative(x, y, yaw, timeout)
            
        elif op == 'arm_ready':
            if self.current_state == self.sit:
                logging.warning("Cannot move arm while sitting")
                return False
            logging.info("Moving arm to ready position")
            if self.dummy_mode:
                return True
            return self.spot_interface.arm_ready()
            
        elif op == 'arm_stow':
            logging.info("Stowing arm")
            if self.dummy_mode:
                return True
            return self.spot_interface.arm_stow()
            
        elif op == 'aim_hand':
            x = action.get('x', 0.0)
            y = action.get('y', 0.0)
            z = action.get('z', 0.0)
            
            if not (0.60 <= x <= 0.90):
                logging.warning(f"Hand camera x={x} outside safe range [0.60, 0.90] m")
                return False
            if not (-0.5 <= y <= 0.5):
                logging.warning(f"Hand camera y={y} outside safe range [-0.5, 0.5] m")
                return False
            if not (-0.2 <= z <= 0.65):
                logging.warning(f"Hand camera z={z} outside safe range [-0.2, 0.65] m")
                return False
            
            logging.info(f"Aiming hand at: x={x}, y={y}, z={z}")
            if self.dummy_mode:
                return True
            return self.spot_interface.aim_hand(x, y, z)
            
        elif op == 'gripper':
            mode = action.get('mode', 'open')
            fraction = action.get('fraction', 0.5)
            
            if mode not in ['open', 'close']:
                logging.warning(f"Invalid gripper mode: {mode}")
                return False
            
            fraction = max(0.0, min(1.0, fraction))
            
            logging.info(f"Gripper: mode={mode}, fraction={fraction}")
            if self.dummy_mode:
                return True
            return self.spot_interface.gripper(mode, fraction)
            
        elif op == 'open_gripper':
            """Convenience action to open gripper fully."""
            logging.info("Opening gripper fully")
            if self.dummy_mode:
                return True
            return self.spot_interface.gripper('open')
            
        elif op == 'close_gripper':
            """Convenience action to close gripper fully."""
            logging.info("Closing gripper fully")
            if self.dummy_mode:
                return True
            return self.spot_interface.gripper('close')
            
        elif op == 'detect_label':
            label = action.get('label', '')
            provider = action.get('provider', 'ultralytics')
            conf = action.get('conf', 0.25)
            image_source = action.get('image_source', 'hand_color_image')
            
            if not label:
                logging.warning("No label specified for detection")
                return False
            if not (0.0 <= conf <= 1.0):
                logging.warning(f"Confidence {conf} outside valid range [0.0, 1.0]")
                return False
            
            logging.info(f"Detecting label: {label} with {provider}")
            if self.dummy_mode:
                self.last_detection = {
                    'label': label,
                    'confidence': conf,
                    'bbox': [100, 100, 200, 200],
                    'center': [150, 150]
                }
                return True
            
            # Get all detected objects and save debug image
            all_detections = self._detect_label_with_list(label, provider, conf, image_source)
            if all_detections:
                # Store the target detection
                self.last_detection = all_detections['target']
                # Also store all detections for reference
                self.all_detections = all_detections['all']
                logging.info(f"Detection successful: {self.last_detection}")
                logging.info(f"All objects detected: {len(self.all_detections)} objects")
                return True
            else:
                # Even if target label not found, we might have detected other objects
                # Check if we have any detections from the interface
                if hasattr(self.spot_interface, 'all_detections') and self.spot_interface.all_detections:
                    self.all_detections = self.spot_interface.all_detections
                    # Create a dummy target detection for the requested label
                    self.last_detection = {
                        'label': label,
                        'confidence': 0.0,
                        'bbox': [0, 0, 0, 0],
                        'center': [0, 0]
                    }
                    logging.info(f"Target label '{label}' not found, but {len(self.all_detections)} other objects detected")
                    return True
                else:
                    logging.warning(f"No detection found for label: {label}")
                    return False
                
        elif op == 'list_objects':
            """List all currently detected objects without running new detection."""
            logging.info("Listing currently detected objects")
            self.list_detected_objects()
            return True
            
        elif op == 'list_debug_images':
            """List all debug images saved to the debug_images directory."""
            logging.info("Listing debug images")
            if not self.dummy_mode and self.spot_interface:
                self.spot_interface.list_debug_images()
            else:
                print("DUMMY MODE: Debug image listing simulated")
            return True
            
        elif op == 'pick_from_pixel':
            # Pick protection check
            command_name = "pick_from_pixel"
            if self._is_pick_command_duplicate(command_name):
                print(f"WARNING: Skipping duplicate {command_name} command")
                return False
            
            if self._pick_executing:
                print("Pick operation already in progress, skipping...")
                return False
            
            # Check if we have detection data and require confirmation
            if not hasattr(self, 'all_detections') or not self.all_detections:
                print("No detection data available. Please run detect_label first.")
                return False
            
            # Show all detected objects and require confirmation
            print("\n=== OBJECT DETECTION SUMMARY ===")
            print(f"Target object: {self.last_detection['label']} at pixel {self.last_detection['center']}")
            print(f"All detected objects ({len(self.all_detections)}):")
            for i, det in enumerate(self.all_detections):
                print(f"  {i+1}. {det['label']} (confidence: {det['score']:.3f}) at pixel {det['center']}")
            
            # Require user confirmation before picking
            print(f"\nAbout to pick: {self.last_detection['label']} at pixel {self.last_detection['center']}")
            print("Press 'y' or 'yes' to confirm, any other key to cancel:")
            
            try:
                user_input = input().strip().lower()
                if user_input not in ['y', 'yes']:
                    print("Pick operation cancelled by user")
                    
                    # Offer to retry detection with a new image
                    print("\nWould you like to retry detection with a new image?")
                    print("Press 'y' or 'yes' to retry detection, any other key to cancel:")
                    
                    try:
                        retry_input = input().strip().lower()
                        if retry_input in ['y', 'yes']:
                            print("Retrying detection with new image...")
                            
                            # Get the original detection parameters from the last detection
                            if hasattr(self, 'last_detection') and self.last_detection:
                                original_label = self.last_detection['label']
                                
                                # Retry detection with the same parameters
                                retry_action = {
                                    'op': 'detect_label',
                                    'label': original_label,
                                    'provider': 'ultralytics',
                                    'conf': 0.01,  # Use low confidence for retry
                                    'image_source': 'hand_color_image'
                                }
                                
                                print(f"Retrying detection for: {original_label}")
                                retry_result = self._execute_action(retry_action)
                                
                                if retry_result:
                                    print("Retry detection completed. You can now try pick_from_pixel again.")
                                    return False  # Return False to indicate pick was cancelled, but detection was retried
                                else:
                                    print("Retry detection failed.")
                                    return False
                            else:
                                print("No previous detection data available for retry.")
                                return False
                        else:
                            print("Detection retry cancelled by user")
                            return False
                    except (EOFError, KeyboardInterrupt):
                        print("Detection retry cancelled (no input)")
                        return False
                        
                print("Pick operation confirmed by user")
            except (EOFError, KeyboardInterrupt):
                print("Pick operation cancelled (no input)")
                return False
            
            # Set protection flags
            self._set_pick_protection_flags(command_name)
            
            try:
                image_source = action.get('image_source', 'hand_color_image')
                x = action.get('x', 0)
                y = action.get('y', 0)
                top_down = action.get('top_down', True)
                auto_walk = action.get('auto_walk', False)
                timeout = action.get('timeout', 25.0)
                
                # Safety: Validate pixel coordinates
                if not (0 <= x <= 1920):  # Assuming max image width
                    print(f"Pixel x={x} outside valid range [0, 1920]")
                    self._reset_pick_protection_flags()
                    return False
                if not (0 <= y <= 1080):  # Assuming max image height
                    print(f"Pixel y={y} outside valid range [0, 1080]")
                    self._reset_pick_protection_flags()
                    return False
                timeout = max(5.0, min(60.0, timeout))  # 5-60 seconds
                
                print(f"Picking from pixel: x={x}, y={y}, top_down={top_down}")
                if self.dummy_mode:
                    print("DUMMY MODE: Pick from pixel command simulated")
                    self._reset_pick_protection_flags()
                    return True
                
                result = self.spot_interface.pick_from_pixel(image_source, x, y, top_down, auto_walk, timeout)
                
                # If pick was successful, automatically open gripper to release the object
                if result:
                    print("Pick operation successful! Automatically opening gripper to release object...")
                    try:
                        gripper_result = self.spot_interface.gripper('open')
                        if gripper_result:
                            print("Gripper opened successfully - object released")
                        else:
                            print("Warning: Failed to open gripper after pick")
                    except Exception as e:
                        print(f"Warning: Error opening gripper after pick: {e}")
                else:
                    print("Pick operation failed")
                
                self._reset_pick_protection_flags()
                return result

            except Exception as e:
                print(f"Pick from pixel failed: {e}")
                self._reset_pick_protection_flags()
                return False
            
        elif op == 'grasp_body_point':
            x = action.get('x', 0.0)
            y = action.get('y', 0.0)
            z = action.get('z', 0.0)
            approach = action.get('approach', 0.12)
            lift = action.get('lift', 0.15)
            
            # Safety: Check scripted BODY grasp limits
            r = math.hypot(x, y)
            if not (0.30 <= r <= 0.85):
                print(f"Grasp target r={r:.2f} outside safe range [0.30, 0.85] m")
                return False
            if not (-0.45 <= z <= 1.0):
                print(f"Grasp target z={z} outside safe range [-0.45, 1.0] m")
                return False
            
            # Safety: Validate approach and lift distances
            approach = max(0.05, min(0.30, approach))  # 5-30cm
            lift = max(0.05, min(0.50, lift))  # 5-50cm
            
            # Require user confirmation for grasp operations
            print(f"\n=== GRASP OPERATION CONFIRMATION ===")
            print(f"About to grasp at body coordinates: x={x:.3f}, y={y:.3f}, z={z:.3f}")
            print(f"Approach distance: {approach:.3f}m, Lift distance: {lift:.3f}m")
            print("Press 'y' or 'yes' to confirm, any other key to cancel:")
            
            try:
                user_input = input().strip().lower()
                if user_input not in ['y', 'yes']:
                    print("Grasp operation cancelled by user")
                    
                    # Offer to retry detection with a new image
                    print("\nWould you like to retry detection with a new image?")
                    print("Press 'y' or 'yes' to retry detection, any other key to cancel:")
                    
                    try:
                        retry_input = input().strip().lower()
                        if retry_input in ['y', 'yes']:
                            print("Retrying detection with new image...")
                            
                            # Get the original detection parameters from the last detection
                            if hasattr(self, 'last_detection') and self.last_detection:
                                original_label = self.last_detection['label']
                                
                                # Retry detection with the same parameters
                                retry_action = {
                                    'op': 'detect_label',
                                    'label': original_label,
                                    'provider': 'ultralytics',
                                    'conf': 0.01,  # Use low confidence for retry
                                    'image_source': 'hand_color_image'
                                }
                                
                                print(f"Retrying detection for: {original_label}")
                                retry_result = self._execute_action(retry_action)
                                
                                if retry_result:
                                    print("Retry detection completed. You can now try grasp_body_point again.")
                                    return False  # Return False to indicate grasp was cancelled, but detection was retried
                                else:
                                    print("Retry detection failed.")
                                    return False
                            else:
                                print("No previous detection data available for retry.")
                                return False
                        else:
                            print("Detection retry cancelled by user")
                            return False
                    except (EOFError, KeyboardInterrupt):
                        print("Detection retry cancelled (no input)")
                        return False
                        
                print("Grasp operation confirmed by user")
            except (EOFError, KeyboardInterrupt):
                print("Grasp operation cancelled (no input)")
                return False
            
            print(f"Grasping body point: x={x}, y={y}, z={z}")
            if self.dummy_mode:
                print("DUMMY MODE: Grasp body point command simulated")
                return True
            return self.spot_interface.grasp_body_point(x, y, z, approach, lift)
            
        elif op == 'sleep':
            seconds = action.get('seconds', 1.0)
            # Safety: Limit sleep duration
            seconds = max(0.1, min(60.0, seconds))  # 0.1-60 seconds
            print(f"Sleeping for {seconds} seconds")
            if self.dummy_mode:
                print("DUMMY MODE: Sleep command simulated")
                time.sleep(0.1)  # Short sleep in dummy mode
            else:
                time.sleep(seconds)
            return True
            
        elif op == 'pick_apriltag':
            # Pick protection check
            command_name = "pick_apriltag"
            if self._is_pick_command_duplicate(command_name):
                print(f"WARNING: Skipping duplicate {command_name} command")
                return False
            
            if self._pick_executing:
                print("Pick operation already in progress, skipping...")
                return False
            
            # Set protection flags
            self._set_pick_protection_flags(command_name)
            
            try:
                tag_id = action.get('tag_id', None)
                z_offset = action.get('z_offset', 0.20)
                approach = action.get('approach', 0.12)
                lift = action.get('lift', 0.15)
                
                # Safety: Validate parameters
                z_offset = max(0.05, min(0.50, z_offset))  # 5-50cm above tag
                approach = max(0.05, min(0.30, approach))  # 5-30cm approach
                lift = max(0.05, min(0.50, lift))  # 5-50cm lift
                
                if self.dummy_mode:
                    print("DUMMY MODE: Pick Apriltag command simulated")
                    self._reset_pick_protection_flags()
                    return True
                
                # Get WorldObject client
                wo = self.spot_interface.world_object_client
                if wo is None:
                    print("WorldObject client not available")
                    return False
                    
                # Query for AprilTags
                resp = wo.list_world_objects(object_type=[world_object_pb2.WORLD_OBJECT_APRILTAG])
                
                if not resp.world_objects:
                    print("No Apriltags found")
                    return False
                
                # Get robot state for transforms
                snapshot = self.spot_interface.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
                vision_T_body = get_a_tform_b(snapshot, "vision", "body")
                
                # Select best tag
                best = None
                best_d2 = float("inf")
                for obj in resp.world_objects:
                    # Find fiducial frame in transforms_snapshot
                    ts = obj.transforms_snapshot
                    fid_name = None
                    for name in ts.child_to_parent_edge_map.keys():
                        if "fiducial" in name:
                            fid_name = name
                            break
                    if not fid_name:
                        continue
                    
                    vision_T_tag = get_a_tform_b(ts, "vision", fid_name)
                    this_id = obj.apriltag_properties.tag_id if obj.HasField("apriltag_properties") else None
                    
                    if tag_id is not None and this_id != tag_id:
                        continue
                
                    dx = vision_T_tag.x - vision_T_body.x
                    dy = vision_T_tag.y - vision_T_body.y
                    d2 = dx*dx + dy*dy
                    if d2 < best_d2:
                        best_d2 = d2
                        best = (vision_T_tag, this_id)
                
                if not best:
                    print("Requested Apriltag not found")
                    self._reset_pick_protection_flags()
                    return False
                
                tag_pose_vision, sel_id = best
                goal_vision = SE3Pose(tag_pose_vision.x, tag_pose_vision.y, tag_pose_vision.z + z_offset, Quat(1,0,0,0))
                
                # Transform vision to body
                body_T_vision = get_a_tform_b(snapshot, BODY_FRAME_NAME, "vision")
                body_goal = body_T_vision * goal_vision
                
                # Safety: Check scripted BODY grasp limits
                r = math.hypot(body_goal.x, body_goal.y)
                if not (0.30 <= r <= 0.85):
                    print(f"Apriltag target r={r:.2f} outside safe range [0.30, 0.85] m. Move base first.")
                    self._reset_pick_protection_flags()
                    return False

                if not (-0.45 <= body_goal.z <= 1.0):
                    print(f"Apriltag target z={body_goal.z:.2f} outside safe range [-0.45, 1.0] m")
                    self._reset_pick_protection_flags()
                    return False

                print(f"Picking Apriltag {sel_id} at BODY ({body_goal.x:.3f},{body_goal.y:.3f},{body_goal.z:.3f})")
                
                # Require user confirmation for AprilTag pick operations
                print(f"\n=== APRILTAG PICK CONFIRMATION ===")
                print(f"About to pick AprilTag {sel_id} at body coordinates: x={body_goal.x:.3f}, y={body_goal.y:.3f}, z={body_goal.z:.3f}")
                print(f"Approach distance: {approach:.3f}m, Lift distance: {lift:.3f}m")
                print("Press 'y' or 'yes' to confirm, any other key to cancel:")
                
                try:
                    user_input = input().strip().lower()
                    if user_input not in ['y', 'yes']:
                        print("AprilTag pick operation cancelled by user")
                        
                        # Offer to retry detection with a new image
                        print("\nWould you like to retry detection with a new image?")
                        print("Press 'y' or 'yes' to retry detection, any other key to cancel:")
                        
                        try:
                            retry_input = input().strip().lower()
                            if retry_input in ['y', 'yes']:
                                print("Retrying detection with new image...")
                                
                                # Get the original detection parameters from the last detection
                                if hasattr(self, 'last_detection') and self.last_detection:
                                    original_label = self.last_detection['label']
                                    
                                    # Retry detection with the same parameters
                                    retry_action = {
                                        'op': 'detect_label',
                                        'label': original_label,
                                        'provider': 'ultralytics',
                                        'conf': 0.01,  # Use low confidence for retry
                                        'image_source': 'hand_color_image'
                                    }
                                    
                                    print(f"Retrying detection for: {original_label}")
                                    retry_result = self._execute_action(retry_action)
                                    
                                    if retry_result:
                                        print("Retry detection completed. You can now try pick_apriltag again.")
                                        self._reset_pick_protection_flags()
                                        return False  # Return False to indicate pick was cancelled, but detection was retried
                                    else:
                                        print("Retry detection failed.")
                                        self._reset_pick_protection_flags()
                                        return False
                                else:
                                    print("No previous detection data available for retry.")
                                    self._reset_pick_protection_flags()
                                    return False
                            else:
                                print("Detection retry cancelled by user")
                                self._reset_pick_protection_flags()
                                return False
                        except (EOFError, KeyboardInterrupt):
                            print("Detection retry cancelled (no input)")
                            self._reset_pick_protection_flags()
                            return False
                            
                        self._reset_pick_protection_flags()
                        return False
                    print("AprilTag pick operation confirmed by user")
                except (EOFError, KeyboardInterrupt):
                    print("AprilTag pick operation cancelled (no input)")
                    self._reset_pick_protection_flags()
                    return False
                
                if self.dummy_mode:
                    print("DUMMY MODE: Pick Apriltag command simulated")
                    self._reset_pick_protection_flags()
                    return True
                
                # Execute scripted grasp
                result = self.spot_interface.grasp_body_point(body_goal.x, body_goal.y, body_goal.z, approach=approach, lift=lift)
                
                # If grasp was successful, automatically open gripper to release the object
                if result:
                    print("Grasp operation successful! Automatically opening gripper to release object...")
                    try:
                        gripper_result = self.spot_interface.gripper('open')
                        if gripper_result:
                            print("Gripper opened successfully - object released")
                        else:
                            print("Warning: Failed to open gripper after grasp")
                    except Exception as e:
                        print(f"Warning: Error opening gripper after grasp: {e}")
                else:
                    print("Grasp operation failed")
                
                self._reset_pick_protection_flags()
                return result
                
            except Exception as e:
                print(f"Apriltag pick error: {e}")
                self._reset_pick_protection_flags()
                return False

        else:
            raise ValueError(f"Unknown action: {op}")
    
    def _detect_label(self, label, provider, conf, image_source):
        """Detect a label in the image and store the result."""
        try:
            success = self.spot_interface.detect_label(label, provider, conf, image_source)
            if success:
                if hasattr(self.spot_interface, 'last_detection') and self.spot_interface.last_detection:
                    self.last_detection = self.spot_interface.last_detection
                    logging.info(f"Detection successful: {self.last_detection}")
                else:
                    self.last_detection = {
                        'label': label,
                        'confidence': conf,
                        'bbox': [100, 100, 200, 200],
                        'center': [150, 150]
                    }
                    logging.info(f"Detection successful (default): {self.last_detection}")
                return True
            else:
                logging.warning(f"No detection found for label: {label}")
                return False
        except Exception as e:
            logging.error(f"Detection failed: {e}")
        return False

    def _detect_label_with_list(self, label, provider, conf, image_source):
        """Detect a label and return both target and all detections."""
        try:
            # Call the enhanced detection method that returns all objects
            if hasattr(self.spot_interface, 'detect_label_with_list'):
                result = self.spot_interface.detect_label_with_list(label, provider, conf, image_source)
                if result:
                    return result
                else:
                    return None
            
            # Fallback to regular detection if enhanced method not available
            success = self.spot_interface.detect_label(label, provider, conf, image_source)
            if success and hasattr(self.spot_interface, 'last_detection') and self.spot_interface.last_detection:
                # Create a basic structure with the target detection
                target = self.spot_interface.last_detection
                all_detections = [target]  # Just the target for now
                return {
                    'target': target,
                    'all': all_detections
                }
            else:
                return None
                
        except Exception as e:
            logging.error(f"Enhanced detection failed: {e}")
            return None

    def get_last_detection(self):
        """Get the last successful detection."""
        return self.last_detection
    
    def get_all_detections(self):
        """Get all objects detected in the last detection run."""
        if hasattr(self, 'all_detections'):
            return self.all_detections
        return []
    
    def list_detected_objects(self):
        """List all currently detected objects."""
        if hasattr(self, 'all_detections') and self.all_detections:
            print(f"\n=== CURRENTLY DETECTED OBJECTS ({len(self.all_detections)}) ===")
            for i, det in enumerate(self.all_detections):
                print(f"  {i+1}. {det['label']} (confidence: {det['score']:.3f}) at pixel {det['center']}")
            if hasattr(self, 'last_detection') and self.last_detection:
                print(f"\nTarget object: {self.last_detection['label']} at pixel {self.last_detection['center']}")
        else:
            print("No objects currently detected. Run detect_label first.")
    
    def _check_robot_ready(self):
        """Check if robot is ready for actions."""
        if self.dummy_mode:
            return True
        
        try:
            if not self.spot_interface:
                logging.warning("Spot interface not available")
                return False

            robot_state = self.spot_interface.robot_state_client.get_robot_state()
            if not robot_state.power_state.motor_power_state == robot_state.power_state.STATE_ON:
                logging.warning("Robot motors not powered on")
                return False

            if robot_state.estop_states:
                for estop_state in robot_state.estop_states:
                    if estop_state.state == estop_state.STATE_ESTOPPED:
                        logging.warning("Robot is estopped")
                        return False

            return True
            
        except Exception as e:
            logging.error(f"Error checking robot readiness: {e}")
            return False

    def _check_action_preconditions(self, op):
        """Check if action is allowed in current state."""
        current_state = self.current_state
        
        always_allowed = ['stand', 'sit', 'sleep']
        if op in always_allowed:
            return True

        standing_required = [
            'move_relative', 'arm_ready', 'arm_stow', 'aim_hand', 
            'gripper', 'open_gripper', 'close_gripper', 'detect_label', 'list_objects', 'list_debug_images', 'pick_from_pixel', 'grasp_body_point', 'pick_apriltag'
        ]
        
        if op in standing_required:
            if current_state == self.sit:
                logging.warning(f"Action {op} requires standing, but robot is sitting")
                return False
        
        if op == 'arm_ready' and current_state != self.stand:
            logging.warning("Arm ready only allowed from stand state")
            return False

        if op == 'arm_stow' and current_state != self.stand:
            logging.warning("Arm stow only allowed from stand state")
            return False

        return True

    def _is_pick_command_duplicate(self, command_name):
        """Check if pick command is a duplicate within the time window."""
        current_time = time.time()
        return (self._last_pick_command == command_name and 
                current_time - self._pick_command_time < self.PICK_DUPLICATE_WINDOW)

    def _is_in_pick_state(self):
        """Check if robot is currently in a pick state."""
        return self._pick_executing

    def _set_pick_protection_flags(self, command_name):
        """Set protection flags for pick operation."""
        self._pick_executing = True
        self._last_pick_command = command_name
        self._pick_command_time = time.time()

    def _reset_pick_protection_flags(self):
        """Reset pick operation protection flags."""
        self._pick_executing = False
        self._last_pick_command = None
        self._pick_command_time = 0