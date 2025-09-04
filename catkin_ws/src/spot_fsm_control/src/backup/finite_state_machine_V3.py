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
        
        # Arm state tracking (separate from robot state)
        self._arm_state = 'stowed'  # 'stowed' or 'extended'
        self._last_arm_action = None
        
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
            'move_relative', 'arm_stow', 'aim_hand', 
            'detect_label', 'pick_from_pixel', 'grasp_body_point', 'pick_apriltag'
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
            print(f"Current arm state: {self._arm_state}")
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
            
            yaw = math.atan2(math.sin(yaw), math.cos(yaw))
            
            logging.info(f"Moving relative: x={x}, y={y}, yaw={yaw:.3f}")
            if self.dummy_mode:
                return True
            return self.spot_interface.move_relative(x, y, yaw, timeout)
            
        elif op == 'arm_stow':
            logging.info("Stowing arm")
            if self.dummy_mode:
                self._arm_state = 'stowed'
                self._last_arm_action = 'arm_stow'
                return True
            
            # Use the same direct approach that works during shutdown
            try:
                # Call arm_stow directly on the interface (bypassing FSM complexity)
                result = self.spot_interface.arm_stow()
                if result:
                    self._arm_state = 'stowed'
                    self._last_arm_action = 'arm_stow'
                    logging.info("Direct arm_stow succeeded (like during shutdown)")
                    return True
                else:
                    logging.warning("Direct arm_stow returned False - command failed")
            except Exception as e:
                logging.warning(f"Direct arm_stow failed with exception: {e}")
            
            # If direct stow fails, try moving to a safe position first
            logging.info("Direct arm_stow failed, trying safe intermediate position...")
            try:
                # Move arm to the known safe position before stowing
                safe_result = self.spot_interface.aim_hand(0.60, 0.0, 0.2)  # Safe position: x=0.60, y=0, z=0.2
                if safe_result:
                    time.sleep(1.0)  # Wait for arm to stabilize
                    # Now try to stow again using the same direct approach
                    result = self.spot_interface.arm_stow()
                    if result:
                        self._arm_state = 'stowed'
                        self._last_arm_action = 'arm_stow'
                        logging.info("arm_stow succeeded after moving to safe position")
                        return True
            except Exception as e:
                logging.warning(f"Safe intermediate position approach failed: {e}")
            
            # If all else fails, try using arm_ready as a fallback
            logging.info("All stow attempts failed, trying arm_ready as fallback...")
            try:
                result = self.spot_interface.arm_ready()
                if result:
                    # Consider this a partial success - arm is in a safe position
                    self._arm_state = 'extended'  # Not fully stowed, but safe
                    self._last_arm_action = 'arm_ready'
                    logging.warning("Arm stow failed, but arm is now in safe ready position")
                    return True
            except Exception as e:
                logging.error(f"All arm positioning attempts failed: {e}")
            
            return False
            
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
            if not (-0.2 <= z <= 0.3):
                logging.warning(f"Hand camera z={z} outside safe range [-0.2, 0.3] m")
                return False
            
            logging.info(f"Aiming hand at: x={x}, y={y}, z={z}")
            if self.dummy_mode:
                self._arm_state = 'extended'
                self._last_arm_action = 'aim_hand'
                return True
            result = self.spot_interface.aim_hand(x, y, z)
            if result:
                self._arm_state = 'extended'
                self._last_arm_action = 'aim_arm'
            return result
            

        elif op == 'detect_label':
            label = action.get('label', '')
            provider = action.get('provider', 'ultralytics')
            conf = action.get('conf', 0.01)
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
            
            # Use V1-style label detection and picking
            return self._pick_by_label_pixel(label, image_source, provider, conf, auto_walk=False)
            
        elif op == 'pick_from_pixel':
            # Pick protection check
            command_name = "pick_from_pixel"
            if self._is_pick_command_duplicate(command_name):
                print(f"WARNING: Skipping duplicate {command_name} command")
                return False
            
            if self._pick_executing:
                print("Pick operation already in progress, skipping...")
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
                
                if self.dummy_mode:
                    print("DUMMY MODE: Pick Apriltag command simulated")
                    return True
                
                # Execute scripted grasp
                result = self.spot_interface.grasp_body_point(body_goal.x, body_goal.y, body_goal.z, approach=approach, lift=lift)
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

    def get_last_detection(self):
        """Get the last successful detection."""
        return self.last_detection
    
    def get_arm_state(self):
        """Get current arm state."""
        return self._arm_state
    
    def is_arm_extended(self):
        """Check if arm is currently extended."""
        return self._arm_state == 'extended'
    
    def is_arm_stowed(self):
        """Check if arm is currently stowed."""
        return self._arm_state == 'stowed'
    
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
            'detect_label', 'pick_from_pixel', 'grasp_body_point', 'pick_apriltag'
        ]
        
        if op in standing_required:
            if current_state == self.sit:
                logging.warning(f"Action {op} requires standing, but robot is sitting")
                return False

        if op == 'arm_stow' and current_state != self.stand:
            logging.warning("Arm stow only allowed from stand state")
            return False

        return True
    
    def should_stow_arm_before_action(self, op):
        """Check if arm should be stowed before executing this action."""
        # Actions that require arm to be stowed for safety
        safety_required = ['move_relative', 'sit']
        
        # Only require stow if arm is currently extended
        if op in safety_required and self._arm_state == 'extended':
            return True
        
        return False
    
    def get_safety_warning(self, op):
        """Get safety warning if action requires arm stow."""
        if self.should_stow_arm_before_action(op):
            return f"WARNING: Arm is extended. Consider stowing arm before {op} for safety."
        return None

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

    def _pick_by_label_pixel(self, target_label="bottle", image_source="hand_color_image",
                            provider="ultralytics", conf=0.1, auto_walk=True, use_retry=True,
                            arm_only=True):
        """
        Detect <label> and grasp by pixel using Manipulation API PickObjectInImage.
        This is the only label-based picking method now.
        """
        if self._is_in_pick_state():
            print("Cannot pick by pixel. A pick is already in progress.")
            return False

        print(f"Detect and grasp-by-pixel: label={target_label}, source={image_source}, provider={provider}")

        try:
            # Use current hand position - no repositioning needed
            print(f"Using current hand position for detection")

            # Get image using V2 interface method
            bgr = self.spot_interface.get_bgr_image(image_source)
            if bgr is None:
                print("Failed to get image from camera")
                return False

            if provider == "ultralytics":
                det = self.spot_interface._detect_with_ultralytics(bgr, target_label=target_label, conf=conf, image_source=image_source)
            else:
                det = self.spot_interface._detect_with_opencv(bgr, target_label=target_label, conf=conf)

            if det is None:
                print(f"No '{target_label}' found for pixel grasp.")
                return False

            cx, cy, score, bbox = det
            print(f"Grasp-by-pixel for '{target_label}' at pixel ({cx}, {cy})")
            
            # Get depth at the detected pixel for logging
            try:
                depth_client = self.spot_interface.robot_sdk.ensure_client(ImageClient.default_service_name)
                depth_resp = depth_client.get_image([build_image_request("hand_depth_in_hand_color_frame")])[0]
                depth_img = depth_resp.shot.image
                if depth_img.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                    depth_data = np.frombuffer(depth_img.data, dtype=np.uint16).reshape(depth_img.rows, depth_img.cols).astype(np.float32) / 1000.0
                    # Scale pixel coordinates to depth image size
                    dw, dh = depth_img.cols, depth_img.rows
                    cw, ch = bgr.shape[1], bgr.shape[0]
                    dx = int(cx * dw / cw)
                    dy = int(cy * dh / ch)
                    dx = max(0, min(dx, dw - 1))
                    dy = max(0, min(dy, dh - 1))
                    depth_at_pixel = depth_data[dy, dx]
                    print(f"📍 Depth at pixel ({cx}, {cy}): {depth_at_pixel:.3f}m")
                else:
                    print(f"📍 Depth format not supported: {depth_img.pixel_format}")
            except Exception as e:
                print(f"📍 Could not get depth: {e}")
                depth_at_pixel = None

            # Check if detection confidence is strong enough
            min_conf_ok = max(0.10, conf)
            if score < min_conf_ok:
                print(f"Detection too weak for '{target_label}': {score:.3f} < {min_conf_ok}")
                return False

            # Prefer Manipulation API with the hand camera, with walking disabled
            use_hand = image_source.startswith("hand_")
            if use_hand:
                print("Hand camera detected: trying PickObjectInImage with walking disabled")
                
                # Show pick coordinates and ask for user confirmation
                print("\n" + "="*60)
                print("🎯 PICK COORDINATES CONFIRMATION")
                print("="*60)
                print(f"Target object: {target_label}")
                print(f"Pixel coordinates: ({cx}, {cy})")
                if depth_at_pixel is not None:
                    print(f"Detected depth: {depth_at_pixel:.3f}m")
                    print(f"PickObjectInImage will target: vision frame at depth {depth_at_pixel:.3f}m")
                else:
                    print(f"PickObjectInImage will target: vision frame coordinates")
                print(f"Method: PickObjectInImage (built-in API)")
                print(f"✅ Using same camera ({image_source}) for detection and PickObjectInImage")
                print("="*60)
                
                # Get user confirmation
                while True:
                    response = input("Do you want to proceed with this pick? (yes/no): ").strip().lower()
                    if response in ['yes', 'y']:
                        print("✅ User confirmed - proceeding with PickObjectInImage...")
                        break
                    elif response in ['no', 'n']:
                        print("❌ User cancelled - exiting program")
                        import sys
                        sys.exit(0)
                    else:
                        print("Please type 'yes' or 'no'")
                
                poi_ok = self.spot_interface.pick_from_pixel(
                    image_source=image_source,
                    x=cx, y=cy,
                    top_down=True,
                    auto_walk=False,      # keep base fixed
                    timeout=25.0
                )
                if poi_ok:
                    print(f"✅ PickObjectInImage succeeded for {target_label}")
                    print(f"📍 PickObjectInImage used coordinates: pixel=({cx}, {cy})")
                    if depth_at_pixel is not None:
                        print(f"📍 PickObjectInImage target depth: {depth_at_pixel:.3f}m (from depth camera)")
                    print(f"📍 PickObjectInImage target position: vision frame coordinates")
                    # After successful pick, lift the arm up to hold the object
                    self._lift_arm_after_successful_pick()
                    return True

                # Try a few nearby pixels inside the bbox
                if use_retry and bbox is not None:
                    print("Center pixel failed, scanning nearby pixels...")
                    
                    # Show retry coordinates and ask for user confirmation
                    print("\n" + "="*60)
                    print("🔄 RETRY PICK COORDINATES CONFIRMATION")
                    print("="*60)
                    print(f"Target object: {target_label}")
                    print(f"Original pixel: ({cx}, {cy})")
                    print(f"Will try multiple nearby pixels within bounding box")
                    if depth_at_pixel is not None:
                        print(f"Detected depth: {depth_at_pixel:.3f}m")
                    print(f"Method: PickObjectInImage retry (built-in API)")
                    print("="*60)
                    
                    # Get user confirmation
                    while True:
                        response = input("Do you want to proceed with retry attempts? (yes/no): ").strip().lower()
                        if response in ['yes', 'y']:
                            print("✅ User confirmed - proceeding with retry attempts...")
                            break
                        elif response in ['no', 'n']:
                            print("❌ User cancelled - exiting program")
                            import sys
                            sys.exit(0)
                        else:
                            print("Please type 'yes' or 'no'")
                    
                    if self._try_multiple_pick_points(image_source, bbox, target_label, auto_walk=False):
                        print(f"📍 Multiple pick points succeeded at pixel ({cx}, {cy})")
                        # After successful pick, lift the arm up to hold the object
                        self._lift_arm_after_successful_pick()
                        return True

            # Fall back to arm-only backprojection with -0.6m offset
            print("PickObjectInImage failed, falling back to backprojection with -0.6m Z offset")
            
            # Show backprojection coordinates and ask for user confirmation
            print("\n" + "="*60)
            print("🔄 BACKPROJECTION FALLBACK CONFIRMATION")
            print("="*60)
            print(f"Target object: {target_label}")
            print(f"Pixel coordinates: ({cx}, {cy})")
            if depth_at_pixel is not None:
                print(f"Detected depth: {depth_at_pixel:.3f}m")
            print(f"Method: Backprojection with -0.6m Z offset")
            print("="*60)
            
            # Get user confirmation
            while True:
                response = input("Do you want to proceed with backprojection? (yes/no): ").strip().lower()
                if response in ['yes', 'y']:
                    print("✅ User confirmed - proceeding with backprojection...")
                    break
                elif response in ['no', 'n']:
                    print("❌ User cancelled - exiting program")
                    import sys
                    sys.exit(0)
                else:
                    print("Please type 'yes' or 'no'")
            
            # Use backprojection with -0.6m Z offset
            backprojection_ok = self._backproject_and_script_grasp(
                color_source=image_source,
                pixel_xy=(cx, cy),
                z_offset=-0.6,
                approach=0.12,
                lift=0.15
            )
            
            if backprojection_ok:
                print(f"✅ Backprojection succeeded for {target_label}")
                print(f"📍 Backprojection used coordinates: pixel=({cx}, {cy})")
                if depth_at_pixel is not None:
                    print(f"📍 Backprojection target depth: {depth_at_pixel:.3f}m (from depth camera)")
                print(f"📍 Backprojection target position: -0.6m Z offset from hand")
                # After successful pick, lift the arm up to hold the object
                self._lift_arm_after_successful_pick()
                return True

            print(f"❌ All picking methods failed for {target_label}")
            return False

        except Exception as e:
            print(f"Error in pick_by_label_pixel: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _aim_hand_at_ground_ahead(self, distance=0.65, z=0.65, y=-0.10):
        """Aim hand at ground ahead for better object detection."""
        try:
            # Use the spot interface to aim hand
            return self.spot_interface.aim_hand(distance, y, z)
        except Exception as e:
            print(f"Error aiming hand at ground: {e}")
            return False

    def _lift_arm_after_successful_pick(self):
        """Lift arm after successful pick to hold object."""
        try:
            # Lift arm to a higher position
            return self.spot_interface.aim_hand(0.75, 0.0, 0.3)
        except Exception as e:
            print(f"Error lifting arm after pick: {e}")
            return False

    def _try_multiple_pick_points(self, image_source, bbox, target_label, auto_walk=False):
        """Try multiple pick points within the bounding box."""
        try:
            x1, y1, x2, y2 = bbox
            # Try center and corners of bounding box
            test_points = [
                (int((x1 + x2) / 2), int((y1 + y2) / 2)),  # center
                (x1 + 10, y1 + 10),  # top-left with offset
                (x2 - 10, y1 + 10),  # top-right with offset
                (x1 + 10, y2 - 10),  # bottom-left with offset
                (x2 - 10, y2 - 10),  # bottom-right with offset
            ]
            
            for i, (cx, cy) in enumerate(test_points):
                print(f"Trying pick point {i+1}: ({cx}, {cy})")
                try:
                    poi_ok = self.spot_interface.pick_from_pixel(
                        image_source=image_source,
                        x=cx, y=cy,
                        top_down=True,
                        auto_walk=auto_walk,
                        timeout=25.0
                    )
                    if poi_ok:
                        print(f"✅ Multiple pick points succeeded at ({cx}, {cy})")
                        return True
                except Exception as e:
                    print(f"Pick point {i+1} failed: {e}")
                    continue
            
            return False
        except Exception as e:
            print(f"Error in multiple pick points: {e}")
            return False

    def _backproject_and_script_grasp(self, color_source, pixel_xy, z_offset=0.0,
                                      approach=0.12, lift=0.15):
        """Arm-only grasp: use hand depth to back-project pixel to 3D and do a scripted grasp."""
        try:
            # For now, just return False to indicate fallback not implemented
            print("Backprojection fallback not implemented in V2")
            return False
        except Exception as e:
            print(f"Error in backprojection: {e}")
            return False