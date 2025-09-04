#!/usr/bin/env python

import time
import math
import json
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
        self.action_result = None  # Store action execution result
        self.last_detection = None  # Store last detection for pick_from_pixel
        
        # Command deduplication and pick protection
        self.PICK_DUPLICATE_WINDOW = 10.0  # seconds
        self._last_pick_command = None
        self._pick_command_time = 0
        self._pick_executing = False
        
        # Data logging
        self.data_logger = None
        
        if dummy_mode:
            print("FSM running in DUMMY MODE - no real robot commands will be sent")
        else:
            print("FSM running in REAL ROBOT MODE")
        
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
                print(f"⚠️  Skipping duplicate {op} command within time window")
                return False
        
        print(f"Enqueueing action: {action_dict}")
        self.pending_action = action_dict
        self.action_result = None  # Store the result of the action
        
        # Auto-stand if currently sitting and action requires standing
        standing_required = [
            'move_relative', 'arm_ready', 'arm_stow', 'aim_hand', 
            'gripper', 'detect_label', 'pick_from_pixel', 'grasp_body_point', 'pick_apriltag'
        ]
        
        if self.current_state == self.sit and op in standing_required:
            print("Auto-standing before action that requires standing")
            if not self.dummy_mode and self.spot_interface:
                self.spot_interface.stand(height=0.0)
            self.send('sit_to_stand')
        
        # Start action state
        if self.current_state == self.stand:
            self.send('stand_to_action')
        elif self.current_state == self.sit and op not in standing_required:
            # Only transition if action doesn't require standing
            self.send('sit_to_action')
        
        # Wait for action to complete and return result
        start_time = time.time()
        while self.pending_action and time.time() - start_time < 30.0:  # 30 second timeout
            time.sleep(0.1)
        
        # Return the stored result (True if action completed successfully)
        return self.action_result is True
    
    def on_enter_action(self):
        """Execute the pending action when entering action state."""
        if not self.pending_action:
            print("No pending action, returning to stand")
            self.send('action_to_stand')
            return
            
        action = self.pending_action
        op = action.get('op')
        
        print(f"Executing action: {op} with params: {action}")
        
        try:
            result = self._execute_action(action)
            self.action_result = result  # Store the result
            
            if result:
                print(f"Action {op} completed successfully")
            else:
                print(f"Action {op} failed")
            
            self.pending_action = None
            
            # Return to stand state (unless action was sit)
            if op == 'sit':
                self.send('action_to_sit')
            else:
                self.send('action_to_stand')
                
        except Exception as e:
            print(f"Action {op} failed: {e}")
            self.action_result = False  # Store failure result
            self.pending_action = None
            
            # Return to stand state on error (unless we were trying to sit)
            if op == 'sit':
                self.send('action_to_sit')
            else:
                self.send('action_to_stand')
    
    def _execute_action(self, action):
        """Execute a single action based on the operation type with safety checks."""
        op = action.get('op')
        
        # Check if robot is powered on and ready
        if not self._check_robot_ready():
            print("Robot not ready for action")
            return False
        
        # Check state-based preconditions
        if not self._check_action_preconditions(op):
            print(f"Action {op} not allowed in current state {self.current_state}")
            return False
        
        if op == 'stand':
            height = action.get('height', 0.0)
            # Safety: Limit height to reasonable range
            height = max(-0.2, min(0.5, height))  # -20cm to +50cm
            print(f"Standing with height: {height}")
            if self.dummy_mode:
                print("DUMMY MODE: Stand command simulated")
                return True
            return self.spot_interface.stand(height)
            
        elif op == 'sit':
            print("Sitting down")
            if self.dummy_mode:
                print("DUMMY MODE: Sit command simulated")
                return True
            return self.spot_interface.sit()
            
        elif op == 'move_relative':
            x = action.get('x', 0.0)
            y = action.get('y', 0.0)
            yaw = action.get('yaw', 0.0)
            timeout = action.get('timeout', 10.0)
            
            # Safety: Limit movement distances
            x = max(-5.0, min(5.0, x))  # Max 5m forward/backward
            y = max(-3.0, min(3.0, y))  # Max 3m left/right
            timeout = max(1.0, min(60.0, timeout))  # 1-60 seconds
            
            # Normalize yaw to [-pi, pi]
            yaw = math.atan2(math.sin(yaw), math.cos(yaw))
            
            print(f"Moving relative: x={x}, y={y}, yaw={yaw:.3f}")
            if self.dummy_mode:
                print("DUMMY MODE: Move relative command simulated")
                return True
            return self.spot_interface.move_relative(x, y, yaw, timeout)
            
        elif op == 'arm_ready':
            # Safety: Ensure robot is standing before arm operations
            if self.current_state == self.sit:
                print("Cannot move arm while sitting")
                return False
            print("Moving arm to ready position")
            if self.dummy_mode:
                print("DUMMY MODE: Arm ready command simulated")
                return True
            return self.spot_interface.arm_ready()
            
        elif op == 'arm_stow':
            print("Stowing arm")
            if self.dummy_mode:
                print("DUMMY MODE: Arm stow command simulated")
                return True
            return self.spot_interface.arm_stow()
            
        elif op == 'aim_hand':
            x = action.get('x', 0.0)
            y = action.get('y', 0.0)
            z = action.get('z', 0.0)
            
            # Safety: Check hand camera observation pose limits
            if not (0.60 <= x <= 0.90):
                print(f"Hand camera x={x} outside safe range [0.60, 0.90] m")
                return False
            if not (-0.5 <= y <= 0.5):
                print(f"Hand camera y={y} outside safe range [-0.5, 0.5] m")
                return False
            if not (-0.2 <= z <= 0.75):
                print(f"Hand camera z={z} outside safe range [-0.2, 0.75] m")
                return False
            
            print(f"Aiming hand at: x={x}, y={y}, z={z}")
            if self.dummy_mode:
                print("DUMMY MODE: Aim hand command simulated")
                return True
            return self.spot_interface.aim_hand(x, y, z)
            
        elif op == 'gripper':
            mode = action.get('mode', 'open')
            fraction = action.get('fraction', 0.5)
            
            # Safety: Validate gripper mode
            if mode not in ['open', 'close']:
                print(f"Invalid gripper mode: {mode}")
                return False
            
            # Clamp fraction between 0.0 and 1.0
            fraction = max(0.0, min(1.0, fraction))
            
            print(f"Gripper: mode={mode}, fraction={fraction}")
            if self.dummy_mode:
                print("DUMMY MODE: Gripper command simulated")
                return True
            return self.spot_interface.gripper(mode, fraction)
            
        elif op == 'detect_label':
            label = action.get('label', '')
            provider = action.get('provider', 'ultralytics')
            conf = action.get('conf', 0.25)
            image_source = action.get('image_source', 'hand_color_image')
            
            # Safety: Validate parameters
            if not label:
                print("No label specified for detection")
                return False
            if not (0.0 <= conf <= 1.0):
                print(f"Confidence {conf} outside valid range [0.0, 1.0]")
                return False
            
            print(f"Detecting label: {label} with {provider}")
            if self.dummy_mode:
                print("DUMMY MODE: Detect label command simulated")
                # Simulate detection result
                self.last_detection = {
                    'label': label,
                    'confidence': conf,
                    'bbox': [100, 100, 200, 200],
                    'center': [150, 150]
                }
                return True
            return self._detect_label(label, provider, conf, image_source)
            
        elif op == 'pick_from_pixel':
            # Pick protection check
            command_name = "pick_from_pixel"
        if self._is_pick_command_duplicate(command_name):
            print(f"⚠️  Skipping duplicate {command_name} command")
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
            print(f"⚠️  Skipping duplicate {command_name} command")
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
                # Get the detection from spot_interface if available
                if hasattr(self.spot_interface, 'last_detection') and self.spot_interface.last_detection:
                    self.last_detection = self.spot_interface.last_detection
                    print(f"Detection successful: {self.last_detection}")
                else:
                    # Create a default detection object
                    self.last_detection = {
                        'label': label,
                        'confidence': conf,
                        'bbox': [100, 100, 200, 200],
                        'center': [150, 150]
                    }
                    print(f"Detection successful (default): {self.last_detection}")
                return True
            else:
                print(f"No detection found for label: {label}")
                return False
        except Exception as e:
            print(f"Detection failed: {e}")
        return False

    def get_last_detection(self):
        """Get the last successful detection."""
        return self.last_detection
    
    def _check_robot_ready(self):
        """Check if robot is ready for actions."""
        if self.dummy_mode:
            print("DUMMY MODE: Robot ready check skipped")
                return True
        
        try:
            # Check if spot interface is available
            if not self.spot_interface:
                print("Spot interface not available")
        return False

            # Check if robot is powered on
            robot_state = self.spot_interface.robot_state_client.get_robot_state()
            if not robot_state.power_state.motor_power_state == robot_state.power_state.STATE_ON:
                print("Robot motors not powered on")
            return False

            # Check if robot is not estopped
            if robot_state.estop_states:
                for estop_state in robot_state.estop_states:
                    if estop_state.state == estop_state.STATE_ESTOPPED:
                        print("Robot is estopped")
                return False

            return True
            
            except Exception as e:
            print(f"Error checking robot readiness: {e}")
                return False

    def _check_action_preconditions(self, op):
        """Check if action is allowed in current state."""
        current_state = self.current_state
        
        # Actions allowed from any state
        always_allowed = ['stand', 'sit', 'sleep']
        if op in always_allowed:
                    return True

        # Actions that require standing (not sitting)
        standing_required = [
            'move_relative', 'arm_ready', 'arm_stow', 'aim_hand', 
            'gripper', 'detect_label', 'pick_from_pixel', 'grasp_body_point', 'pick_apriltag'
        ]
        
        if op in standing_required:
            if current_state == self.sit:
                print(f"Action {op} requires standing, but robot is sitting")
                return False
        
        # Specific state requirements
        if op == 'arm_ready' and current_state != self.stand:
            print("Arm ready only allowed from stand state")
            return False

        if op == 'arm_stow' and current_state != self.stand:
            print("Arm stow only allowed from stand state")
            return False

        return True

    def _is_pick_command_duplicate(self, command_name):
        """Check if pick command is a duplicate within the time window."""
        current_time = time.time()
        return (self._last_pick_command == command_name and 
                current_time - self._pick_command_time < self.PICK_DUPLICATE_WINDOW)

    def _is_in_pick_state(self):
        """Check if robot is currently in a pick state."""
        # In V2, all pick operations go through the 'action' state
        # We check if we're currently executing a pick operation
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