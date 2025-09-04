#!/usr/bin/env python

import time
import math
import cv2
from statemachine import StateMachine, State
import numpy as np
from spot_hololens_llm_interface.arm_impedance_control_helpers import get_root_T_ground_body
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b, BODY_FRAME_NAME
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import geometry_pb2, manipulation_api_pb2, world_object_pb2, image_pb2
from bosdyn.util import seconds_to_duration

class SpotStateMachine(StateMachine):
    """
    A state machine for Spot to control all the inputs from the NUI to a command to the robot
    The state machine is used to increase the safety of operation and prevent initiation of states that cannot match with other states.
    """
    # Pick operation constants
    PICK_DUPLICATE_WINDOW = 10.0  # seconds
    PICK_X_DISTANCE = 0.45        # meters forward
    PICK_Z_HEIGHT = 0.20          # meters above body origin
    PICK_Y_POSITIONS = {
        "green": 0.0,    # Center
        "red": -0.5,     # Left (negative Y) - moved further left
        "blue": 0.5      # Right (positive Y) - moved further right
    }
    
    # Hand camera safety constants (based on successful QR pick position)
    HAND_SAFE_X_MIN = 0.70   # meters ahead of BODY origin (like QR pick: 0.743)
    HAND_SAFE_Z_MIN = 0.60   # meters above BODY origin (20cm higher for stability)
    HAND_PEEK_Y = -0.10      # slight left offset (like QR pick: -0.100)
    
    # QR code picking constants
    QR_SPHERE_OFFSET_Z = 1      # 50 cm boven de tag
    QR_SPHERE_RADIUS   = 0.06      # 6 cm visualisatie
    QR_SPHERE_RGBA     = [255, 165, 0, 255]  # oranje (RGBA values 0-255)
    

    
    sit = State(initial=True)

    # Stand
    stand = State()
    stand_high = State()
    stand_low = State()
    sit_down = ( stand.to(sit) | stand_high.to(sit) | stand_low.to(sit) )
    stand_up = ( sit.to(stand) | stand_high.to(stand) | stand_low.to(stand) )
    stand_up_high = (stand.to(stand_high) | sit.to(stand_high)|stand_low.to(stand_high))
    stand_up_low = (stand.to(stand_low) | sit.to(stand_low) | stand_high.to(stand_low))

    walk_forward = State()
    walk_backward = State()
    walk_left = State()
    walk_right = State()

    walk_to_forward = stand.to(walk_forward)
    walk_to_backward = stand.to(walk_backward)
    walk_to_left = stand.to(walk_left)
    walk_to_right = stand.to(walk_right)

    deitic_location_movement = State()
    move_to_location = stand.to(deitic_location_movement)

    face_operator = State()
    face_to_operator = stand.to(face_operator)
    
    pick_object = State()
    pick_up_object = stand.to(pick_object)
    
    pick_green = State()
    pick_red = State()
    pick_blue = State()
    pick_up_green = stand.to(pick_green)
    pick_up_red = stand.to(pick_red)
    pick_up_blue = stand.to(pick_blue)
    
    # Nieuw: QR pick state
    pick_qr = State()
    pick_up_qr = stand.to(pick_qr)
    
    # Nieuw: Label-based pick state
    pick_label = State()
    pick_up_label = stand.to(pick_label)
    


    turn_left = State()
    turn_right = State()
    turn_to_left = stand.to(turn_left)
    turn_to_right = stand.to(turn_right)
    
    gaze_control = State()
    arm_trajectory = State()
    start_gaze = stand.to(gaze_control)
    start_trajectory = stand.to(arm_trajectory)
    
    direct_arm_control = State()
    start_direct_arm_control = stand.to(direct_arm_control)
    
    stop_action = (
        walk_forward.to(stand) |
        walk_backward.to(stand) |
        walk_left.to(stand) |
        walk_right.to(stand) |
        turn_right.to(stand) |
        turn_left.to(stand) |
        deitic_location_movement.to(stand) |
        face_operator.to(stand) |
        pick_object.to(stand) |
        pick_green.to(stand) |
        pick_red.to(stand) |
        pick_blue.to(stand) |
        pick_qr.to(stand) |
        pick_label.to(stand) |
        arm_trajectory.to(stand) |
        gaze_control.to(stand) |
        direct_arm_control.to(stand)
    )
    
    turn_off = State(final=True)
    turn_off_robot = sit.to(turn_off)

    def __init__(self, robot):
        self.robot = robot
        super().__init__()
        self.robot_speed = 0.3
        self.movement_duration = 2
        
        # Pick operation protection flags
        self._pick_executing = False
        self._last_pick_command = None
        self._pick_command_time = 0
        
    def after_stop_action(self):
        """Stop all robot actions and reset pick operation flags."""
        self.robot.stop()
        self.robot.ready_or_stow_arm(stow=True)
        
        # Reset pick operation protection flags
        self._pick_executing = False
        self._last_pick_command = None
        self._pick_command_time = 0
        
        print("Action stopped.")
        
    def on_enter_walk_forward(self):
        self.robot.forward = self.robot_speed
        self.robot.two_d_location_body_frame_command(1.5, 0, 0)
        print("move forward")

    def on_enter_walk_backward(self):
        self.robot.forward = -1 * self.robot_speed
        self.robot.two_d_location_body_frame_command(-1.5, 0, 0)
        print("move backward")

    def on_enter_turn_left(self):
        self.robot.rotate = self.robot_speed
        self.robot.two_d_location_body_frame_command(0, 0, math.pi/2)
        print("Rotate left")

    def on_enter_turn_right(self):
        self.robot.rotate = -1 * self.robot_speed
        self.robot.two_d_location_body_frame_command(0, 0, -math.pi/2)
        print("Rotate right")

    def on_enter_walk_left(self):
        self.robot.strafe = self.robot_speed
        self.robot.two_d_location_body_frame_command(0, 0.75, 0)
        print("Move left")

    def on_enter_walk_right(self):
        self.robot.strafe = -1 * self.robot_speed
        self.robot.two_d_location_body_frame_command(0, -0.75, 0)
        print("Move right")

    def on_exit_arm_trajectory(self):
        self.robot.stand(0.0)

    def on_enter_stand(self):
        if self.robot:
            self.robot.stand(0.0)
            print(f"Standing")
    
    def on_enter_stand_high(self):
        self.robot.stand(0.1)
        print(f"Standing high")

    def on_enter_stand_low(self):
        self.robot.stand(-0.1)
        print(f"Standing low")

    def on_enter_sit(self):
        if self.robot:
            self.robot.sit_down()
            print(f"Sit down.")

    def on_enter_gaze_control(self):
        print("Gaze Control.")
        self.robot.gaze_control()

    def on_enter_arm_trajectory(self):
        self.robot.arm_trajectory()

    def on_enter_direct_arm_control(self):
        self.robot.ready_or_stow_arm()
        
        task_T_tool_desired = SE3Pose(0.75, 0, 0.45, Quat(1, 0, 0, 0))
        odom_T_task = get_root_T_ground_body(robot_state=self.robot.robot_state_client.get_robot_state(),
                                             root_frame_name=GRAV_ALIGNED_BODY_FRAME_NAME)
        wr1_T_tool = SE3Pose(0, 0, 0, Quat.from_pitch(-math.pi / 2))
        
        self.robot.move_to_cartesian_pose_rt_task(task_T_tool_desired, odom_T_task, wr1_T_tool)
        
        time.sleep(0.5)  # Reduced delay (was 1)
    
        self.robot.init_pos_empty = True
        self.robot.current_state_direct_control = True
        
    def on_exit_direct_arm_control(self):
        print("Exiting direct control")
        self.robot.current_state_direct_control = False
        time.sleep(0.5)  # Reduced delay (was 1)
        self.robot.stand(0.0)
        time.sleep(1)  # Reduced delay (was 2)
        self.robot.ready_or_stow_arm(stow=True)
        time.sleep(0.5)  # Reduced delay (was 1)

    def on_enter_turn_off(self):
        self.robot.stand(0.0)
        self.robot.sit_down()

    def _is_pick_command_duplicate(self, command_name):
        """Check if pick command is a duplicate within the time window."""
        current_time = time.time()
        return (self._last_pick_command == command_name and 
                current_time - self._pick_command_time < self.PICK_DUPLICATE_WINDOW)

    def _is_in_pick_state(self):
        """Check if robot is currently in a pick state."""
        return self.current_state in ['pick_green', 'pick_red', 'pick_blue', 'pick_object', 'pick_qr', 'pick_label']

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

    def on_enter_pick_object(self):
        """
        Execute pick object action at a predefined position.
        
        The robot arm will grasp an object at a fixed position:
        - x: 0.45m forward (within arm reach)
        - y: 0.0m (center line)
        - z: 0.20m above body origin (for clearance)
        """
        command_name = "pick_up_object"
        
        # Prevent duplicate command execution
        if self._is_pick_command_duplicate(command_name):
            print(f"⚠️  Skipping duplicate {command_name} command")
            return
        
        # Check if already in a pick state
        if self._is_in_pick_state():
            print(f"{command_name} not possible - already in {self.current_state} state")
            return
            
        # Prevent concurrent pick operations
        if self._pick_executing:
            print("Pick operation already in progress, skipping...")
            return
        
        # Set protection flags
        self._set_pick_protection_flags(command_name)
        
        try:
            # Target position in body frame
            x = self.PICK_X_DISTANCE   # meters forward
            y = 0.0                    # center line
            z = self.PICK_Z_HEIGHT     # meters above body origin

            print(f"Executing pick object action")
            print(f"Target position: ({x:.2f}, {y:.2f}, {z:.2f}) "
                  f"(x={x:.2f}m forward, y={y:.2f}m left/right, z={z:.2f}m up)")

            # Execute scripted grasp instead of automatic pick
            self._scripted_grasp_body_point(x, y, z, approach=0.12, lift=0.15)

        except Exception as e:
            print(f"Pick object failed: {e}")

    def on_enter_pick_green(self):
        """Execute pick green sphere action when robot is standing."""
        self._handle_colored_pick("green", (0, 200, 0), "green_sphere")

    def on_enter_pick_red(self):
        """Execute pick red sphere action when robot is standing."""
        self._handle_colored_pick("red", (200, 0, 0), "red_sphere")

    def on_enter_pick_blue(self):
        """Execute pick blue sphere action when robot is standing."""
        self._handle_colored_pick("blue", (0, 0, 200), "blue_sphere")

    def on_enter_pick_qr(self):
        """
        Zoek een AprilTag in de omgeving, teken een oranje bol 20 cm erboven in 'vision',
        transformeer naar je task-frame en voer een scripted grasp uit op die plek.
        """
        command_name = "pick_up_qr"

        # Duplicate guard
        if self._is_pick_command_duplicate(command_name):
            print("⚠️  Skipping duplicate pick_up_qr command")
            return

        if self._is_in_pick_state():
            print(f"{command_name} not possible - already in {self.current_state} state")
            return

        if self._pick_executing:
            print("Pick operation already in progress, skipping...")
            return

        self._set_pick_protection_flags(command_name)

        try:
            print("Searching for AprilTags in World Objects...")
            
            # 1) Vind dichtstbijzijnde AprilTag en reken doelpunt uit in 'vision'
            try:
                tag_pose_vision = self._find_nearest_apriltag_pose_in_vision()
                print("AprilTag detection completed successfully")
            except Exception as e:
                print(f"ERROR in AprilTag detection: {e}")
                self._reset_pick_protection_flags()
                return
                
            if tag_pose_vision is None:
                print("ERROR: No AprilTags detected in World Objects!")
                print("   - Make sure AprilTags are visible to the robot's cameras")
                print("   - Check that AprilTag detection is enabled in the World Object service")
                print("   - Verify the tags are within the robot's field of view")
                self._reset_pick_protection_flags()
                return

            print(f"AprilTag found at vision coordinates: ({tag_pose_vision.x:.3f}, {tag_pose_vision.y:.3f}, {tag_pose_vision.z:.3f})")

            try:
                # Calculate target height: QR_SPHERE_OFFSET_Z above tag, but minimum PICK_Z_HEIGHT above body origin (like colored spheres)
                target_z = max(tag_pose_vision.z + self.QR_SPHERE_OFFSET_Z, self.PICK_Z_HEIGHT)
                
                goal_vision = SE3Pose(
                    tag_pose_vision.x,
                    tag_pose_vision.y,
                    target_z,
                    Quat(1, 0, 0, 0)
                )
                print("Goal pose calculation completed successfully")
            except Exception as e:
                print(f"ERROR in goal pose calculation: {e}")
                self._reset_pick_protection_flags()
                return

            height_offset = target_z - tag_pose_vision.z
            print(f"Target pick position: ({goal_vision.x:.3f}, {goal_vision.y:.3f}, {goal_vision.z:.3f})")
            print(f"   - Tag at Z: {tag_pose_vision.z:.3f}m")
            print(f"   - Pick at Z: {target_z:.3f}m (offset: {height_offset:.3f}m)")
            if target_z == self.PICK_Z_HEIGHT:
                print(f"   - Using minimum height constraint ({self.PICK_Z_HEIGHT}m above body origin)")
            else:
                print(f"   - Using {self.QR_SPHERE_OFFSET_Z*100:.0f}cm above tag position")

            print("Skipping visualization sphere for debugging")

            # 3) Transformeer naar body-frame en grijp (zoals de andere pick functies)
            try:
                snapshot = self.robot.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
                body_T_vision = get_a_tform_b(snapshot, BODY_FRAME_NAME, "vision")  # body<-vision
                body_T_goal   = body_T_vision * goal_vision
                print("Frame transformation completed successfully")
            except Exception as e:
                print(f"ERROR in frame transformation: {e}")
                self._reset_pick_protection_flags()
                return

            print(f"Executing scripted grasp at body coordinates: ({body_T_goal.x:.3f}, {body_T_goal.y:.3f}, {body_T_goal.z:.3f})")
            self._scripted_grasp_body_point(body_T_goal.x, body_T_goal.y, body_T_goal.z,
                                            approach=0.12, lift=0.15)

        except Exception as e:
            print(f"pick_qr failed: {e}")
            import traceback
            traceback.print_exc()
            self._reset_pick_protection_flags()

    def on_enter_pick_label(self):
        """Execute label-based pick action when robot is standing."""
        command_name = "pick_up_label"
        
        # Prevent duplicate command execution
        if self._is_pick_command_duplicate(command_name):
            print(f"⚠️  Skipping duplicate {command_name} command")
            return
        
        # Check if already in a pick state
        if self._is_in_pick_state():
            print(f"{command_name} not possible - already in {self.current_state} state")
            return
            
        # Prevent concurrent pick operations
        if self._pick_executing:
            print("Pick operation already in progress, skipping...")
            return
        
        # Set protection flags
        self._set_pick_protection_flags(command_name)
        
        try:
            # Get target label from robot instance or use default
            target_label = getattr(self.robot, 'target_label', 'bottle')
            image_source = getattr(self.robot, 'image_source', 'frontleft_fisheye_image')
            provider = getattr(self.robot, 'detection_provider', 'ultralytics')
            conf = getattr(self.robot, 'detection_confidence', 0.1)
            
            print(f"Executing pixel-based pick for '{target_label}'")
            print(f"Using {provider} detector with confidence {conf}")
            
            image_source = 'hand_color_image'  # Use hand camera for detection to match depth
            allow_base = getattr(self.robot, 'allow_base_motion_during_pick', False)
            success = self.pick_by_label_pixel(
                target_label=target_label,
                image_source=image_source,          # hand camera
                provider=provider,
                conf=conf,
                auto_walk=allow_base,               # only allow base motion if you really want it
                use_retry=True,
                arm_only=not allow_base             # default to arm-only
            )
            
            if success:
                print(f"✅ Successfully picked {target_label}")
            else:
                print(f"❌ Failed to pick {target_label}")
                
        except Exception as e:
            print(f"Pixel-based pick failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._reset_pick_protection_flags()



    def _handle_colored_pick(self, color_name, rgb_color, sphere_name):
        """
        Common handler for colored pick operations with protection logic.
        
        Args:
            color_name: Color of the sphere ('green', 'red', 'blue')
            rgb_color: RGB color tuple for visualization
            sphere_name: Name identifier for the sphere
        """
        command_name = f"pick_up_{color_name}"
        
        # Prevent duplicate command execution
        if self._is_pick_command_duplicate(command_name):
            print(f"⚠️  Skipping duplicate {command_name} command")
            return
        
        # Check if already in a pick state
        if self._is_in_pick_state():
            print(f"{command_name} not possible - already in {self.current_state} state")
            return
            
        # Prevent concurrent pick operations
        if self._pick_executing:
            print("Pick operation already in progress, skipping...")
            return
            
        # Set protection flags
        self._set_pick_protection_flags(command_name)
        self._execute_colored_pick(color_name, rgb_color, sphere_name)

    def _scripted_grasp_body_point(self, x, y, z, approach=0.12, lift=0.15):
        """
        Execute a scripted grasp sequence with explicit gripper control.
        
        Args:
            x, y, z: Target position in body frame (meters)
            approach: Distance above target for pre-grasp position (meters)
            lift: Distance to lift after grasp (meters)
        """
        # Deploy arm and orient tool downward
        self.robot.ready_or_stow_arm(stow=False)

        # Get task frame transformation
        odom_T_task = get_root_T_ground_body(
            robot_state=self.robot.robot_state_client.get_robot_state(),
            root_frame_name=GRAV_ALIGNED_BODY_FRAME_NAME
        )
        
        # Orient WR1-tool so gripper points downward
        wr1_T_tool = SE3Pose(0, 0, 0, Quat.from_pitch(-math.pi/2))

        print(f"Executing scripted grasp at ({x:.2f}, {y:.2f}, {z:.2f})")
        print(f"🎯 SCRIPTED GRASP COORDINATES:")
        print(f"   Target position: X={x:.3f}m, Y={y:.3f}m, Z={z:.3f}m")
        print(f"   Pre-grasp: X={x:.3f}m, Y={y:.3f}m, Z={z + approach:.3f}m")
        print(f"   Grasp point: X={x:.3f}m, Y={y:.3f}m, Z={z:.3f}m")
        print(f"   Post-lift: X={x:.3f}m, Y={y:.3f}m, Z={z + lift:.3f}m")
        print(f"   Gripper strength: 50% (gentle grip)")
        
        # 1) Pre-grasp position - approach above target
        pre = SE3Pose(x, y, z + approach, Quat(1, 0, 0, 0))
        print(f"1. Moving to pre-grasp position: ({x:.2f}, {y:.2f}, {z + approach:.2f})")
        self.robot.move_to_cartesian_pose_rt_task(pre, odom_T_task, wr1_T_tool)
        time.sleep(0.2)  # Reduced delay (was 0.3)

        # 2) Open gripper
        print("2. Opening gripper")
        if not self.robot.gripper("open"):
            print("❌ Failed to open gripper")
            return False
        time.sleep(0.5)  # Delay to ensure gripper opens fully

        # 3) Descend to grasp point
        down = SE3Pose(x, y, z, Quat(1, 0, 0, 0))
        print(f"3. Descending to grasp point: ({x:.2f}, {y:.2f}, {z:.2f})")
        self.robot.move_to_cartesian_pose_rt_task(down, odom_T_task, wr1_T_tool)
        time.sleep(0.5)  # Delay to ensure arm reaches position

        # 4) Close gripper to 50% (not fully closed for gentler grip)
        print("4. Closing gripper to 50% strength")
        if not self.robot.gripper("close"):
            print("❌ Failed to close gripper")
            return False
        time.sleep(1.0)  # Longer delay to ensure gripper closes fully

        # 5) Lift with grasped object
        up = SE3Pose(x, y, z + lift, Quat(1, 0, 0, 0))
        print(f"5. Lifting to: ({x:.2f}, {y:.2f}, {z + lift:.2f})")
        self.robot.move_to_cartesian_pose_rt_task(up, odom_T_task, wr1_T_tool)
        time.sleep(0.3)  # Delay for lift movement
        
        print("✅ Scripted grasp sequence completed successfully")
        return True

    def _lift_arm_after_successful_pick(self):
        """Lift the arm up after a successful PickObjectInImage to hold the object."""
        try:
            print("🔄 Lifting arm to regular stand position...")
            
            # Get task frame transformation
            odom_T_task = get_root_T_ground_body(
                robot_state=self.robot.robot_state_client.get_robot_state(),
                root_frame_name=GRAV_ALIGNED_BODY_FRAME_NAME
            )
            
            # Orient WR1-tool so gripper points downward
            wr1_T_tool = SE3Pose(0, 0, 0, Quat.from_pitch(-math.pi/2))
            
            # Move to retracted position above the body (like during startup)
            retract_pose = SE3Pose(0.6, 0.0, 0.6, Quat(1, 0, 0, 0))
            
            print(f"Moving to retracted position above body: (0.60, 0.00, 0.60)")
            self.robot.move_to_cartesian_pose_rt_task(retract_pose, odom_T_task, wr1_T_tool)
            time.sleep(1.0)  # Wait for arm to reach retracted position
            
            print("✅ Arm retracted above body")
            
            # Return to standing state after successful pick
            print("🔄 Returning to standing state with arm retracted above body...")
            self.send("stop_action")
            
        except Exception as e:
            print(f"❌ Failed to lift arm after pick: {e}")

    def _execute_colored_pick(self, color_name, rgb_color, sphere_name):
        """
        Execute pick action for colored spheres at different positions.
        
        Args:
            color_name: Color of the sphere ('green', 'red', 'blue')
            rgb_color: RGB color tuple for visualization
            sphere_name: Name identifier for the sphere
        """
        # Prevent concurrent pick operations
        if self._pick_executing:
            print(f"Pick operation already in progress, skipping {color_name} sphere...")
            return
        
        self._pick_executing = True
        
        try:
            # Calculate target position in body frame
            x = self.PICK_X_DISTANCE
            y = self.PICK_Y_POSITIONS[color_name]
            z = self.PICK_Z_HEIGHT
            
            print(f"Executing pick {color_name} sphere action")
            print(f"Target position: ({x:.2f}, {y:.2f}, {z:.2f}) "
                  f"(x={x:.2f}m forward, y={y:.2f}m left/right, z={z:.2f}m up)")

            # Execute scripted grasp instead of automatic pick
            self._scripted_grasp_body_point(x, y, z, approach=0.12, lift=0.15)
            
        except Exception as e:
            print(f"Pick {color_name} failed: {e}")

    def _find_nearest_apriltag_pose_in_vision(self):
        """
        Geef de pose van de dichtstbijzijnde AprilTag terug als SE3Pose in 'vision'.
        Return None als er geen tags zijn.
        """
        try:
            # WorldObject client
            print("Creating WorldObject client...")
            wo_client = self.robot.robot_sdk.ensure_client(WorldObjectClient.default_service_name)
            print("WorldObject client created successfully")
            
            print("Listing AprilTag world objects...")
            resp = wo_client.list_world_objects(object_type=[world_object_pb2.WORLD_OBJECT_APRILTAG])
            print("World object list request completed")
            
            print(f"Found {len(resp.world_objects)} AprilTag(s) in World Objects")
            
            if not resp.world_objects:
                print("   - No AprilTags detected in the last ~15 seconds")
                return None

            # Huidige transforms
            print("Getting robot state...")
            robot_state = self.robot.robot_state_client.get_robot_state()
            snapshot = robot_state.kinematic_state.transforms_snapshot
            print("Robot state obtained successfully")

            # Body in vision om afstand te bepalen
            print("Calculating vision to body transform...")
            vision_T_body = get_a_tform_b(snapshot, "vision", "body")
            print("Vision to body transform calculated")

            best_pose_vision = None
            best_dist2 = float("inf")
            best_tag_id = None

            print(f"Processing {len(resp.world_objects)} AprilTag objects...")
            for i, wo in enumerate(resp.world_objects):
                print(f"Processing tag {i+1}/{len(resp.world_objects)}...")
                
                # Debug: Print protobuf fields
                print(f"   World object type: {type(wo)}")
                print(f"   Protobuf fields: {wo.ListFields()}")
                
                # Check if transforms_snapshot field exists (this is the correct field for AprilTags)
                if wo.HasField('transforms_snapshot'):
                    transforms_snapshot = wo.transforms_snapshot
                    print(f"   Found transforms_snapshot with {len(transforms_snapshot.child_to_parent_edge_map)} transforms")
                    
                    # Look for the fiducial frame in the transforms snapshot
                    fiducial_frame = None
                    for frame_name, transform in transforms_snapshot.child_to_parent_edge_map.items():
                        if 'fiducial' in frame_name:
                            fiducial_frame = frame_name
                            print(f"   Found fiducial frame: {fiducial_frame}")
                            break
                    
                    if fiducial_frame is None:
                        print(f"   ERROR: No fiducial frame found in transforms snapshot")
                        print(f"   Available frames: {list(transforms_snapshot.child_to_parent_edge_map.keys())}")
                        continue
                    
                    # Get the transform from vision to the fiducial frame
                    try:
                        vision_T_fiducial = get_a_tform_b(transforms_snapshot, "vision", fiducial_frame)
                        print(f"   Vision_T_fiducial: ({vision_T_fiducial.x:.3f}, {vision_T_fiducial.y:.3f}, {vision_T_fiducial.z:.3f})")
                        vision_T_tag = vision_T_fiducial
                    except Exception as e:
                        print(f"   ERROR getting vision_T_fiducial transform: {e}")
                        continue
                    
                else:
                    print(f"   ERROR: No 'transforms_snapshot' field found on world object")
                    print(f"   Available fields: {[field.name for field in wo.DESCRIPTOR.fields]}")
                    continue
                
                # Get tag ID if available
                tag_id = "unknown"
                if wo.HasField('apriltag_properties'):
                    tag_id = wo.apriltag_properties.tag_id
                print(f"   Tag ID: {tag_id}")

                # Calculate distance from robot to tag
                dx = vision_T_tag.x - vision_T_body.x
                dy = vision_T_tag.y - vision_T_body.y
                dist2 = dx*dx + dy*dy
                distance = math.sqrt(dist2)
                
                print(f"   - Tag {tag_id}: distance {distance:.2f}m, position ({vision_T_tag.x:.2f}, {vision_T_tag.y:.2f}, {vision_T_tag.z:.2f})")
                
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_pose_vision = vision_T_tag
                    best_tag_id = tag_id

            if best_pose_vision:
                best_distance = math.sqrt(best_dist2)
                print(f"Selected nearest tag {best_tag_id} at distance {best_distance:.2f}m")

            return best_pose_vision
            
        except Exception as e:
            print(f"ERROR in _find_nearest_apriltag_pose_in_vision: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _draw_orange_sphere_vision(self, name, pose_vision: SE3Pose, radius=0.06, rgba=None):
        """
        Teken een bol in de World Object service op een bepaalde pose in 'vision'.
        Alleen voor visualisatie/debugging.
        """
        if rgba is None:
            rgba = self.QR_SPHERE_RGBA
            
        wo_client = self.robot.robot_sdk.ensure_client(WorldObjectClient.default_service_name)
        # Handige helper uit de SDK
        # draw_sphere(name, x_rt_frame_name, y_rt_frame_name, z_rt_frame_name, frame_name, radius=..., rgba=[R,G,B,A])
        wo_client.draw_sphere(
            name=name,
            x_rt_frame_name=pose_vision.x,
            y_rt_frame_name=pose_vision.y,
            z_rt_frame_name=pose_vision.z,
            frame_name="vision",
            radius=radius,
            rgba=rgba,
            list_objects_now=True
        )

    # ============================================================================
    # LABEL-BASED PICKING METHODS
    # ============================================================================

    def _bd_image_to_cv2(self, img_resp):
        """Convert Boston Dynamics image response to OpenCV format."""
        img = img_resp.shot.image
        
        # Handle JPEG format
        if img.format == image_pb2.Image.FORMAT_JPEG:
            data = np.frombuffer(img.data, dtype=np.uint8)
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return bgr
            
        # Handle RAW format with different pixel formats
        h, w = img.rows, img.cols
        if img.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            arr = np.frombuffer(img.data, dtype=np.uint8).reshape(h, w, 3)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if img.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            arr = np.frombuffer(img.data, dtype=np.uint8).reshape(h, w)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if img.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
            arr = np.frombuffer(img.data, dtype=np.uint8).reshape(h, w, 4)
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            
        # If format is unknown, try to decode as JPEG anyway
        if img.format == image_pb2.Image.FORMAT_UNKNOWN:
            try:
                data = np.frombuffer(img.data, dtype=np.uint8)
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if bgr is not None:
                    return bgr
            except Exception as e:
                print(f"Failed to decode as JPEG: {e}")
                
        raise RuntimeError(f"Unsupported image format for detector. Format: {img.format}, Pixel format: {img.pixel_format}")

    def _pick_from_pixel(self, image_source="hand_color_image", pixel_xy=(960, 540),
                         top_down=True, auto_walk=True, timeout_s=25.0):
        """Execute PickObjectInImage at specific pixel coordinates."""
        img_client = self.robot.robot_sdk.ensure_client(ImageClient.default_service_name)
        manip_client = self.robot.robot_sdk.ensure_client(ManipulationApiClient.default_service_name)

        img_resp = img_client.get_image([build_image_request(image_source)])[0]
        req = manipulation_api_pb2.ManipulationApiRequest()
        pick = req.pick_object_in_image

        pick.transforms_snapshot_for_camera.CopyFrom(img_resp.shot.transforms_snapshot)
        pick.frame_name_image_sensor = img_resp.shot.frame_name_image_sensor

        # --- camera model: copy the active oneof directly into pick.camera_model ---
        src = img_resp.source

        # For visibility
        print("frame_name_image_sensor:", img_resp.shot.frame_name_image_sensor)
        print("image_source:", image_source)

        # Figure out which oneof is used on this SDK
        oneof_name = None
        for cand in ("camera_model", "model", "projection"):
            try:
                if src.WhichOneof(cand):
                    oneof_name = cand
                    break
            except Exception:
                pass

        active_field = src.WhichOneof(oneof_name) if oneof_name else None
        print("camera oneof name:", oneof_name, " active field:", active_field)

        # Helper to attempt compatible copies in a safe order
        def _try_copy_model(field_name: str) -> bool:
            try:
                if not field_name:
                    return False
                if not src.HasField(field_name):
                    return False
            except Exception:
                # HasField may not be implemented for all generated types; fallback on getattr truthiness
                if not hasattr(src, field_name) or getattr(src, field_name) is None:
                    return False

            try:
                # IMPORTANT: pick.camera_model expects the concrete submessage (e.g., PinholeModel)
                pick.camera_model.CopyFrom(getattr(src, field_name))
                print(f"Using camera model from source field: {field_name}")
                return True
            except Exception as e:
                print(f"Incompatible camera model field '{field_name}':", e)
                return False

        copied = False

        # 1) First try the actually active field on this image (best case)
        copied = _try_copy_model(active_field)

        # 2) If for any reason that fails, try common alternates in a sane order
        if not copied:
            for alt in ("pinhole", "kannala_brandt", "pinhole_brown_conrady", "fisheye"):
                if _try_copy_model(alt):
                    copied = True
                    break

        if not copied:
            raise RuntimeError(
                f"Could not set a compatible camera model for {image_source} "
                f"({img_resp.shot.frame_name_image_sensor})."
            )
        # --- end camera model selection ---

        # Clamp pixel to image bounds before assigning
        w = img_resp.shot.image.cols
        h = img_resp.shot.image.rows
        x = max(0, min(int(pixel_xy[0]), w - 1))
        y = max(0, min(int(pixel_xy[1]), h - 1))
        pick.pixel_xy.CopyFrom(geometry_pb2.Vec2(x=float(x), y=float(y)))
        
        print(f"📍 PickObjectInImage target: pixel=({x}, {y}) in {w}x{h} image")
        print(f"📍 PickObjectInImage will use vision frame coordinates")

        if top_down:
            axis_on_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)  # gripper x-axis
            axis_in_vision  = geometry_pb2.Vec3(x=0, y=0, z=-1) # point down in vision

            cons = pick.grasp_params.allowable_orientation.add()
            vat = cons.vector_alignment_with_tolerance

            # Required vectors
            vat.axis_on_gripper_ewrt_gripper.CopyFrom(axis_on_gripper)
            vat.axis_to_align_with_ewrt_frame.CopyFrom(axis_in_vision)

            # ---- Frame name handling across SDK versions ----
            # Some SDKs have 'frame_name' on the constraint; some require it on grasp_params.
            # Try constraint-level first only if it exists; otherwise set it on grasp_params.
            try:
                # Will raise if field does not exist on this SDK
                getattr(vat, "frame_name")
                vat.frame_name = "vision"
                print("Orientation frame set on constraint: vision")
            except Exception:
                if hasattr(pick.grasp_params, "grasp_params_frame_name"):
                    pick.grasp_params.grasp_params_frame_name = "vision"
                    print("Orientation frame set on grasp_params: vision")
                else:
                    print("No place to set orientation frame; using API default")

            # (Optional) Set an angle tolerance if this SDK exposes such a field.
            for tol_name in ("threshold_radians", "tolerance_radians", "max_angle_radians"):
                if hasattr(vat, tol_name):
                    setattr(vat, tol_name, math.radians(25.0))
                    break

        def _set_walk_gaze_mode(pick, auto_walk: bool):
            """
            Set walk/gaze behavior across SDK versions.
            Tries nested enum on PickObjectInImage, then module-level names,
            then falls back to 'disable_walk' if available.
            """
            # Candidates: (container_attr, constant_name) in priority order
            on = [
                ("PickObjectInImage", "WALK_GAZE_MODE_MOVE_AND_GAZE"),
                ("PickObjectInImage", "WALK_GAZE_MODE_AUTO"),
                ("", "PICK_AUTO_WALK_AND_GAZE"),
            ]
            off = [
                ("PickObjectInImage", "WALK_GAZE_MODE_GAZE"),
                ("PickObjectInImage", "WALK_GAZE_MODE_NONE"),
                ("", "PICK_GAZE_ONLY"),
            ]

            def _try(constants):
                for scope, name in constants:
                    try:
                        container = manipulation_api_pb2
                        if scope:
                            container = getattr(container, scope)
                        val = getattr(container, name)
                        pick.walk_gaze_mode = val
                        print(f"walk_gaze_mode set: {scope + '.' if scope else ''}{name} ({val})")
                        return True
                    except Exception:
                        pass
                return False

            if _try(on if auto_walk else off):
                return

            # Fallback (older SDKs): boolean flag
            if hasattr(pick, "disable_walk"):
                pick.disable_walk = (not auto_walk)
                print(f"disable_walk set to {pick.disable_walk} (fallback)")
            else:
                print("No walk/gaze control available; using API default")

        # Use it here:
        _set_walk_gaze_mode(pick, auto_walk)

        resp = manip_client.manipulation_api_command(req)
        cmd_id = resp.manipulation_cmd_id

        fb_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(manipulation_cmd_id=cmd_id)
        t0 = time.time()

        # Resolve enum container (supports older/newer SDKs)
        EnumNew = getattr(manipulation_api_pb2, "ManipulationApiFeedbackState", None)
        EnumOld = getattr(manipulation_api_pb2, "ManipulationFeedbackState", None)
        if EnumNew:
            ENUM = EnumNew
            SUCCESS = getattr(ENUM, "STATE_GRASP_SUCCEEDED", None)
            FAILED  = getattr(ENUM, "STATE_FAILED", None)
            UNKNOWN = getattr(ENUM, "STATE_UNKNOWN", None)
            name_fn = ENUM.Name
        else:
            ENUM = EnumOld
            SUCCESS = getattr(ENUM, "MANIP_STATE_GRASP_SUCCEEDED", None)
            FAILED  = getattr(ENUM, "MANIP_STATE_FAILED", None)
            UNKNOWN = getattr(ENUM, "MANIP_STATE_UNKNOWN", None)
            name_fn = ENUM.Name if ENUM else (lambda v: str(v))

        while time.time() - t0 < timeout_s:
            fb = manip_client.manipulation_api_feedback_command(fb_req)
            # Field name differs by SDK
            state_val = getattr(fb, "current_state", None)
            if state_val is None:
                state_val = getattr(fb, "state", None)

            try:
                print("Manipulation state:", name_fn(state_val))
            except Exception:
                print("Manipulation state (raw):", state_val)

            if state_val in tuple(v for v in (SUCCESS, FAILED, UNKNOWN) if v is not None):
                return state_val == SUCCESS
            time.sleep(0.2)

        print("Manipulation timed out")
        return False

    def _backproject_and_script_grasp(self, color_source, pixel_xy, z_offset=0.0,
                                      approach=0.12, lift=0.15):
        """Arm-only grasp: use hand depth to back-project pixel to 3D and do a scripted grasp."""
        img_client = self.robot.robot_sdk.ensure_client(ImageClient.default_service_name)

        # B) Use a depth stream that is explicitly aligned to the hand color
        try:
            src_names = [s.name for s in img_client.list_image_sources()]
        except Exception:
            src_names = []

        depth_candidates = [
            "hand_depth_in_hand_color_frame",
            "hand_depth_in_color_frame", 
            "hand_depth_in_hand_color_frame_image",
        ]
        depth_source = next((n for n in depth_candidates if n in src_names), None)
        if depth_source is None:
            print("Arm-only: no aligned hand depth source available.")
            return False

        color = img_client.get_image([build_image_request(color_source)])[0]
        depth = img_client.get_image([build_image_request(depth_source)])[0]

        dimg = depth.shot.image
        if dimg.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16 or \
           dimg.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
            depth_arr = np.frombuffer(dimg.data, dtype=np.uint16).reshape(dimg.rows, dimg.cols).astype(np.float32) / 1000.0
        else:
            print(f"Arm-only: unsupported depth pixel format {dimg.pixel_format}")
            return False

        # D) Verify depth/colour resolutions match; scale pixel if needed
        cw, ch = color.shot.image.cols, color.shot.image.rows
        dw, dh = depth.shot.image.cols, depth.shot.image.rows
        ux = int(pixel_xy[0] * dw / cw)
        uy = int(pixel_xy[1] * dh / ch)
        ux = max(0, min(ux, dw - 1))
        uy = max(0, min(uy, dh - 1))

        z = float(depth_arr[uy, ux])
        if not np.isfinite(z) or z <= 0:
            # Neighborhood median fallback
            patch = depth_arr[max(0, uy-2):min(dh, uy+3), max(0, ux-2):min(dw, ux+3)]
            good = patch[np.isfinite(patch) & (patch > 0)]
            z = float(np.median(good)) if good.size else 0.0
        if z <= 0:
            print("Arm-only: no valid depth at/near pixel.")
            return False

        # C) Extract intrinsics robustly with improved camera model detection
        src = color.source
        
        # Enhanced camera model detection with better debugging
        print(f"Debug: ImageSource type: {type(src)}")
        print(f"Debug: ImageSource attributes: {[attr for attr in dir(src) if not attr.startswith('_')]}")
        
        # Try different field names for camera model with more robust detection
        oneof = None
        model = None
        
        # Method 1: Try WhichOneof approach
        for field_name in ["camera_model", "model", "projection"]:
            try:
                if hasattr(src, "WhichOneof") and src.WhichOneof(field_name):
                    oneof = src.WhichOneof(field_name)
                    print(f"Debug: Found oneof field '{field_name}' with value '{oneof}'")
                    break
            except Exception as e:
                print(f"Debug: WhichOneof failed for '{field_name}': {e}")
                continue
        
        # Method 2: Direct attribute access if WhichOneof fails
        if not oneof:
            print("Debug: WhichOneof approach failed, trying direct attribute access")
            for field_name in ["camera_model", "model", "projection"]:
                try:
                    if hasattr(src, field_name):
                        attr_value = getattr(src, field_name)
                        if attr_value is not None:
                            print(f"Debug: Found direct attribute '{field_name}': {type(attr_value)}")
                            model = attr_value
                            break
                except Exception as e:
                    print(f"Debug: Direct access failed for '{field_name}': {e}")
                    continue
        
        # Method 3: Try to get model from oneof if we found it
        if oneof and not model:
            try:
                model = getattr(src, oneof)
                print(f"Debug: Got model from oneof '{oneof}': {type(model)}")
            except Exception as e:
                print(f"Debug: Failed to get model from oneof '{oneof}': {e}")
        
        # Method 4: Fallback to hardcoded intrinsics for hand camera
        if not model:
            print("Debug: No camera model found, using fallback intrinsics for hand camera")
            # Use typical hand camera intrinsics as fallback
            # These are approximate values - adjust based on your specific hand camera
            fx, fy = 400.0, 400.0  # focal length in pixels
            cx, cy = cw / 2.0, ch / 2.0  # principal point (center of image)
            print(f"Debug: Using fallback intrinsics - fx={fx}, fy={fy}, cx={cx}, cy={cy}")
        else:
            # Extract intrinsics from the model
            intr = getattr(model, "intrinsics", None) if model else None
            if not intr or not hasattr(intr, "focal_length") or not hasattr(intr, "principal_point"):
                print("Arm-only: missing pinhole intrinsics on hand_color_image; using fallback.")
                # Use fallback intrinsics
                fx, fy = 400.0, 400.0
                cx, cy = cw / 2.0, ch / 2.0
            else:
                fx, fy = intr.focal_length.x, intr.focal_length.y
                cx, cy = intr.principal_point.x, intr.principal_point.y
                print(f"Debug: Using model intrinsics - fx={fx}, fy={fy}, cx={cx}, cy={cy}")

        # In hand sensor frame
        Xc = (ux - cx) * z / fx
        Yc = (uy - cy) * z / fy
        Zc = z + z_offset  # Apply the specified Z offset (-0.6m for label-based picks)

        print(f"Debug: Z offset applied: depth={z:.3f}m + offset={z_offset:.3f}m = {Zc:.3f}m")
        print(f"Debug: Backprojected 3D point in sensor frame: ({Xc:.3f}, {Yc:.3f}, {Zc:.3f})")
        print(f"Debug: Camera frame pickup point: X={Xc:.3f}m, Y={Yc:.3f}m, Z={Zc:.3f}m")

        # Transform to body
        snapshot = color.shot.transforms_snapshot
        body_T_sensor = get_a_tform_b(snapshot, BODY_FRAME_NAME, color.shot.frame_name_image_sensor)
        p_body = body_T_sensor * SE3Pose(Xc, Yc, Zc, Quat(1, 0, 0, 0))

        print(f"Debug: Transformed to body frame: ({p_body.x:.3f}, {p_body.y:.3f}, {p_body.z:.3f})")
        print(f"Debug: Body frame pickup point: X={p_body.x:.3f}m, Y={p_body.y:.3f}m, Z={p_body.z:.3f}m")
        print(f"Debug: Pickup coordinates - X: {p_body.x:.3f}m forward/backward, Y: {p_body.y:.3f}m left/right, Z: {p_body.z:.3f}m up/down")

        # E) Sanity-clamp the 3-D point before commanding the arm
        r = math.hypot(p_body.x, p_body.y)

        # Accept ground objects and give a 2 cm clearance if z is below body origin
        GROUND_Z_MIN = -0.45  # about floor height relative to BODY for normal stand
        GROUND_CLEAR = 0.02   # lift a bit above the floor to avoid scraping
        Z_MAX = 1.0  # Maximum Z for 1.0m offset (depth + 1.0m = max 1.0m above body)
        R_MIN, R_MAX = 0.30, 0.85

        if not (R_MIN <= r <= R_MAX) or not (GROUND_Z_MIN <= p_body.z <= Z_MAX):
            print(f"Arm-only: target out of safe reach (r={r:.2f}, z={p_body.z:.2f}); aborting.")
            print(f"   Safe range: r=[{R_MIN:.2f}, {R_MAX:.2f}], z=[{GROUND_Z_MIN:.2f}, {Z_MAX:.2f}]")
            return False

        # If the point is near or below the floor plane, nudge up a bit for grasp
        target_z = p_body.z if p_body.z > 0 else p_body.z + GROUND_CLEAR
        if target_z != p_body.z:
            print(f"Ground object detected: adjusting Z from {p_body.z:.3f} to {target_z:.3f} (+{GROUND_CLEAR:.3f}m clearance)")

        print(f"🎯 FINAL PICKUP POINT:")
        print(f"   Camera frame: X={Xc:.3f}m, Y={Yc:.3f}m, Z={Zc:.3f}m")
        print(f"   Body frame: X={p_body.x:.3f}m, Y={p_body.y:.3f}m, Z={target_z:.3f}m")
        print(f"   Approach height: {target_z + approach:.3f}m")
        print(f"   Grasp height: {target_z:.3f}m")
        print(f"   Lift height: {target_z + lift:.3f}m")

        # Scripted grasp = no base motion
        self._scripted_grasp_body_point(p_body.x, p_body.y, target_z,
                                        approach=approach, lift=lift)
        return True

    def _aim_hand_at_ground_ahead(self, distance=0.22, z=0.30, y=HAND_PEEK_Y):
        """Aim hand camera straight down at a ground patch ahead, but keep clear of the torso."""
        self.robot.ready_or_stow_arm(stow=False)

        # Bring the hand a bit closer and lower to reduce range error
        x = max(float(distance), 0.60)  # Closer for better sphere detection
        z = max(float(z), 0.50)         # Lower for better ground view

        odom_T_task = get_root_T_ground_body(
            robot_state=self.robot.robot_state_client.get_robot_state(),
            root_frame_name=GRAV_ALIGNED_BODY_FRAME_NAME
        )
        # Point gripper/tool z-axis straight down
        wr1_T_tool = SE3Pose(0, 0, 0, Quat.from_pitch(-math.pi/2))

        observe_pose = SE3Pose(x, float(y), z, Quat(1, 0, 0, 0))
        self.robot.move_to_cartesian_pose_rt_task(observe_pose, odom_T_task, wr1_T_tool)
        # Wait for arm movement to complete and settle
        time.sleep(0.8)  # Reduced wait to ensure arm is stable (was 1.5)

    def _detect_with_ultralytics(self, bgr, target_label="bottle", model_path="yolov8n.pt", conf=0.1, image_source="hand_color_image"):
        """Detect objects using Ultralytics YOLO."""
        try:
            from ultralytics import YOLO
        except Exception as e:
            print("Ultralytics niet beschikbaar:", e)
            return None

        if not hasattr(self, "_yolo_model"):
            try:
                self._yolo_model = YOLO(model_path)
                print(f"YOLO model loaded: {model_path}")
            except Exception as e:
                print(f"Failed to load YOLO model {model_path}: {e}")
                return None

        # First, let's see what classes the model actually supports
        names = self._yolo_model.model.names
        print(f"\n=== YOLO MODEL INFO ===")
        print(f"Model: {model_path}")
        print(f"Available classes: {list(names.values())}")
        print(f"Looking for: '{target_label}'")
        
        # Check if target_label exists in the model
        if target_label not in names.values():
            print(f"⚠️  WARNING: '{target_label}' not found in model classes!")
            print(f"   Available similar classes: {[name for name in names.values() if 'ball' in name.lower() or 'sport' in name.lower()]}")
            print(f"   Try using one of these instead: {[name for name in names.values() if 'ball' in name.lower() or 'sport' in name.lower() or 'baseball' in name.lower()]}")
        
        # Try with lower confidence first to see what's detected
        results = self._yolo_model.predict(bgr, conf=0.1, verbose=False)  # Lower confidence for debugging
        if not results:
            print("No objects detected at all (even with low confidence)")
            return None

        # Debug: Show all detected objects
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
        
        # Sort by confidence score (highest first)
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
            if det['label'] == target_label and det['score'] > best_score:
                best = (det['center'][0], det['center'][1], det['score'], det['bbox'])
                best_score = det['score']
        
        if best:
            print(f"✅ Found target '{target_label}' with confidence {best[2]:.3f}")
        else:
            print(f"❌ Target '{target_label}' not found in detected objects")
            if all_detections:
                print(f"   Available objects: {[det['label'] for det in all_detections]}")
        
        # Always save debug image (success or failure)
        self._save_debug_image(bgr, all_detections, target_label, image_source, save_path="/catkin_ws/debug_images/debug_detection.jpg")
            
        return best  # (cx, cy, score, bbox) of None

    def _save_debug_image(self, bgr, detections, target_label, image_source, save_path="/catkin_ws/debug_images/debug_detection.jpg"):
        """Save debug image with detection results."""
        try:
            import cv2
            import os
            
            # Ensure the debug_images directory exists
            debug_dir = os.path.dirname(save_path)
            os.makedirs(debug_dir, exist_ok=True)
            
            # Create a copy of the image for drawing
            debug_img = bgr.copy()
            
            # Draw all detections
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                label = det['label']
                score = det['score']
                
                # Color based on whether it's the target
                if label == target_label:
                    color = (0, 255, 0)  # Green for target
                    thickness = 3
                else:
                    color = (0, 0, 255)  # Red for other objects
                    thickness = 2
                
                # Draw bounding box
                cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
                
                # Draw label and confidence
                text = f"{label}: {score:.2f}"
                cv2.putText(debug_img, text, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Add image info
            info_text = f"Camera: {image_source} | Target: {target_label} | Objects: {len(detections)}"
            cv2.putText(debug_img, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add note about coordinate system
            coord_text = f"Coordinates: Same camera for detection and backprojection"
            cv2.putText(debug_img, coord_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save image
            cv2.imwrite(save_path, debug_img)
            print(f"Debug image saved to: {save_path}")
            
        except Exception as e:
            print(f"Failed to save debug image: {e}")

    def _detect_with_opencv(self, bgr, target_label="bottle", onnx_path="yolov8n.onnx", conf=0.1):
        """Detect objects using OpenCV DNN with ONNX model."""
        if not hasattr(self, "_cv_dnn"):
            try:
                self._cv_dnn = cv2.dnn.readNetFromONNX(onnx_path)
                # labels van het model horen in een aparte file, vereenvoudigd voorbeeld:
                self._cv_labels = {39: "bottle"}  # pas aan aan jouw model
                print(f"OpenCV DNN model loaded: {onnx_path}")
            except Exception as e:
                print(f"Failed to load OpenCV DNN model {onnx_path}: {e}")
                return None

        blob = cv2.dnn.blobFromImage(bgr, scalefactor=1/255.0, size=(640, 640), swapRB=True, crop=False)
        self._cv_dnn.setInput(blob)
        out = self._cv_dnn.forward()  # vorm hangt af van model

        H, W = bgr.shape[:2]
        best = None
        best_score = -1.0
        # Pseudo parsing. Pas aan op jouw ONNX output.
        for det in out[0]:
            score = det[4]
            if score < conf:
                continue
            cls_scores = det[5:]
            cls_id = int(np.argmax(cls_scores))
            cls_conf = cls_scores[cls_id]
            total = score * cls_conf
            label = self._cv_labels.get(cls_id, str(cls_id))
            if label != target_label or total < conf:
                continue
            cx, cy, w, h = det[:4]
            x1 = int((cx - w / 2) * W)
            y1 = int((cy - h / 2) * H)
            x2 = int((cx + w / 2) * W)
            y2 = int((cy + h / 2) * H)
            if total > best_score:
                best_score = float(total)
                best = (int((x1 + x2) / 2), int((y1 + y2) / 2), best_score, (x1, y1, x2, y2))
        return best

    def _try_multiple_pick_points(self, image_source, bbox, target_label, auto_walk=True):
        """Try multiple points within the bounding box for better pick success."""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Define multiple pick points within the bounding box
        pick_points = [
            (int((x1 + x2) / 2), int((y1 + y2) / 2)),  # Center
            (int((x1 + x2) / 2), int((y1 + y2) / 2) + int(height * 0.1)),  # 10% down from center
            (int((x1 + x2) / 2), int((y1 + y2) / 2) + int(height * 0.15)),  # 15% down from center
            (int((x1 + x2) / 2) - int(width * 0.1), int((y1 + y2) / 2)),  # 10% left from center
            (int((x1 + x2) / 2) + int(width * 0.1), int((y1 + y2) / 2)),  # 10% right from center
        ]
        
        print(f"Trying {len(pick_points)} pick points for {target_label}")
        
        for i, (cx, cy) in enumerate(pick_points):
            print(f"  Point {i+1}: ({cx}, {cy})")
            success = self._pick_from_pixel(
                image_source=image_source, 
                pixel_xy=(cx, cy),
                top_down=True, 
                auto_walk=auto_walk,
                timeout_s=15.0  # Shorter timeout for retry attempts
            )
            if success:
                print(f"✅ Pick successful at point {i+1}")
                return True
            else:
                print(f"❌ Pick failed at point {i+1}")
                time.sleep(0.5)  # Reduced pause between attempts (was 1.0)
        
        print(f"❌ All pick attempts failed for {target_label}")
        return False

    def _try_multiple_pick_points_backprojection(self, image_source, bbox, target_label, z_offset=-0.6):
        """Try multiple points within the bounding box using backprojection with Z offset."""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        # Define multiple pick points within the bounding box
        pick_points = [
            (int((x1 + x2) / 2), int((y1 + y2) / 2)),  # Center
            (int((x1 + x2) / 2), int((y1 + y2) / 2) + int(height * 0.1)),  # 10% down from center
            (int((x1 + x2) / 2), int((y1 + y2) / 2) + int(height * 0.15)),  # 15% down from center
            (int((x1 + x2) / 2) - int(width * 0.1), int((y1 + y2) / 2)),  # 10% left from center
            (int((x1 + x2) / 2) + int(width * 0.1), int((y1 + y2) / 2)),  # 10% right from center
        ]
        
        print(f"Trying {len(pick_points)} backprojection pick points for {target_label}")
        
        for i, (cx, cy) in enumerate(pick_points):
            print(f"  Point {i+1}: ({cx}, {cy}) with z_offset={z_offset}m")
            success = self._backproject_and_script_grasp(
                image_source=image_source, 
                pixel_xy=(cx, cy),
                z_offset=z_offset,
                approach=0.12, 
                lift=0.15
            )
            if success:
                print(f"✅ Backprojection pick successful at point {i+1}")
                return True
            else:
                print(f"❌ Backprojection pick failed at point {i+1}")
                time.sleep(0.5)  # Reduced pause between attempts
        
        print(f"❌ All backprojection pick attempts failed for {target_label}")
        return False





    def pick_by_label_pixel(self, target_label="bottle", image_source="hand_color_image",
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
            # If we're using the hand camera, aim it down first.
            if image_source.startswith("hand_"):
                # Use optimized position for ground object detection - 20cm higher for better label detection
                self._aim_hand_at_ground_ahead(distance=0.65, z=0.65, y=-0.10)  # 20cm higher (0.45 + 0.20) for better view
                # Optional: lower the body a bit for a better viewing angle
                try:
                    self.robot.stand(-0.08)  # Lower stance for better ground visibility
                except Exception:
                    pass
                # Wait for arm to settle and robot to stop moving
                time.sleep(0.8)  # Wait to ensure stability

            img_client = self.robot.robot_sdk.ensure_client(ImageClient.default_service_name)
            img_resp = img_client.get_image([build_image_request(image_source)])[0]
            bgr = self._bd_image_to_cv2(img_resp)
            if bgr is None:
                print("Failed to get image from camera")
                return False

            if provider == "ultralytics":
                det = self._detect_with_ultralytics(bgr, target_label=target_label, conf=conf, image_source=image_source)
            else:
                det = self._detect_with_opencv(bgr, target_label=target_label, conf=conf)

            if det is None:
                print(f"No '{target_label}' found for pixel grasp.")
                return False

            cx, cy, score, bbox = det
            print(f"Grasp-by-pixel for '{target_label}' at pixel ({cx}, {cy})")
            
            # Get depth at the detected pixel for logging
            try:
                depth_client = self.robot.robot_sdk.ensure_client(ImageClient.default_service_name)
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
                
                poi_ok = self._pick_from_pixel(
                    image_source=image_source,
                    pixel_xy=(cx, cy),
                    top_down=True,
                    auto_walk=False,      # keep base fixed
                    timeout_s=25.0
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
                print(f"With -0.6m Z offset: target Z = {depth_at_pixel - 0.6:.3f}m")
                print(f"   (Negative offset in camera frame = higher position in body frame)")
            else:
                print(f"Will apply -0.6m Z offset to detected depth")
            print(f"Method: Backprojection with scripted grasp (Z offset WILL be applied)")
            print(f"Safety: -0.6m offset ensures ~0.20-0.25m body frame Z (no floor bumping)")
            print("="*60)
            
            # Get user confirmation
            while True:
                response = input("Do you want to proceed with backprojection fallback? (yes/no): ").strip().lower()
                if response in ['yes', 'y']:
                    print("✅ User confirmed - proceeding with backprojection fallback...")
                    break
                elif response in ['no', 'n']:
                    print("❌ User cancelled - exiting program")
                    import sys
                    sys.exit(0)
                else:
                    print("Please type 'yes' or 'no'")
            
            if self._backproject_and_script_grasp(image_source, (cx, cy),
                                                  z_offset=-0.6, approach=0.12, lift=0.15):
                print(f"✅ Backprojection fallback succeeded for {target_label}")
                print(f"📍 Backprojection used z_offset=-0.6m successfully")
                # After successful pick, lift the arm up to hold the object
                self._lift_arm_after_successful_pick()
                return True

            # Try a few nearby pixels inside the bbox if first attempt failed
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
                    print(f"Method: PickObjectInImage retry (no Z offset applied)")
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

            # If backprojection failed, try multiple nearby pixels
            print("Backprojection failed, trying multiple nearby pixels...")
            
            # Show retry coordinates and ask for user confirmation
            print("\n" + "="*60)
            print("🔄 RETRY PICK COORDINATES CONFIRMATION")
            print("="*60)
            print(f"Target object: {target_label}")
            print(f"Original pixel: ({cx}, {cy})")
            print(f"Will try multiple nearby pixels within bounding box")
            if depth_at_pixel is not None:
                print(f"Detected depth: {depth_at_pixel:.3f}m")
                print(f"Will apply -0.6m Z offset to all attempts")
            print(f"Method: Backprojection retry (Z offset WILL be applied)")
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
            
            # Try multiple pick points with backprojection
            if self._try_multiple_pick_points_backprojection(image_source, bbox, target_label, z_offset=-0.6):
                print(f"📍 Multiple pick points succeeded with backprojection")
                # After successful pick, lift the arm up to hold the object
                self._lift_arm_after_successful_pick()
                return True

            print("❌ All attempts failed for label pick")
            print("💡 Tip: If you want Spot to take a tiny step to center the ball, set:")
            print("   self.robot.allow_base_motion_during_pick = True")
            print("   Then use auto_walk=True in the PickObjectInImage calls above")
            return False

        except Exception as e:
            print(f"Error in pick_by_label_pixel: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    spot = SpotStateMachine(robot=None)

    img_path = "docs/images/readme_spotstatemachine1.png"
    spot._graph().write_png(img_path)
    
    msg = spot.send("stand_up")
    print(msg)

    spot.send("stand_up_high")
    
    ## How to error handle wrong actions to state machine
    try:
        spot.send("start_trajectory")
    except:
        try:
            spot.send("stop_walking")
            spot.send("start_trajectory")
        except:
            try:
                spot.send("stand_up")
                print("Stand up first")
                spot.send("start_trajectory")
            except:
                print("Start trajectory not possible")
                ## Do some handling or more feedback to user
    
    img_path = "docs/images/readme_spotstatemachine2.png"
    spot._graph().write_png(img_path)


