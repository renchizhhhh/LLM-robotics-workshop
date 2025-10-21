#!/usr/bin/env python

import rospy
import math
import time
import json
import sys
import os
from statemachine import StateMachine, State
import actionlib
from std_srvs.srv import Trigger
from std_msgs.msg import String
from spot_hololens_llm_interface.srv import MoveToPosition, MoveToPositionRequest, GetImage, GetImageRequest, GetInitialPose, GetInitialPoseRequest, GetRobotPose, GetRobotPoseRequest, ArmCommand, ArmCommandRequest
from spot_hololens_llm_interface.msg import AutomatedGraspAction, AutomatedGraspGoal, MoveArmPoseAction, MoveArmPoseGoal

# Add the scripts directory to the path to import timing_utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts'))
from timing_utils import recorder

class SpotStateMachine(StateMachine):
    """
    Minimal state machine for Spot robot control.
    Acts as a bridge between high-level commands and ROS services.
    """
    # States
    # Start in an `unknown` state and probe the robot at startup. This is the
    # safest option for mixed deployments: if the robot manager is already up
    # the FSM will transition to `connected`; otherwise it will move to
    # `disconnected` and wait for explicit connect commands.
    unknown = State(initial=True)
    connected = State()
    disconnected = State()
    powered_off = State()
    sit = State()
    stand = State()
    moving = State()
    rotating = State()  # New state for grid-based rotation
    grasping = State()
    carry = State()
    move_arm_pose = State()
    arm_command = State()

    # Transitions
    discover_connected = unknown.to(connected)
    discover_disconnected = unknown.to(disconnected)
    discover_powered_off = unknown.to(powered_off)
    discover_stand = unknown.to(stand)
    discover_sit = unknown.to(sit)
    connect = disconnected.to(connected)
    power_on = connected.to(powered_off)
    stand_up = (powered_off.to(stand) | sit.to(stand))
    sit_down = (stand.to(sit) | moving.to(sit) | rotating.to(sit) | grasping.to(sit))
    start_moving = (stand.to(moving) | moving.to(moving) | carry.to(moving))
    stop_moving = moving.to(stand)
    start_rotating = (stand.to(rotating) | rotating.to(rotating) | carry.to(rotating) | moving.to(rotating))  # New grid-based rotation
    stop_rotating = rotating.to(stand)
    get_image = (stand.to(stand) | moving.to(moving) | rotating.to(rotating) | sit.to(sit))
    get_initial_pose = (stand.to(stand) | moving.to(moving) | rotating.to(rotating) | sit.to(sit))
    start_arm_command = (stand.to(arm_command) | moving.to(arm_command) | rotating.to(arm_command) | carry.to(arm_command))
    finish_arm_command = (arm_command.to(stand))
    start_automated_grasp = (stand.to(grasping) | moving.to(grasping) | rotating.to(grasping))
    finish_automated_grasp = (grasping.to(carry))
    start_move_arm_pose = (stand.to(move_arm_pose) | moving.to(move_arm_pose) | rotating.to(move_arm_pose) | carry.to(move_arm_pose))
    finish_move_arm_pose = (move_arm_pose.to(stand))
    start_drop_off = (stand.to(move_arm_pose) | moving.to(move_arm_pose) | rotating.to(move_arm_pose) | carry.to(move_arm_pose))
    finish_drop_off = (move_arm_pose.to(stand))
    carry_to_stand = carry.to(stand)
    stand_to_carry = stand.to(carry)
    power_off_from_stand = stand.to(powered_off)
    power_off_from_sit = sit.to(powered_off)
    disconnect = powered_off.to(disconnected)

    def create_service_proxy(self, name, srv_type, timeout=5.0, retry_interval=1.0):
        """Wait for a ROS service and create a ServiceProxy for it."""
        while not rospy.is_shutdown():
            try:
                rospy.wait_for_service(name, timeout=timeout)
                break
            except rospy.ROSException:
                rospy.logwarn(f"FSM: Service {name} not available yet, retrying in {retry_interval}s")
                rospy.sleep(retry_interval)
        try:
            proxy = rospy.ServiceProxy(name, srv_type)
            return proxy
        except Exception as e:
            rospy.logerr(f"FSM: Failed to create ServiceProxy for {name}: {e}")
            return rospy.ServiceProxy(name, srv_type)

    def __init__(self, dummy_mode=None):
        # Check for dummy mode parameter
        self.dummy_mode = dummy_mode if dummy_mode is not None else rospy.get_param('~dummy_mode', False)
        
        # Timing utilities will be imported from timing_utils module
        
        if self.dummy_mode:
            rospy.loginfo("FSM: Running in DUMMY MODE - using dummy services")
        else:
            rospy.loginfo("FSM: Running in REAL MODE - using real robot services")
        
        # Setup ROS service connections (wait for services and create proxies)
        self.connect_srv = self.create_service_proxy('/spot_entrance/connect', Trigger)
        self.power_on_srv = self.create_service_proxy('/spot_entrance/power_on', Trigger)
        self.stand_srv = self.create_service_proxy('/spot_entrance/stand', Trigger)
        self.sit_srv = self.create_service_proxy('/spot_entrance/sit', Trigger)
        self.move_srv = self.create_service_proxy('/spot_entrance/move_to_position', MoveToPosition)
        self.get_image_srv = self.create_service_proxy('/spot_entrance/get_image', GetImage)
        self.get_initial_pose_srv = self.create_service_proxy('/spot_entrance/get_initial_pose', GetInitialPose)
        self.get_robot_pose_srv = self.create_service_proxy('/spot_entrance/get_robot_pose', GetRobotPose)
        self.arm_command_srv = self.create_service_proxy('/spot_entrance/arm_command', ArmCommand)
        self.power_off_srv = self.create_service_proxy('/spot_entrance/power_off', Trigger)
        self.disconnect_srv = self.create_service_proxy('/spot_entrance/disconnect', Trigger)
        
        # Setup action clients
        self.automated_grasp_client = actionlib.SimpleActionClient('automated_grasp', AutomatedGraspAction)
        self.move_arm_pose_client = actionlib.SimpleActionClient('move_arm_pose', MoveArmPoseAction)
        self.drop_off_client = actionlib.SimpleActionClient('drop_off_object', MoveArmPoseAction)
        if not self.dummy_mode:
            self.automated_grasp_client.wait_for_server(rospy.Duration(5.0))
            self.move_arm_pose_client.wait_for_server(rospy.Duration(5.0))
            self.drop_off_client.wait_for_server(rospy.Duration(5.0))
            rospy.loginfo("FSM: Connected to action servers")
        
        # Initialize the state machine after setting up services
        super().__init__()
        self._startup_probe(timeout=3.0)
        
        # Default movement parameters (used as fallback)
        self.move_x = 0.0
        self.move_y = 0.0
        self.move_yaw = 0.0
        self.move_frame = "vision"

        # Grid origin alignment for real mode: ensure first real pose maps to GUI start cell
        # GUI defaults to start at cell (4,4); allow override via params
        self.grid_start_row = rospy.get_param('/nl_control/start_cell_row', 4)
        self.grid_start_col = rospy.get_param('/nl_control/start_cell_col', 4)
        self._grid_origin_x = None
        self._grid_origin_y = None
        # Yaw origin: normalize initial heading to 0 at startup
        self._yaw_origin = None

    def _startup_probe(self, timeout=3.0):
        """Check `/spot_entrance/robot_state` and align FSM to the most specific state."""
        try:
            if self.dummy_mode:
                rospy.loginfo('FSM: Dummy mode - simulating full startup sequence')
                self.send('discover_connected')
                # Auto-power on in dummy mode for easier testing
                rospy.sleep(0.1)  # Small delay to ensure state transition
                self.send('power_on')
                return
            else:
                rospy.loginfo('FSM: Real robot mode - starting automatic connection sequence')
                # Auto-connect and power on for real robot
                self.send('discover_connected')
                rospy.sleep(0.5)  # Allow connection to complete
                self.send('power_on')
                return

            msg = rospy.wait_for_message('/spot_entrance/robot_state', String, timeout=timeout)

            if msg and isinstance(msg.data, str):
                try:
                    rospy.loginfo(f'FSM: Startup probe got robot_state: {msg.data}')
                    fields = dict(part.split(':', 1) for part in msg.data.split(','))
                    connected_val = fields.get('connected', 'false').strip().lower()
                    powered_val = fields.get('powered', 'false').strip().lower()
                    standing_val = fields.get('standing', 'false').strip().lower()

                    is_connected = connected_val in ('true', '1', 'yes')
                    is_powered = powered_val in ('true', '1', 'yes')
                    is_standing = standing_val in ('true', '1', 'yes')

                    if not is_connected:
                        self.send('discover_disconnected')
                        return
                    if not is_powered:
                        self.send('discover_powered_off')
                        return
                    if is_standing:
                        self.send('discover_stand')
                        return
                    else:
                        self.send('discover_sit')
                        return
                except Exception:
                    rospy.logwarn('FSM: Failed to parse robot_state message, falling back')

            self.send('discover_disconnected')
        except Exception as e:
            rospy.logwarn(f'FSM: Startup probe failed: {e} - staying in unknown')

    # State entry methods - call services when states change
    def on_enter_connected(self):
        rospy.loginfo("FSM: Connecting to robot")
        # Note: Timing events for connect are handled in nl_control.py initialization
        result = self._call_service(self.connect_srv, "Connect")

    def on_enter_powered_off(self):
        rospy.loginfo("FSM: Powering on robot")
        # Note: Timing events for power_on are handled in nl_control.py initialization
        result = self._call_service(self.power_on_srv, "Power on")

    def on_enter_stand(self):
        rospy.loginfo("FSM: Standing up")
        recorder.publish_event('start_stand_up')
        result = self._call_service(self.stand_srv, "Stand")
        recorder.publish_event('stop_stand_up')

    def on_enter_sit(self):
        rospy.loginfo("FSM: Sitting down")
        recorder.publish_event('start_sit_down')
        result = self._call_service(self.sit_srv, "Sit")
        recorder.publish_event('stop_sit_down')

    def on_enter_carry(self):
        rospy.loginfo("FSM: Moving to carry position")
        recorder.publish_event('start_carry')
        self._call_arm_command_service("carry")
        recorder.publish_event('stop_carry')

    def on_enter_moving(self):
        # Get parameters from kwargs if available, otherwise use stored ones
        kwargs = getattr(self, '_current_kwargs', {})
        
        # Check if this is a grid-based movement (row/col) or continuous (x/y)
        if 'row' in kwargs and 'col' in kwargs:
            # Grid-based movement: use new grid conversion method
            row = kwargs.get('row', 0)
            col = kwargs.get('col', 0)
            
            recorder.publish_event('start_moving')
            result = self._call_move_to_cell(row, col)
            recorder.publish_event('stop_moving')
        else:
            # Continuous movement (legacy support)
            move_x = kwargs.get('x', self.move_x)
            move_y = kwargs.get('y', self.move_y)
            move_yaw = kwargs.get('yaw', self.move_yaw)
            move_frame = kwargs.get('frame', self.move_frame)
            
            rospy.loginfo(f"FSM: Continuous movement x={move_x}, y={move_y}, yaw={move_yaw}")
            
            recorder.publish_event('start_moving')
            result = self._call_move_service(move_x, move_y, move_yaw, move_frame)
            recorder.publish_event('stop_moving')
    
    def on_enter_rotating(self):
        # Handle grid-based rotation to cardinal directions
        kwargs = getattr(self, '_current_kwargs', {})
        direction = kwargs.get('direction', 'N')
        
        rospy.loginfo(f"FSM: Rotating to face {direction}")
        recorder.publish_event('start_rotating')
        result = self._call_rotate_to(direction)
        recorder.publish_event('stop_rotating')
    
    def on_enter_get_image(self):
        # Get image source from kwargs, default to frontleft_fisheye_image
        kwargs = getattr(self, '_current_kwargs', {})
        image_source = kwargs.get('image_source', 'frontleft_fisheye_image')
        
        rospy.loginfo(f"FSM: Getting image from {image_source}")
        recorder.publish_event('start_get_image')
        self._call_get_image_service(image_source)
        recorder.publish_event('stop_get_image')

    def on_enter_get_initial_pose(self):
        rospy.loginfo("FSM: Getting initial pose")
        recorder.publish_event('start_get_initial_pose')
        self._call_get_initial_pose_service()
        recorder.publish_event('stop_get_initial_pose')

    def on_enter_arm_command(self):
        # Get command type from kwargs, default to "stow"
        kwargs = getattr(self, '_current_kwargs', {})
        command_type = kwargs.get('command_type', 'stow')
        
        rospy.loginfo(f"FSM: Executing arm command: {command_type}")
        recorder.publish_event('start_arm_command')
        self._call_arm_command_service(command_type)
        recorder.publish_event('stop_arm_command')
        
        # Always return to stand state after arm command
        self.finish_arm_command()

    def on_enter_move_arm_pose(self):
        # Get parameters from kwargs
        kwargs = getattr(self, '_current_kwargs', {})
        is_drop_off = kwargs.get('_is_drop_off', False)
        
        if is_drop_off:
            # Use drop_off action which handles full sequence
            rospy.loginfo("FSM: Starting drop-off sequence")
            recorder.publish_event('start_drop_off')
            result = self._call_drop_off_action()
            recorder.publish_event('stop_drop_off')
            self.finish_drop_off()
        else:
            # Regular move_arm_pose action
            x = kwargs.get('x', 0.8)
            y = kwargs.get('y', 0.0)
            z = kwargs.get('z', 0.3)
            qw = kwargs.get('qw', 0.7071)
            qx = kwargs.get('qx', 0.7071)
            qy = kwargs.get('qy', 0.0)
            qz = kwargs.get('qz', 0.0)
            duration = kwargs.get('duration', 3.0)
            open_gripper = kwargs.get('open_gripper', False)
            
            rospy.loginfo(f"FSM: Moving arm to pose ({x}, {y}, {z})")
            recorder.publish_event('start_move_arm_pose')
            result = self._call_move_arm_pose_action(x, y, z, qw, qx, qy, qz, duration, open_gripper)
            recorder.publish_event('stop_move_arm_pose')
            
            # Always return to stand state after move_arm_pose
            self.finish_move_arm_pose()

    def on_enter_grasping(self):
        # Get parameters from kwargs
        kwargs = getattr(self, '_current_kwargs', {})
        object_type = kwargs.get('object_type', 'cup')
        image_source = kwargs.get('image_source', 'hand_color_image')
        force_top_down_grasp = kwargs.get('force_top_down_grasp', False)
        force_horizontal_grasp = kwargs.get('force_horizontal_grasp', True)
        force_45_angle_grasp = kwargs.get('force_45_angle_grasp', False)
        force_squeeze_grasp = kwargs.get('force_squeeze_grasp', False)
        return_to_initial_pose = kwargs.get('return_to_initial_pose', False)
        
        rospy.loginfo(f"FSM: Starting automated grasp for {object_type} using {image_source}")
        recorder.publish_event('start_automated_grasp')
        
        result = self._call_automated_grasp(
            image_source, 
            object_type, 
            force_top_down_grasp,
            force_horizontal_grasp,
            force_45_angle_grasp,
            force_squeeze_grasp,
            return_to_initial_pose
        )
        
        recorder.publish_event('stop_automated_grasp')
        if result:
            self.finish_automated_grasp()
        else:
            try:
                self.sit_down()
            except Exception:
                rospy.logwarn("FSM: Failed to transition to sit state after grasp failure")

    def on_enter_powered_off_from_stand(self):
        rospy.loginfo("FSM: Powering off robot")
        recorder.publish_event('start_power_off')
        result = self._call_service(self.power_off_srv, "Power off")
        recorder.publish_event('stop_power_off')

    def on_enter_powered_off_from_sit(self):
        rospy.loginfo("FSM: Powering off robot")
        recorder.publish_event('start_power_off')
        result = self._call_service(self.power_off_srv, "Power off")
        recorder.publish_event('stop_power_off')

    def on_enter_disconnected(self):
        rospy.loginfo("FSM: Disconnecting from robot")
        recorder.publish_event('start_disconnect')
        result = self._call_service(self.disconnect_srv, "Disconnect")
        recorder.publish_event('stop_disconnect')


    def send(self, event, **kwargs):
        """Override send method to pass keyword arguments to state entry methods"""
        # Store kwargs for potential use in state entry methods
        self._current_kwargs = kwargs
        return super().send(event)
    
    def _call_get_image_service(self, image_source):
        """Helper to call get image service"""
        try:
            req = GetImageRequest()
            req.image_source = image_source
            resp = self.get_image_srv(req)
            if resp.success:
                mode_text = " (dummy)" if self.dummy_mode else ""
                rospy.loginfo(f"FSM: Image retrieved successfully{mode_text}")
            else:
                mode_text = " (dummy)" if self.dummy_mode else ""
                rospy.logerr(f"FSM: Failed to get image{mode_text}: {resp.message}")
        except Exception as e:
            mode_text = " (dummy)" if self.dummy_mode else ""
            rospy.logerr(f"FSM: Image service call failed{mode_text}: {e}")

    def _call_get_initial_pose_service(self):
        """Helper to call get initial pose service"""
        try:
            req = GetInitialPoseRequest()
            resp = self.get_initial_pose_srv(req)
            if resp.success:
                mode_text = " (dummy)" if self.dummy_mode else ""
                rospy.loginfo(f"FSM: Initial pose retrieved successfully{mode_text}")
            else:
                mode_text = " (dummy)" if self.dummy_mode else ""
                rospy.logerr(f"FSM: Failed to get initial pose{mode_text}: {resp.message}")
        except Exception as e:
            mode_text = " (dummy)" if self.dummy_mode else ""
            rospy.logerr(f"FSM: Initial pose service call failed{mode_text}: {e}")

    def _call_arm_command_service(self, command_type):
        """Helper to call arm command service"""
        try:
            req = ArmCommandRequest()
            req.command_type = command_type
            resp = self.arm_command_srv(req)
            if resp.success:
                mode_text = " (dummy)" if self.dummy_mode else ""
                rospy.loginfo(f"FSM: Arm command '{command_type}' successful{mode_text}")
            else:
                mode_text = " (dummy)" if self.dummy_mode else ""
                rospy.logerr(f"FSM: Arm command '{command_type}' failed{mode_text}: {resp.message}")
        except Exception as e:
            mode_text = " (dummy)" if self.dummy_mode else ""
            rospy.logerr(f"FSM: Arm command service call failed{mode_text}: {e}")
            
    def _call_automated_grasp(self, image_source, object_type, force_top_down_grasp=False, 
                             force_horizontal_grasp=True, force_45_angle_grasp=False, 
                             force_squeeze_grasp=False, return_to_initial_pose=True):
        """Helper to call automated grasp action"""
        # TODO: remove return to initial_pose 
        if self.dummy_mode:
            rospy.loginfo(f"FSM: DUMMY MODE - Simulating automated grasp for {object_type}")
            
            # Simulate object detection and grasping process
            rospy.sleep(1.0)  # Simulate detection time
            
            # In dummy mode, we'll assume the object is found and grasped
            # This matches the standalone behavior where objects are always available
            rospy.loginfo(f"FSM: DUMMY MODE - Object {object_type} detected and grasped")
            rospy.sleep(1.0)  # Simulate grasp time
            
            return True
            
        try:
            # Create goal for automated grasp action
            goal = AutomatedGraspGoal()
            goal.image_source = image_source
            goal.object_type = object_type
            goal.force_top_down_grasp = force_top_down_grasp
            goal.force_horizontal_grasp = force_horizontal_grasp
            goal.force_45_angle_grasp = force_45_angle_grasp
            goal.force_squeeze_grasp = force_squeeze_grasp
            goal.return_to_initial_pose = return_to_initial_pose
            
            # Send goal and wait for result (with timeout)
            rospy.loginfo(f"FSM: Sending automated grasp goal for {object_type}")
            self.automated_grasp_client.send_goal(goal)
            
            # Wait for result with timeout (60 seconds is a reasonable timeout for grasp)
            finished = self.automated_grasp_client.wait_for_result(rospy.Duration(60.0))
            
            if not finished:
                rospy.logerr("FSM: Automated grasp action timed out")
                self.automated_grasp_client.cancel_goal()
                return False
            
            # Get and process result
            result = self.automated_grasp_client.get_result()
            if result and result.success:
                rospy.loginfo(f"FSM: Automated grasp succeeded! Grasped at ({result.selected_pixel_x}, {result.selected_pixel_y})")
                return True
            else:
                rospy.logerr(f"FSM: Automated grasp failed: {result.message if result else 'Unknown error'}")
                return False
                
        except Exception as e:
            rospy.logerr(f"FSM: Automated grasp action failed with exception: {e}")
            return False

    def _call_move_arm_pose_action(self, x, y, z, qw, qx, qy, qz, duration, open_gripper):
        """Helper to call move arm pose action"""
        if self.dummy_mode:
            rospy.loginfo(f"FSM: DUMMY MODE - Simulating move arm pose to ({x}, {y}, {z})")
            rospy.sleep(2.0)  # Simulate time for arm movement in dummy mode
            return True
            
        try:
            # Create goal for move arm pose action
            goal = MoveArmPoseGoal()
            goal.x = x
            goal.y = y
            goal.z = z
            goal.qw = qw
            goal.qx = qx
            goal.qy = qy
            goal.qz = qz
            goal.duration = duration
            goal.open_gripper = open_gripper
            
            # Send goal and wait for result (with timeout)
            rospy.loginfo(f"FSM: Sending move arm pose goal to ({x}, {y}, {z})")
            self.move_arm_pose_client.send_goal(goal)
            
            # Wait for result with timeout (10 seconds is reasonable for arm movement)
            finished = self.move_arm_pose_client.wait_for_result(rospy.Duration(10.0))
            
            if not finished:
                rospy.logerr("FSM: Move arm pose action timed out")
                self.move_arm_pose_client.cancel_goal()
                return False
            
            # Get and process result
            result = self.move_arm_pose_client.get_result()
            if result and result.success:
                rospy.loginfo(f"FSM: Move arm pose succeeded! {result.message}")
                return True
            else:
                rospy.logerr(f"FSM: Move arm pose failed: {result.message if result else 'Unknown error'}")
                return False
                
        except Exception as e:
            rospy.logerr(f"FSM: Move arm pose action failed with exception: {e}")
            return False

    def _call_drop_off_action(self):
        """Helper to call drop_off_object action (move to position, open gripper, close, stow)"""
        if self.dummy_mode:
            rospy.loginfo("FSM: DUMMY MODE - Simulating drop-off sequence")
            rospy.sleep(4.0)  # Simulate time for full drop-off sequence in dummy mode
            return True
            
        try:
            # Create goal for drop_off action
            # Default drop-off position (can be customized via kwargs if needed)
            goal = MoveArmPoseGoal()
            goal.x = 0.8      # 80cm forward
            goal.y = 0.0      # No lateral movement
            goal.z = 0.3      # 30cm above body
            goal.qw = 0.7071  # Gripper pointing down
            goal.qx = 0.7071
            goal.qy = 0.0
            goal.qz = 0.0
            goal.duration = 1.0
            goal.open_gripper = False  # Ignored by drop_off action
            
            # Send goal and wait for result (with timeout)
            rospy.loginfo("FSM: Sending drop-off goal (move + open + close + stow)")
            self.drop_off_client.send_goal(goal)
            
            # Wait for result with longer timeout for full sequence (20 seconds)
            finished = self.drop_off_client.wait_for_result(rospy.Duration(20.0))
            
            if not finished:
                rospy.logerr("FSM: Drop-off action timed out")
                self.drop_off_client.cancel_goal()
                return False
            
            # Get and process result
            result = self.drop_off_client.get_result()
            if result and result.success:
                rospy.loginfo(f"FSM: Drop-off sequence succeeded! {result.message}")
                return True
            else:
                rospy.logerr(f"FSM: Drop-off sequence failed: {result.message if result else 'Unknown error'}")
                return False
                
        except Exception as e:
            rospy.logerr(f"FSM: Drop-off action failed with exception: {e}")
            return False

    # Helper methods
    def _call_service(self, service, name):
        """Helper to call a service and log result"""
        try:
            resp = service()
            if resp.success:
                mode_text = " (dummy)" if self.dummy_mode else ""
                rospy.loginfo(f"FSM: {name} successful{mode_text}")
                return True
            else:
                mode_text = " (dummy)" if self.dummy_mode else ""
                rospy.logerr(f"FSM: {name} failed{mode_text}: {resp.message}")
                return False
        except Exception as e:
            mode_text = " (dummy)" if self.dummy_mode else ""
            rospy.logerr(f"FSM: {name} service call failed{mode_text}: {e}")
            return False

    def get_robot_pose(self):
        """Get current robot pose in vision frame."""
        try:
            req = GetRobotPoseRequest()
            resp = self.get_robot_pose_srv(req)
            return resp
        except Exception as e:
            rospy.logerr(f"FSM: Robot pose service call failed: {e}")
            return None

    def _call_move_service(self, x=None, y=None, yaw=None, frame=None):
        """Helper to call move service with provided or default parameters"""
        try:
            # Use provided parameters or fall back to stored ones
            move_x = x if x is not None else self.move_x
            move_y = y if y is not None else self.move_y
            move_yaw = yaw if yaw is not None else self.move_yaw
            move_frame = frame if frame is not None else self.move_frame
            
            req = MoveToPositionRequest()
            req.target_pose.position.x = move_x
            req.target_pose.position.y = move_y
            req.target_pose.position.z = 0.0
            req.target_pose.orientation.w = math.cos(move_yaw/2)
            req.target_pose.orientation.z = math.sin(move_yaw/2)
            req.frame_name = move_frame
            
            resp = self.move_srv(req)
            if resp.success:
                mode_text = " (dummy)" if self.dummy_mode else ""
                rospy.loginfo(f"FSM: Move successful{mode_text}: {resp.message}")
                return True
            else:
                mode_text = " (dummy)" if self.dummy_mode else ""
                rospy.logerr(f"FSM: Move failed{mode_text}: {resp.message}")
                return False
        except Exception as e:
            mode_text = " (dummy)" if self.dummy_mode else ""
            rospy.logerr(f"FSM: Move service call failed{mode_text}: {e}")
            return False

    def _grid_to_body_frame(self, target_row, target_col):
        """Convert grid coordinates to body-frame movement for real robot."""
        # Grid cell size: 0.3m per cell
        cell_size = 0.3
        
        # Get current robot pose to calculate relative movement
        current_pose = self.get_robot_pose()
        if not current_pose:
            rospy.logerr("FSM: Cannot get current robot pose for grid conversion")
            return None, None, None
        
        # Convert current pose to grid coordinates
        # Service returns a GetRobotPoseResponse with PoseStamped at response.robot_pose
        try:
            current_x = current_pose.robot_pose.pose.position.x
            current_y = current_pose.robot_pose.pose.position.y
        except Exception:
            rospy.logwarn("FSM: Unexpected robot pose format; using zeros for dx/dy")
            current_x, current_y = 0.0, 0.0
        # Initialize grid origin on first use so current real pose maps to configured start cell
        if self._grid_origin_x is None or self._grid_origin_y is None:
            self._grid_origin_x = current_x - (self.grid_start_col * cell_size)
            self._grid_origin_y = current_y - (self.grid_start_row * cell_size)
            rospy.loginfo(f"FSM: Grid origin set to ({self._grid_origin_x:.2f},{self._grid_origin_y:.2f}) so current pose maps to cell ({self.grid_start_row},{self.grid_start_col})")

        # Compute current grid cell using fixed origin
        current_col = round((current_x - self._grid_origin_x) / cell_size)
        current_row = round((current_y - self._grid_origin_y) / cell_size)
        
        # Calculate target position in meters
        target_x = (target_col * cell_size) + self._grid_origin_x
        target_y = (target_row * cell_size) + self._grid_origin_y
        
        # Calculate relative movement in vision frame
        dx_v = target_x - current_x
        dy_v = target_y - current_y

        # Transform vision-frame delta into body frame using current yaw
        try:
            import tf.transformations as tft
            qx = current_pose.robot_pose.pose.orientation.x
            qy = current_pose.robot_pose.pose.orientation.y
            qz = current_pose.robot_pose.pose.orientation.z
            qw = current_pose.robot_pose.pose.orientation.w
            _, _, yaw = tft.euler_from_quaternion((qx, qy, qz, qw))
        except Exception:
            yaw = 0.0

        cos_y = math.cos(-yaw)
        sin_y = math.sin(-yaw)
        dx_b = cos_y * dx_v - sin_y * dy_v
        dy_b = sin_y * dx_v + cos_y * dy_v
        
        rospy.loginfo(f"FSM: Grid movement: ({current_row},{current_col}) -> ({target_row},{target_col})")
        rospy.loginfo(f"FSM: Vision delta: dx={dx_v:.2f}m, dy={dy_v:.2f}m; Body delta: dx={dx_b:.2f}m (forward +), dy={dy_b:.2f}m (left +)")
        
        return dx_b, dy_b, 0.0  # No yaw change for move_to_cell

    def _direction_to_yaw(self, direction):
        """Convert cardinal direction to yaw angle."""
        direction_map = {
            # Convention per user:
            # North (forward) = 0
            # East (right) = -pi/2
            # West (left) = +pi/2
            # South (back) = pi
            'N': 0.0,
            'E': -math.pi / 2,
            'W': math.pi / 2,
            'S': math.pi
        }
        return direction_map.get(direction, 0.0)

    def _call_move_to_cell(self, row, col):
        """Handle move_to_cell command with grid-to-body-frame conversion."""
        if self.dummy_mode:
            # In dummy mode, just simulate the movement
            rospy.loginfo(f"FSM: Dummy move_to_cell to ({row}, {col})")
            rospy.sleep(2.0)  # Simulate movement time
            return True
        else:
            # Convert grid delta to body-frame relative movement (x forward +, y left +)
            cell_size = 0.3
            current_pose = self.get_robot_pose()
            if not current_pose:
                rospy.logerr("FSM: Cannot get current robot pose for grid conversion")
                return False

            try:
                current_x = current_pose.robot_pose.pose.position.x
                current_y = current_pose.robot_pose.pose.position.y
            except Exception:
                current_x, current_y = 0.0, 0.0

            # Initialize grid origin if needed (so current pose aligns to configured start cell)
            if self._grid_origin_x is None or self._grid_origin_y is None:
                self._grid_origin_x = current_x - (self.grid_start_col * cell_size)
                self._grid_origin_y = current_y - (self.grid_start_row * cell_size)
                rospy.loginfo(f"FSM: Grid origin set to ({self._grid_origin_x:.2f},{self._grid_origin_y:.2f}) so current pose maps to cell ({self.grid_start_row},{self.grid_start_col})")

            current_col = round((current_x - self._grid_origin_x) / cell_size)
            current_row = round((current_y - self._grid_origin_y) / cell_size)

            drow = row - current_row
            dcol = col - current_col

            # Map grid deltas to body-frame deltas: forward (row-1) => +x; right (col+1) => -y
            dx_b = (-drow) * cell_size
            dy_b = (-dcol) * cell_size

            rospy.loginfo(f"FSM: Grid movement: ({current_row},{current_col}) -> ({row},{col}) => drow={drow}, dcol={dcol}")
            rospy.loginfo(f"FSM: Body-frame delta: dx={dx_b:.2f}m (forward +), dy={dy_b:.2f}m (left +)")

            return self._call_move_service(x=dx_b, y=dy_b, yaw=0.0, frame="body")

    def _call_rotate_to(self, direction):
        """Handle rotate_to command with direction-to-yaw conversion."""
        if self.dummy_mode:
            # In dummy mode, just simulate the rotation
            rospy.loginfo(f"FSM: Dummy rotate_to {direction}")
            rospy.sleep(1.0)  # Simulate rotation time
            return True
        else:
            # Compute relative yaw needed from current yaw to target cardinal
            try:
                pose_resp = self.get_robot_pose()
                qx = pose_resp.robot_pose.pose.orientation.x
                qy = pose_resp.robot_pose.pose.orientation.y
                qz = pose_resp.robot_pose.pose.orientation.z
                qw = pose_resp.robot_pose.pose.orientation.w
                import tf.transformations as tft
                _, _, current_yaw = tft.euler_from_quaternion((qx, qy, qz, qw))
            except Exception:
                current_yaw = 0.0

            # Initialize yaw origin so startup heading is treated as 0
            if self._yaw_origin is None:
                self._yaw_origin = current_yaw
                rospy.loginfo(f"FSM: Yaw origin set to {self._yaw_origin:.2f} rad (startup heading -> 0)")

            # Work in yaw relative to origin
            current_rel = math.atan2(math.sin(current_yaw - self._yaw_origin), math.cos(current_yaw - self._yaw_origin))
            desired_rel = self._direction_to_yaw(direction)
            delta = desired_rel - current_rel
            # Normalize to [-pi, pi]
            delta = math.atan2(math.sin(delta), math.cos(delta))

            rospy.loginfo(f"FSM: Rotating (rel): current={current_rel:.2f} -> desired={desired_rel:.2f} (delta={delta:.2f}) for {direction}")

            # Use body frame for relative rotation
            return self._call_move_service(x=0.0, y=0.0, yaw=delta, frame="body")


if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node('spot_finite_state_machine')
    
    # Create FSM (will auto-detect dummy mode from parameter)
    spot = SpotStateMachine()
    
    if spot.dummy_mode:
        rospy.loginfo("Spot Finite State Machine (DUMMY) started and ready for commands")
    else:
        rospy.loginfo("Spot Finite State Machine (REAL) started and ready for commands")
    
    rospy.loginfo("Use test_real_spot_manager.py to test the FSM")
    
    # Keep the node running
    rospy.spin()

