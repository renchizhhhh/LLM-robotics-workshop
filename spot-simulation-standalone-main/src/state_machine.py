#!/usr/bin/env python3
"""
Finite State Machine for Spot Robot Control (Standalone Version)
"""

import math
import time
import json
import sys
import os
from typing import Optional
from statemachine import StateMachine, State

# Import our standalone modules
from timing_utils import recorder
from robot_simulator import robot_simulator
from message_bus import rospy, Publisher, String
from pose_tracker import PoseTracker, GridFrameMapper


class SpotStateMachine(StateMachine):
    """
    Minimal state machine for Spot robot control.
    Acts as a bridge between high-level commands and robot simulator.
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
    rotating = State()
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
    sit_down = (stand.to(sit) | moving.to(sit) | grasping.to(sit))
    start_moving = (stand.to(moving) | moving.to(moving) | carry.to(moving))
    stop_moving = moving.to(stand)
    start_rotating = (stand.to(rotating) | moving.to(rotating) | carry.to(rotating))
    stop_rotating = rotating.to(stand)
    get_image = (stand.to(stand) | moving.to(moving) | sit.to(sit))
    get_initial_pose = (stand.to(stand) | moving.to(moving) | sit.to(sit))
    start_arm_command = (stand.to(arm_command) | moving.to(arm_command) | carry.to(arm_command))
    finish_arm_command = (arm_command.to(stand))
    start_automated_grasp = (stand.to(grasping) | moving.to(grasping) | rotating.to(grasping) | carry.to(grasping))
    finish_automated_grasp = (grasping.to(carry))
    start_move_arm_pose = (stand.to(move_arm_pose) | moving.to(move_arm_pose) | carry.to(move_arm_pose))
    finish_move_arm_pose = (move_arm_pose.to(stand))
    start_drop_off = (stand.to(move_arm_pose) | moving.to(move_arm_pose) | carry.to(move_arm_pose) | rotating.to(move_arm_pose))
    finish_drop_off = (move_arm_pose.to(stand))
    carry_to_stand = carry.to(stand)
    stand_to_carry = stand.to(carry)
    power_off_from_stand = stand.to(powered_off)
    power_off_from_sit = sit.to(powered_off)
    disconnect = powered_off.to(disconnected)
    def __init__(self, dummy_mode=None, pose_tracker: Optional[PoseTracker] = None):
        self.dummy_mode = self._detect_robot_mode(dummy_mode)
        self.pose_tracker = pose_tracker or PoseTracker(grid_mapper=GridFrameMapper())
        self.grid_mapper = self.pose_tracker.grid_mapper

        # Initialize feedback publisher
        self.feedback_pub = Publisher('/spot/execution_feedback')
        
        if self.dummy_mode:
            rospy.loginfo("FSM: Running in SIMULATION MODE - using robot simulator")
            # In dummy mode, we need to publish robot state ourselves
            self.robot_state_pub = Publisher('/spot_entrance/robot_state')
            self._start_robot_state_publishing()
            # Initialize action clients to None for simulation mode
            self.automated_grasp_client = None
            self.drop_off_client = None
        else:
            rospy.loginfo("FSM: Running in REAL ROBOT MODE - using ROS services")
            self._setup_real_robot_services()
            # In real robot mode, the spot_entrance service publishes robot state
            # We don't need to publish our own state
        
        # Initialize the state machine after setting up services
        super().__init__()
        self._startup_probe(timeout=3.0)
        
        # Default movement parameters (used as fallback)
        self.move_x = 0.0
        self.move_y = 0.0
        self.move_yaw = 0.0
        self.move_frame = "vision"
        
        # Initialize yaw origin for rotation tracking
        self._yaw_origin = None

    def _start_robot_state_publishing(self):
        """Start publishing robot state updates (dummy mode only)."""
        import threading
        import time
        
        def publish_state():
            while not rospy.is_shutdown():
                try:
                    # Only publish state in dummy mode - in real robot mode, rely on spot_entrance service
                    if self.dummy_mode:
                        # Check if real robot services are available (even in dummy mode)
                        try:
                            rospy.wait_for_service('/spot_entrance/get_robot_pose', timeout=0.1)
                            # Real robot services are available, don't publish our state
                            time.sleep(1.0)
                            continue
                        except rospy.ROSException:
                            # No real robot services, we can publish our state
                            pass
                        
                        # Use FSM state for dummy mode
                        current_state = str(self.current_state)
                        connected = current_state not in ["unknown", "disconnected"]
                        powered = current_state not in ["unknown", "disconnected", "connected"]
                        standing = current_state in ["stand", "moving", "rotating", "grasping", "carry", "move_arm_pose", "arm_command"]
                        
                        # Publish robot state
                        state_msg = f"connected:{str(connected).lower()},powered:{str(powered).lower()},standing:{str(standing).lower()}"
                        self.robot_state_pub.publish(String(data=state_msg))
                        
                        # Publish feedback for state changes
                        current_state_str = f"connected={connected}, powered={powered}, standing={standing}"
                        if hasattr(self, '_last_published_state') and self._last_published_state != current_state_str:
                            self.feedback_pub.publish(String(data=f"Robot state: {current_state_str}"))
                            self._last_published_state = current_state_str
                    else:
                        # In real robot mode, don't publish our own state - let spot_entrance handle it
                        # Just sleep to avoid busy waiting
                        time.sleep(1.0)
                    
                except Exception as e:
                    rospy.logwarn(f"Failed to publish robot state: {e}")
                
                time.sleep(1.0)  # Update every second
        
        # Start state publishing in background thread
        self.state_thread = threading.Thread(target=publish_state, daemon=True)
        self.state_thread.start()

    def _detect_robot_mode(self, dummy_mode):
        """Detect if real robot services are available."""
        if dummy_mode is not None:
            return dummy_mode
        
        # Check if ROS services are available
        try:
            from message_bus import is_ros_master_available
            if not is_ros_master_available():
                return True
            
            # Check for key robot services
            services_to_check = [
                '/spot_entrance/connect',
                '/spot_entrance/stand',
                '/spot_entrance/move_to_position'
            ]
            
            for service in services_to_check:
                try:
                    import rospy as real_rospy
                    real_rospy.wait_for_service(service, timeout=1.0)
                except Exception:
                    rospy.loginfo(f"FSM: Service {service} not available - using simulation mode")
                    return True
            
            rospy.loginfo("FSM: Real robot services detected - using real robot mode")
            return False
            
        except Exception as e:
            rospy.loginfo(f"FSM: Error detecting robot services: {e} - using simulation mode")
            return True

    def _setup_real_robot_services(self):
        """Setup service proxies for real robot control."""
        try:
            from std_srvs.srv import Trigger
            from spot_hololens_llm_interface.srv import MoveToPosition, GetImage, GetInitialPose, GetRobotPose, ArmCommand
            
            # Basic robot control services
            self.connect_srv = rospy.ServiceProxy('/spot_entrance/connect', Trigger)
            self.power_on_srv = rospy.ServiceProxy('/spot_entrance/power_on', Trigger)
            self.stand_srv = rospy.ServiceProxy('/spot_entrance/stand', Trigger)
            self.sit_srv = rospy.ServiceProxy('/spot_entrance/sit', Trigger)
            
            # Movement and manipulation services
            self.move_srv = rospy.ServiceProxy('/spot_entrance/move_to_position', MoveToPosition)
            self.get_image_srv = rospy.ServiceProxy('/spot_entrance/get_image', GetImage)
            self.get_initial_pose_srv = rospy.ServiceProxy('/spot_entrance/get_initial_pose', GetInitialPose)
            self.get_robot_pose_srv = rospy.ServiceProxy('/spot_entrance/get_robot_pose', GetRobotPose)
            self.arm_command_srv = rospy.ServiceProxy('/spot_entrance/arm_command', ArmCommand)
            
            rospy.loginfo("FSM: Real robot service proxies created")
            
            # Setup action clients for automated grasp and drop-off
            try:
                import actionlib
                from spot_hololens_llm_interface.msg import AutomatedGraspAction, AutomatedGraspGoal, MoveArmPoseAction, MoveArmPoseGoal
                
                # Import real ROS rospy for Duration objects (needed for actionlib compatibility)
                import rospy as ros_rospy
                
                rospy.loginfo("FSM: Creating action clients...")
                self.automated_grasp_client = actionlib.SimpleActionClient('automated_grasp', AutomatedGraspAction)
                self.drop_off_client = actionlib.SimpleActionClient('drop_off_object', MoveArmPoseAction)
                rospy.loginfo("FSM: Action clients created successfully")
                
                # Wait for action servers (with longer timeout to allow spot_entrance to initialize)
                rospy.loginfo("FSM: Waiting for action servers (this may take 10-15 seconds if spot_entrance is starting)...")
                rospy.loginfo("FSM: Checking for automated_grasp action server...")
                # Use real ROS Duration for actionlib compatibility
                try:
                    automated_grasp_connected = self.automated_grasp_client.wait_for_server(ros_rospy.Duration(15.0))
                except Exception as wait_error:
                    rospy.logerr(f"FSM: Error waiting for automated_grasp server: {wait_error}")
                    rospy.logerr(f"FSM: Error type: {type(wait_error)}")
                    import traceback
                    rospy.logerr(f"FSM: Traceback: {traceback.format_exc()}")
                    automated_grasp_connected = False
                if automated_grasp_connected:
                    rospy.loginfo("FSM: ✓ Connected to automated_grasp action server")
                else:
                    rospy.logwarn("FSM: ✗ automated_grasp action server not available after 15s wait")
                    rospy.logwarn("FSM: Make sure spot_entrance node has arm_action_server=true in launch file")
                    rospy.logwarn("FSM: Check spot_entrance logs for 'Spot Robot Manager initialized' message")
                    rospy.logwarn("FSM: Action client created but will retry when action is called")
                    # Keep client so we can retry later - don't set to None
                
                rospy.loginfo("FSM: Checking for drop_off_object action server...")
                try:
                    drop_off_connected = self.drop_off_client.wait_for_server(ros_rospy.Duration(15.0))
                except Exception as wait_error:
                    rospy.logerr(f"FSM: Error waiting for drop_off_object server: {wait_error}")
                    rospy.logerr(f"FSM: Error type: {type(wait_error)}")
                    import traceback
                    rospy.logerr(f"FSM: Traceback: {traceback.format_exc()}")
                    drop_off_connected = False
                if drop_off_connected:
                    rospy.loginfo("FSM: ✓ Connected to drop_off_object action server")
                else:
                    rospy.logwarn("FSM: ✗ drop_off_object action server not available after 15s wait")
                    rospy.logwarn("FSM: Make sure spot_entrance node has arm_action_server=true in launch file")
                    rospy.logwarn("FSM: Check spot_entrance logs for 'Spot Robot Manager initialized' message")
                    rospy.logwarn("FSM: Action client created but will retry when action is called")
                    # Keep client so we can retry later - don't set to None
                    
            except ImportError as e:
                rospy.logwarn(f"FSM: Could not import action types: {e} - will use simulator for grasp/drop_off")
                self.automated_grasp_client = None
                self.drop_off_client = None
            except Exception as e:
                rospy.logwarn(f"FSM: Failed to setup action clients: {e} - will use simulator for grasp/drop_off")
                self.automated_grasp_client = None
                self.drop_off_client = None
            
        except Exception as e:
            rospy.logerr(f"FSM: Failed to setup real robot services: {e}")
            self.dummy_mode = True  # Fallback to simulation mode
            self.automated_grasp_client = None
            self.drop_off_client = None

    def _move_to_cell_real_robot(self, target_row, target_col, current_row=None, current_col=None):
        """Convert grid movement to real robot body frame movement."""
        try:
            from spot_hololens_llm_interface.srv import MoveToPositionRequest

            # Ensure the mapper knows about NL control offsets if they were published
            if not self.grid_mapper.has_offsets():
                try:
                    row_offset = rospy.get_param('/nl_control/grid_offset_row', None)
                    col_offset = rospy.get_param('/nl_control/grid_offset_col', None)
                    if row_offset is not None and col_offset is not None:
                        self.grid_mapper.set_offsets(row_offset, col_offset)
                        rospy.loginfo(
                            f"FSM: Loaded grid offsets from params ({float(row_offset):.2f}, {float(col_offset):.2f})"
                        )
                except Exception as param_error:
                    rospy.logwarn(f"FSM: Unable to read NL Control grid offsets: {param_error}")

            pose_response = self.get_robot_pose_srv()
            if not pose_response.success:
                rospy.logerr("FSM: Failed to get current robot pose")
                return {"success": False, "message": pose_response.message}

            current_pose = pose_response.robot_pose.pose

            # Establish offsets from the live pose if they are still missing
            offsets_initialised = self.grid_mapper.ensure_offsets_from_pose(current_pose)
            if offsets_initialised:
                rospy.loginfo(
                    f"FSM: Initialised grid offsets from pose ({float(self.grid_mapper.grid_offset_row):.2f}, "
                    f"{float(self.grid_mapper.grid_offset_col):.2f})"
                )

            # Convert the target cell into body-frame deltas
            body_x, body_y = self.grid_mapper.grid_to_body_delta(target_row, target_col, current_pose)

            # Log current estimate for visibility when NL control passes a tracked position
            try:
                current_row_est, current_col_est = self.grid_mapper.vision_xy_to_grid(
                    current_pose.position.x, current_pose.position.y
                )
            except ValueError:
                current_row_est, current_col_est = float('nan'), float('nan')

            rospy.loginfo(
                f"FSM: Grid movement: ({current_row_est:.2f}, {current_col_est:.2f}) -> "
                f"({float(target_row):.2f}, {float(target_col):.2f}) | Body delta: x={body_x:.2f}m, y={body_y:.2f}m"
            )

            request = MoveToPositionRequest()
            request.target_pose.position.x = body_x
            request.target_pose.position.y = body_y
            request.target_pose.position.z = 0.0
            request.target_pose.orientation.w = 1.0
            request.target_pose.orientation.x = 0.0
            request.target_pose.orientation.y = 0.0
            request.target_pose.orientation.z = 0.0
            request.frame_name = "body"

            result = self.move_srv(request)
            return {"success": result.success, "message": result.message}

        except Exception as exc:
            rospy.logerr(f"FSM: Real robot grid movement failed: {exc}")
            return {"success": False, "message": str(exc)}

    def _move_real_robot(self, x, y, yaw, frame):
        """Move real robot using body frame coordinates."""
        try:
            from spot_hololens_llm_interface.srv import MoveToPositionRequest
            import math
            rospy.loginfo(f"FSM: _move_real_robot request -> frame={frame}, x={x:.3f}, y={y:.3f}, yaw={yaw:.3f} rad")
            req = MoveToPositionRequest()
            req.target_pose.position.x = x
            req.target_pose.position.y = y
            req.target_pose.position.z = 0.0
            req.target_pose.orientation.w = math.cos(yaw/2)
            req.target_pose.orientation.x = 0.0
            req.target_pose.orientation.y = 0.0
            req.target_pose.orientation.z = math.sin(yaw/2)
            req.frame_name = frame
            
            result = self.move_srv(req)
            try:
                rospy.loginfo(f"FSM: _move_real_robot response -> success={result.success}, message={getattr(result, 'message', '')}")
            except Exception:
                pass
            return {"success": result.success, "message": result.message}
            
        except Exception as e:
            rospy.logerr(f"FSM: Real robot movement failed: {e}")
            return {"success": False, "message": str(e)}

    def _rotate_to_direction_real_robot(self, direction):
        """Rotate in place to face a cardinal direction without translating.

        Implementation notes:
        - Use body-frame rotation so x=y=0 implies no translation.
        - Compute the delta from current relative yaw to the desired cardinal yaw.
        - If current yaw isn't available, fall back to using the target relative yaw as the command.
        """
        try:
            import math

            # Desired facing relative to yaw origin (grid-aligned):
            direction_to_rel_yaw = {
                'N': 0.0,
                'E': -math.pi / 2,
                'S': math.pi,
                'W': math.pi / 2,
            }

            if direction not in direction_to_rel_yaw:
                return {"success": False, "message": f"Invalid direction: {direction}"}

            target_rel = float(direction_to_rel_yaw[direction])

            # Current relative yaw from pose tracker, default 0.0 if missing
            current_rel = 0.0
            pt = getattr(self, 'pose_tracker', None)
            if pt is not None:
                try:
                    current_rel = float(getattr(pt, 'current_yaw_rel', 0.0))
                except Exception:
                    current_rel = 0.0

            def _wrap(a: float) -> float:
                return math.atan2(math.sin(a), math.cos(a))

            delta_yaw = _wrap(target_rel - current_rel)
            rospy.loginfo(
                f"FSM: Rotating to direction {direction} using body frame: "
                f"current_rel={current_rel:.2f} rad, target_rel={target_rel:.2f} rad, delta={delta_yaw:.2f} rad"
            )

            # Body frame so the service treats (x=0,y=0,yaw=delta) as an in-place turn
            return self._move_real_robot(0.0, 0.0, delta_yaw, "body")

        except Exception as e:
            rospy.logerr(f"FSM: Real robot rotation failed: {e}")
            return {"success": False, "message": str(e)}

    def _startup_probe(self, timeout=3.0):
        """Check robot state and align FSM to the most specific state."""
        try:
            rospy.loginfo('FSM: Standalone mode - simulating full startup sequence')
            self.send('discover_connected')
            # Auto-power on in standalone mode for easier testing
            rospy.sleep(0.1)  # Small delay to ensure state transition
            self.send('power_on')
            return
        except Exception as e:
            rospy.logwarn(f'FSM: Startup probe failed: {e} - staying in unknown')

    # State entry methods - call robot simulator when states change
    def on_enter_connected(self):
        rospy.loginfo("FSM: Connecting to robot")
        if self.dummy_mode:
            result = robot_simulator.connect()
        else:
            try:
                result = self.connect_srv()
                if not result.success:
                    rospy.logerr(f"FSM: Connect failed: {result.message}")
            except Exception as e:
                rospy.logerr(f"FSM: Connect service call failed: {e}")

    def on_enter_powered_off(self):
        rospy.loginfo("FSM: Powering on robot")
        if self.dummy_mode:
            result = robot_simulator.power_on()
        else:
            try:
                result = self.power_on_srv()
                if not result.success:
                    rospy.logerr(f"FSM: Power on failed: {result.message}")
            except Exception as e:
                rospy.logerr(f"FSM: Power on service call failed: {e}")

    def on_enter_stand(self):
        rospy.loginfo("FSM: Standing up")
        if self.dummy_mode:
            # Skip if simulator already considers the robot standing
            if getattr(robot_simulator, 'standing', False):
                rospy.loginfo("FSM: Robot already standing - skipping stand command")
                return
            recorder.publish_event('start_stand_up')
            result = robot_simulator.stand()
            recorder.publish_event('stop_stand_up')
        else:
            try:
                recorder.publish_event('start_stand_up')
                result = self.stand_srv()
                recorder.publish_event('stop_stand_up')
                if not result.success:
                    rospy.logerr(f"FSM: Stand failed: {result.message}")
            except Exception as e:
                rospy.logerr(f"FSM: Stand service call failed: {e}")

    def on_enter_sit(self):
        rospy.loginfo("FSM: Sitting down")
        if self.dummy_mode:
            recorder.publish_event('start_sit_down')
            result = robot_simulator.sit()
            recorder.publish_event('stop_sit_down')
        else:
            try:
                recorder.publish_event('start_sit_down')
                result = self.sit_srv()
                recorder.publish_event('stop_sit_down')
                if not result.success:
                    rospy.logerr(f"FSM: Sit failed: {result.message}")
            except Exception as e:
                rospy.logerr(f"FSM: Sit service call failed: {e}")

    def on_enter_carry(self):
        rospy.loginfo("FSM: Moving to carry position")
        recorder.publish_event('start_carry')
        robot_simulator.arm_command("carry")
        recorder.publish_event('stop_carry')

    def on_enter_moving(self):
        # Get parameters from kwargs if available, otherwise use stored ones
        kwargs = getattr(self, '_current_kwargs', {})
        
        # Check if this is a grid-based movement (new system)
        if 'row' in kwargs and 'col' in kwargs:
            target_row = kwargs.get('row')
            target_col = kwargs.get('col')
            current_row = kwargs.get('current_row')
            current_col = kwargs.get('current_col')
            rospy.loginfo(f"FSM: Moving to cell ({target_row}, {target_col})")
            recorder.publish_event('start_moving')
            
            if self.dummy_mode:
                result = robot_simulator.move_to_cell(target_row, target_col)
            else:
                # Convert grid coordinates to body frame for real robot
                result = self._move_to_cell_real_robot(target_row, target_col, current_row, current_col)
            
            recorder.publish_event('stop_moving')
        else:
            # Fallback to old coordinate system
            move_x = kwargs.get('x', self.move_x)
            move_y = kwargs.get('y', self.move_y)
            move_yaw = kwargs.get('yaw', self.move_yaw)
            move_frame = kwargs.get('frame', self.move_frame)
            
            rospy.loginfo(f"FSM: Moving x={move_x}, y={move_y}, yaw={move_yaw}")
            recorder.publish_event('start_moving')
            
            if self.dummy_mode:
                # Check if move_to_position method exists, otherwise use move_to_cell
                if hasattr(robot_simulator, 'move_to_position'):
                    result = robot_simulator.move_to_position(move_x, move_y, move_yaw, move_frame)
                else:
                    rospy.logwarn("FSM: move_to_position not available, using move_to_cell fallback")
                    result = {"success": False, "message": "Grid-based movement not supported in this mode"}
            else:
                # Use real robot service for body frame movement
                result = self._move_real_robot(move_x, move_y, move_yaw, move_frame)
            
            recorder.publish_event('stop_moving')
    
    def on_enter_rotating(self):
        # Get parameters from kwargs if available
        kwargs = getattr(self, '_current_kwargs', {})
        
        # Check if this is a grid-based rotation (new system)
        if 'direction' in kwargs:
            direction = kwargs.get('direction')
            rospy.loginfo(f"FSM: Rotating to face {direction}")
            recorder.publish_event('start_rotating')
            
            if self.dummy_mode:
                result = robot_simulator.rotate_to_direction(direction)
            else:
                # Convert direction to yaw rotation for real robot
                result = self._rotate_to_direction_real_robot(direction)
            
            recorder.publish_event('stop_rotating')
            if isinstance(result, dict) and result.get('success'):
                # Automatically transition back to stand once rotation completes
                try:
                    self.send('stop_rotating')
                except Exception as transition_error:
                    rospy.logerr(f"FSM: Failed to stop rotating after success: {transition_error}")
            else:
                rospy.logwarn(f"FSM: Rotation reported failure, staying in rotating: {result}")
        else:
            # Fallback to old coordinate system
            move_yaw = kwargs.get('yaw', 0.0)
            rospy.loginfo(f"FSM: Rotating yaw={move_yaw}")
            recorder.publish_event('start_rotating')
            
            if self.dummy_mode:
                rospy.logwarn("FSM: Grid-based rotation not supported in this mode")
                result = {"success": False, "message": "Grid-based rotation not supported in this mode"}
            else:
                # Use real robot rotation
                result = self._move_real_robot(0.0, 0.0, move_yaw, "vision")
            
            recorder.publish_event('stop_rotating')
    
    def on_enter_get_image(self):
        # Get image source from kwargs, default to frontleft_fisheye_image
        kwargs = getattr(self, '_current_kwargs', {})
        image_source = kwargs.get('image_source', 'frontleft_fisheye_image')
        
        rospy.loginfo(f"FSM: Getting image from {image_source}")
        recorder.publish_event('start_get_image')
        robot_simulator.get_image(image_source)
        recorder.publish_event('stop_get_image')

    def on_enter_get_initial_pose(self):
        rospy.loginfo("FSM: Getting initial pose")
        recorder.publish_event('start_get_initial_pose')
        robot_simulator.get_initial_pose()
        recorder.publish_event('stop_get_initial_pose')

    def on_enter_arm_command(self):
        # Get command type from kwargs, default to "stow"
        kwargs = getattr(self, '_current_kwargs', {})
        command_type = kwargs.get('command_type', 'stow')
        
        rospy.loginfo(f"FSM: Executing arm command: {command_type}")
        recorder.publish_event('start_arm_command')
        robot_simulator.arm_command(command_type)
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
            
            # Try to use real robot action client if available, otherwise use simulator
            has_client = hasattr(self, 'drop_off_client') and self.drop_off_client is not None
            # Check if client is actually connected to a server
            server_connected = False
            if has_client:
                try:
                    # Use real ROS Duration for actionlib compatibility
                    import rospy as ros_rospy
                    # Check if server is available (quick check, don't wait)
                    server_connected = self.drop_off_client.wait_for_server(ros_rospy.Duration(0.1))
                except:
                    server_connected = False
            
            rospy.loginfo(f"FSM: dummy_mode={self.dummy_mode}, has_drop_off_client={has_client}, server_connected={server_connected}")
            
            if not self.dummy_mode and has_client and server_connected:
                rospy.loginfo(f"FSM: Using real robot action client for drop-off")
                result = self._call_drop_off_action()
            else:
                if not self.dummy_mode and has_client and not server_connected:
                    rospy.logwarn(f"FSM: Action client exists but server not connected, retrying...")
                    # Try to reconnect
                    result = self._call_drop_off_action()
                else:
                    rospy.loginfo(f"FSM: Using simulator for drop-off (dummy_mode={self.dummy_mode}, has_client={has_client}, server_connected={server_connected})")
            result = robot_simulator.drop_off()
            
            recorder.publish_event('stop_drop_off')
            self.finish_drop_off()
        else:
            # Regular move_arm_pose action - simplified for standalone
            rospy.loginfo("FSM: Moving arm to pose")
            recorder.publish_event('start_move_arm_pose')
            # In standalone mode, we don't need complex arm positioning
            rospy.sleep(1.0)  # Simulate arm movement time
            recorder.publish_event('stop_move_arm_pose')
            
            # Always return to stand state after move_arm_pose
            self.finish_move_arm_pose()

    def on_enter_grasping(self):
        # Get parameters from kwargs
        kwargs = getattr(self, '_current_kwargs', {})
        object_type = kwargs.get('object_type', 'cup')
        
        rospy.loginfo(f"FSM: Starting automated grasp for {object_type}")
        recorder.publish_event('start_automated_grasp')
        
        # Try to use real robot action client if available, otherwise use simulator
        has_client = hasattr(self, 'automated_grasp_client') and self.automated_grasp_client is not None
        # Check if client is actually connected to a server
        server_connected = False
        if has_client:
            try:
                # Use real ROS Duration for actionlib compatibility
                import rospy as ros_rospy
                # Check if server is available (quick check, don't wait)
                server_connected = self.automated_grasp_client.wait_for_server(ros_rospy.Duration(0.1))
            except:
                server_connected = False
        
        rospy.loginfo(f"FSM: dummy_mode={self.dummy_mode}, has_client={has_client}, server_connected={server_connected}")
        
        if not self.dummy_mode and has_client and server_connected:
            rospy.loginfo(f"FSM: Using real robot action client for automated grasp")
            result = self._call_automated_grasp_action(object_type)
        else:
            if not self.dummy_mode and has_client and not server_connected:
                rospy.logwarn(f"FSM: Action client exists but server not connected, retrying...")
                # Try to reconnect
                result = self._call_automated_grasp_action(object_type)
            else:
                rospy.loginfo(f"FSM: Using simulator for automated grasp (dummy_mode={self.dummy_mode}, has_client={has_client}, server_connected={server_connected})")
        result = robot_simulator.automated_grasp(object_type)
        
        recorder.publish_event('stop_automated_grasp')
        
        # Always transition to carry state after grasping
        self.finish_automated_grasp()

    def _call_service(self, service, operation_name):
        """Call a robot simulator service."""
        try:
            if operation_name == "Connect":
                return robot_simulator.connect()
            elif operation_name == "Power on":
                return robot_simulator.power_on()
            elif operation_name == "Stand":
                return robot_simulator.stand()
            elif operation_name == "Sit":
                return robot_simulator.sit()
            elif operation_name == "Power off":
                return robot_simulator.power_off()
            elif operation_name == "Disconnect":
                return robot_simulator.disconnect()
            else:
                rospy.logwarn(f"Unknown service operation: {operation_name}")
                return {"success": False, "message": f"Unknown operation: {operation_name}"}
        except Exception as e:
            rospy.logerr(f"Service call failed for {operation_name}: {e}")
            return {"success": False, "message": str(e)}

    def _call_move_service(self, x, y, yaw, frame):
        """Call move service with parameters."""
        try:
            return robot_simulator.move_to_position(x, y, yaw, frame)
        except Exception as e:
            rospy.logerr(f"Move service call failed: {e}")
            return {"success": False, "message": str(e)}

    def _call_arm_command_service(self, command_type):
        """Call arm command service."""
        try:
            return robot_simulator.arm_command(command_type)
        except Exception as e:
            rospy.logerr(f"Arm command service call failed: {e}")
            return {"success": False, "message": str(e)}

    def _call_get_image_service(self, image_source):
        """Call get image service."""
        try:
            return robot_simulator.get_image(image_source)
        except Exception as e:
            rospy.logerr(f"Get image service call failed: {e}")
            return {"success": False, "message": str(e)}

    def _call_get_initial_pose_service(self):
        """Call get initial pose service."""
        try:
            return robot_simulator.get_initial_pose()
        except Exception as e:
            rospy.logerr(f"Get initial pose service call failed: {e}")
            return {"success": False, "message": str(e)}

    def _call_automated_grasp(self, object_type):
        """Call automated grasp service (deprecated - use _call_automated_grasp_action)."""
        try:
            return robot_simulator.automated_grasp(object_type)
        except Exception as e:
            rospy.logerr(f"Automated grasp service call failed: {e}")
            return {"success": False, "message": str(e)}
    
    def _call_automated_grasp_action(self, object_type):
        """Call automated grasp action using real ROS action server or simulator."""
        # Fallback to simulator if no action client available
        if self.dummy_mode:
            rospy.loginfo(f"FSM: Using simulator for automated grasp of {object_type} (dummy_mode=True)")
            return robot_simulator.automated_grasp(object_type)
        
        # Try to create or reconnect to action client if needed
        if not hasattr(self, 'automated_grasp_client') or self.automated_grasp_client is None:
            rospy.logwarn("FSM: automated_grasp action client not initialized, attempting to create...")
            try:
                import actionlib
                from spot_hololens_llm_interface.msg import AutomatedGraspAction
                self.automated_grasp_client = actionlib.SimpleActionClient('automated_grasp', AutomatedGraspAction)
                rospy.loginfo("FSM: Created automated_grasp action client, checking for server...")
            except Exception as e:
                rospy.logerr(f"FSM: Failed to create automated_grasp action client: {e}")
                rospy.logwarn("FSM: Falling back to simulator for automated grasp")
                return robot_simulator.automated_grasp(object_type)
        
        # Check if action server is actually available (with longer timeout for retry)
        rospy.loginfo("FSM: Waiting for automated_grasp action server (up to 10s)...")
        # Use real ROS Duration for actionlib compatibility
        import rospy as ros_rospy
        if not self.automated_grasp_client.wait_for_server(ros_rospy.Duration(10.0)):
            rospy.logerr("FSM: automated_grasp action server not available after 10s wait")
            rospy.logerr("FSM: Make sure spot_entrance node is running with arm_action_server=true")
            rospy.logerr("FSM: You can check with: rostopic list | grep automated_grasp")
            rospy.logwarn("FSM: Falling back to simulator for automated grasp")
            return robot_simulator.automated_grasp(object_type)
        
        rospy.loginfo("FSM: automated_grasp action server is available, proceeding with real robot grasp")
        
        try:
            from spot_hololens_llm_interface.msg import AutomatedGraspGoal
            
            # Create goal for automated grasp action
            goal = AutomatedGraspGoal()
            goal.image_source = "hand_color_image"  # Default image source
            goal.object_type = object_type
            goal.force_top_down_grasp = False
            goal.force_horizontal_grasp = True
            goal.force_45_angle_grasp = False
            goal.force_squeeze_grasp = False
            goal.return_to_initial_pose = True
            
            # Send goal and wait for result (with timeout)
            rospy.loginfo(f"FSM: Sending automated grasp goal for {object_type}")
            rospy.loginfo(f"FSM: Robot will capture image from current arm position, detect object with Gemini AI, and execute grasp")
            self.automated_grasp_client.send_goal(goal)
            
            # Wait for result with timeout (60 seconds is reasonable for grasp)
            # Use real ROS Duration for actionlib compatibility
            import rospy as ros_rospy
            finished = self.automated_grasp_client.wait_for_result(ros_rospy.Duration(60.0))
            
            if not finished:
                rospy.logerr("FSM: Automated grasp action timed out")
                self.automated_grasp_client.cancel_goal()
                return {"success": False, "message": "Automated grasp timed out"}
            
            # Get and process result
            result = self.automated_grasp_client.get_result()
            if result and result.success:
                rospy.loginfo(f"FSM: Automated grasp succeeded! Grasped at ({result.selected_pixel_x}, {result.selected_pixel_y})")
                return {"success": True, "message": f"Grasped {object_type}"}
            else:
                rospy.logerr(f"FSM: Automated grasp failed: {result.message if result else 'Unknown error'}")
                return {"success": False, "message": result.message if result else "Unknown error"}
                
        except Exception as e:
            rospy.logerr(f"FSM: Automated grasp action failed with exception: {e}")
            # Fallback to simulator on error
            rospy.logwarn("FSM: Falling back to simulator for automated grasp")
            return robot_simulator.automated_grasp(object_type)

    def _call_move_arm_pose_action(self, x, y, z, qw, qx, qy, qz, duration, open_gripper):
        """Call move arm pose action."""
        try:
            # Simplified for standalone mode
            rospy.sleep(duration)
            return {"success": True, "message": "Arm moved to pose"}
        except Exception as e:
            rospy.logerr(f"Move arm pose action failed: {e}")
            return {"success": False, "message": str(e)}

    def _call_drop_off_action(self):
        """Call drop off action using real ROS action server or simulator."""
        # Fallback to simulator if no action client available
        if self.dummy_mode:
            rospy.loginfo("FSM: Using simulator for drop-off (dummy_mode=True)")
            return robot_simulator.drop_off()
        
        # Try to create or reconnect to action client if needed
        if not hasattr(self, 'drop_off_client') or self.drop_off_client is None:
            rospy.logwarn("FSM: drop_off_object action client not initialized, attempting to create...")
            try:
                import actionlib
                from spot_hololens_llm_interface.msg import MoveArmPoseAction
                self.drop_off_client = actionlib.SimpleActionClient('drop_off_object', MoveArmPoseAction)
                rospy.loginfo("FSM: Created drop_off_object action client, checking for server...")
            except Exception as e:
                rospy.logerr(f"FSM: Failed to create drop_off_object action client: {e}")
                rospy.logwarn("FSM: Falling back to simulator for drop-off")
                return robot_simulator.drop_off()
        
        # Check if action server is actually available (with longer timeout for retry)
        rospy.loginfo("FSM: Waiting for drop_off_object action server (up to 10s)...")
        # Use real ROS Duration for actionlib compatibility
        import rospy as ros_rospy
        if not self.drop_off_client.wait_for_server(ros_rospy.Duration(10.0)):
            rospy.logerr("FSM: drop_off_object action server not available after 10s wait")
            rospy.logerr("FSM: Make sure spot_entrance node is running with arm_action_server=true")
            rospy.logerr("FSM: You can check with: rostopic list | grep drop_off_object")
            rospy.logwarn("FSM: Falling back to simulator for drop-off")
            return robot_simulator.drop_off()
        
        rospy.loginfo("FSM: drop_off_object action server is available, proceeding with real robot drop-off")
        
        try:
            from spot_hololens_llm_interface.msg import MoveArmPoseGoal
            
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
            # Use real ROS Duration for actionlib compatibility
            import rospy as ros_rospy
            finished = self.drop_off_client.wait_for_result(ros_rospy.Duration(20.0))
            
            if not finished:
                rospy.logerr("FSM: Drop-off action timed out")
                self.drop_off_client.cancel_goal()
                return {"success": False, "message": "Drop-off timed out"}
            
            # Get and process result
            result = self.drop_off_client.get_result()
            if result and result.success:
                rospy.loginfo(f"FSM: Drop-off sequence succeeded! {result.message}")
                return {"success": True, "message": "Object dropped off"}
            else:
                rospy.logerr(f"FSM: Drop-off sequence failed: {result.message if result else 'Unknown error'}")
                return {"success": False, "message": result.message if result else "Unknown error"}
                
        except Exception as e:
            rospy.logerr(f"FSM: Drop-off action failed with exception: {e}")
            # Fallback to simulator on error
            rospy.logwarn("FSM: Falling back to simulator for drop-off")
            return robot_simulator.drop_off()

    def get_robot_pose(self):
        """Get current robot pose."""
        try:
            if self.dummy_mode:
                # In dummy mode, use robot simulator
                return robot_simulator.get_robot_pose()
            else:
                # In real robot mode, call the actual robot service
                if hasattr(self, 'get_robot_pose_srv'):
                    from spot_hololens_llm_interface.srv import GetRobotPoseRequest
                    req = GetRobotPoseRequest()
                    resp = self.get_robot_pose_srv(req)
                    return resp
                else:
                    rospy.logerr("FSM: Robot pose service not available")
                    return {"success": False, "message": "Robot pose service not available"}
        except Exception as e:
            rospy.logerr(f"Get robot pose failed: {e}")
            return {"success": False, "message": str(e)}

    def send(self, event, **kwargs):
        """Send event to state machine with parameters."""
        # Store kwargs for use in state entry methods
        self._current_kwargs = kwargs
        
        # Call the parent send method
        return super().send(event)

    def get_current_state_name(self):
        """Get current state name as string."""
        return str(self.current_state.name)

    def is_connected(self):
        """Check if robot is connected."""
        return self.current_state in [self.connected, self.powered_off, self.stand, self.sit, self.moving, self.grasping, self.carry, self.move_arm_pose, self.arm_command]

    def is_powered(self):
        """Check if robot is powered."""
        return self.current_state in [self.powered_off, self.stand, self.sit, self.moving, self.grasping, self.carry, self.move_arm_pose, self.arm_command]

    def is_standing(self):
        """Check if robot is standing."""
        return self.current_state in [self.stand, self.moving, self.grasping, self.carry, self.move_arm_pose, self.arm_command]
