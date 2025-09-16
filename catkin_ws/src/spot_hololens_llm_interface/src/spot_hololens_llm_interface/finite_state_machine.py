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
from spot_hololens_llm_interface.srv import MoveToPosition, MoveToPositionRequest, GetImage, GetImageRequest, GetInitialPose, GetInitialPoseRequest, ArmCommand, ArmCommandRequest
from spot_hololens_llm_interface.msg import AutomatedGraspAction, AutomatedGraspGoal

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
    grasping = State()

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
    start_moving = (stand.to(moving) | moving.to(moving))
    stop_moving = moving.to(stand)
    get_image = (stand.to(stand) | moving.to(moving) | sit.to(sit))
    get_initial_pose = (stand.to(stand) | moving.to(moving) | sit.to(sit))
    arm_command = (stand.to(stand) | moving.to(moving))
    start_automated_grasp = (stand.to(grasping))
    finish_automated_grasp = (grasping.to(stand))
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
        self.arm_command_srv = self.create_service_proxy('/spot_entrance/arm_command', ArmCommand)
        self.power_off_srv = self.create_service_proxy('/spot_entrance/power_off', Trigger)
        self.disconnect_srv = self.create_service_proxy('/spot_entrance/disconnect', Trigger)
        
        # Setup action client for automated grasp
        self.automated_grasp_client = actionlib.SimpleActionClient('automated_grasp', AutomatedGraspAction)
        if not self.dummy_mode:
            self.automated_grasp_client.wait_for_server(rospy.Duration(5.0))
            rospy.loginfo("FSM: Connected to automated_grasp action server")
        
        # Initialize the state machine after setting up services
        super().__init__()
        self._startup_probe(timeout=3.0)
        
        # Default movement parameters (used as fallback)
        self.move_x = 0.0
        self.move_y = 0.0
        self.move_yaw = 0.0
        self.move_frame = "vision"

    def _startup_probe(self, timeout=3.0):
        """Check `/spot_entrance/robot_state` and align FSM to the most specific state."""
        try:
            if self.dummy_mode:
                rospy.loginfo('FSM: Dummy mode - assuming connected')
                self.send('discover_connected')
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

    def on_enter_moving(self):
        # Get parameters from kwargs if available, otherwise use stored ones
        kwargs = getattr(self, '_current_kwargs', {})
        move_x = kwargs.get('x', self.move_x)
        move_y = kwargs.get('y', self.move_y)
        move_yaw = kwargs.get('yaw', self.move_yaw)
        move_frame = kwargs.get('frame', self.move_frame)
        
        rospy.loginfo(f"FSM: Moving x={move_x}, y={move_y}, yaw={move_yaw}")
        recorder.publish_event('start_moving')
        result = self._call_move_service(move_x, move_y, move_yaw, move_frame)
        recorder.publish_event('stop_moving')
    
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
            rospy.sleep(2.0)  # Simulate time for grasp in dummy mode
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

