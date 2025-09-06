#!/usr/bin/env python

import rospy
import math
from statemachine import StateMachine, State
from std_srvs.srv import Trigger
from spot_hololens_llm_interface.srv import MoveToPosition, MoveToPositionRequest, GetImage, GetImageRequest, GetInitialPose, GetInitialPoseRequest, ArmCommand, ArmCommandRequest

class SpotStateMachine(StateMachine):
    """
    Minimal state machine for Spot robot control.
    Acts as a bridge between high-level commands and ROS services.
    """
    # States
    disconnected = State(initial=True)
    connected = State()
    powered_off = State()
    sit = State()
    stand = State()
    moving = State()
    
    # Transitions
    connect = disconnected.to(connected)
    power_on = connected.to(powered_off)
    stand_up = (powered_off.to(stand) | sit.to(stand))
    sit_down = (stand.to(sit) | moving.to(sit))
    start_moving = (stand.to(moving) | moving.to(moving))
    stop_moving = moving.to(stand)
    get_image = (stand.to(stand) | moving.to(moving) | sit.to(sit))
    get_initial_pose = (stand.to(stand) | moving.to(moving) | sit.to(sit))
    arm_command = (stand.to(stand) | moving.to(moving))
    power_off_from_stand = stand.to(powered_off)
    power_off_from_sit = sit.to(powered_off)
    disconnect = powered_off.to(disconnected)

    def __init__(self, dummy_mode=None):
        # Check for dummy mode parameter
        self.dummy_mode = dummy_mode if dummy_mode is not None else rospy.get_param('~dummy_mode', False)
        
        if self.dummy_mode:
            rospy.loginfo("FSM: Running in DUMMY MODE - using dummy services")
        else:
            rospy.loginfo("FSM: Running in REAL MODE - using real robot services")
        
        # Setup ROS service connections
        self.connect_srv = rospy.ServiceProxy('/spot_entrance/connect', Trigger)
        self.power_on_srv = rospy.ServiceProxy('/spot_entrance/power_on', Trigger)
        self.stand_srv = rospy.ServiceProxy('/spot_entrance/stand', Trigger)
        self.sit_srv = rospy.ServiceProxy('/spot_entrance/sit', Trigger)
        self.move_srv = rospy.ServiceProxy('/spot_entrance/move_to_position', MoveToPosition)
        self.get_image_srv = rospy.ServiceProxy('/spot_entrance/get_image', GetImage)
        self.get_initial_pose_srv = rospy.ServiceProxy('/spot_entrance/get_initial_pose', GetInitialPose)
        self.arm_command_srv = rospy.ServiceProxy('/spot_entrance/arm_command', ArmCommand)
        self.power_off_srv = rospy.ServiceProxy('/spot_entrance/power_off', Trigger)
        self.disconnect_srv = rospy.ServiceProxy('/spot_entrance/disconnect', Trigger)
        
        # Initialize the state machine after setting up services
        super().__init__()
        
        # Default movement parameters (used as fallback)
        self.move_x = 0.0
        self.move_y = 0.0
        self.move_yaw = 0.0
        self.move_frame = "body"
        
    # State entry methods - call services when states change
    def on_enter_connected(self):
        rospy.loginfo("FSM: Connecting to robot")
        self._call_service(self.connect_srv, "Connect")

    def on_enter_powered_off(self):
        rospy.loginfo("FSM: Powering on robot")
        self._call_service(self.power_on_srv, "Power on")

    def on_enter_stand(self):
        rospy.loginfo("FSM: Standing up")
        self._call_service(self.stand_srv, "Stand")

    def on_enter_sit(self):
        rospy.loginfo("FSM: Sitting down")
        self._call_service(self.sit_srv, "Sit")

    def on_enter_moving(self):
        # Get parameters from kwargs if available, otherwise use stored ones
        kwargs = getattr(self, '_current_kwargs', {})
        move_x = kwargs.get('x', self.move_x)
        move_y = kwargs.get('y', self.move_y)
        move_yaw = kwargs.get('yaw', self.move_yaw)
        move_frame = kwargs.get('frame', self.move_frame)
        
        rospy.loginfo(f"FSM: Moving x={move_x}, y={move_y}, yaw={move_yaw}")
        self._call_move_service(move_x, move_y, move_yaw, move_frame)
    
    def on_enter_get_image(self):
        # Get image source from kwargs, default to frontleft_fisheye_image
        kwargs = getattr(self, '_current_kwargs', {})
        image_source = kwargs.get('image_source', 'frontleft_fisheye_image')
        
        rospy.loginfo(f"FSM: Getting image from {image_source}")
        self._call_get_image_service(image_source)

    def on_enter_get_initial_pose(self):
        rospy.loginfo("FSM: Getting initial pose")
        self._call_get_initial_pose_service()

    def on_enter_arm_command(self):
        # Get command type from kwargs, default to "stow"
        kwargs = getattr(self, '_current_kwargs', {})
        command_type = kwargs.get('command_type', 'stow')
        
        rospy.loginfo(f"FSM: Executing arm command: {command_type}")
        self._call_arm_command_service(command_type)

    def on_enter_powered_off_from_stand(self):
        rospy.loginfo("FSM: Powering off robot")
        self._call_service(self.power_off_srv, "Power off")

    def on_enter_powered_off_from_sit(self):
        rospy.loginfo("FSM: Powering off robot")
        self._call_service(self.power_off_srv, "Power off")

    def on_enter_disconnected(self):
        rospy.loginfo("FSM: Disconnecting from robot")
        self._call_service(self.disconnect_srv, "Disconnect")

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

    # Helper methods
    def _call_service(self, service, name):
        """Helper to call a service and log result"""
        try:
            resp = service()
            if resp.success:
                mode_text = " (dummy)" if self.dummy_mode else ""
                rospy.loginfo(f"FSM: {name} successful{mode_text}")
            else:
                mode_text = " (dummy)" if self.dummy_mode else ""
                rospy.logerr(f"FSM: {name} failed{mode_text}: {resp.message}")
        except Exception as e:
            mode_text = " (dummy)" if self.dummy_mode else ""
            rospy.logerr(f"FSM: {name} service call failed{mode_text}: {e}")

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
            else:
                mode_text = " (dummy)" if self.dummy_mode else ""
                rospy.logerr(f"FSM: Move failed{mode_text}: {resp.message}")
        except Exception as e:
            mode_text = " (dummy)" if self.dummy_mode else ""
            rospy.logerr(f"FSM: Move service call failed{mode_text}: {e}")


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

