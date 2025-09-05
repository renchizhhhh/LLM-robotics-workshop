#!/usr/bin/env python3

import rospy
import time

import bosdyn.client
import bosdyn.client.util
import bosdyn.client.lease
import bosdyn.client.estop
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.frame_helpers import get_se2_a_tform_b, ODOM_FRAME_NAME, BODY_FRAME_NAME
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.api import estop_pb2
from std_srvs.srv import Trigger, TriggerResponse
from std_msgs.msg import Bool, String

from spot_hololens_llm_interface.spot_shared_services import SpotSharedServices
from spot_hololens_llm_interface.spot_grasp_action_server import SpotGraspActionServer

class SpotRobotManager:
    def __init__(self, ready_for_command=True, start_services=True, arm_action_server=True):
        # Get parameters
        self.hostname = rospy.get_param('~hostname', None)
        self.verbose = rospy.get_param('~verbose', False)
        
        if self.hostname is None:
            rospy.logerr("No hostname specified!")
            return
            
        # Initialize robot connection
        bosdyn.client.util.setup_logging(self.verbose)
        self.sdk = bosdyn.client.create_standard_sdk('SpotRobotManager')
        self.robot = self.sdk.create_robot(self.hostname)
        bosdyn.client.util.authenticate(self.robot)
        
        # Initialize clients
        self.lease_client = self.robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        self.lease_keep_alive = None
        self.command_client = None
        self.robot_state_client = None
        self.image_client = None
        self.manipulation_api_client = None
        
        # Robot state tracking
        self.is_connected = False
        self.is_powered = False
        self.is_standing = False
        self.initial_pose = None
        
        # Setup basic services
        self.srv_connect = rospy.Service('~connect', Trigger, self.handle_connect)
        self.srv_disconnect = rospy.Service('~disconnect', Trigger, self.handle_disconnect)
        self.srv_power_on = rospy.Service('~power_on', Trigger, self.handle_power_on)
        self.srv_power_off = rospy.Service('~power_off', Trigger, self.handle_power_off)
        self.srv_stand = rospy.Service('~stand', Trigger, self.handle_stand)
        self.srv_sit = rospy.Service('~sit', Trigger, self.handle_sit)

        # Publishers
        self.pub_robot_state = rospy.Publisher('~robot_state', String, queue_size=1)
        self.pub_power_state = rospy.Publisher('~power_state', Bool, queue_size=1)
        
        # Status publisher timer
        self.status_timer = rospy.Timer(rospy.Duration(1.0), self.publish_status)
        
        # Initialize shared services
        if start_services:
            self.shared_services = SpotSharedServices(self)

        if arm_action_server:
            self.grasp_action_server = SpotGraspActionServer(self)

        if ready_for_command:
            self.handle_connect(None)
            self.handle_power_on(None)

        rospy.loginfo(f"Spot Robot Manager initialized. Ready for commands: {ready_for_command}; "
                      f"Services started: {start_services}; Arm action server: {arm_action_server}")

    def verify_estop(self):
        """Verify the robot is not estopped"""
        client = self.robot.ensure_client(bosdyn.client.estop.EstopClient.default_service_name)
        if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
            error_message = 'Robot is estopped. Please use an external E-Stop client to configure E-Stop.'
            rospy.logerr(error_message)
            raise Exception(error_message)
            
    def get_clients(self):
        """Get all robot clients if connected"""
        if self.is_connected:
            return {
                'command': self.command_client,
                'robot_state': self.robot_state_client,
                'image': self.image_client,
                'manipulation_api': self.manipulation_api_client,
                'lease': self.lease_client
            }
        return None

    def handle_connect(self, req):
        """Connect to the robot and acquire lease"""
        try:
            self.robot.time_sync.wait_for_sync()
            self.verify_estop()
            
            self._lease_keepalive_running = True
            # is force take safe?
            self.lease_client.take()
            self.lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(
                self.lease_client,
                must_acquire=True,
                return_at_exit=True,
                keep_running_cb=lambda: self._lease_keepalive_running
            )
            
            self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
            self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
            self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
            self.manipulation_api_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)
            
            self.is_connected = True
            rospy.loginfo("Connected to robot and acquired lease")
            return TriggerResponse(success=True, message="Connected to robot")
        except Exception as e:
            rospy.logerr(f"Failed to connect to robot: {str(e)}")
            return TriggerResponse(success=False, message=f"Failed to connect: {str(e)}")
    
    def handle_disconnect(self, req):
        """Disconnect from the robot and release lease"""
        try:
            if self.is_powered:
                self.handle_power_off(req)

            self._lease_keepalive_running = False
            if self.lease_keep_alive:
                self.lease_keep_alive.shutdown()
                self.lease_keep_alive = None
            
            self.is_connected = False
            rospy.loginfo("Disconnected from robot")
            return TriggerResponse(success=True, message="Disconnected from robot")
        except Exception as e:
            rospy.logerr(f"Failed to disconnect from robot: {str(e)}")
            return TriggerResponse(success=False, message=f"Failed to disconnect: {str(e)}")
    
    def handle_power_on(self, req):
        """Power on the robot"""
        try:
            if not self.is_connected:
                return TriggerResponse(success=False, message="Not connected to robot")
                
            rospy.loginfo("Powering on robot...")
            self.robot.power_on(timeout_sec=20)
            if not self.robot.is_powered_on():
                return TriggerResponse(success=False, message="Failed to power on robot")
                
            self.is_powered = True
            rospy.loginfo("Robot powered on")
            return TriggerResponse(success=True, message="Robot powered on")
        except Exception as e:
            rospy.logerr(f"Failed to power on robot: {str(e)}")
            return TriggerResponse(success=False, message=f"Failed to power on: {str(e)}")
    
    def handle_power_off(self, req):
        """Power off the robot"""
        try:
            if not self.is_connected:
                return TriggerResponse(success=False, message="Not connected to robot")
                
            if self.is_standing:
                self.handle_sit(req)
                
            rospy.loginfo("Powering off robot...")
            self.robot.power_off(cut_immediately=False, timeout_sec=20)
            if self.robot.is_powered_on():
                return TriggerResponse(success=False, message="Failed to power off robot")
                
            self.is_powered = False
            rospy.loginfo("Robot powered off")
            return TriggerResponse(success=True, message="Robot powered off")
        except Exception as e:
            rospy.logerr(f"Failed to power off robot: {str(e)}")
            return TriggerResponse(success=False, message=f"Failed to power off: {str(e)}")
    
    def handle_stand(self, req):
        """Command the robot to stand"""
        try:
            if not self.is_powered:
                return TriggerResponse(success=False, message="Robot not powered on")
                
            rospy.loginfo("Commanding robot to stand...")
            blocking_stand(self.command_client, timeout_sec=10)
            
            # Store initial pose
            robot_state = self.robot_state_client.get_robot_state()
            self.initial_pose = get_se2_a_tform_b(
                robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME, BODY_FRAME_NAME)
                
            self.is_standing = True
            rospy.loginfo("Robot standing")
            return TriggerResponse(success=True, message="Robot standing")
        except Exception as e:
            rospy.logerr(f"Failed to stand robot: {str(e)}")
            return TriggerResponse(success=False, message=f"Failed to stand: {str(e)}")
    
    def handle_sit(self, req):
        """Command the robot to sit"""
        try:
            if not self.is_powered:
                return TriggerResponse(success=False, message="Robot not powered on")
            
            rospy.loginfo("Commanding robot to sit...")
            sit_command = RobotCommandBuilder.synchro_sit_command()
            self.command_client.robot_command(sit_command)
            time.sleep(2)  # Give time for the robot to sit
            
            self.is_standing = False
            rospy.loginfo("Robot sitting")
            return TriggerResponse(success=True, message="Robot sitting")
        except Exception as e:
            rospy.logerr(f"Failed to sit robot: {str(e)}")
            return TriggerResponse(success=False, message=f"Failed to sit: {str(e)}")
    
    def publish_status(self, event):
        """Publish robot status information; TODO: align with the state machine"""
        if self.is_connected:
            self.pub_power_state.publish(self.is_powered)
            
            state = "DISCONNECTED"
            if self.is_connected:
                if not self.is_powered:
                    state = "CONNECTED_OFF"
                elif self.is_standing:
                    state = "STANDING"
                else:
                    state = "SITTING"
            
            self.pub_robot_state.publish(state)
    
    def shutdown(self):
        """Shutdown the robot manager"""
        if self.is_connected:
            if self.is_powered:
                try:
                    self.handle_power_off(None)
                except:
                    rospy.logwarn("Failed to power off robot during shutdown")
                    
            if self.lease_keep_alive:
                self.lease_keep_alive.shutdown()
                
        rospy.loginfo("Robot manager shutdown")

if __name__ == '__main__':
    rospy.init_node('spot_robot_manager')
    manager = SpotRobotManager()
    rospy.on_shutdown(manager.shutdown)
    rospy.spin()