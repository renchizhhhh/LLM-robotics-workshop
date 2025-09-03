#!/usr/bin/env python

import os
import time
import sys
import json
import rospy
from std_msgs.msg import String
import logging

from spot_fsm_control.finite_state_machine import SpotStateMachine
from spot_fsm_control.data_logger import DataLogger

logging.basicConfig(format="[LINE:%(lineno)d] %(levelname)-8s [%(asctime)s]  %(message)s", level=logging.INFO)


class FsmNode:
    """
    ROS node for Spot FSM control.
    Listens to /fsm_commands for JSON commands and publishes feedback on /fsm_feedback.
    """

    def __init__(self, hostname="192.168.80.3", dummy_mode=False):
        rospy.init_node('spot_fsm_node', anonymous=True)
        
        self.hostname = hostname
        self.dummy_mode = dummy_mode
        self.spot_interface = None
        self.state_machine = None
        
        # Publishers and subscribers
        self.feedback_pub = rospy.Publisher('/fsm_feedback', String, queue_size=10)
        self.command_sub = rospy.Subscriber('/fsm_commands', String, self.command_callback)
        
        # Legacy command mapping for backward compatibility
        self.legacy_commands = {
            'stand': {'op': 'stand', 'height': 0.0},
            'sit': {'op': 'sit'},
            'stop': {'op': 'stop'},
            'arm_ready': {'op': 'arm_ready'},
            'arm_stow': {'op': 'arm_stow'},
        }
        
        # Data logging
        self.data_logger = None
        
        print("Spot FSM Node initialized")
        print("Listening on /fsm_commands for JSON commands")
        print("Publishing feedback on /fsm_feedback")
        
    def initialize_robot(self):
        """Initialize Spot robot connection."""
        # This method is now simplified since initialization is done in main()
        return True
    
    def command_callback(self, msg):
        """Handle incoming commands from /fsm_commands topic."""
        try:
            command_str = msg.data.strip()
            print(f"Received command: {command_str}")
            
            # Try to parse as JSON first
            try:
                command = json.loads(command_str)
            except json.JSONDecodeError:
                # Try legacy string command
                if command_str in self.legacy_commands:
                    command = self.legacy_commands[command_str]
                    print(f"Mapped legacy command '{command_str}' to: {command}")
                else:
                    print(f"Invalid JSON and unknown legacy command: {command_str}")
                    self._publish_feedback("error", "invalid_command", f"Invalid command: {command_str}")
                    return
            
            # Validate command
            if 'op' not in command:
                print("Missing 'op' field in command")
                self._publish_feedback("error", "invalid_command", "Missing 'op' field")
                return
            
            # Execute command
            self._execute_command(command)
            
        except Exception as e:
            print(f"Error processing command: {e}")
            self._publish_feedback("error", "processing_error", str(e))
    
    def _execute_command(self, command):
        """Execute a command through the state machine."""
        op = command.get('op')
        
        # Start action timer for logging
        if self.data_logger:
            self.data_logger.start_action_timer()
        
        try:
            print(f"Executing command: {op}")
            
            # Log action start
            if self.data_logger:
                self.data_logger.log_action(op, command, success=None, robot_state=str(self.state_machine.current_state))
            
            # Enqueue action in state machine
            success = self.state_machine.enqueue_action(command)
            
            # Calculate execution time
            execution_time = None
            if self.data_logger:
                execution_time = self.data_logger.end_action_timer()
            
            if success:
                # Action was executed successfully
                self._publish_feedback("ok", op, "Action completed successfully")
                
                # Log successful action
                if self.data_logger:
                    self.data_logger.log_action(op, command, success=True, execution_time=execution_time, robot_state=str(self.state_machine.current_state))
            else:
                # Action failed
                self._publish_feedback("error", op, "Action failed")
                
                # Log failed action
                if self.data_logger:
                    self.data_logger.log_action(op, command, success=False, execution_time=execution_time, robot_state=str(self.state_machine.current_state))
                    self.data_logger.log_error("execution_failed", f"Action {op} failed", op, str(self.state_machine.current_state))
            
        except Exception as e:
            print(f"Command execution failed: {e}")
            self._publish_feedback("error", op, str(e))
            
            # Log error
            if self.data_logger:
                execution_time = self.data_logger.end_action_timer()
                self.data_logger.log_action(op, command, success=False, execution_time=execution_time, robot_state=str(self.state_machine.current_state))
                self.data_logger.log_error("exception", str(e), op, str(self.state_machine.current_state))
    
    def _publish_feedback(self, status, operation, message=""):
        """Publish feedback message."""
        feedback = {
            "status": status,
            "op": operation,
            "timestamp": time.time()
        }
        
        if message:
            feedback["msg"] = message
            
        feedback_str = json.dumps(feedback)
        self.feedback_pub.publish(feedback_str)
        print(f"Feedback: {feedback_str}")
        
        # Log feedback
        if self.data_logger:
            self.data_logger.log_feedback(operation, status, message)
    
    def shutdown(self):
        """Clean shutdown of the node."""
        print("Shutting down FSM node...")
        
        try:
            if self.spot_interface:
                # First stow the arm (like V1)
                print("Stowing arm...")
                try:
                    self.spot_interface.arm_stow()
                    time.sleep(1.0)  # Give time for arm to stow
                except Exception as e:
                    print(f"Warning: Could not stow arm: {e}")
                
                # Then sit down robot (like V1)
                print("Sitting down robot...")
                try:
                    self.spot_interface.sit()
                    time.sleep(2.0)  # Give time for robot to sit
                except Exception as e:
                    print(f"Warning: Could not sit down: {e}")
                
                # Return lease if available (like V1)
                if hasattr(self.spot_interface, 'robot_sdk') and self.spot_interface.robot_sdk:
                    if hasattr(self.spot_interface.robot_sdk, 'lease_client'):
                        print("Returning lease...")
                        try:
                            self.spot_interface.robot_sdk.lease_client.return_lease()
                        except Exception as e:
                            print(f"Warning: Could not return lease: {e}")
                    
                    # Clean up lease keep-alive (like V1)
                    if hasattr(self.spot_interface.robot_sdk, 'lease_keep_alive'):
                        print("Cleaning up lease keep-alive...")
                        try:
                            self.spot_interface.robot_sdk.lease_keep_alive.shutdown()
                        except Exception as e:
                            print(f"Warning: Could not shutdown lease keep-alive: {e}")
                
                # Finally shutdown the interface
                try:
                    self.spot_interface.shutdown(power_off=False)
                except Exception as e:
                    print(f"Warning: Could not shutdown interface: {e}")
            
        except Exception as e:
            print(f"Error during shutdown: {e}")
        
        # Close data logger
        if self.data_logger:
            try:
                self.data_logger.close()
            except Exception as e:
                print(f"Error closing data logger: {e}")
        
        print("FSM node shutdown complete")


def main():
    """Main function."""
    try:
        # Get parameters
        hostname = rospy.get_param('~hostname', '192.168.80.3')
        
        # Simple dummy mode toggle like V1 - set this manually
        DUMMY_MODE = False  # Set to True for dummy mode, False for real robot
        
        if DUMMY_MODE:
            print("DUMMY MODE: robotInterface = None")
            dummy_mode = True
            spot_interface = None
        else:
            print("REAL ROBOT MODE: robotInterface = SpotControlInterface")
            dummy_mode = False
            # Only import SpotControlInterface if not in dummy mode
            try:
                from spot_fsm_control.spot_control_interface import SpotControlInterface
                import bosdyn.client
                import bosdyn.client.lease
                import bosdyn.client.util
                
                print("Connecting to Spot...")
                
                # Create SpotControlInterface first (like V1)
                spot_interface = SpotControlInterface(15)  # direct_control_frequency like V1
                
                # Create SDK and robot (like V1)
                sdk = bosdyn.client.create_standard_sdk('SpotControlInterface')
                # Try the hostname that V1 actually used
                robot = sdk.create_robot("192.168.1.109")  # V1 hostname
                spot_interface.robot_sdk = robot
                spot_interface.robot = robot  # Also set robot for move_relative
                
                # Authenticate
                bosdyn.client.util.authenticate(robot)
                print("Authentication successful")
                
                # Time sync
                robot.time_sync.wait_for_sync()
                print("Time sync completed")
                
                # Check E-stop
                assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client."
                print("E-stop check passed")
                
                # Take lease (like V1)
                lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
                lease = lease_client.acquire()
                print("Lease acquired")
                
                # Store lease client and lease for cleanup (like V1)
                robot.lease_client = lease_client
                robot.lease = lease
                
                # Use lease keep-alive context manager (like V1)
                lease_keep_alive = bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)
                robot.lease_keep_alive = lease_keep_alive
                
                # Power on robot (like V1)
                print("Powering on robot... This may take several seconds.")
                robot.power_on(timeout_sec=20)
                assert robot.is_powered_on(), "Robot power on failed."
                print("Robot powered on successfully")
                
                # Initialize clients (like V1)
                spot_interface.image_client = robot.ensure_client(bosdyn.client.image.ImageClient.default_service_name)
                spot_interface.command_client = robot.ensure_client(bosdyn.client.robot_command.RobotCommandClient.default_service_name)
                spot_interface.robot_state_client = robot.ensure_client(bosdyn.client.robot_state.RobotStateClient.default_service_name)
                spot_interface.manipulation_api_client = robot.ensure_client(bosdyn.client.manipulation_api_client.ManipulationApiClient.default_service_name)
                spot_interface.world_object_client = robot.ensure_client(bosdyn.client.world_object.WorldObjectClient.default_service_name)
                
                # Wait a bit (like V1)
                time.sleep(1)
                
                # Sit down robot (like V1)
                print("Sitting down robot...")
                spot_interface.sit()
                print("Robot is now sitting")
                
                print("Robot connected and ready!")
                
            except Exception as e:
                print(f"Failed to connect to robot: {e}")
                print("Falling back to dummy mode")
                spot_interface = None
        
        # Create FSM node
        fsm_node = FsmNode(hostname, dummy_mode=(spot_interface is None))
        
        # Initialize with the interface (or None for dummy)
        fsm_node.spot_interface = spot_interface
        fsm_node.state_machine = SpotStateMachine(spot_interface, dummy_mode=(spot_interface is None))
        
        # Initialize data logger only when not in dummy mode
        if spot_interface is not None:  # Only log when using real robot
            try:
                participant_id = rospy.get_param('~participant_id', -1)
                condition = rospy.get_param('~condition', 'default')
                fsm_node.data_logger = DataLogger(participant_id=participant_id, condition=condition)
                fsm_node.state_machine.data_logger = fsm_node.data_logger
                
                # Log experiment info
                fsm_node.data_logger.log_experiment_info({
                    "description": "Spot FSM Control V2",
                    "robot_type": "Boston Dynamics Spot",
                    "software_version": "V2",
                    "dummy_mode": False,
                    "hostname": hostname
                })
                print("Data logging enabled for real robot operation")
            except Exception as e:
                print(f"Warning: Could not initialize data logger: {e}")
                fsm_node.data_logger = None
        else:
            print("Data logging disabled in dummy mode")
            fsm_node.data_logger = None
        
        if spot_interface is None:
            print("FSM node ready in DUMMY MODE. Press Ctrl+C to exit.")
        else:
            print("FSM node ready with REAL ROBOT. Press Ctrl+C to exit.")
        
        # Spin until interrupted
        rospy.spin()
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        if 'fsm_node' in locals():
            fsm_node.shutdown()


if __name__ == '__main__':
    main()
        