#!/usr/bin/env python3
# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""A small ROS node wrapper for the Spot estop keep-alive.

Provides services:
  /estop/stop             (std_srvs/Trigger) -> trigger estop (cut power)
  /estop/allow            (std_srvs/Trigger) -> release estop (allow)
  /estop/settle_then_cut  (std_srvs/Trigger) -> settle then cut

Publishes:
  /estop/status (std_msgs/String) -> "NOT_STOPPED" / "STOPPED" / "ERROR"
"""
import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse

import bosdyn.client.util
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.robot_state import RobotStateClient


def make_estop_keepalive(robot, timeout_sec, name='ros_estop'):
    estop_client = robot.ensure_client(EstopClient.default_service_name)
    ep = EstopEndpoint(estop_client, name, timeout_sec)
    ep.force_simple_setup()
    keep = EstopKeepAlive(ep)
    # start allowed
    keep.allow()
    return keep, estop_client


class EstopRosNode(object):
    def __init__(self):
        rospy.init_node('estop_node')
        hostname = rospy.get_param('~hostname', rospy.get_param('hostname', None))
        timeout = float(rospy.get_param('~timeout', rospy.get_param('timeout', 5.0)))
        dummy_mode = rospy.get_param('~dummy_mode', False)
        
        if dummy_mode:
            rospy.loginfo("Estop node starting in DUMMY MODE - no real robot connection")
            self.dummy_mode = True
            self.estop_keep_alive = None
            self.state_client = None
            self.robot = None
        else:
            if hostname is None:
                rospy.logerr("No 'hostname' param set (robot hostname/IP). Set with <param name='hostname' value='x.x.x.x'/>")
                raise rospy.ROSInitException("hostname param required")

            self.dummy_mode = False
            sdk = bosdyn.client.create_standard_sdk('ros_estop_node')
            robot = sdk.create_robot(hostname)
            
            # Get credentials from ROS parameters and set as environment variables
            username = rospy.get_param('~username', 'user')
            password = rospy.get_param('~password', '')
            
            # Set environment variables for authentication
            import os
            os.environ['BOSDYN_CLIENT_USERNAME'] = username
            os.environ['BOSDYN_CLIENT_PASSWORD'] = password
            
            try:
                bosdyn.client.util.authenticate(robot)
                rospy.loginfo("Estop node authenticated with robot using provided credentials")
            except Exception as e:
                rospy.logwarn("Authentication with robot may have failed (exception: %s). Ensure credentials are available.", e)

            # Create keepalive & robot-state client
            try:
                self.estop_keep_alive, _ = make_estop_keepalive(robot, timeout, 'ROS E-Stop')
            except Exception as e:
                rospy.logerr("Failed to create estop endpoint: %s", e)
                raise

            try:
                self.state_client = robot.ensure_client(RobotStateClient.default_service_name)
            except Exception as e:
                rospy.logwarn("Could not create RobotStateClient: %s", e)
                self.state_client = None

        # Services
        self.svc_stop = rospy.Service('estop/stop', Trigger, self.handle_stop)
        self.svc_allow = rospy.Service('estop/allow', Trigger, self.handle_allow)
        self.svc_settle = rospy.Service('estop/settle_then_cut', Trigger, self.handle_settle)

        # Publisher
        self.pub_status = rospy.Publisher('estop/status', String, queue_size=1)

        rospy.on_shutdown(self.on_shutdown)
        rospy.loginfo("estop_node ready (hostname=%s timeout=%s)", hostname, timeout)

        self.loop()

    def handle_stop(self, req):
        if self.dummy_mode:
            rospy.loginfo("[DUMMY] Estop triggered (stopped)")
            return TriggerResponse(success=True, message="Estop triggered (stopped) - dummy mode")
        
        try:
            self.estop_keep_alive.stop()
            return TriggerResponse(success=True, message="Estop triggered (stopped).")
        except Exception as e:
            return TriggerResponse(success=False, message="Failed to stop: %s" % e)

    def handle_allow(self, req):
        if self.dummy_mode:
            rospy.loginfo("[DUMMY] Estop released (allowed)")
            return TriggerResponse(success=True, message="Estop released (allowed) - dummy mode")
        
        try:
            self.estop_keep_alive.allow()
            return TriggerResponse(success=True, message="Estop released (allowed).")
        except Exception as e:
            return TriggerResponse(success=False, message="Failed to allow: %s" % e)

    def handle_settle(self, req):
        if self.dummy_mode:
            rospy.loginfo("[DUMMY] Settle then cut issued")
            return TriggerResponse(success=True, message="Settle then cut issued - dummy mode")
        
        try:
            self.estop_keep_alive.settle_then_cut()
            return TriggerResponse(success=True, message="Settle then cut issued.")
        except Exception as e:
            return TriggerResponse(success=False, message="Failed to settle_then_cut: %s" % e)

    def loop(self):
        rate_hz = rospy.get_param('~status_rate', 2.0)
        rate = rospy.Rate(rate_hz)
        while not rospy.is_shutdown():
            status = "UNKNOWN"
            try:
                if self.dummy_mode:
                    status = "NOT_STOPPED"  # Always allow in dummy mode
                elif self.state_client is not None:
                    state = self.state_client.get_robot_state()
                    estop_states = state.estop_states
                    # default NOT_STOPPED
                    status = "NOT_STOPPED"
                    for est in estop_states:
                        state_str = est.State.Name(est.state)
                        if state_str == 'STATE_ESTOPPED':
                            status = "STOPPED"
                            break
                        elif state_str == 'STATE_UNKNOWN':
                            status = "ERROR"
                else:
                    status = "NO_STATE_CLIENT"
            except Exception as e:
                status = "ERROR"
                rospy.logdebug("Error reading robot_state: %s", e)

            try:
                self.pub_status.publish(String(status))
            except Exception:
                pass

            rate.sleep()

    def on_shutdown(self):
        rospy.loginfo("Shutting down estop_node, ending keep-alive.")
        try:
            self.estop_keep_alive.end_periodic_check_in()
        except Exception:
            # Best-effort
            pass


if __name__ == '__main__':
    try:
        EstopRosNode()
    except rospy.ROSInitException:
        # already logged
        pass
    except Exception as e:
        rospy.logfatal("estop_node failed: %s", e)
        raise
