#!/usr/bin/env python3
"""
Timing utilities for robot performance analysis.

Provides centralized timing event publishing and rosbag recording
for precise latency measurements across the robot pipeline.
"""

import os
import time
import rospy
import rosbag
from std_msgs.msg import String


class TimingRecorder:
    """Manages timing event recording to rosbag with precise ROS timestamps."""
    
    def __init__(self):
        self.bag = None
        self.subscriber = None
        self.publisher = None
        self.recording_enabled = False
        self.rosbag_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '..', 'rosbag'
        )
    
    def start_recording(self):
        """Initialize rosbag recording for timing events."""
        if self.bag is not None:
            return
        
        os.makedirs(self.rosbag_dir, exist_ok=True)
        filename = f"timing_events_{time.strftime('%Y%m%d_%H%M%S')}.bag"
        bag_path = os.path.join(self.rosbag_dir, filename)
        
        self.bag = rosbag.Bag(bag_path, 'w')
        self.subscriber = rospy.Subscriber('/timing_events', String, self._bridge_callback)
        self.recording_enabled = True
        
        rospy.loginfo(f"Recording timing events to: {filename}")
    
    def stop_recording(self):
        """Stop recording and close rosbag."""
        self.recording_enabled = False
        
        if self.bag:
            self.bag.close()
            self.bag = None
        
        if self.subscriber:
            self.subscriber.unregister()
            self.subscriber = None
    
    def publish_event(self, event_type):
        """Publish timing event to ROS topic and record to rosbag."""
        if self.publisher is None:
            self.publisher = rospy.Publisher('/timing_events', String, queue_size=10)
        
        msg = String(data=event_type)
        self.publisher.publish(msg)
        
        if self.recording_enabled and self.bag:
            self.bag.write('/timing_events', msg)
    
    def _bridge_callback(self, msg):
        """Bridge external ROS events to rosbag."""
        if self.bag and self.recording_enabled:
            self.bag.write('/timing_events', msg)


# Global instance for direct access
recorder = TimingRecorder()