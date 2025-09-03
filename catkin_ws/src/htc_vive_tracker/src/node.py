#!/usr/bin/env python

import time
import math
import rospy
from std_msgs.msg import Float32MultiArray
from  htc_vive_tracker import triad_openvr



class HTCViveTrackerNode:

    def __init__(self):
        ## HTC tracker init
        self.vive_tracker = triad_openvr.triad_openvr()
        self.vive_tracker.print_discovered_objects()
        
        self.pub = rospy.Publisher('robot_pose', Float32MultiArray, queue_size=3)
        rospy.init_node('robot_pose_node', anonymous=True)
            
    def run(self):
        rate = rospy.Rate(2)
        try:
            while True:
                try:
                    pose = self.vive_tracker.devices["tracker_1"].get_pose_euler()
                    if pose is not None:
                        msg = Float32MultiArray()
                        msg.data = pose
                        self.pub.publish(msg)

                    rate.sleep()
                except Exception as e:
                    print(e)
                    rate.sleep()
        except KeyboardInterrupt:
            print("stopping tracker node")

        

if __name__ == "__main__":
    node =  HTCViveTrackerNode()
    node.run()