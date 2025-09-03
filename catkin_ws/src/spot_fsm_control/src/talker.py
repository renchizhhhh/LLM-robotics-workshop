#!/usr/bin/env python

import rospy
from std_msgs.msg import String

def talker():
    pub = rospy.Publisher('spot_fsm_control_msgs/Action', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(0.2) # 10hz
    actions = ['stand_up', 'sit_down']
    i = 0
    while not rospy.is_shutdown():
        hello_str = actions[(i % 2)]
        i += 1
        
        rospy.loginfo(hello_str)
        pub.publish(hello_str)
        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass