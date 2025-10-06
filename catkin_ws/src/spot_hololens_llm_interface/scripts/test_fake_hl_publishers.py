import json
import rospy
from std_msgs.msg import String, Empty

class HoloLensInterface:
    def __init__(self):
        self.current_plan_id = None
        # Initialize ROS node and publishers
        rospy.init_node('fake_hololens_interface', anonymous=True)
        self.topic_approval = '/hl/approval'
        self.topic_soft_stop = '/hl/stop'
        self.approval_publisher = rospy.Publisher(self.topic_approval, String, queue_size=10)
        self.soft_stop_publisher = rospy.Publisher(self.topic_soft_stop, Empty, queue_size=10)

        timeout = rospy.Time.now() + rospy.Duration(.5)
        while self.approval_publisher.get_num_connections() == 0 and rospy.Time.now() < timeout and not rospy.is_shutdown():
            rospy.sleep(0.02)

    def publish_approval(self, approved, override_plan_id=None):
        plan_id = override_plan_id if override_plan_id else self.current_plan_id
        
        payload = {
            "plan_id": plan_id,
            "approved": approved,
        }
        
        json_payload = json.dumps(payload)

        self.approval_publisher.publish(String(data=json_payload))
        self.append(f"[Approval] → {self.topic_approval} {json_payload}")

    def publish_soft_stop(self):
        self.soft_stop_publisher.publish(Empty())
        self.append(f"[Soft Stop] → {self.topic_soft_stop} <Empty>")
    
    def append(self, message):
        print(message)

# Example usage
if __name__ == "__main__":
    interface = HoloLensInterface()
    interface.current_plan_id = "test_plan_123"
    interface.publish_approval(True, None)
    timeout = rospy.Time.now() + rospy.Duration(5)
    while rospy.Time.now() < timeout and not rospy.is_shutdown():
        rospy.sleep(1)
        print("Waiting 1 second before sending soft stop...")
    interface.publish_soft_stop()

