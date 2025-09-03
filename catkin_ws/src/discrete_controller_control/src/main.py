import pygame
from time import sleep
import rospy
from std_msgs.msg import String  # Replace with your desired ROS message type

class PlayStationController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()

        self.joystick_count = pygame.joystick.get_count()
        if self.joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        else:
            print("No joystick detected")
            exit()

        # Initialize ROS node
        rospy.init_node('playstation_controller', anonymous=True)
        self.publisher = rospy.Publisher('chatter', String, queue_size=1)  # Replace 'String' with your message type

    def run(self):
        while not rospy.is_shutdown():
            for event in pygame.event.get():
                try:
                    if event.button == 11:
                        print("Joystick button pressed up")
                        self.publish_ros_message("walk_to_forward")
                        break
                    elif event.button == 12:
                        print("Joystick button pressed down")
                        self.publish_ros_message("walk_to_backward")
                        break
                    elif event.button == 13:
                        print("Joystick button pressed left")
                        self.publish_ros_message("turn_to_left")
                        break
                    if event.button == 14:
                        print("Joystick button pressed right")
                        self.publish_ros_message("turn_to_right")
                        break
                except:
                    pass
            sleep(1)

        pygame.quit()

    def publish_ros_message(self, message):
        ros_msg = String()  # Replace with your ROS message
        ros_msg.data = message
        self.publisher.publish(ros_msg)
        rospy.loginfo("Published ROS message: " + message)

if __name__ == "__main__":
    controller = PlayStationController()
    controller.run()
