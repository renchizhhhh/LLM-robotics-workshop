#!/usr/bin/env python3

import rospy
import actionlib
import os
import cv2
import numpy as np
from spot_hololens_llm_interface.msg import AutomatedGraspAction, AutomatedGraspGoal, AutomatedGraspFeedback
from spot_hololens_llm_interface.srv import GetImage, GetImageRequest
from std_srvs.srv import Trigger

def robot_stand():
    """Make robot stand up"""
    try:
        rospy.wait_for_service('/spot_entrance/stand', timeout=5.0)
        stand_srv = rospy.ServiceProxy('/spot_entrance/stand', Trigger)
        resp = stand_srv()
        if resp.success:
            rospy.loginfo("Robot standing up")
            rospy.sleep(3.0)  # Wait for robot to stand
            return True
        else:
            rospy.logerr(f"Failed to stand: {resp.message}")
            return False
    except Exception as e:
        rospy.logerr(f"Stand service call failed: {e}")
        return False

def robot_sit():
    """Make robot sit down"""
    try:
        rospy.wait_for_service('/spot_entrance/sit', timeout=5.0)
        sit_srv = rospy.ServiceProxy('/spot_entrance/sit', Trigger)
        resp = sit_srv()
        if resp.success:
            rospy.loginfo("Robot sitting down")
            rospy.sleep(3.0)  # Wait for robot to sit
            return True
        else:
            rospy.logerr(f"Failed to sit: {resp.message}")
            return False
    except Exception as e:
        rospy.logerr(f"Sit service call failed: {e}")
        return False

def get_image_from_robot(image_source):
    """Get image from robot using ROS service"""
    try:
        from spot_hololens_llm_interface.srv import GetImage, GetImageRequest
        rospy.wait_for_service('/spot_entrance/get_image', timeout=5.0)
        get_image_srv = rospy.ServiceProxy('/spot_entrance/get_image', GetImage)
        
        req = GetImageRequest()
        req.image_source = image_source
        resp = get_image_srv(req)
        
        if resp.success:
            import cv_bridge
            bridge = cv_bridge.CvBridge()
            img = bridge.imgmsg_to_cv2(resp.image, "bgr8")
            return img
        return None
    except Exception as e:
        rospy.logerr(f"Failed to get image: {e}")
        return None

def save_grasp_image(img, pixel_x, pixel_y, object_type):
    """Save image with grasp visualization"""
    try:
        os.makedirs('/Docker-LLM-Spot-Image/catkin_ws/images', exist_ok=True)
        
        box_size = 50
        x1, y1 = max(0, pixel_x - box_size), max(0, pixel_y - box_size)
        x2, y2 = min(img.shape[1], pixel_x + box_size), min(img.shape[0], pixel_y + box_size)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(img, (pixel_x, pixel_y), 5, (0, 0, 255), -1)
        cv2.putText(img, f"Grasp {object_type} at ({pixel_x}, {pixel_y})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        filename = f"/Docker-LLM-Spot-Image/catkin_ws/images/grasp_{object_type}_{rospy.Time.now().to_sec():.0f}.jpg"
        cv2.imwrite(filename, img)
        rospy.loginfo(f"Grasp image saved: {filename}")
        return True
    except Exception as e:
        rospy.logerr(f"Failed to save image: {e}")
        return False

def test_automated_grasp():
    """Test automated grasp: stand -> detect object + grasp -> sit"""
    # First, make robot stand
    rospy.loginfo("Making robot stand...")
    if not robot_stand():
        rospy.logerr("Failed to make robot stand")
        return False
    
    client = actionlib.SimpleActionClient('automated_grasp', AutomatedGraspAction)
    
    if not client.wait_for_server(rospy.Duration(10.0)):
        rospy.logerr("Automated grasp action server not available")
        robot_sit()  # Try to sit even if action server fails
        return False

    if not os.getenv('GOOGLE_API_KEY'):
        rospy.logerr("GOOGLE_API_KEY not set")
        robot_sit()  # Try to sit even if API key is missing
        return False

    image_source = "hand_color_image"
    img = get_image_from_robot(image_source)
    if img is None:
        rospy.logwarn("Could not get image for visualization")
        img = None

    goal = AutomatedGraspGoal()
    goal.image_source = image_source
    goal.object_type = "Tomato"
    goal.force_top_down_grasp = False
    goal.force_horizontal_grasp = True
    goal.force_45_angle_grasp = False
    goal.force_squeeze_grasp = False
    goal.return_to_initial_pose = True

    rospy.loginfo("Starting automated grasp...")
    client.send_goal(goal)

    finished = client.wait_for_result(rospy.Duration(60.0))
    if not finished:
        rospy.logwarn("Automated grasp timed out")
        client.cancel_goal()
        robot_sit()  # Make sure to sit even if grasp times out
        return False

    result = client.get_result()
    grasp_success = False
    if result and result.success:
        rospy.loginfo(f"Automated grasp successful! Grasped at ({result.selected_pixel_x}, {result.selected_pixel_y})")
        
        if img is not None:
            save_grasp_image(img, result.selected_pixel_x, result.selected_pixel_y, "tomato")
        
        grasp_success = True
    else:
        rospy.logwarn(f"Automated grasp failed: {result.message if result else 'No result'}")
    
    # Finally, make robot sit
    rospy.loginfo("Making robot sit...")
    if not robot_sit():
        rospy.logerr("Failed to make robot sit")
        return False
    
    return grasp_success

def main():
    rospy.init_node('automated_grasp_test')
    rospy.loginfo("Automated Grasp Test - Make sure cup is visible in camera")
    
    if not os.getenv('GOOGLE_API_KEY'):
        rospy.logerr("Set GOOGLE_API_KEY environment variable")
        return
    
    try:
        rospy.sleep(2.0)  # Wait for services
        success = test_automated_grasp()
        rospy.loginfo("Test completed successfully!" if success else "Test failed")
    except rospy.ROSInterruptException:
        rospy.loginfo("Test interrupted")
    except Exception as e:
        rospy.logerr(f"Test failed: {e}")

if __name__ == '__main__':
    main()