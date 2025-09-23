#!/usr/bin/python3

import sys
import os
# Add ROS Python path so system Python can find ROS modules
sys.path.insert(0, '/opt/ros/noetic/lib/python3/dist-packages')
# Add workspace packages to path
sys.path.insert(0, '/Docker-LLM-Spot-Image/catkin_ws/devel/lib/python3/dist-packages')

import rospy
import math
import json
import threading
import time
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped

# Check for display availability before importing tkinter
try:
    import tkinter as tk
    from tkinter import ttk
    GUI_AVAILABLE = True
except Exception as e:
    rospy.logwarn(f"GUI not available: {e}")
    GUI_AVAILABLE = False

class SpotSimulationGUI:
    def __init__(self):
        rospy.init_node('spot_simulation_gui', anonymous=True)
        
        # Check if GUI is available
        if not GUI_AVAILABLE:
            rospy.logwarn("GUI not available - running in headless mode (logging only)")
            self.headless_mode = True
            self.setup_headless_mode()
            return
        else:
            self.headless_mode = False
        
        # Robot state
        self.robot_x = 0.0  # Body frame position
        self.robot_y = 0.0
        self.robot_yaw = -math.pi/2  # Radians - robot starts facing left (Y direction) so length is parallel to X
        self.robot_state = "unknown"
        self.current_action = "idle"
        self.has_object = False
        self.carried_object_name = None
        self.arm_status = "stowed"
        self.gripper_status = "closed"
        
        # World objects (in vision frame coordinates) - positioned diagonally around pickup location
        self.objects = {
            "tomato_can": {"x": 3.0, "y": -1.0, "present": True, "color": "#f85149"},
            "apple": {"x": 2.8, "y": -0.8, "present": True, "color": "#39d353"},
            "bottle": {"x": 3.2, "y": -1.2, "present": True, "color": "#58a6ff"}
        }
        
        # World landmarks (in vision frame) - updated positions
        self.pickup_location = {"x": 3.0, "y": -1.0}  # Pickup at (3, -1)
        self.dropoff_location = {"x": 2.0, "y": 1.5}  # Dropoff at (2, 1.5)
        
        # Position tracking - convert vision frame to display coordinates
        self.vision_to_body_offset_x = 0.0
        self.vision_to_body_offset_y = 0.0
        self.start_position_set = False
        
        # GUI setup
        try:
            self.root = tk.Tk()
            self.root.title("Spot Robot Simulation - Bird's Eye View")
        except Exception as e:
            rospy.logerr(f"Failed to create GUI window: {e}")
            self.headless_mode = True
            self.setup_headless_mode()
            return
        
        # Try different fullscreen methods for cross-platform compatibility
        try:
            self.root.state('zoomed')  # Windows/some Linux
        except:
            try:
                self.root.attributes('-zoomed', True)  # Linux
            except:
                try:
                    self.root.attributes('-fullscreen', True)  # Cross-platform
                except:
                    self.root.geometry("1400x1000")  # Fallback large window
        
        # Dark mode styling
        self.root.configure(bg='#1a1a1a')
        
        # Get screen dimensions for fullscreen canvas
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Create canvas for visualization (larger, fullscreen)
        canvas_width = screen_width - 50
        canvas_height = screen_height - 200  # Leave space for controls
        self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height, bg='#0d1117')
        self.canvas.pack(pady=20)
        
        # Info panel with dark theme
        self.info_frame = tk.Frame(self.root, bg='#1a1a1a')
        self.info_frame.pack(fill=tk.X, padx=20, pady=15)
        
        # Status labels with futuristic styling
        self.state_label = tk.Label(self.info_frame, text="State: unknown", font=("Arial", 14, "bold"), 
                                   bg='#1a1a1a', fg='#00d4aa')
        self.state_label.grid(row=0, column=0, sticky=tk.W, padx=15, pady=5)
        
        self.position_label = tk.Label(self.info_frame, text="Position: (0.0, 0.0, 0.0°)", font=("Arial", 14, "bold"), 
                                      bg='#1a1a1a', fg='#58a6ff')
        self.position_label.grid(row=0, column=1, sticky=tk.W, padx=15, pady=5)
        
        self.action_label = tk.Label(self.info_frame, text="Action: idle", font=("Arial", 14, "bold"), 
                                    bg='#1a1a1a', fg='#f85149')
        self.action_label.grid(row=1, column=0, sticky=tk.W, padx=15, pady=5)
        
        self.arm_label = tk.Label(self.info_frame, text="Arm: stowed | Gripper: closed", font=("Arial", 14, "bold"), 
                                 bg='#1a1a1a', fg='#ffa657')
        self.arm_label.grid(row=1, column=1, sticky=tk.W, padx=15, pady=5)
        
        
        # Control buttons with dark theme
        button_frame = tk.Frame(self.root, bg='#1a1a1a')
        button_frame.pack(fill=tk.X, padx=20, pady=15)
        
        self.reset_button = tk.Button(button_frame, text="Reset Simulation", command=self.reset_simulation,
                                     bg='#21262d', fg='#f0f6fc', font=("Arial", 12, "bold"),
                                     activebackground='#30363d', activeforeground='#f0f6fc',
                                     relief='flat', padx=20, pady=10)
        self.reset_button.pack(side=tk.LEFT, padx=10)
        
        # ROS subscribers
        self.sub_robot_state = rospy.Subscriber('/spot_entrance/robot_state', String, self.on_robot_state)
        self.sub_feedback = rospy.Subscriber('/spot/execution_feedback', String, self.on_execution_feedback)
        
        # Threading for GUI updates
        self.update_lock = threading.Lock()
        
        # Start drawing
        self.draw_world()
        self.update_display()
        
        # Schedule periodic updates
        self.root.after(100, self.periodic_update)
        
    def vision_to_canvas(self, x, y):
        """Convert vision frame coordinates to canvas coordinates with 90° rotation"""
        # Canvas center (dynamic based on screen size)
        canvas_center_x = self.canvas.winfo_width() // 2 if self.canvas.winfo_width() > 1 else 600
        canvas_center_y = self.canvas.winfo_height() // 2 if self.canvas.winfo_height() > 1 else 400
        
        # Scale: 1 meter = 80 pixels (larger for better visibility)
        scale = 80
        
        # Rotate 90 degrees counterclockwise: (x,y) -> (-y,x)
        # Vision frame rotated: x=forward becomes upward, y=left becomes backward
        rotated_x = -y
        rotated_y = x
        
        canvas_x = canvas_center_x + rotated_x * scale
        canvas_y = canvas_center_y - rotated_y * scale  # Invert Y for screen coordinates
        
        return canvas_x, canvas_y
    
    def body_to_vision(self, body_x, body_y, body_yaw):
        """Convert body frame movement to vision frame position"""
        # For visualization, we track cumulative position in vision frame
        # Body frame movements are relative to current robot orientation
        
        # Current robot orientation in vision frame
        vision_yaw = self.robot_yaw
        
        # Rotate body frame movement to vision frame
        # The robot's body frame is x=forward, y=left
        # Since we rotated the display 90° counterclockwise, we need to account for this
        # The robot's body frame is still x=forward, y=left, but the display is rotated
        cos_yaw = math.cos(vision_yaw)
        sin_yaw = math.sin(vision_yaw)
        
        vision_dx = cos_yaw * body_x - sin_yaw * body_y
        vision_dy = sin_yaw * body_x + cos_yaw * body_y
        
        return vision_dx, vision_dy
    
    def draw_world(self):
        """Draw the static world elements"""
        self.canvas.delete("all")
        
        # Draw coordinate system (rotated)
        center_x, center_y = self.vision_to_canvas(0, 0)
        
        # X-axis (forward) - now points upward after rotation - cyan
        self.canvas.create_line(center_x, center_y, center_x, center_y - 120, 
                               fill="#00d4aa", width=3, arrow=tk.LAST)
        self.canvas.create_text(center_x + 15, center_y - 130, text="X (forward)", fill="#00d4aa", font=("Arial", 12, "bold"))
        
        # Y-axis (left) - now points left after rotation - blue  
        self.canvas.create_line(center_x, center_y, center_x - 120, center_y, 
                               fill="#58a6ff", width=3, arrow=tk.LAST)
        self.canvas.create_text(center_x - 130, center_y - 15, text="Y (left)", fill="#58a6ff", font=("Arial", 12, "bold"))
        
        # Draw grid with futuristic styling
        canvas_width = self.canvas.winfo_width() if self.canvas.winfo_width() > 1 else 1200
        canvas_height = self.canvas.winfo_height() if self.canvas.winfo_height() > 1 else 800
        
        for i in range(-10, 11):
            x, y = self.vision_to_canvas(i, 0)
            if 0 <= x <= canvas_width:
                self.canvas.create_line(x, 0, x, canvas_height, fill="#21262d", width=1)
            x, y = self.vision_to_canvas(0, i)
            if 0 <= y <= canvas_height:
                self.canvas.create_line(0, y, canvas_width, y, fill="#21262d", width=1)
        
        # Draw pickup location with futuristic styling
        pickup_x, pickup_y = self.vision_to_canvas(self.pickup_location["x"], self.pickup_location["y"])
        self.canvas.create_rectangle(pickup_x-25, pickup_y-25, pickup_x+25, pickup_y+25, 
                                   fill="#ffa657", outline="#ff7b72", width=3)
        self.canvas.create_text(pickup_x, pickup_y-40, text="PICKUP", font=("Arial", 14, "bold"), fill="#ffa657")
        
        # Draw dropoff location with futuristic styling
        dropoff_x, dropoff_y = self.vision_to_canvas(self.dropoff_location["x"], self.dropoff_location["y"])
        self.canvas.create_rectangle(dropoff_x-25, dropoff_y-25, dropoff_x+25, dropoff_y+25,
                                   fill="#00d4aa", outline="#39d353", width=3)
        self.canvas.create_text(dropoff_x, dropoff_y-40, text="DROPOFF", font=("Arial", 14, "bold"), fill="#00d4aa")
        
        # Draw objects with better visibility
        for obj_name, obj_data in self.objects.items():
            if obj_data["present"]:
                obj_x, obj_y = self.vision_to_canvas(obj_data["x"], obj_data["y"])
                self.canvas.create_oval(obj_x-12, obj_y-12, obj_x+12, obj_y+12,
                                      fill=obj_data["color"], outline="#f0f6fc", width=2)
                self.canvas.create_text(obj_x, obj_y+25, text=obj_name, font=("Arial", 10, "bold"), fill="#f0f6fc")
    
    def draw_robot(self):
        """Draw the robot at current position with realistic gripper"""
        # Convert robot position to canvas coordinates
        robot_canvas_x, robot_canvas_y = self.vision_to_canvas(self.robot_x, self.robot_y)
        
        # Robot dimensions (scaled larger)
        robot_width = 56  # 0.7m * 80 pixels/m
        robot_length = 112  # 1.4m * 80 pixels/m
        
        # Calculate robot corner points based on orientation
        # Robot's length should be parallel to X-axis (forward direction) when yaw=0
        # Positive yaw is counterclockwise (left)
        display_yaw = self.robot_yaw
        cos_yaw = math.cos(display_yaw)
        sin_yaw = math.sin(display_yaw)
        
        # Robot corners relative to center (length along X-axis, width along Y-axis)
        # Front is in positive X direction - robot is oriented correctly
        corners = [
            (robot_length/2, -robot_width/2),   # front left  
            (-robot_length/2, -robot_width/2),  # rear left
            (-robot_length/2, robot_width/2),   # rear right
            (robot_length/2, robot_width/2),    # front right
        ]
        
        # Rotate and translate corners
        robot_corners = []
        for dx, dy in corners:
            rotated_dx = cos_yaw * dx - sin_yaw * dy
            rotated_dy = sin_yaw * dx + cos_yaw * dy
            robot_corners.extend([robot_canvas_x + rotated_dx, robot_canvas_y + rotated_dy])
        
        # Draw robot body with futuristic colors
        robot_color = "#58a6ff"
        if self.robot_state == "sit":
            robot_color = "#1f6feb"
        elif self.robot_state in ["moving", "grasping"]:
            robot_color = "#79c0ff"
        elif self.robot_state in ["powered_off", "disconnected"]:
            robot_color = "#6e7681"
            
        self.canvas.create_polygon(robot_corners, fill=robot_color, outline="#f0f6fc", width=3, tags="robot")
        
        # Draw direction indicator (front of robot) - points in positive X direction
        front_x = robot_canvas_x + cos_yaw * (robot_length/2 + 15)
        front_y = robot_canvas_y + sin_yaw * (robot_length/2 + 15)
        self.canvas.create_oval(front_x-8, front_y-8, front_x+8, front_y+8,
                              fill="#f85149", outline="#ff7b72", width=2, tags="robot")
        
        # Draw robot center point
        self.canvas.create_oval(robot_canvas_x-4, robot_canvas_y-4, robot_canvas_x+4, robot_canvas_y+4,
                              fill="#ffa657", outline="#f0f6fc", width=2, tags="robot")
        
        # Draw realistic gripper based on arm status
        self.draw_gripper(robot_canvas_x, robot_canvas_y, cos_yaw, sin_yaw, robot_length)
        
        # Draw carried object
        if self.has_object:
            carried_x = robot_canvas_x + cos_yaw * 25
            carried_y = robot_canvas_y + sin_yaw * 25
            self.canvas.create_oval(carried_x-8, carried_y-8, carried_x+8, carried_y+8,
                                  fill="#d2a8ff", outline="#f0f6fc", width=2, tags="robot")
            self.canvas.create_text(carried_x, carried_y+20, text="OBJ", font=("Arial", 10, "bold"), fill="#f0f6fc", tags="robot")
    
    def draw_gripper(self, robot_x, robot_y, cos_yaw, sin_yaw, robot_length):
        """Draw realistic gripper based on arm status"""
        if self.arm_status == "stowed":
            # Gripper at robot center when stowed
            gripper_x = robot_x
            gripper_y = robot_y
            gripper_size = 8
            arm_length = 0
        elif self.arm_status == "carry":
            # Gripper at front edge of robot when in carry position
            gripper_x = robot_x + cos_yaw * (robot_length/2 - 10)
            gripper_y = robot_y + sin_yaw * (robot_length/2 - 10)
            gripper_size = 10
            arm_length = robot_length/2 - 10
        else:  # extended, grasping, etc.
            # Gripper extended further from robot body
            gripper_x = robot_x + cos_yaw * (robot_length/2 + 30)
            gripper_y = robot_y + sin_yaw * (robot_length/2 + 30)
            gripper_size = 12
            arm_length = robot_length/2 + 30
        
        # Draw arm if extended
        if arm_length > 0:
            self.canvas.create_line(robot_x, robot_y, gripper_x, gripper_y,
                                  fill="#d2a8ff", width=6, tags="robot")
        
        # Draw gripper base (pink/purple like in the image)
        self.canvas.create_oval(gripper_x - gripper_size, gripper_y - gripper_size, 
                              gripper_x + gripper_size, gripper_y + gripper_size,
                              fill="#ff69b4", outline="#f0f6fc", width=2, tags="robot")
        
        # Draw gripper jaws
        if self.gripper_status == "open":
            # Open gripper - two separated jaws
            jaw_offset = gripper_size + 3
            # Left jaw
            jaw1_x = gripper_x - sin_yaw * jaw_offset
            jaw1_y = gripper_y + cos_yaw * jaw_offset
            self.canvas.create_oval(jaw1_x - 4, jaw1_y - 4, jaw1_x + 4, jaw1_y + 4,
                                  fill="#00d4aa", outline="#f0f6fc", width=1, tags="robot")
            # Right jaw
            jaw2_x = gripper_x + sin_yaw * jaw_offset
            jaw2_y = gripper_y - cos_yaw * jaw_offset
            self.canvas.create_oval(jaw2_x - 4, jaw2_y - 4, jaw2_x + 4, jaw2_y + 4,
                                  fill="#00d4aa", outline="#f0f6fc", width=1, tags="robot")
        else:
            # Closed gripper - single smaller indicator
            self.canvas.create_oval(gripper_x - 3, gripper_y - 3, gripper_x + 3, gripper_y + 3,
                                  fill="#f85149", outline="#f0f6fc", width=1, tags="robot")
    
    def setup_headless_mode(self):
        """Setup headless mode with minimal state tracking"""
        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = -math.pi/2  # Robot starts facing left (Y direction) so length is parallel to X
        self.robot_state = "unknown"
        self.current_action = "idle"
        self.has_object = False
        self.carried_object_name = None
        self.arm_status = "stowed"
        self.gripper_status = "closed"
        
        # Objects for state tracking - positioned diagonally around pickup location
        self.objects = {
            "tomato_can": {"x": 3.0, "y": -1.0, "present": True, "color": "#f85149"},
            "apple": {"x": 2.8, "y": -0.8, "present": True, "color": "#39d353"},
            "bottle": {"x": 3.2, "y": -1.2, "present": True, "color": "#58a6ff"}
        }
        
        self.pickup_location = {"x": 3.0, "y": -1.0}
        self.dropoff_location = {"x": 2.0, "y": 1.5}
        
        # ROS subscribers for state tracking
        self.sub_robot_state = rospy.Subscriber('/spot_entrance/robot_state', String, self.on_robot_state_headless)
        self.sub_feedback = rospy.Subscriber('/spot/execution_feedback', String, self.on_execution_feedback_headless)
        
        rospy.loginfo("Spot Simulation running in headless mode - state tracking via ROS logs")
    
    def on_robot_state_headless(self, msg):
        """Log robot state changes in headless mode"""
        try:
            fields = dict(part.split(':', 1) for part in msg.data.split(','))
            connected = fields.get('connected', 'false').lower() == 'true'
            powered = fields.get('powered', 'false').lower() == 'true'
            standing = fields.get('standing', 'false').lower() == 'true'
            
            if not connected:
                self.robot_state = "disconnected"
            elif not powered:
                self.robot_state = "powered_off"
            elif standing:
                self.robot_state = "stand"
            else:
                self.robot_state = "sit"
                
            rospy.loginfo(f"[HEADLESS] Robot State: {self.robot_state}")
        except Exception as e:
            rospy.logwarn(f"Failed to parse robot state in headless mode: {e}")
    
    def on_execution_feedback_headless(self, msg):
        """Log execution feedback in headless mode"""
        feedback = msg.data
        rospy.loginfo(f"[HEADLESS] Execution: {feedback}")
        
        # Simple state tracking for major events
        if "start_moving" in feedback:
            self.current_action = "moving"
        elif "start_automated_grasp" in feedback:
            self.current_action = "grasping"
        elif "start_arm_command" in feedback:
            if "open" in feedback:
                self.gripper_status = "open"
            elif "close" in feedback:
                self.gripper_status = "closed"
        elif "✓" in feedback:
            self.current_action = "idle"
    
    def on_robot_state(self, msg):
        """Update robot state from /spot_entrance/robot_state"""
        try:
            # Parse state message: "connected:true,powered:true,standing:true"
            with self.update_lock:
                fields = dict(part.split(':', 1) for part in msg.data.split(','))
                connected = fields.get('connected', 'false').lower() == 'true'
                powered = fields.get('powered', 'false').lower() == 'true'
                standing = fields.get('standing', 'false').lower() == 'true'
                
                if not connected:
                    self.robot_state = "disconnected"
                elif not powered:
                    self.robot_state = "powered_off"
                elif standing:
                    self.robot_state = "stand"
                else:
                    self.robot_state = "sit"
                    
        except Exception as e:
            rospy.logwarn(f"Failed to parse robot state: {e}")
    
    def on_execution_feedback(self, msg):
        """Update robot state from execution feedback"""
        try:
            feedback = msg.data
            
            with self.update_lock:
                # Parse different types of feedback
                if "[exec] Step" in feedback:
                    # Extract action from step feedback
                    if "start_moving" in feedback:
                        self.current_action = "moving"
                        # Extract movement parameters if available
                        try:
                            # Look for x=, y=, yaw= in the feedback
                            import re
                            x_match = re.search(r'x=([+-]?\d*\.?\d+)', feedback)
                            y_match = re.search(r'y=([+-]?\d*\.?\d+)', feedback)
                            yaw_match = re.search(r'yaw=([+-]?\d*\.?\d+)', feedback)
                            
                            if x_match and y_match:
                                body_x = float(x_match.group(1))
                                body_y = float(y_match.group(1))
                                body_yaw = float(yaw_match.group(1)) if yaw_match else 0.0
                                
                                # Convert body frame movement to vision frame displacement
                                vision_dx, vision_dy = self.body_to_vision(body_x, body_y, body_yaw)
                                
                                # Update robot position
                                # Since the display is rotated 90° counterclockwise, we need to adjust the movement
                                # World X (forward) becomes display Y (upward)
                                # World Y (left) becomes display -X (rightward)
                                # So we need to rotate the movement by 90° counterclockwise
                                rotated_dx = -vision_dy  # World Y becomes display -X
                                rotated_dy = vision_dx   # World X becomes display Y
                                
                                self.robot_x += rotated_dx
                                self.robot_y += rotated_dy
                                self.robot_yaw -= body_yaw  # Negative for counterclockwise rotation
                                
                                # Normalize yaw to [-π, π]
                                while self.robot_yaw > math.pi:
                                    self.robot_yaw -= 2 * math.pi
                                while self.robot_yaw < -math.pi:
                                    self.robot_yaw += 2 * math.pi
                                
                                rospy.loginfo(f"Robot moved by ({body_x}, {body_y}, {body_yaw}) in body frame -> "
                                            f"({vision_dx:.2f}, {vision_dy:.2f}) in vision frame. "
                                            f"New position: ({self.robot_x:.2f}, {self.robot_y:.2f}, {self.robot_yaw:.2f})")
                                rospy.loginfo(f"Current robot yaw: {self.robot_yaw:.2f} radians ({math.degrees(self.robot_yaw):.1f} degrees)")
                        except Exception as e:
                            rospy.logwarn(f"Failed to parse movement parameters: {e}")
                            
                    elif "start_automated_grasp" in feedback:
                        self.current_action = "grasping"
                        self.arm_status = "extended"
                        # Extract target object type from feedback
                        target_object = None
                        try:
                            # Look for object_type="something" in the feedback
                            import re
                            match = re.search(r'object_type[=:]"([^"]+)"', feedback)
                            if match:
                                target_object = match.group(1)
                        except Exception as e:
                            rospy.logwarn(f"Could not extract target object: {e}")
                        
                        # Check if there's an object at current location to grasp
                        self.simulate_grasp_attempt(target_object)
                        
                    elif "start_arm_command" in feedback:
                        if "open" in feedback:
                            self.gripper_status = "open"
                            if self.has_object:
                                self.simulate_object_drop()
                        elif "close" in feedback:
                            self.gripper_status = "closed"
                        elif "stow" in feedback:
                            self.arm_status = "stowed"
                        elif "carry" in feedback:
                            self.arm_status = "carry"
                            
                    elif "stand_up" in feedback:
                        self.current_action = "standing"
                        
                    elif "sit_down" in feedback:
                        self.current_action = "sitting"
                        
                elif "✓" in feedback:
                    # Action completed
                    if self.current_action == "moving":
                        self.current_action = "idle"
                    elif self.current_action == "grasping":
                        self.current_action = "idle"
                        
        except Exception as e:
            rospy.logwarn(f"Failed to parse execution feedback: {e}")
    
    def on_execution_feedback_headless(self, msg):
        """Log execution feedback in headless mode"""
        feedback = msg.data
        rospy.loginfo(f"[HEADLESS] Execution: {feedback}")
        
        # Simple state tracking for major events
        if "start_moving" in feedback:
            self.current_action = "moving"
        elif "start_automated_grasp" in feedback:
            self.current_action = "grasping"
        elif "start_arm_command" in feedback:
            if "open" in feedback:
                self.gripper_status = "open"
            elif "close" in feedback:
                self.gripper_status = "closed"
        elif "✓" in feedback:
            self.current_action = "idle"
    
    def simulate_grasp_attempt(self, target_object=None):
        """Simulate grasping objects at current location"""
        grasp_distance = 1.5  # Increased to 1.5m for more reliable grasping
        
        rospy.loginfo(f"Attempting to grasp {target_object} at robot position ({self.robot_x:.2f}, {self.robot_y:.2f})")
        
        # If a specific target object is specified, try to grasp that one first
        if target_object:
            for obj_name, obj_data in self.objects.items():
                # More precise matching - check if target object name is contained in object name
                # but also check if object name contains the target (to avoid partial matches)
                target_lower = target_object.lower().strip()
                obj_lower = obj_name.lower().strip()
                
                # Check for exact match or if target is a complete word in the object name
                is_match = (target_lower == obj_lower or 
                           target_lower in obj_lower.split('_') or 
                           target_lower in obj_lower.split(' '))
                
                if obj_data["present"] and is_match:
                    distance = math.sqrt((self.robot_x - obj_data["x"])**2 + (self.robot_y - obj_data["y"])**2)
                    if distance <= grasp_distance:
                        # Successful grasp of target object
                        obj_data["present"] = False
                        self.has_object = True
                        self.gripper_status = "closed"
                        self.carried_object_name = obj_name
                        rospy.loginfo(f"Successfully grasped target {obj_name} at distance {distance:.2f}m")
                        return
        
        # Fallback: grasp any object within range
        for obj_name, obj_data in self.objects.items():
            if obj_data["present"]:
                distance = math.sqrt((self.robot_x - obj_data["x"])**2 + (self.robot_y - obj_data["y"])**2)
                if distance <= grasp_distance:
                    # Successful grasp
                    obj_data["present"] = False
                    self.has_object = True
                    self.gripper_status = "closed"
                    self.carried_object_name = obj_name
                    rospy.loginfo(f"Successfully grasped {obj_name} at distance {distance:.2f}m")
                    return
    
    def simulate_object_drop(self):
        """Simulate dropping the carried object"""
        if self.has_object:
            # Check if robot is near drop-off location
            dropoff_distance = math.sqrt((self.robot_x - self.dropoff_location["x"])**2 + 
                                       (self.robot_y - self.dropoff_location["y"])**2)
            
            
            if dropoff_distance <= 1.0:  # Within 1m of drop-off location
                # Drop object at the drop-off location
                drop_x = self.dropoff_location["x"]
                drop_y = self.dropoff_location["y"]
                rospy.loginfo(f"Dropping {self.carried_object_name} at drop-off location ({drop_x:.2f}, {drop_y:.2f})")
            else:
                # Drop object at current robot location
                drop_x = self.robot_x
                drop_y = self.robot_y
                rospy.loginfo(f"Dropping {self.carried_object_name} at current location ({drop_x:.2f}, {drop_y:.2f})")
            
            # Create dropped object
            object_name = self.carried_object_name if self.carried_object_name else "dropped_object"
            dropped_obj = {
                "x": drop_x,
                "y": drop_y,
                "present": True,
                "color": "#d2a8ff"  # Purple color for dropped objects
            }
            self.objects[f"{object_name}_dropped_{int(time.time())}"] = dropped_obj
            self.has_object = False
            self.carried_object_name = None
    
    def reset_simulation(self):
        """Reset the simulation to initial state"""
        with self.update_lock:
            # Reset robot position and state
            self.robot_x = 0.0
            self.robot_y = 0.0
            self.robot_yaw = -math.pi/2  # Robot starts facing left (Y direction) so length is parallel to X
            self.robot_state = "stand"
            self.current_action = "idle"
            self.has_object = False
            self.carried_object_name = None
            self.arm_status = "stowed"
            self.gripper_status = "closed"
            
            # Reset objects to initial state (positioned diagonally around pickup location)
            self.objects = {
                "tomato_can": {"x": 3.0, "y": -1.0, "present": True, "color": "#f85149"},
                "apple": {"x": 2.8, "y": -0.8, "present": True, "color": "#39d353"},
                "bottle": {"x": 3.2, "y": -1.2, "present": True, "color": "#58a6ff"}
            }
            
            rospy.loginfo("Simulation reset to initial state")
    
    def update_display(self):
        """Update the GUI display"""
        with self.update_lock:
            # Clear previous robot drawing
            self.canvas.delete("robot")
            
            # Redraw world (to refresh objects)
            self.draw_world()
            
            # Draw robot
            self.draw_robot()
            
            # Update status labels
            self.state_label.config(text=f"State: {self.robot_state}")
            # Display yaw with 90 degree offset for display purposes
            display_yaw = math.degrees(self.robot_yaw) + 90.0
            self.position_label.config(text=f"Position: ({self.robot_x:.2f}, {self.robot_y:.2f}, {display_yaw:.1f}°)")
            self.action_label.config(text=f"Action: {self.current_action}")
            arm_text = f"Arm: {self.arm_status} | Gripper: {self.gripper_status}"
            if self.has_object:
                arm_text += " | Carrying object"
            self.arm_label.config(text=arm_text)
    
    def periodic_update(self):
        """Periodic GUI update"""
        self.update_display()
        self.root.after(100, self.periodic_update)  # Update every 100ms
    
    def run(self):
        """Start the GUI or headless mode"""
        if self.headless_mode:
            rospy.loginfo("Spot Simulation running in headless mode")
            try:
                rospy.spin()  # Keep node alive for ROS message processing
            except rospy.ROSInterruptException:
                rospy.loginfo("Spot Simulation headless mode shutting down")
        else:
            rospy.loginfo("Spot Simulation GUI started")
            
            def on_closing():
                rospy.loginfo("Spot Simulation GUI shutting down")
                rospy.signal_shutdown("GUI closed")
                self.root.destroy()
            
            self.root.protocol("WM_DELETE_WINDOW", on_closing)
            self.root.mainloop()

if __name__ == '__main__':
    try:
        gui = SpotSimulationGUI()
        gui.run()
    except rospy.ROSInterruptException:
        pass
