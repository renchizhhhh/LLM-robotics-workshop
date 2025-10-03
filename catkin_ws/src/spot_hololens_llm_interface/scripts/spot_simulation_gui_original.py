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
from std_msgs.msg import String, Empty
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
        self.robot_x = 0.0  # World frame position (same as NL_Control)
        self.robot_y = 0.0
        self.robot_yaw = 0.0  # Radians - robot starts facing forward (X direction)
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
        
        # World landmarks (in vision frame) - pickup rechts, dropoff links
        self.pickup_location = {"x": 3.0, "y": -1.0}  # Pickup at (3, -1) - rechts van robot
        self.dropoff_location = {"x": 2.0, "y": 1.5}  # Dropoff at (2, 1.5) - links van robot
        
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
        
        # Natural language feedback section (middle panel)
        self.feedback_frame = tk.Frame(self.root, bg='#2d2d2d', relief='flat', bd=0)
        self.feedback_frame.place(relx=0.33, rely=0.82, relwidth=0.33, relheight=0.12)
        
        # Create scrollable text widget for NL feedback
        self.feedback_text = tk.Text(self.feedback_frame, 
                                   font=("SF Pro Text", 11), bg='#2d2d2d', fg='#f0f6fc',
                                   wrap=tk.WORD, height=4, width=40,
                                   insertbackground='#f0f6fc', selectbackground='#404040',
                                   relief='flat', bd=0, padx=8, pady=8)
        
        # Add scrollbar
        scrollbar = tk.Scrollbar(self.feedback_frame, orient=tk.VERTICAL, command=self.feedback_text.yview)
        self.feedback_text.configure(yscrollcommand=scrollbar.set)
        
        # Pack text widget and scrollbar
        self.feedback_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Insert initial text
        initial_text = """Ready for commands"""
        self.feedback_text.insert(tk.END, initial_text)
        self.feedback_text.config(state=tk.DISABLED)  # Make read-only
        
        # Bind mouse wheel scrolling
        self.feedback_text.bind("<MouseWheel>", self._on_mousewheel)
        
        # Control panel (right side)
        self.control_frame = tk.Frame(self.root, bg='#1a1a1a')
        self.control_frame.place(relx=0.7, rely=0.05, relwidth=0.28, relheight=0.75)
        
        # Control title
        control_title = tk.Label(self.control_frame, text="Robot Controls", font=("Arial", 16, "bold"), 
                                bg='#1a1a1a', fg='#f0f6fc')
        control_title.pack(pady=10)
        
        # Emergency stop button
        self.emergency_stop = tk.Button(self.control_frame, text="EMERGENCY STOP", 
                                       font=("Arial", 14, "bold"), bg='#f85149', fg='#ffffff',
                                       relief='flat', bd=0, padx=20, pady=15,
                                       command=self.emergency_stop)
        self.emergency_stop.pack(pady=10)
        
        # Status display
        status_frame = tk.Frame(self.control_frame, bg='#2d2d2d', relief='flat', bd=1)
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(status_frame, text="Robot Status", font=("Arial", 12, "bold"), 
                bg='#2d2d2d', fg='#f0f6fc').pack(pady=5)
        
        self.status_text = tk.Text(status_frame, height=8, width=30, font=("Arial", 10),
                                 bg='#1a1a1a', fg='#f0f6fc', relief='flat', bd=0, padx=5, pady=5)
        self.status_text.pack(padx=5, pady=5)
        
        # Initialize status
        self.update_status("Robot initialized\nWaiting for commands...")
        
        # ROS subscribers
        self.sub_position = rospy.Subscriber('/nl_control/robot_position', String, self.on_position_update)
        self.sub_feedback = rospy.Subscriber('/spot/execution_feedback', String, self.on_execution_feedback)
        
        # Start the visualization loop
        self.root.after(100, self.update_visualization)
        
        rospy.loginfo("Spot Simulation GUI initialized")
    
    def setup_headless_mode(self):
        """Setup headless mode with minimal state tracking"""
        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0  # Robot starts facing forward (X direction)
        self.robot_state = "unknown"
        self.current_action = "idle"
        self.has_object = False
        self.carried_object_name = None
        self.arm_status = "stowed"
        self.gripper_status = "closed"
        
        # World objects (in vision frame coordinates)
        self.objects = {
            "tomato_can": {"x": 3.0, "y": -1.0, "present": True, "color": "#f85149"},
            "apple": {"x": 2.8, "y": -0.8, "present": True, "color": "#39d353"},
            "bottle": {"x": 3.2, "y": -1.2, "present": True, "color": "#58a6ff"}
        }
        
        # World landmarks (in vision frame)
        self.pickup_location = {"x": 3.0, "y": -1.0}
        self.dropoff_location = {"x": 2.0, "y": 1.5}
        
        # Position tracking
        self.vision_to_body_offset_x = 0.0
        self.vision_to_body_offset_y = 0.0
        self.start_position_set = False
        
        # ROS subscribers for headless mode
        self.sub_position = rospy.Subscriber('/nl_control/robot_position', String, self.on_position_update_headless)
        self.sub_feedback = rospy.Subscriber('/spot/execution_feedback', String, self.on_execution_feedback_headless)
        
        rospy.loginfo("Spot Simulation headless mode initialized")
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling for feedback text"""
        self.feedback_text.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def vision_to_canvas(self, x, y):
        """Convert vision frame coordinates to canvas coordinates"""
        # Scale factor: 1 meter = 100 pixels
        scale = 100
        
        # Canvas center
        canvas_width = self.canvas.winfo_width() if self.canvas.winfo_width() > 1 else 1200
        canvas_height = self.canvas.winfo_height() if self.canvas.winfo_height() > 1 else 800
        center_x = canvas_width // 2
        center_y = canvas_height // 2
        
        # Convert to canvas coordinates (Y-axis flipped for display)
        canvas_x = center_x + x * scale
        canvas_y = center_y - y * scale  # Flip Y-axis for display
        
        return canvas_x, canvas_y
    
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
        """Draw the robot at its current position"""
        # Convert robot position to canvas coordinates
        robot_canvas_x, robot_canvas_y = self.vision_to_canvas(self.robot_x, self.robot_y)
        
        # Robot dimensions (scaled)
        robot_length = 40  # pixels
        robot_width = 20   # pixels
        
        # Calculate robot corners based on orientation
        cos_yaw = math.cos(self.robot_yaw)
        sin_yaw = math.sin(self.robot_yaw)
        
        # Robot body corners (relative to center)
        half_length = robot_length / 2
        half_width = robot_width / 2
        
        # Calculate corners in robot frame
        corners = [
            (-half_length, -half_width),  # Front-left
            (half_length, -half_width),   # Front-right
            (half_length, half_width),    # Rear-right
            (-half_length, half_width)   # Rear-left
        ]
        
        # Rotate corners to world frame
        rotated_corners = []
        for x, y in corners:
            # Rotate around robot center
            new_x = x * cos_yaw - y * sin_yaw
            new_y = x * sin_yaw + y * cos_yaw
            rotated_corners.append((robot_canvas_x + new_x, robot_canvas_y + new_y))
        
        # Draw robot body
        self.canvas.create_polygon(rotated_corners, fill="#2d2d2d", outline="#f0f6fc", width=2, tags="robot")
        
        # Draw robot direction indicator (arrow pointing forward)
        arrow_length = 30
        arrow_x = robot_canvas_x + arrow_length * cos_yaw
        arrow_y = robot_canvas_y - arrow_length * sin_yaw  # Negative because Y is flipped
        self.canvas.create_line(robot_canvas_x, robot_canvas_y, arrow_x, arrow_y, 
                               fill="#00d4aa", width=4, arrow=tk.LAST, tags="robot")
        
        # Draw gripper if arm is not stowed
        if self.arm_status != "stowed":
            self.draw_gripper(robot_canvas_x, robot_canvas_y, cos_yaw, sin_yaw, robot_length)
    
    def draw_gripper(self, robot_x, robot_y, cos_yaw, sin_yaw, robot_length):
        """Draw the robot's gripper"""
        # Gripper position (in front of robot)
        gripper_distance = robot_length * 0.8
        gripper_x = robot_x + gripper_distance * cos_yaw
        gripper_y = robot_y - gripper_distance * sin_yaw  # Negative because Y is flipped
        
        if self.gripper_status == "open":
            # Open gripper - two parallel lines
            gripper_width = 15
            left_x = gripper_x - gripper_width * sin_yaw
            left_y = gripper_y - gripper_width * cos_yaw
            right_x = gripper_x + gripper_width * sin_yaw
            right_y = gripper_y + gripper_width * cos_yaw
            
            self.canvas.create_line(gripper_x, gripper_y, left_x, left_y, 
                                   fill="#f85149", width=3, tags="robot")
            self.canvas.create_line(gripper_x, gripper_y, right_x, right_y, 
                                   fill="#f85149", width=3, tags="robot")
        else:
            # Closed gripper - single smaller indicator
            self.canvas.create_oval(gripper_x - 3, gripper_y - 3, gripper_x + 3, gripper_y + 3,
                                  fill="#f85149", outline="#f0f6fc", width=1, tags="robot")
    
    def update_visualization(self):
        """Update the visualization"""
        if not self.headless_mode:
            self.draw_world()
            self.draw_robot()
        
        # Schedule next update
        self.root.after(100, self.update_visualization)
    
    def update_status(self, message):
        """Update the status display"""
        if not self.headless_mode:
            self.status_text.config(state=tk.NORMAL)
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(tk.END, message)
            self.status_text.config(state=tk.DISABLED)
    
    def emergency_stop(self):
        """Emergency stop button"""
        rospy.logwarn("EMERGENCY STOP ACTIVATED")
        self.update_status("EMERGENCY STOP ACTIVATED\nAll operations halted")
        
        # Publish emergency stop message
        pub = rospy.Publisher('/hl/stop', Empty, queue_size=1)
        pub.publish(Empty())
    
    def on_position_update(self, msg):
        """Handle position updates from NL control"""
        try:
            position_data = json.loads(msg.data)
            self.robot_x = position_data['x']
            self.robot_y = position_data['y']
            self.robot_yaw = position_data['yaw']
            
            # Update position display
            self.position_label.config(text=f"Position: ({self.robot_x:.2f}, {self.robot_y:.2f}, {math.degrees(self.robot_yaw):.1f}°)")
            
            # Update status
            status_msg = f"Robot at ({self.robot_x:.2f}, {self.robot_y:.2f})\nOrientation: {math.degrees(self.robot_yaw):.1f}°\nAction: {self.current_action}"
            if self.has_object:
                status_msg += f"\nCarrying: {self.carried_object_name}"
            self.update_status(status_msg)
            
        except Exception as e:
            rospy.logwarn(f"Failed to parse position update: {e}")
    
    def on_position_update_headless(self, msg):
        """Handle position updates in headless mode"""
        try:
            position_data = json.loads(msg.data)
            self.robot_x = position_data['x']
            self.robot_y = position_data['y']
            self.robot_yaw = position_data['yaw']
            rospy.loginfo(f"Robot position: ({self.robot_x:.2f}, {self.robot_y:.2f}, {math.degrees(self.robot_yaw):.1f}°)")
        except Exception as e:
            rospy.logwarn(f"Failed to parse position update: {e}")
    
    def on_execution_feedback(self, msg):
        """Handle execution feedback from the robot"""
        feedback = msg.data
        
        # Add feedback to the text widget
        self.feedback_text.config(state=tk.NORMAL)
        self.feedback_text.insert(tk.END, f"\n{feedback}")
        self.feedback_text.see(tk.END)
        self.feedback_text.config(state=tk.DISABLED)
        
        # Parse feedback for state updates
        try:
            if "start_moving" in feedback:
                self.current_action = "moving"
                self.action_label.config(text=f"Action: moving", fg='#f85149')
                
                # Extract movement parameters
                try:
                    if "x=" in feedback:
                        x_val = float(feedback.split("x=")[1].split()[0])
                        y_val = float(feedback.split("y=")[1].split()[0]) if "y=" in feedback else 0.0
                        yaw_val = float(feedback.split("yaw=")[1].split()[0]) if "yaw=" in feedback else 0.0
                        
                        # Update robot position (simplified)
                        self.robot_x += x_val
                        self.robot_y += y_val
                        self.robot_yaw += yaw_val
                        
                        # Update position display
                        self.position_label.config(text=f"Position: ({self.robot_x:.2f}, {self.robot_y:.2f}, {math.degrees(self.robot_yaw):.1f}°)")
                except Exception as e:
                    rospy.logwarn(f"Could not extract movement parameters: {e}")
                    
            elif "start_automated_grasp" in feedback:
                self.current_action = "grasping"
                self.action_label.config(text=f"Action: grasping", fg='#ffa657')
                
                # Extract target object
                try:
                    if "object_type=" in feedback:
                        target_object = feedback.split("object_type=")[1].strip()
                        rospy.loginfo(f"Attempting to grasp: {target_object}")
                        
                        # Check if there's an object at current location to grasp
                        self.simulate_grasp_attempt(target_object)
                        
                except Exception as e:
                    rospy.logwarn(f"Could not extract target object: {e}")
                
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
            
            if dropoff_distance <= 2.0:  # Within 2 meters of drop-off
                # Successful drop
                self.has_object = False
                self.carried_object_name = None
                self.gripper_status = "open"
                rospy.loginfo(f"Successfully dropped object at drop-off location (distance: {dropoff_distance:.2f}m)")
            else:
                rospy.logwarn(f"Too far from drop-off location to drop object (distance: {dropoff_distance:.2f}m)")
    
    def run(self):
        """Main loop"""
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
