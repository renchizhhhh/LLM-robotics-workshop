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

# Import world manager for dynamic world loading
from world_manager import WorldManager

# Check for display availability before importing tkinter
try:
    import tkinter as tk
    from tkinter import ttk
    GUI_AVAILABLE = True
except Exception as e:
    rospy.logwarn(f"GUI not available: {e}")
    GUI_AVAILABLE = False

class SpotSimulationGUI:
    def __init__(self, world_id=None):
        rospy.init_node('spot_simulation_gui', anonymous=True)
        
        # Initialize world manager
        self.world_manager = WorldManager()
        
        # Load world configuration
        if world_id and world_id in self.world_manager.worlds:
            self.world_config = self.world_manager.get_world_config(world_id)
            self.world_name, self.world_description = self.world_manager.get_world_info(world_id)
            rospy.loginfo(f"GUI loaded world: {self.world_name} - {self.world_description}")
        else:
            # Default to simple world if no world_id provided
            self.world_config = self.world_manager.get_world_config("1")
            self.world_name, self.world_description = self.world_manager.get_world_info("1")
            rospy.loginfo(f"GUI using default world: {self.world_name}")
        
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
        
        # Dynamic world objects based on configuration
        self.objects = self._create_dynamic_objects()
        
        # Dynamic world landmarks based on configuration
        self.waypoints = self.world_config.get("waypoints", {})
        self.zones = self.world_config.get("zones", {})
        
        # Position tracking - convert vision frame to display coordinates
        self.vision_to_body_offset_x = 0.0
        self.vision_to_body_offset_y = 0.0
        self.start_position_set = False
        
        # GUI setup
        try:
            self.root = tk.Tk()
            self.root.title(f"Spot Robot Simulation - {self.world_name}")
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
        
        # World info label
        self.world_label = tk.Label(self.info_frame, text=f"World: {self.world_name}", font=("Arial", 12, "bold"), 
                                   bg='#1a1a1a', fg='#39d353')
        self.world_label.grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=15, pady=5)
        
        # World selection button
        self.world_button = tk.Button(self.info_frame, text="Change World", font=("Arial", 10, "bold"),
                                     bg='#2d2d2d', fg='#f0f6fc', relief='flat', bd=1,
                                     command=self.show_world_selection)
        self.world_button.grid(row=2, column=2, sticky=tk.E, padx=15, pady=5)
        
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
        
        # Robot control section (right panel)
        self.ui_frame = tk.Frame(self.root, bg='#2d2d2d', relief='flat', bd=0)
        self.ui_frame.place(relx=0.66, rely=0.82, relwidth=0.33, relheight=0.12)
        
        # Title
        title_label = tk.Label(self.ui_frame, text="Robot Control", 
                             font=("SF Pro Display", 12, "bold"), bg='#2d2d2d', fg='#f0f6fc')
        title_label.pack(pady=(8, 0))
        
        # Text input
        self.command_entry = tk.Entry(self.ui_frame, font=("SF Pro Text", 11), 
                                    bg='#3a3a3a', fg='#f0f6fc', insertbackground='#f0f6fc',
                                    relief='flat', bd=8, highlightthickness=0)
        self.command_entry.pack(fill=tk.X, padx=12, pady=6)
        self.command_entry.bind('<Return>', self._on_send_command)
        
        # Button frame
        button_frame = tk.Frame(self.ui_frame, bg='#2d2d2d')
        button_frame.pack(fill=tk.X, padx=12, pady=(0, 8))
        
        # Buttons with Apple-like styling
        self.refresh_button = tk.Button(button_frame, text="Clear", command=self._on_refresh,
                                      bg='#3a3a3a', fg='#f0f6fc', font=("SF Pro Text", 10, "bold"),
                                      activebackground='#4a4a4a', activeforeground='#f0f6fc',
                                      relief='flat', bd=0, padx=12, pady=6, highlightthickness=0)
        self.refresh_button.pack(side=tk.LEFT, padx=3)
        
        self.send_button = tk.Button(button_frame, text="Send", command=self._on_send_command,
                                   bg='#007AFF', fg='white', font=("SF Pro Text", 10, "bold"),
                                   activebackground='#0056CC', activeforeground='white',
                                   relief='flat', bd=0, padx=12, pady=6, highlightthickness=0)
        self.send_button.pack(side=tk.LEFT, padx=3)
        
        self.approve_button = tk.Button(button_frame, text="Approve", command=self._on_approve,
                                      bg='#34C759', fg='white', font=("SF Pro Text", 10, "bold"),
                                      activebackground='#28A745', activeforeground='white',
                                      relief='flat', bd=0, padx=12, pady=6, highlightthickness=0)
        self.approve_button.pack(side=tk.LEFT, padx=3)
        
        self.decline_button = tk.Button(button_frame, text="Dismiss", command=self._on_decline,
                                      bg='#FF3B30', fg='white', font=("SF Pro Text", 10, "bold"),
                                      activebackground='#D70015', activeforeground='white',
                                      relief='flat', bd=0, padx=12, pady=6, highlightthickness=0)
        self.decline_button.pack(side=tk.LEFT, padx=3)
        
        # Initially disable approve/decline buttons
        self.approve_button.config(state='disabled')
        self.decline_button.config(state='disabled')
        
        # Control buttons with dark theme
        button_frame = tk.Frame(self.root, bg='#1a1a1a')
        button_frame.pack(fill=tk.X, padx=20, pady=15)
        
        self.reset_button = tk.Button(button_frame, text="Reset Simulation", command=self.reset_simulation,
                                     bg='#21262d', fg='#f0f6fc', font=("Arial", 12, "bold"),
                                     activebackground='#30363d', activeforeground='#f0f6fc',
                                     relief='flat', padx=20, pady=10)
        self.reset_button.pack(side=tk.LEFT, padx=10)
        
        self.stop_button = tk.Button(button_frame, text="Stop", command=self._on_stop,
                                    bg='#f85149', fg='#f0f6fc', font=("Arial", 12, "bold"),
                                    activebackground='#ff7b72', activeforeground='#f0f6fc',
                                    relief='flat', padx=20, pady=10)
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        # ROS subscribers
        self.sub_robot_state = rospy.Subscriber('/spot_entrance/robot_state', String, self.on_robot_state)
        self.sub_feedback = rospy.Subscriber('/spot/execution_feedback', String, self.on_execution_feedback)
        self.sub_nl_position = rospy.Subscriber('/nl_control/robot_position', String, self.on_nl_position_update)
        self.sub_interpretation = rospy.Subscriber('/llm_int/interpretation', String, self._on_interpretation)
        
        # ROS publishers for GUI interface
        self.pub_user_speech = rospy.Publisher('/hl/user_speech', String, queue_size=1)
        self.pub_approval = rospy.Publisher('/hl/approval', String, queue_size=1)
        self.pub_stop = rospy.Publisher('/hl/stop', Empty, queue_size=1)
        
        # Threading for GUI updates
        self.update_lock = threading.Lock()
        
        # Start drawing
        self.draw_world()
        self.update_display()
        
        # Schedule periodic updates
        self.root.after(100, self.periodic_update)
        
    def vision_to_canvas(self, x, y):
        """Convert vision frame coordinates to canvas coordinates with natural mapping"""
        # Canvas center (dynamic based on screen size)
        canvas_center_x = self.canvas.winfo_width() // 2 if self.canvas.winfo_width() > 1 else 600
        canvas_center_y = self.canvas.winfo_height() // 2 if self.canvas.winfo_height() > 1 else 400
        
        # Scale: 1 meter = 80 pixels (larger for better visibility)
        scale = 80
        
        # Natural mapping for intuitive display:
        # X (forward) -> Y (upward on screen)
        # Y (left) -> X (leftward on screen)
        # This makes the simulation feel natural for participants
        canvas_x = canvas_center_x - y * scale  # Y (left) -> X (leftward), Y (right) -> X (rightward)
        canvas_y = canvas_center_y - x * scale  # X (forward) -> Y (upward), invert for screen coords
        
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
        
        # Draw waypoints dynamically
        for waypoint_name, waypoint_data in self.waypoints.items():
            x, y = waypoint_data["x"], waypoint_data["y"]
            canvas_x, canvas_y = self.vision_to_canvas(x, y)
            
            # Determine waypoint type and styling
            if waypoint_data.get("pick_yaw") is not None and waypoint_data.get("drop_yaw") is not None:
                # Both pickup and dropoff
                color = "#ffa657"
                outline = "#ff7b72"
                label = "PICK/DROP"
            elif waypoint_data.get("pick_yaw") is not None:
                # Pickup only
                color = "#ffa657"
                outline = "#ff7b72"
                label = "PICKUP"
            elif waypoint_data.get("drop_yaw") is not None:
                # Dropoff only
                color = "#00d4aa"
                outline = "#39d353"
                label = "DROPOFF"
            else:
                # General waypoint
                color = "#58a6ff"
                outline = "#39d353"
                label = waypoint_name.upper()
            
            # Draw waypoint rectangle
            self.canvas.create_rectangle(canvas_x-25, canvas_y-25, canvas_x+25, canvas_y+25, 
                                       fill=color, outline=outline, width=3)
            self.canvas.create_text(canvas_x, canvas_y-40, text=label, font=("Arial", 12, "bold"), fill=color)
            
            # Draw orientation indicator if specified
            if waypoint_data.get("pick_yaw") is not None:
                yaw = waypoint_data["pick_yaw"]
                arrow_length = 20
                arrow_x = canvas_x + arrow_length * math.cos(yaw)
                arrow_y = canvas_y - arrow_length * math.sin(yaw)
                self.canvas.create_line(canvas_x, canvas_y, arrow_x, arrow_y, fill="#ffffff", width=2, arrow=tk.LAST)
        
        # Draw zones dynamically
        for zone_name, zone_data in self.zones.items():
            centroid = zone_data["centroid"]
            x, y = centroid["x"], centroid["y"]
            canvas_x, canvas_y = self.vision_to_canvas(x, y)
            
            # Draw zone as a larger circle
            self.canvas.create_oval(canvas_x-40, canvas_y-40, canvas_x+40, canvas_y+40,
                                   fill="#2d2d2d", outline="#58a6ff", width=2, stipple="gray25")
            self.canvas.create_text(canvas_x, canvas_y, text=zone_name.upper(), font=("Arial", 10, "bold"), fill="#58a6ff")
            
            # Draw orientation hint if available
            if zone_data.get("yaw_hint") is not None:
                yaw = zone_data["yaw_hint"]
                arrow_length = 30
                arrow_x = canvas_x + arrow_length * math.cos(yaw)
                arrow_y = canvas_y - arrow_length * math.sin(yaw)
                self.canvas.create_line(canvas_x, canvas_y, arrow_x, arrow_y, fill="#58a6ff", width=3, arrow=tk.LAST)
        
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
        # Subtract 90 degree offset so robot faces up (X forward) at start
        # Positive yaw is counterclockwise (left) - invert for correct display
        display_yaw = -self.robot_yaw - math.pi/2
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
    
    def _create_dynamic_objects(self):
        """Create dynamic objects based on world configuration."""
        objects = {}
        
        # Create objects near waypoints that have pick_yaw (pickup locations)
        waypoints = self.world_config.get("waypoints", {})
        colors = ["#f85149", "#39d353", "#58a6ff", "#ffa657", "#ff7b72", "#00d4aa"]
        # Only drink objects for pickup locations
        object_names = ["water_bottle", "soda_can", "juice_box", "coffee_cup", "energy_drink", "sports_drink"]
        
        color_idx = 0
        name_idx = 0
        
        for waypoint_name, waypoint_data in waypoints.items():
            if waypoint_data.get("pick_yaw") is not None:
                # This is a pickup location, add objects nearby
                x = waypoint_data["x"]
                y = waypoint_data["y"]
                
                # Add 2-3 objects near this pickup location
                for i in range(2):
                    offset_x = (i - 0.5) * 0.4  # Spread objects around the waypoint
                    offset_y = (i % 2) * 0.3
                    
                    obj_name = object_names[name_idx % len(object_names)]
                    obj_color = colors[color_idx % len(colors)]
                    
                    objects[obj_name] = {
                        "x": x + offset_x,
                        "y": y + offset_y,
                        "present": True,
                        "color": obj_color,
                        "waypoint": waypoint_name
                    }
                    
                    color_idx += 1
                    name_idx += 1
        
        # If no pickup waypoints found, create default drink objects
        if not objects:
            objects = {
                "water_bottle": {"x": 2.0, "y": -1.0, "present": True, "color": "#f85149", "waypoint": "default"},
                "soda_can": {"x": 2.2, "y": -0.8, "present": True, "color": "#39d353", "waypoint": "default"},
                "juice_box": {"x": 1.8, "y": -1.2, "present": True, "color": "#58a6ff", "waypoint": "default"}
            }
        
        return objects
    
    def show_world_selection(self):
        """Show world selection dialog."""
        # Create world selection window
        world_window = tk.Toplevel(self.root)
        world_window.title("Select World")
        world_window.configure(bg='#1a1a1a')
        world_window.geometry("600x500")
        world_window.transient(self.root)
        world_window.grab_set()
        
        # Center the window
        world_window.geometry("+%d+%d" % (self.root.winfo_rootx() + 50, self.root.winfo_rooty() + 50))
        
        # Title
        title_label = tk.Label(world_window, text="Select World Configuration", 
                               font=("Arial", 16, "bold"), bg='#1a1a1a', fg='#f0f6fc')
        title_label.pack(pady=20)
        
        # Create scrollable frame for world list
        canvas = tk.Canvas(world_window, bg='#1a1a1a', highlightthickness=0)
        scrollbar = tk.Scrollbar(world_window, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#1a1a1a')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add world options
        world_list = self.world_manager.get_world_list()
        selected_world = tk.StringVar()
        
        for world_id, name, description in world_list:
            frame = tk.Frame(scrollable_frame, bg='#2d2d2d', relief='flat', bd=1)
            frame.pack(fill=tk.X, padx=10, pady=5)
            
            radio = tk.Radiobutton(frame, text=f"{world_id}. {name}", 
                                  variable=selected_world, value=world_id,
                                  font=("Arial", 12, "bold"), bg='#2d2d2d', fg='#f0f6fc',
                                  selectcolor='#1a1a1a', activebackground='#2d2d2d')
            radio.pack(anchor=tk.W, padx=10, pady=5)
            
            desc_label = tk.Label(frame, text=description, font=("Arial", 10), 
                                 bg='#2d2d2d', fg='#8b949e', wraplength=500)
            desc_label.pack(anchor=tk.W, padx=30, pady=(0, 10))
        
        # Pack canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        scrollbar.pack(side="right", fill="y")
        
        # Buttons
        button_frame = tk.Frame(world_window, bg='#1a1a1a')
        button_frame.pack(fill=tk.X, padx=20, pady=20)
        
        def on_select():
            world_id = selected_world.get()
            if world_id:
                self.load_world(world_id)
                world_window.destroy()
        
        def on_cancel():
            world_window.destroy()
        
        select_btn = tk.Button(button_frame, text="Select World", command=on_select,
                              font=("Arial", 12, "bold"), bg='#00d4aa', fg='#000000',
                              relief='flat', bd=0, padx=20, pady=10)
        select_btn.pack(side=tk.LEFT, padx=10)
        
        cancel_btn = tk.Button(button_frame, text="Cancel", command=on_cancel,
                              font=("Arial", 12, "bold"), bg='#f85149', fg='#ffffff',
                              relief='flat', bd=0, padx=20, pady=10)
        cancel_btn.pack(side=tk.LEFT, padx=10)
    
    def load_world(self, world_id):
        """Load a new world configuration."""
        if world_id in self.world_manager.worlds:
            # Update world configuration
            self.world_config = self.world_manager.get_world_config(world_id)
            self.world_name, self.world_description = self.world_manager.get_world_info(world_id)
            
            # Update waypoints and zones
            self.waypoints = self.world_config.get("waypoints", {})
            self.zones = self.world_config.get("zones", {})
            
            # Update objects
            self.objects = self._create_dynamic_objects()
            
            # Update GUI elements
            self.root.title(f"Spot Robot Simulation - {self.world_name}")
            self.world_label.config(text=f"World: {self.world_name}")
            
            # Redraw the world
            self.draw_world()
            self.draw_robot()
            
            rospy.loginfo(f"Loaded world: {self.world_name} - {self.world_description}")
    
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
        
        # Objects for state tracking - positioned diagonally around pickup location
        self.objects = {
            "water_bottle": {"x": 3.0, "y": -1.0, "present": True, "color": "#f85149"},
            "soda_can": {"x": 2.8, "y": -0.8, "present": True, "color": "#39d353"},
            "juice_box": {"x": 3.2, "y": -1.2, "present": True, "color": "#58a6ff"}
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
    
    def on_nl_position_update(self, msg):
        """Handle position updates from NL_Control"""
        try:
            # Parse position from NL_Control
            import json
            position_data = json.loads(msg.data)
            self.robot_x = position_data['x']
            self.robot_y = position_data['y'] 
            self.robot_yaw = position_data['yaw']
            rospy.loginfo(f"Updated robot position from NL_Control: ({self.robot_x:.2f}, {self.robot_y:.2f}, {self.robot_yaw:.2f})")
        except Exception as e:
            rospy.logwarn(f"Failed to parse NL_Control position: {e}")

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
                                
                                # Position updates are now handled by NL_Control position subscriber
                                # No need to manually track position here
                                rospy.loginfo(f"Robot moved by ({body_x}, {body_y}, {body_yaw}) in body frame")
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
            self.robot_yaw = 0.0  # Robot starts facing forward (X direction)
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
            # Display the actual NL_Control position (not the display yaw)
            actual_yaw = math.degrees(self.robot_yaw)
            self.position_label.config(text=f"Position: ({self.robot_x:.2f}, {self.robot_y:.2f}, {actual_yaw:.1f}°)")
            self.action_label.config(text=f"Action: {self.current_action}")
            arm_text = f"Arm: {self.arm_status} | Gripper: {self.gripper_status}"
            if self.has_object:
                arm_text += " | Carrying object"
            self.arm_label.config(text=arm_text)
    
    def periodic_update(self):
        """Periodic GUI update"""
        self.update_display()
        self.root.after(100, self.periodic_update)  # Update every 100ms
    
    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling in feedback text"""
        self.feedback_text.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def _on_send_command(self, event=None):
        """Send command to robot"""
        command = self.command_entry.get().strip()
        if not command:
            return
        
        rospy.loginfo(f"GUI: Sending command: '{command}'")
        self.pub_user_speech.publish(String(data=command))
        
        # Clear text entry
        self.command_entry.delete(0, tk.END)
        
        # Enable approve/decline buttons
        self.approve_button.config(state='normal')
        self.decline_button.config(state='normal')
    
    def _on_refresh(self):
        """Clear text entry"""
        self.command_entry.delete(0, tk.END)
        self.approve_button.config(state='disabled')
        self.decline_button.config(state='disabled')
    
    def _on_approve(self):
        """Approve pending plan"""
        rospy.loginfo("GUI: Approving plan")
        import json
        self.pub_approval.publish(String(data=json.dumps({"plan_id": 0, "approved": True})))
        self.approve_button.config(state='disabled')
        self.decline_button.config(state='disabled')
    
    def _on_decline(self):
        """Decline pending plan"""
        rospy.loginfo("GUI: Declining plan")
        import json
        self.pub_approval.publish(String(data=json.dumps({"plan_id": 0, "approved": False})))
        self.approve_button.config(state='disabled')
        self.decline_button.config(state='disabled')
    
    def _on_interpretation(self, msg):
        """Handle natural language interpretation from orchestrator"""
        try:
            interpretation = msg.data.strip()
            if interpretation:
                rospy.loginfo(f"GUI: Received interpretation: {interpretation}")
                # Update feedback text with only natural language interpretation
                self.feedback_text.config(state=tk.NORMAL)
                self.feedback_text.delete(1.0, tk.END)
                
                # Add only the interpretation, no coordinate system info
                self.feedback_text.insert(tk.END, interpretation)
                self.feedback_text.config(state=tk.DISABLED)
        except Exception as e:
            rospy.logwarn(f"Failed to parse interpretation: {e}")
    
    def _on_stop(self):
        """Send stop signal to robot"""
        rospy.loginfo("GUI: Sending stop signal")
        self.pub_stop.publish(Empty())
    
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
    import sys
    
    # Check for world selection argument
    world_id = None
    if len(sys.argv) > 1:
        world_id = sys.argv[1]
        print(f"Starting GUI with world: {world_id}")
    
    try:
        gui = SpotSimulationGUI(world_id=world_id)
        gui.run()
    except rospy.ROSInterruptException:
        pass
