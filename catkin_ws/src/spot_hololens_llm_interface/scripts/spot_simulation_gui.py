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
        self.robot_x = 1.35  # World frame position - center of cell (4,4) = (4.5 * 0.3, 4.5 * 0.3)
        self.robot_y = 1.35
        self.robot_yaw = 0.0  # Radians - robot starts facing forward (X direction)
        self.robot_row = 4  # Grid position - start in true middle (4,4)
        self.robot_col = 4
        self.robot_facing = "N"  # N, S, E, W
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
        
        # Convert grid coordinates to x,y coordinates for GUI
        self.waypoints = self._convert_grid_to_xy(self.waypoints)
        self.zones = self._convert_grid_to_xy(self.zones)
        
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
        self.root.rowconfigure(0, weight=1)  # Canvas row grows with window
        self.root.rowconfigure(1, weight=0)  # Info frame - fixed height
        self.root.rowconfigure(2, weight=0)  # Bottom frame - fixed height
        self.root.columnconfigure(0, weight=1)
        
        # Get screen dimensions for fullscreen canvas
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Create canvas for visualization (larger, fullscreen)
        canvas_width = screen_width - 50
        canvas_height = screen_height - 250  # More space for bottom panels
        self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height, bg='#0d1117')
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=20, pady=10)
        
        # Info panel with dark theme - compact padding
        self.info_frame = tk.Frame(self.root, bg='#1a1a1a')
        self.info_frame.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 5))
        self.root.grid_rowconfigure(1, weight=0)
        # Configure column weights for proper spacing
        self.info_frame.columnconfigure(0, weight=0)  # Left column (status labels)
        self.info_frame.columnconfigure(1, weight=0)  # Middle column (world info)
        self.info_frame.columnconfigure(2, weight=1)  # Right column (model/prompt) - expands
        
        # Status labels - compact size with minimal padding
        self.state_label = tk.Label(self.info_frame, text="State: unknown", font=("Arial", 10, "bold"), 
                                   bg='#1a1a1a', fg='#00d4aa')
        self.state_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=2)
        
        self.position_label = tk.Label(self.info_frame, text="Position: (0.0, 0.0, 0.0°)", font=("Arial", 10, "bold"), 
                                      bg='#1a1a1a', fg='#58a6ff')
        self.position_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=2)
        
        self.action_label = tk.Label(self.info_frame, text="Action: idle", font=("Arial", 10, "bold"), 
                                    bg='#1a1a1a', fg='#f85149')
        self.action_label.grid(row=2, column=0, sticky=tk.W, padx=10, pady=2)
        
        self.arm_label = tk.Label(self.info_frame, text="Arm: stowed | Gripper: closed", font=("Arial", 10, "bold"), 
                                 bg='#1a1a1a', fg='#ffa657')
        self.arm_label.grid(row=3, column=0, sticky=tk.W, padx=10, pady=2)
        
        # World info label - compact
        self.world_label = tk.Label(self.info_frame, text=f"World: {self.world_name}", font=("Arial", 10, "bold"), 
                                   bg='#1a1a1a', fg='#39d353')
        self.world_label.grid(row=0, column=1, sticky=tk.W, padx=10, pady=2)
        
        # World selection button
        self.world_button = tk.Button(self.info_frame, text="Change World", font=("Arial", 10, "bold"),
                                     bg='#2d2d2d', fg='#f0f6fc', relief='flat', bd=1,
                                     command=self.show_world_selection)
        self.world_button.grid(row=1, column=1, sticky=tk.W, padx=15, pady=5)
        
        # LLM Model selector - positioned on right side (compact)
        model_frame = tk.Frame(self.info_frame, bg='#1a1a1a')
        model_frame.grid(row=0, column=2, sticky=tk.E, padx=10, pady=2)
        
        model_label = tk.Label(model_frame, text="Model:", font=("Arial", 10, "bold"),
                              bg='#1a1a1a', fg='#ffa657')
        model_label.pack(side=tk.LEFT, padx=(0, 3))
        
        # Check if GOOGLE_API_KEY is set
        import os
        gemini_available = bool(os.getenv('GOOGLE_API_KEY'))
        
        # Model options with dark mode styling
        self.model_var = tk.StringVar(value='120B Low Reasoning')
        
        # Create custom styled combobox for dark mode
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var,
                                       font=("Arial", 10), state='readonly', width=20)
        
        # Configure dark mode styling for combobox
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox',
                       fieldbackground='#2d2d2d',
                       background='#2d2d2d',
                       foreground='#f0f6fc',
                       borderwidth=1,
                       relief='flat')
        style.map('TCombobox',
                 fieldbackground=[('readonly', '#2d2d2d')],
                 background=[('readonly', '#2d2d2d')],
                 foreground=[('readonly', '#f0f6fc')])
        
        # Set available models
        if gemini_available:
            self.model_combo['values'] = ('120B Low Reasoning', '120B Medium Reasoning', '120B High Reasoning', 'Gemini Robotics 1.5', 'Gemini Pro 2.5')
        else:
            self.model_combo['values'] = ('120B Low Reasoning', '120B Medium Reasoning', '120B High Reasoning', 'Gemini Robotics 1.5 (API key required)', 'Gemini Pro 2.5 (API key required)')
        
        self.model_combo.current(0)
        self.model_combo.pack(side=tk.LEFT)
        
        # Publisher for model selection
        self.pub_model_select = rospy.Publisher('/nl_control/model_select', String, queue_size=1)
        
        # Bind selection event
        def on_model_change(event):
            selection = self.model_combo.get()
            # Map display names to model IDs
            model_map = {
                '120B Low Reasoning': 'base-120b-low',
                '120B Medium Reasoning': 'base-120b-medium', 
                '120B High Reasoning': 'base-120b-high',
                'Gemini Robotics 1.5': 'gemini-robotics-er-1.5-preview',
                'Gemini Robotics 1.5 (API key required)': 'gemini-robotics-er-1.5-preview',
                'Gemini Pro 2.5': 'gemini-2.5-pro',
                'Gemini Pro 2.5 (API key required)': 'gemini-2.5-pro'
            }
            model_id = model_map.get(selection, 'base-120b')
            
            # Only publish if not a disabled option
            if gemini_available or model_id.startswith('base-120b'):
                self.pub_model_select.publish(String(data=model_id))
                rospy.loginfo(f"Model selection published: {model_id}")
            else:
                # Reset to 120B Low Reasoning if GOOGLE_API_KEY not set
                self.model_combo.set('120B Low Reasoning')
                rospy.logwarn("GOOGLE_API_KEY not set - cannot select Gemini models")
        
        self.model_combo.bind('<<ComboboxSelected>>', on_model_change)
        
        # Prompt Type selector - right next to model selector
        prompt_label = tk.Label(model_frame, text="Prompt:", font=("Arial", 10, "bold"),
                               bg='#1a1a1a', fg='#58a6ff')
        prompt_label.pack(side=tk.LEFT, padx=(10, 3))
        
        self.prompt_var = tk.StringVar(value='Base')
        self.prompt_combo = ttk.Combobox(model_frame, textvariable=self.prompt_var,
                                        font=("Arial", 10), state='readonly', width=16)

        # Populate prompt options from discovered prompt files if available
        self._available_prompts = {}
        try:
            # Try to import get_available_prompts from nl_control
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'spot_hololens_llm_interface'))
            from nl_control import get_available_prompts
            if callable(get_available_prompts):
                self._available_prompts = get_available_prompts()  # {name: text}
        except Exception:
            self._available_prompts = {}

        if self._available_prompts:
            # Present user-friendly capitalized names in combobox
            prompt_names = [name.capitalize() for name in sorted(self._available_prompts.keys())]
            self.prompt_combo['values'] = prompt_names
            # Default to 'Base' if present, else first discovered
            if 'base' in self._available_prompts:
                default_index = prompt_names.index('Base') if 'Base' in prompt_names else 0
            else:
                default_index = 0
            self.prompt_combo.current(default_index)
        else:
            # Fallback to the original two-option list
            self.prompt_combo['values'] = ('Base', 'Reasoning')
            self.prompt_combo.current(0)

        self.prompt_combo.pack(side=tk.LEFT)
        
        # Publisher for prompt type selection
        self.pub_prompt_select = rospy.Publisher('/nl_control/prompt_select', String, queue_size=1)
        
        # Bind prompt selection event
        def on_prompt_change(event):
            selection = self.prompt_combo.get()
            # Map display name back to prompt name
            prompt_name = selection.lower()
            # If we have available prompts, ensure mapping exists
            if self._available_prompts:
                # selection was capitalized; convert back to key
                key = prompt_name
                if key in self._available_prompts:
                    prompt_name = key
                else:
                    # attempt to find matching key ignoring case
                    for k in self._available_prompts.keys():
                        if k.lower() == selection.lower():
                            prompt_name = k
                            break

            # Publish the prompt name (e.g., 'base' or 'reasoning') for nl_control
            self.pub_prompt_select.publish(String(data=prompt_name))
            rospy.loginfo(f"Prompt selection published: {prompt_name}")
        
        self.prompt_combo.bind('<<ComboboxSelected>>', on_prompt_change)

        # Publish initial prompt selection to NL control so it can pick up the correct prompt at startup
        try:
            initial_selection = self.prompt_combo.get()
            # Reuse handler logic to determine prompt name
            event = type('E', (), {'widget': None})()
            on_prompt_change(event)
        except Exception:
            pass
        
        # Tooltip for disabled options
        if not gemini_available:
            tooltip_label = tk.Label(model_frame, text="💡 Set GOOGLE_API_KEY to enable Gemini models",
                                    font=("Arial", 9), bg='#1a1a1a', fg='#8b949e')
            tooltip_label.pack(side=tk.LEFT, padx=10)
        
        # Reset and Stop buttons - positioned on the far right below model/prompt selectors
        self.reset_button = tk.Button(model_frame, text="Reset", command=self.reset_simulation,
                                     bg='#21262d', fg='#f0f6fc', font=("Arial", 10, "bold"),
                                     activebackground='#30363d', activeforeground='#f0f6fc',
                                     relief='flat', bd=1, padx=8, pady=4)
        self.reset_button.pack(side=tk.LEFT, padx=(10, 5))
        
        self.stop_button = tk.Button(model_frame, text="Stop", command=self._on_stop,
                                    bg='#f85149', fg='#f0f6fc', font=("Arial", 10, "bold"),
                                    activebackground='#ff7b72', activeforeground='#f0f6fc',
                                    relief='flat', bd=1, padx=8, pady=4)
        self.stop_button.pack(side=tk.LEFT, padx=(5, 0))
        
        # Footer container keeps lower panels anchored regardless of window manager quirks
        self.bottom_frame = tk.Frame(self.root, bg='#1a1a1a')
        self.bottom_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 10))
        self.root.grid_rowconfigure(2, weight=0)
        self.bottom_frame.columnconfigure(0, weight=1)
        self.bottom_frame.columnconfigure(1, weight=1)
        self.bottom_frame.rowconfigure(0, weight=1)
        
        # Natural language feedback section (middle panel)
        self.feedback_frame = tk.Frame(self.bottom_frame, bg='#2d2d2d', relief='flat', bd=0)
        self.feedback_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
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
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        initial_text = f"[{timestamp}] System initialized - Ready for commands"
        self.feedback_text.insert(tk.END, initial_text)
        self.feedback_text.config(state=tk.DISABLED)  # Make read-only
        
        # Bind mouse wheel scrolling
        self.feedback_text.bind("<MouseWheel>", self._on_mousewheel)

        # Helper to append timestamped lines (mirrors standalone)
        def _update_feedback_text(message):
            try:
                import datetime
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                formatted = f"[{timestamp}] {message}"
                self.feedback_text.config(state=tk.NORMAL)
                if self.feedback_text.get(1.0, tk.END).strip():
                    self.feedback_text.insert(tk.END, "\n")
                self.feedback_text.insert(tk.END, formatted)
                self.feedback_text.config(state=tk.DISABLED)
                self.feedback_text.see(tk.END)
            except Exception:
                pass
        # Store as instance method reference
        self._update_feedback_text = _update_feedback_text
        
        # Robot control section (right panel)
        self.ui_frame = tk.Frame(self.bottom_frame, bg='#2d2d2d', relief='flat', bd=0)
        self.ui_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        
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
        
        # ROS subscribers
        self.sub_robot_state = rospy.Subscriber('/spot_entrance/robot_state', String, self.on_robot_state)
        self.sub_feedback = rospy.Subscriber('/spot/execution_feedback', String, self.on_execution_feedback)
        self.sub_nl_position = rospy.Subscriber('/nl_control/robot_position', String, self.on_nl_position_update)
        self.sub_interpretation = rospy.Subscriber('/llm_int/interpretation', String, self._on_interpretation)
        
        # ROS publishers for GUI interface
        self.pub_user_speech = rospy.Publisher('/hl/user_speech', String, queue_size=1)
        self.pub_approval = rospy.Publisher('/hl/approval', String, queue_size=1)
        self.pub_stop = rospy.Publisher('/hl/stop', Empty, queue_size=1)
        self.pub_world_change = rospy.Publisher('/gui/world_change', String, queue_size=1)
        
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
    
    def _get_canvas_dimensions(self):
        """Return current canvas dimensions with fallbacks for early layout."""
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        if width <= 1 or height <= 1:
            width = self.canvas.winfo_reqwidth()
            height = self.canvas.winfo_reqheight()
        if width <= 1 or height <= 1:
            try:
                width = int(self.canvas['width'])
                height = int(self.canvas['height'])
            except Exception:
                width, height = 1200, 800
        return width, height
    
    def cell_to_canvas(self, row, col):
        """Convert grid cell coordinates to canvas coordinates."""
        cell_size = 60
        canvas_width, canvas_height = self._get_canvas_dimensions()
        grid_width = 10 * cell_size
        grid_height = 10 * cell_size
        grid_start_x = (canvas_width - grid_width) // 2
        grid_start_y = (canvas_height - grid_height) // 2
        
        x = grid_start_x + col * cell_size + cell_size // 2
        y = grid_start_y + row * cell_size + cell_size // 2
        return x, y
    
    def draw_world(self):
        """Draw the static world elements"""
        self.canvas.delete("all")
        
        # Draw 10x10 grid
        cell_size = 60
        grid_width = 10 * cell_size
        grid_height = 10 * cell_size
        
        # Get canvas dimensions and center the grid
        canvas_width, canvas_height = self._get_canvas_dimensions()
        grid_start_x = (canvas_width - grid_width) // 2
        grid_start_y = (canvas_height - grid_height) // 2
        
        # Draw cell backgrounds first for better visibility
        for row in range(10):
            for col in range(10):
                x1 = grid_start_x + col * cell_size
                y1 = grid_start_y + row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size
                # Alternate cell colors for better grid visibility
                if (row + col) % 2 == 0:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#2a2a2a", outline="")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="#1a1a1a", outline="")
        
        # Draw wall cells (background layer)
        wall_cells = self.world_config.get("wall_cells", [])
        
        for wall_cell in wall_cells:
            row, col = wall_cell[0], wall_cell[1]
            
            # Get cell position
            cell_x, cell_y = self.cell_to_canvas(row, col)
            cell_size = 60
            
            # Draw wall cell as a filled rectangle
            x1 = cell_x - cell_size // 2
            y1 = cell_y - cell_size // 2
            x2 = cell_x + cell_size // 2
            y2 = cell_y + cell_size // 2
            
            # Draw wall cell with dark color and border
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="#444444", outline="#666666", width=2)
            
            # Add wall pattern/texture
            self.canvas.create_line(x1+10, y1+10, x2-10, y2-10, fill="#666666", width=2)
            self.canvas.create_line(x1+10, y2-10, x2-10, y1+10, fill="#666666", width=2)
        
        # Draw grid lines with much better visibility (after wall cells)
        for i in range(11):  # 11 lines for 10 cells
            # Vertical lines - make them very visible
            x = grid_start_x + i * cell_size
            self.canvas.create_line(x, grid_start_y, x, grid_start_y + grid_height, fill="#ffffff", width=3)
            # Horizontal lines - make them very visible
            y = grid_start_y + i * cell_size
            self.canvas.create_line(grid_start_x, y, grid_start_x + grid_width, y, fill="#ffffff", width=3)
        
        # Draw coordinate labels (on top of walls)
        for i in range(10):
            # Row labels (left side)
            x = grid_start_x - 20
            y = grid_start_y + i * cell_size + cell_size // 2
            self.canvas.create_text(x, y, text=str(i), fill="#58a6ff", font=("Arial", 12, "bold"))
            # Column labels (top side)
            x = grid_start_x + i * cell_size + cell_size // 2
            y = grid_start_y - 20
            self.canvas.create_text(x, y, text=str(i), fill="#58a6ff", font=("Arial", 12, "bold"))
        
        # Draw waypoints dynamically (on top of walls)
        for waypoint_name, waypoint_data in self.waypoints.items():
            # Handle both grid format (row, col) and converted format (x, y)
            if "row" in waypoint_data and "col" in waypoint_data:
                row, col = waypoint_data["row"], waypoint_data["col"]
                canvas_x, canvas_y = self.cell_to_canvas(row, col)
            else:
                # Convert from x,y coordinates back to grid for display
                x, y = waypoint_data["x"], waypoint_data["y"]
                col = round(x / 0.3)
                row = round(y / 0.3)
                canvas_x, canvas_y = self.cell_to_canvas(row, col)
            
            # Determine waypoint type and styling
            if waypoint_data.get("pick_direction") is not None:
                # Pickup waypoint - show category name
                color = "#ffa657"
                outline = "#ff7b72"
                label = waypoint_name.upper()
            else:
                # General waypoint (should not exist anymore since we removed dropoff waypoints)
                color = "#58a6ff"
                outline = "#39d353"
                label = waypoint_name.upper()
            
            # Draw waypoint rectangle
            self.canvas.create_rectangle(canvas_x-25, canvas_y-25, canvas_x+25, canvas_y+25, 
                                       fill=color, outline=outline, width=3)
            self.canvas.create_text(canvas_x, canvas_y-40, text=label, font=("Arial", 12, "bold"), fill=color)
            
            # Draw orientation indicator if specified
            if waypoint_data.get("pick_direction") is not None:
                direction = waypoint_data["pick_direction"]
                arrow_length = 30
                # Calculate arrow direction based on cardinal direction
                if direction == "N":
                    arrow_x, arrow_y = canvas_x, canvas_y - arrow_length
                elif direction == "S":
                    arrow_x, arrow_y = canvas_x, canvas_y + arrow_length
                elif direction == "E":
                    arrow_x, arrow_y = canvas_x + arrow_length, canvas_y
                elif direction == "W":
                    arrow_x, arrow_y = canvas_x - arrow_length, canvas_y
                else:
                    arrow_x, arrow_y = canvas_x, canvas_y - arrow_length
                
                self.canvas.create_line(canvas_x, canvas_y, arrow_x, arrow_y, fill="#ffffff", width=3, arrow=tk.LAST)
        
        # Draw zones dynamically (these are dropoff points) with collision avoidance
        adjusted_zones = self._adjust_zones_for_collisions()
        for zone_name, zone_data in adjusted_zones.items():
            # Handle both grid format (row, col) and converted format (x, y)
            if "row" in zone_data and "col" in zone_data:
                row, col = zone_data["row"], zone_data["col"]
                canvas_x, canvas_y = self.cell_to_canvas(row, col)
            else:
                # Convert from x,y coordinates back to grid for display
                x, y = zone_data["x"], zone_data["y"]
                col = round(x / 0.3)
                row = round(y / 0.3)
                canvas_x, canvas_y = self.cell_to_canvas(row, col)
            
            # Draw zone as a larger circle with dropoff styling
            self.canvas.create_oval(canvas_x-40, canvas_y-40, canvas_x+40, canvas_y+40,
                                   fill="#2d2d2d", outline="#00d4aa", width=3, stipple="gray25")
            self.canvas.create_text(canvas_x, canvas_y, text=zone_name.upper(), font=("Arial", 10, "bold"), fill="#00d4aa")

            # Draw direction arrow indicating required drop-off orientation (match standalone)
            direction = zone_data.get("direction")
            if direction:
                arrow_length = 32
                if direction == "N":
                    arrow_x, arrow_y = canvas_x, canvas_y - arrow_length
                elif direction == "S":
                    arrow_x, arrow_y = canvas_x, canvas_y + arrow_length
                elif direction == "E":
                    arrow_x, arrow_y = canvas_x + arrow_length, canvas_y
                elif direction == "W":
                    arrow_x, arrow_y = canvas_x - arrow_length, canvas_y
                else:
                    arrow_x, arrow_y = canvas_x, canvas_y - arrow_length
                self.canvas.create_line(canvas_x, canvas_y, arrow_x, arrow_y, fill="#00d4aa", width=3, arrow=tk.LAST)
        
        # Draw objects with better visibility (on top of everything)
        for obj_name, obj_data in self.objects.items():
            if obj_data["present"]:
                obj_x, obj_y = self.cell_to_canvas(obj_data["row"], obj_data["col"])
                self.canvas.create_oval(obj_x-12, obj_y-12, obj_x+12, obj_y+12,
                                      fill=obj_data["color"], outline="#f0f6fc", width=2)
                self.canvas.create_text(obj_x, obj_y+25, text=obj_name, font=("Arial", 10, "bold"), fill="#f0f6fc")
    
    def draw_robot(self):
        """Draw the robot at current position with realistic gripper"""
        # Convert robot position to canvas coordinates using grid-based positioning
        robot_canvas_x, robot_canvas_y = self.cell_to_canvas(self.robot_row, self.robot_col)
        
        # Robot dimensions (scaled larger)
        robot_width = 56  # 0.7m * 80 pixels/m
        robot_length = 112  # 1.4m * 80 pixels/m
        
        # Calculate robot corner points based on orientation
        # Use numeric yaw provided by NL/robot (already aligned with convention)
        display_yaw = self.robot_yaw
        cos_yaw = math.cos(display_yaw)
        sin_yaw = math.sin(display_yaw)
        
        # Robot corners relative to center (length along Y-axis, width along X-axis)
        # Front is in positive Y direction to match GUI coordinate system
        corners = [
            (-robot_width/2, robot_length/2),   # front left  
            (-robot_width/2, -robot_length/2),  # rear left
            (robot_width/2, -robot_length/2),   # rear right
            (robot_width/2, robot_length/2),    # front right
        ]
        
        # Rotate and translate corners
        # Use inverted rotation matrix to match GUI coordinate system
        robot_corners = []
        for dx, dy in corners:
            rotated_dx = cos_yaw * dx + sin_yaw * dy
            rotated_dy = -sin_yaw * dx + cos_yaw * dy
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
        
        # Draw direction indicator (front of robot) - points in positive Y direction
        # Robot front is in positive Y direction, so indicator should be at (0, robot_length/2 + 15)
        # Note: GUI canvas Y-axis points down, so we need to negate the Y component
        front_x = robot_canvas_x + cos_yaw * 0 - sin_yaw * (robot_length/2 + 15)
        front_y = robot_canvas_y - (sin_yaw * 0 + cos_yaw * (robot_length/2 + 15))
        self.canvas.create_oval(front_x-8, front_y-8, front_x+8, front_y+8,
                              fill="#f85149", outline="#ff7b72", width=2, tags="robot")
        
        # Draw robot center point
        self.canvas.create_oval(robot_canvas_x-4, robot_canvas_y-4, robot_canvas_x+4, robot_canvas_y+4,
                              fill="#ffa657", outline="#f0f6fc", width=2, tags="robot")
        
        # Draw realistic gripper based on arm status
        self.draw_gripper(robot_canvas_x, robot_canvas_y, cos_yaw, sin_yaw, robot_length)
        
        # Draw carried object
        if self.has_object:
            # Carried object should be in front of robot (positive Y direction)
            carried_x = robot_canvas_x + cos_yaw * 0 - sin_yaw * 25
            carried_y = robot_canvas_y - (sin_yaw * 0 + cos_yaw * 25)
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
            # Robot front is in positive Y direction, so gripper should be at (0, robot_length/2 - 10)
            gripper_x = robot_x + cos_yaw * 0 - sin_yaw * (robot_length/2 - 10)
            gripper_y = robot_y - (sin_yaw * 0 + cos_yaw * (robot_length/2 - 10))
            gripper_size = 10
            arm_length = robot_length/2 - 10
        else:  # extended, grasping, etc.
            # Gripper extended further from robot body
            # Robot front is in positive Y direction, so gripper should be at (0, robot_length/2 + 30)
            gripper_x = robot_x + cos_yaw * 0 - sin_yaw * (robot_length/2 + 30)
            gripper_y = robot_y - (sin_yaw * 0 + cos_yaw * (robot_length/2 + 30))
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
    
    def _convert_grid_to_xy(self, locations):
        """Convert grid coordinates (row, col) to x,y coordinates for GUI display."""
        converted = {}
        for name, data in locations.items():
            converted_data = data.copy()
            if "row" in data and "col" in data:
                # Convert grid coordinates to x,y (30cm per cell)
                converted_data["x"] = data["col"] * 0.3
                converted_data["y"] = data["row"] * 0.3
                # Remove grid coordinates
                converted_data.pop("row", None)
                converted_data.pop("col", None)
            converted[name] = converted_data
        return converted
    
    def _create_dynamic_objects(self):
        """Create dynamic objects based on world configuration without generic placeholders."""
        objects = {}
        
        waypoints = self.world_config.get("waypoints", {})
        colors = ["#f85149", "#39d353", "#58a6ff", "#ffa657", "#ff7b72", "#00d4aa"]
        
        color_idx = 0
        
        for waypoint_name, waypoint_data in waypoints.items():
            if waypoint_data.get("pick_direction") is None:
                continue
            row = waypoint_data["row"]
            col = waypoint_data["col"]
            
            # Get specific object names based on waypoint category (PICKUP_* keys)
            object_names = self._get_objects_for_category(waypoint_name)
            
            # Only place real items; if none are defined for this category, skip
            for i in range(min(3, len(object_names))):
                # Place all objects in the same cell as the waypoint
                obj_name = object_names[i]
                obj_color = colors[color_idx % len(colors)]
                
                objects[obj_name] = {
                    "row": row,  # Same row as waypoint
                    "col": col,  # Same col as waypoint
                    "present": True,
                    "color": obj_color,
                    "waypoint": waypoint_name
                }
                color_idx += 1
        
        return objects
    
    def _get_objects_for_category(self, category_name):
        """Get specific object names for each PICKUP_* category. Never return generic items."""
        name = str(category_name or "").upper()
        objects_by_category = {
            "PICKUP_BEVERAGES": ["water_bottle", "soda_can", "juice_box"],
            "PICKUP_PRODUCE": ["apple", "banana", "orange"],
            "PICKUP_DAIRY": ["milk_carton", "cheddar_block", "yogurt_cup"],
            "PICKUP_INCOMING": ["pallet", "shipping_box", "barcode_label"],
            "PICKUP_ELECTRONICS": ["circuit_board", "power_supply", "hdmi_cable"],
            "PICKUP_TEXTILES": ["fabric_roll", "cotton_bale", "yarn_spool"],
            "PICKUP_TOOLS": ["wrench", "screwdriver", "pliers"],
            "PICKUP_STATIONERY": ["pen", "pencil", "marker"],
            "PICKUP_OFFICE_SUPPLIES": ["stapler", "paper_clips", "folder"],
            "PICKUP_COMPUTERS": ["laptop", "keyboard", "mouse"],
            "PICKUP_COFFEE": ["coffee_cup", "espresso_pod", "sugar_packet"],
            "PICKUP_MEDICATIONS": ["pill_bottle", "syringe", "bandage"],
            "PICKUP_PPE": ["gloves_box", "face_mask", "sanitizer_bottle"],
            "PICKUP_EMERGENCY_SUPPLIES": ["first_aid_kit", "defibrillator", "oxygen_tank"],
            "PICKUP_INGREDIENTS": ["flour_bag", "tomatoes", "spices_jar"],
            "PICKUP_KNIVES": ["chef_knife", "paring_knife", "honing_rod"],
            "PICKUP_UTENSILS": ["fork", "spoon", "tongs"],
            "PICKUP_GLASSWARE": ["test_tube", "beaker", "vial"],
            "PICKUP_MICROSCOPES": ["microscope", "glass_slide", "cover_slip"],
            "PICKUP_SAMPLES": ["blood_sample", "tissue_sample", "culture_dish"],
            "PICKUP_ELECTRONICS_PHONES": ["smartphone", "charger", "earbuds"],
            "PICKUP_APPAREL_TOPS": ["shirt", "t_shirt", "jacket"],
            "PICKUP_TOYS": ["toy_car", "puzzle_box", "plush_bear"],
            "PICKUP_SECURITY_ITEMS": ["security_bin", "tray", "belt_bucket"],
            "PICKUP_TICKETING": ["boarding_pass", "luggage_tag", "passport"],
            "PICKUP_MAPS_INFO": ["terminal_map", "brochure", "guide"],
            "PICKUP_HANDTOOLS": ["hammer", "tape_measure", "chisel"],
            "PICKUP_LUMBER": ["wood_plank", "timber_beam", "plywood_sheet"],
            "PICKUP_SAFETY_HELMETS": ["hard_hat", "safety_vest", "ear_protectors"],
            "PICKUP_MAZE_ITEM": ["maze package"],
        }
        return list(objects_by_category.get(name, []))

    def _adjust_zones_for_collisions(self, min_distance=1.2):
        """Return a copy of zones with centroids nudged to avoid overlap with pickup waypoints."""
        import math as _math
        zones = self.world_config.get("zones", {})
        waypoints = self.world_config.get("waypoints", {})
        adjusted = {}
        for zn, zd in zones.items():
            c = dict(zd.get("centroid", {}))
            zx, zy = float(c.get("x", 0.0)), float(c.get("y", 0.0))
            # push away if close to any pickup waypoint
            for wp_name, wp in waypoints.items():
                if wp.get("pick_yaw") is None:
                    continue
                dx = zx - float(wp.get("x", 0.0))
                dy = zy - float(wp.get("y", 0.0))
                dist = (_math.hypot(dx, dy) or 1e-6)
                if dist < min_distance:
                    # Nudge outward along the vector from waypoint to zone
                    scale = (min_distance - dist) + 0.0
                    zx += (dx / dist) * scale
                    zy += (dy / dist) * scale
            new_zone = dict(zd)
            new_zone["centroid"] = {"x": zx, "y": zy, "z": zd.get("centroid", {}).get("z", 0.0)}
            adjusted[zn] = new_zone
        return adjusted
    
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
            
            # Convert grid coordinates to x,y coordinates for GUI
            self.waypoints = self._convert_grid_to_xy(self.waypoints)
            self.zones = self._convert_grid_to_xy(self.zones)
            
            # Update objects
            self.objects = self._create_dynamic_objects()
            
            # Update GUI elements
            self.root.title(f"Spot Robot Simulation - {self.world_name}")
            self.world_label.config(text=f"World: {self.world_name}")
            
            # Redraw the world
            self.draw_world()
            self.draw_robot()
            
            # Notify NL_Control about world change
            self.pub_world_change.publish(String(data=world_id))
            
            rospy.loginfo(f"Loaded world: {self.world_name} - {self.world_description}")
    
    def setup_headless_mode(self):
        """Setup headless mode with minimal state tracking"""
        # Robot state
        self.robot_x = 1.35  # Center of cell (4,4)
        self.robot_y = 1.35
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
        
        # Use zones from world configuration for drop-off locations
        self.zones = self.world_config.get("zones", {})
        
        # Convert grid coordinates to x,y coordinates for GUI
        self.zones = self._convert_grid_to_xy(self.zones)
        
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
        elif "start_drop_off" in feedback:
            self.current_action = "dropping_off"
            self.arm_status = "extended"
            # Simulate the full drop-off sequence
            rospy.sleep(1.0)
            self.gripper_status = "open"
            if self.has_object:
                self.simulate_object_drop()
            rospy.sleep(0.5)
            self.gripper_status = "closed"
            rospy.sleep(0.5)
            self.arm_status = "stowed"
        elif "start_arm_command" in feedback:
            if "open" in feedback:
                self.gripper_status = "open"
            elif "close" in feedback:
                self.gripper_status = "closed"
            elif "stow" in feedback:
                self.arm_status = "stowed"
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
            # Parse position from NL_Control (grid format)
            import json
            position_data = json.loads(msg.data)
            
            # Convert grid coordinates to GUI coordinates
            if 'row' in position_data and 'col' in position_data:
                # Grid format: row, col, facing
                self.robot_row = position_data['row']
                self.robot_col = position_data['col']
                self.robot_facing = position_data.get('facing', 'N')
                
                # Convert grid cell to meter coordinates (center of cell)
                # Cell (4,4) should be at (0.3 * 4.5, 0.3 * 4.5) = (1.35, 1.35)
                self.robot_x = (float(self.robot_col) + 0.5) * 0.3
                self.robot_y = (float(self.robot_row) + 0.5) * 0.3
                    
                # Prefer explicit yaw from NL if available; otherwise map from facing
                try:
                    if 'yaw_deg' in position_data:
                        import math as _m
                        self.robot_yaw = float(position_data['yaw_deg']) * _m.pi / 180.0
                    else:
                        # Cardinal mapping per system convention: N=0, E=-pi/2, W=+pi/2, S=pi
                        direction_to_yaw = {'N': 0.0, 'E': -1.57, 'S': 3.14, 'W': 1.57}
                        self.robot_yaw = direction_to_yaw.get(self.robot_facing, 0.0)
                except Exception:
                    self.robot_yaw = 0.0
                
            elif 'x' in position_data and 'y' in position_data:
                # Legacy x,y,yaw format
                self.robot_x = position_data['x']
                self.robot_y = position_data['y'] 
                self.robot_yaw = position_data['yaw']
        except Exception as e:
            rospy.logwarn(f"Failed to parse NL_Control position: {e}")

    def on_execution_feedback(self, msg):
        """Update robot state from execution feedback"""
        try:
            feedback = msg.data
            
            with self.update_lock:
                # Mirror standalone: reflect feedback lines in NL panel
                _append_feedback_line = self._update_feedback_text
                
                # Show general pre/post LLM messages in UI too
                if feedback.startswith("[exec]"):
                    # Strip prefix for readability
                    clean = feedback.replace("[exec]", "").strip()
                    if ("Sending command to LLM" in clean) or ("LLM generated" in clean):
                        _append_feedback_line(clean)

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
                        # Append a friendly line
                        _append_feedback_line("Executing movement")
                            
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
                        if target_object:
                            _append_feedback_line(f"Attempting to grasp {target_object}")
                        else:
                            _append_feedback_line("Attempting to grasp object")
                        
                    elif "start_drop_off" in feedback:
                        # Handle drop-off sequence: extend arm, open gripper, drop object, close, stow
                        self.current_action = "dropping_off"
                        self.arm_status = "extended"
                        rospy.loginfo("GUI: Simulating drop-off sequence")
                        _append_feedback_line("Starting drop-off sequence")
                        # Simulate the full sequence
                        import threading
                        def simulate_drop_off_sequence():
                            rospy.sleep(1.0)  # Arm extends
                            self.gripper_status = "open"
                            if self.has_object:
                                self.simulate_object_drop()
                            rospy.sleep(0.5)
                            self.gripper_status = "closed"
                            rospy.sleep(0.5)
                            self.arm_status = "stowed"
                            self.current_action = "idle"
                        threading.Thread(target=simulate_drop_off_sequence, daemon=True).start()
                        
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
                        _append_feedback_line("Movement completed")
                    elif self.current_action == "grasping":
                        self.current_action = "idle"
                        _append_feedback_line("Grasping completed")
                        
                # Plan lifecycle and errors
                if "Plan completed successfully" in feedback:
                    _append_feedback_line("Plan execution completed successfully!")
                    _append_feedback_line("Ready for new commands")
                if "Command parsing failed" in feedback:
                    _append_feedback_line("Command could not be understood")
                    _append_feedback_line("Please try rephrasing your command")
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
                    # Convert object grid coordinates to world coordinates
                    obj_x = obj_data["col"] * 0.3  # Convert col to x meters
                    obj_y = obj_data["row"] * 0.3  # Convert row to y meters
                    distance = math.sqrt((self.robot_x - obj_x)**2 + (self.robot_y - obj_y)**2)
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
                # Convert object grid coordinates to world coordinates
                obj_x = obj_data["col"] * 0.3  # Convert col to x meters
                obj_y = obj_data["row"] * 0.3  # Convert row to y meters
                distance = math.sqrt((self.robot_x - obj_x)**2 + (self.robot_y - obj_y)**2)
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
            # Find the nearest drop-off zone
            nearest_zone = None
            min_distance = float('inf')
            
            for zone_name, zone_data in self.zones.items():
                # Zones are already converted to x,y coordinates
                zone_x = zone_data["x"]
                zone_y = zone_data["y"]
                distance = math.sqrt((self.robot_x - zone_x)**2 + (self.robot_y - zone_y)**2)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_zone = zone_data
            
            # Check if robot is near any drop-off zone
            if nearest_zone and min_distance <= 1.0:  # Within 1m of drop-off zone
                # Drop object at the nearest drop-off zone
                drop_x = nearest_zone["x"]
                drop_y = nearest_zone["y"]
                rospy.loginfo(f"Dropping {self.carried_object_name} at drop-off zone ({drop_x:.2f}, {drop_y:.2f})")
            else:
                # Drop object at current robot location
                drop_x = self.robot_x
                drop_y = self.robot_y
                rospy.loginfo(f"Dropping {self.carried_object_name} at current location ({drop_x:.2f}, {drop_y:.2f})")
            
            # Create dropped object
            object_name = self.carried_object_name if self.carried_object_name else "dropped_object"
            # Convert world coordinates back to grid coordinates for consistency
            drop_row = int(round(drop_y / 0.3))  # Convert y to row
            drop_col = int(round(drop_x / 0.3))  # Convert x to col
            dropped_obj = {
                "row": drop_row,
                "col": drop_col,
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
            self.robot_x = 1.35  # Center of cell (4,4)
            self.robot_y = 1.35
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
        
        # Don't enable buttons here - wait for interpretation to enable them
    
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
        """Handle natural language interpretation from orchestrator or nl_control"""
        try:
            interpretation = msg.data.strip()
            if interpretation:
                rospy.loginfo(f"GUI: Received interpretation: {interpretation}")
                # Append to history like standalone
                self._update_feedback_text("Plan generated:")
                for line in interpretation.split('\n'):
                    if line.strip():
                        self._update_feedback_text(line.strip())
                
                # Enable approve/decline buttons when interpretation is received
                self.approve_button.config(state='normal')
                self.decline_button.config(state='normal')
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