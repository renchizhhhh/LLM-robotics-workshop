#!/usr/bin/env python3
"""
GUI Core Module - Main GUI Coordinator
Handles initialization, window setup, and component lifecycle.
"""

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

from world_manager import WorldManager
from message_bus import Publisher, String, Empty, rospy

# Import our modular components
from gui_world import GUIWorldManager
from gui_canvas import GUICanvasRenderer
from gui_preview import GUIPreviewManager
from gui_widgets import GUIWidgets
from gui_handlers import GUIHandlers

logger = logging.getLogger(__name__)

# Check for display availability before importing tkinter
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    GUI_AVAILABLE = True
except Exception as e:
    rospy.logwarn(f"GUI not available: {e}")
    GUI_AVAILABLE = False


class SpotSimulationGUI:
    """Main GUI coordinator class."""
    
    def __init__(self, world_id=None):
        # Don't initialize ROS node here - let NL control handle it
        # This prevents conflicts with multiple node initialization
        
        # Initialize world manager
        self.world_manager = WorldManager()
        
        # Load world configuration
        if world_id and (world_id in self.world_manager.worlds or world_id in self.world_manager.custom_worlds):
            self.current_world_id = world_id
            self.world_config = self.world_manager.get_world_config(world_id)
            self.world_name, self.world_description = self.world_manager.get_world_info(world_id)
            rospy.loginfo(f"GUI loaded world: {self.world_name} - {self.world_description}")
        else:
            # Default to custom_sim if no world_id provided or invalid world_id
            self.current_world_id = "custom_sim"
            self.world_config = self.world_manager.get_world_config("custom_sim")
            self.world_name, self.world_description = self.world_manager.get_world_info("custom_sim")
            rospy.loginfo(f"GUI using default world: {self.world_name}")
        
        # Debug: Check initial world config
        obstacle_cells = self.world_config.get("obstacle_cells", [])
        logger.debug("Initial world config has %d wall cells", len(obstacle_cells))
        
        # Internal stop flag for graceful shutdown
        self._stopping = False

        # Check if GUI is available
        if not GUI_AVAILABLE:
            rospy.logerr("GUI not available - cannot run without display")
            raise RuntimeError("GUI not available")
        
        # Initialize core state
        self._initialize_state()
        
        # Setup GUI components
        self._setup_gui()
        
        # Initialize modular components
        self._initialize_components()
        
        # Setup ROS communication
        self._setup_ros_communication()
    
    def _get_robot_start_position(self):
        """Get robot start position from current world configuration."""
        try:
            # Check if world has robot_start_position defined
            if self.current_world_id in self.world_manager.custom_worlds:
                world_data = self.world_manager.custom_worlds[self.current_world_id]
                robot_start = world_data.get("robot_start_position", {})
                if robot_start:
                    return robot_start
            elif self.current_world_id in self.world_manager.worlds:
                world_data = self.world_manager.worlds[self.current_world_id]
                # Built-in worlds might have robot_start_position in their config
                robot_start = world_data.get("robot_start_position", {})
                if robot_start:
                    return robot_start
        except Exception as e:
            rospy.logwarn(f"Failed to get robot start position: {e}")
        
        # Default fallback
        return {"row": 4, "col": 4, "facing": "N"}
    
    def _initialize_state(self):
        """Initialize core robot and GUI state."""
        # Get robot start position from world configuration
        robot_start = self._get_robot_start_position()
        
        # Robot state
        self.robot_row = float(robot_start.get("row", 4.0))
        self.robot_col = float(robot_start.get("col", 4.0))
        self.robot_facing = robot_start.get("facing", "N")  # N, S, E, W
        self.robot_state = "unknown"
        self.current_action = "idle"
        self.has_object = False
        self.carried_object_name = None
        self.arm_status = "stowed"
        self.gripper_status = "closed"
        
        # Flag to indicate position changed (thread-safe for Windows)
        self.position_changed = False
        self.update_lock = threading.Lock()

        self.grid_size = 10
        self.cell_size = 45  # Reduced from 60 to make grid more compact
        self.scale_factor = 0.90

        # Plan preview state for animation and path overlay
        self.pending_plan_actions = []
        self.pending_plan_start = None
        self.preview_path_coords = None
        self.preview_waypoints = []
        self._preview_frames = []
        self._preview_animation_running = False
        self._preview_after_id = None
        self._preview_original_state = None
        self._preview_frame_index = 0
        self._preview_frame_delay_ms = 350
        self._preview_cancelled = False

        # Aggregated preview overlay state
        self.multi_preview_paths = []
        self.multi_preview_counts = {}
        self.all_preview_visible = False

        # Batch input planning state
        self.loaded_inputs = []
        self.input_plan_cache = {}
        self.batch_preview_queue = []
        self.batch_preview_inflight = None
        self.batch_preview_limit = 5
        
        # Position tracking - convert vision frame to display coordinates
        self.vision_to_body_offset_x = 0.0
        self.vision_to_body_offset_y = 0.0
        self.start_position_set = False
        
        # Initialize tree state flag
        self.inputs_tree_disabled = False
        
        # Presenter mode state
        self.presenter_mode = False
    
    def _initialize_components(self):
        """Initialize all modular components."""
        # Initialize world manager
        self.world_manager_component = GUIWorldManager(self.world_manager, self.world_config)
        
        # Initialize canvas renderer
        self.canvas_renderer = GUICanvasRenderer(
            self.canvas, 
            self.grid_size, 
            self.cell_size, 
            self.scale_factor
        )
        # Initialize renderer grid dimensions from world if provided
        try:
            # Prefer authoritative dimensions from WorldManager using current world id
            if hasattr(self, 'world_manager') and hasattr(self, 'current_world_id'):
                r, c = self.world_manager.get_world_dimensions(self.current_world_id)
                self.canvas_renderer.set_dimensions(int(r), int(c))
        except Exception:
            pass
        
        # Initialize preview manager
        self.preview_manager = GUIPreviewManager(self.canvas_renderer, self)
        
        # Initialize widgets
        self.widgets = GUIWidgets(
            self.root, 
            self.main_frame, 
            self.sidebar_frame, 
            self.bottom_frame,
            self.world_manager,
            self.current_world_id,
            self.world_name,
            self  # Pass gui_core reference for presenter mode
        )
        
        # Initialize handlers
        self.handlers = GUIHandlers(
            self, 
            self.widgets, 
            self.world_manager_component, 
            self.preview_manager
        )
        
        # Setup ROS handlers
        self.handlers.setup_ros_handlers()
        
        # Add world change handler
        self.sub_world_change = rospy.Subscriber('/gui/world_change', String, self.on_world_change)
        
        # Request current position after GUI is ready
        self.root.after(1000, self._request_current_position)
        
        # Connect widget buttons to handlers (with safe checks for new layout)
        if hasattr(self.widgets, 'load_inputs_button') and self.widgets.load_inputs_button:
            self.widgets.load_inputs_button.config(command=self.handlers._on_load_inputs)
        if hasattr(self.widgets, 'refresh_inputs_button') and self.widgets.refresh_inputs_button:
            self.widgets.refresh_inputs_button.config(command=self.handlers._on_refresh_input_files)
        if hasattr(self.widgets, 'copy_user_data_button') and self.widgets.copy_user_data_button:
            self.widgets.copy_user_data_button.config(command=self.handlers._on_copy_and_clear_user_data)
        if hasattr(self.widgets, 'save_baseline_button') and self.widgets.save_baseline_button:
            self.widgets.save_baseline_button.config(command=self.handlers._on_save_as_baseline)
        if hasattr(self.widgets, 'show_all_trajectories_button') and self.widgets.show_all_trajectories_button:
            self.widgets.show_all_trajectories_button.config(command=self.handlers._on_show_all_previews)
        if hasattr(self.widgets, 'heatmap_toggle_button') and self.widgets.heatmap_toggle_button:
            self.widgets.heatmap_toggle_button.config(command=self.handlers._on_toggle_heatmap)
        if getattr(self.widgets, "strict_scoring_toggle", None):
            self.widgets.strict_scoring_toggle.config(command=self.handlers._on_toggle_strict_scoring)
        if getattr(self.widgets, "wall_enforcement_toggle", None):
            self.widgets.wall_enforcement_toggle.config(command=self.handlers._on_toggle_wall_enforcement)
        if hasattr(self.widgets, 'preview_trajectory_button') and self.widgets.preview_trajectory_button:
            self.widgets.preview_trajectory_button.config(command=self.handlers._on_preview_trajectory)
        if hasattr(self.widgets, 'batch_approve_button') and self.widgets.batch_approve_button:
            self.widgets.batch_approve_button.config(command=self.handlers._on_approve)
        if hasattr(self.widgets, 'send_button') and self.widgets.send_button:
            self.widgets.send_button.config(command=self.handlers._on_send_single_command)
        if hasattr(self.widgets, 'clear_button') and self.widgets.clear_button:
            self.widgets.clear_button.config(command=self.handlers._on_clear_command)
        if hasattr(self.widgets, 'single_preview_trajectory_button') and self.widgets.single_preview_trajectory_button:
            self.widgets.single_preview_trajectory_button.config(command=self.handlers._on_single_preview_trajectory)
        if hasattr(self.widgets, 'single_approve_button') and self.widgets.single_approve_button:
            self.widgets.single_approve_button.config(command=self.handlers._on_single_approve)
        if hasattr(self.widgets, 'single_dismiss_button') and self.widgets.single_dismiss_button:
            self.widgets.single_dismiss_button.config(command=self.handlers._on_single_dismiss)
        
        # Baseline mode buttons
        if hasattr(self.widgets, 'baseline_show_button') and self.widgets.baseline_show_button:
            self.widgets.baseline_show_button.config(command=self.handlers._on_show_baseline)
        if hasattr(self.widgets, 'baseline_execute_button') and self.widgets.baseline_execute_button:
            self.widgets.baseline_execute_button.config(command=self.handlers._on_execute_baseline)
        
        # Set up Enter key binding for command entry (prompt_entry in new layout)
        # For Text widget, use Ctrl+Enter to send, regular Enter creates new line
        try:
            entry_widget = None
            if hasattr(self.widgets, 'command_entry') and self.widgets.command_entry is not None:
                entry_widget = self.widgets.command_entry
                # Entry widget - Enter key sends
                entry_widget.bind('<Return>', self.handlers._on_send_single_command)
            elif hasattr(self.widgets, 'prompt_entry') and self.widgets.prompt_entry is not None:
                entry_widget = self.widgets.prompt_entry
                # Text widget - Ctrl+Enter sends, Enter creates new line
                entry_widget.bind('<Control-Return>', self.handlers._on_send_single_command)
            
            if entry_widget is None:
                logger.warning("No command entry widget found for Enter key binding")
        except Exception as e:
            logger.warning(f"Failed to bind Enter key to command entry: {e}")

        # Initial display update to draw the world and robot (with delay to ensure canvas is ready)
        logger.debug("Calling initial update_display()")
        self.root.after(100, self.update_display)  # Delay to ensure canvas is ready
        logger.debug("Initial update_display() scheduled")
        
        # Start periodic updates
        logger.debug("Starting periodic updates")
        self.periodic_update()
        logger.debug("Periodic updates started")
    
    def _setup_gui(self):
        """Setup the main GUI window and layout."""
        # GUI setup
        try:
            self.root = tk.Tk()
            self.root.title(f"Spot Robot Simulation - {self.world_name}")
        except Exception as e:
            rospy.logerr(f"Failed to create GUI window: {e}")
            raise
        
        # Set a reasonable window size that fits on most screens
        self.root.geometry("1500x900")
        self.root.minsize(1300, 700)  # Minimum size to ensure usability
        
        # Dark mode styling
        self.root.configure(bg='#1a1a1a')
        self.root.rowconfigure(0, weight=1)  # Main content row grows
        self.root.rowconfigure(1, weight=0)  # Control bar row fixed (will be created in widgets)
        self.root.columnconfigure(0, weight=1)
        
        # Main layout: canvas on left, right sidebar with tabs
        self.main_frame = tk.Frame(self.root, bg='#1a1a1a')
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Configure main frame: canvas on left, sidebar on right
        self.main_frame.columnconfigure(0, weight=1)  # Canvas column grows
        self.main_frame.columnconfigure(1, weight=0)  # Sidebar fixed width
        self.main_frame.rowconfigure(0, weight=1)

        # Canvas on the left - takes most of the space
        canvas_width = 900
        canvas_height = 700
        self.canvas = tk.Canvas(self.main_frame, width=canvas_width, height=canvas_height, bg='#0d1117', 
                               highlightthickness=0, relief='flat')
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        
        # Force canvas to be visible and properly sized
        self.canvas.update_idletasks()
        logger.debug("Canvas created with size %dx%d", canvas_width, canvas_height)

        # Track canvas resize events so grid stays centered after window adjustments
        self._last_canvas_size = None
        self._canvas_resize_after = None
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Right sidebar with tabs (will be created in widgets)
        self.sidebar_frame = tk.Frame(self.main_frame, bg='#161b22', padx=0, pady=0)
        self.sidebar_frame.grid(row=0, column=1, sticky="nsew")
        self.sidebar_frame.config(width=460)  # Increased width for better button visibility
        try:
            self.sidebar_frame.pack_propagate(False)
        except Exception:
            pass

        # Footer container for logs (legacy, kept for compatibility but hidden)
        # The control bar will be created in widgets and placed at row 1
        self.bottom_frame = tk.Frame(self.root, bg='#1a1a1a')
        # Don't grid it - it's legacy and not used in new layout
        self.bottom_frame.columnconfigure(0, weight=1)
        self.bottom_frame.columnconfigure(1, weight=1)
        self.bottom_frame.rowconfigure(0, weight=1)
    
    def _setup_ros_communication(self):
        """Setup ROS publishers and subscribers."""
        # ROS subscribers
        self.sub_robot_state = rospy.Subscriber('/spot_entrance/robot_state', String, self.handlers.on_robot_state)
        self.sub_feedback = rospy.Subscriber('/spot/execution_feedback', String, self.handlers.on_execution_feedback)
        self.sub_nl_position = rospy.Subscriber('/nl_control/robot_position', String, self.handlers.on_nl_position_update)
        self.sub_interpretation = rospy.Subscriber('/llm_int/interpretation', String, self.handlers._on_interpretation)
        self.sub_plan_preview = rospy.Subscriber('/nl_control/plan_preview', String, self.handlers._on_plan_preview)
        self.sub_batch_plan_preview = rospy.Subscriber('/nl_control/batch_plan_preview', String, self.handlers._on_batch_plan_preview)
        self.sub_batch_interpretation = rospy.Subscriber('/nl_control/batch_plan_interpretation', String, self.handlers._on_batch_interpretation)

        # ROS publishers
        self.pub_user_speech = rospy.Publisher('/hl/user_speech', String, queue_size=1)
        self.pub_approval = rospy.Publisher('/hl/approval', String, queue_size=1)
        self.pub_stop = rospy.Publisher('/hl/stop', Empty, queue_size=1)
        self.pub_world_change = rospy.Publisher('/gui/world_change', String, queue_size=1)
        self.pub_position_reset = rospy.Publisher('/gui/position_reset', String, queue_size=1)
        self.pub_batch_preview_request = rospy.Publisher('/gui/batch_preview_request', String, queue_size=1)
        self.pub_interpretation = rospy.Publisher('/llm_int/interpretation', String, queue_size=1)
        self.pub_plan = rospy.Publisher('/nl_control/plan', String, queue_size=1)
    
    def _on_canvas_resize(self, event):
        """Handle canvas resize events."""
        self.canvas_renderer._on_canvas_resize(event)
        # Trigger a redraw after resize
        self.update_display()
    
    def on_world_change(self, msg):
        """Handle world change from GUI widgets."""
        world_id = msg.data
        rospy.loginfo(f"GUI Core: World change received: {world_id}")
        
        # Update world configuration
        if world_id in self.world_manager.worlds or world_id in self.world_manager.custom_worlds:
            self.current_world_id = world_id
            self.world_config = self.world_manager.get_world_config(world_id)
            self.world_name, self.world_description = self.world_manager.get_world_info(world_id)
            
            # Update canvas dimensions
            try:
                r, c = self.world_manager.get_world_dimensions(world_id)
                self.canvas_renderer.set_dimensions(int(r), int(c))
                rospy.loginfo(f"GUI Core: Updated canvas dimensions to {r}x{c}")
            except Exception as e:
                rospy.logwarn(f"GUI Core: Failed to update canvas dimensions: {e}")
            
            # Update world manager component
            self.world_manager_component.load_world(world_id)
            
            # Update display
            self.update_display()
            
            rospy.loginfo(f"GUI Core: Successfully switched to world {world_id}")
        else:
            rospy.logwarn(f"GUI Core: Invalid world ID: {world_id}")
    
    def periodic_update(self):
        """Periodic GUI update."""
        try:
            # If stopping, do not schedule further updates
            if getattr(self, '_stopping', False):
                return
            # Schedule next tick
            self.root.after(100, self.periodic_update)
            # Update display when position changed (triggered by position updates from robot)
            # This runs on the main GUI thread, so it's safe on Windows
            if self.position_changed:
                with self.update_lock:
                    self.position_changed = False
                self.update_display()
        except Exception as e:
            rospy.logwarn(f"Periodic update failed: {e}")
    
    def update_display(self):
        """Update the GUI display."""
        # Debug: Check world config
        obstacle_cells = self.world_config.get("obstacle_cells", [])
        logger.debug("update_display called with %d wall cells", len(obstacle_cells))
        
        # Draw world using canvas renderer
        # Check if path should be hidden
        preview_path = None if self.preview_manager._path_hidden else self.preview_manager.preview_path_coords
        preview_waypoints = None if self.preview_manager._path_hidden else self.preview_manager.preview_waypoints
        
        self.canvas_renderer.draw_world(
            self.world_config,
            self.world_manager_component.objects,
            self.world_manager_component.waypoints,
            self.world_manager_component.zones,
            self.preview_manager.multi_preview_paths,
            self.preview_manager.multi_preview_counts,
            self.preview_manager.heatmap_visible,
            self.preview_manager.all_preview_visible,
            preview_path,
            preview_waypoints
        )
        
        # Draw robot using canvas renderer
        self.canvas_renderer.draw_robot(
            self.robot_row,
            self.robot_col,
            self.robot_facing,
            self.robot_state,
            self.has_object,
            self.carried_object_name,
            self.arm_status,
            self.gripper_status
        )
    
    def run(self):
        """Start the GUI."""
        rospy.loginfo("Spot Simulation GUI started")
        
        # Force initial display update
        self.update_display()
        
        # Make sure the window is visible
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        
        def on_closing():
            rospy.loginfo("Spot Simulation GUI shutting down")
            rospy.signal_shutdown("GUI closed")
            self.root.destroy()
        
        self.root.protocol("WM_DELETE_WINDOW", on_closing)
        self.root.mainloop()

    def stop(self):
        """Gracefully stop GUI updates, unsubscribe, and close the window."""
        try:
            rospy.loginfo("GUI: stop() called - stopping updates and closing UI")
            self._stopping = True
            try:
                self._cancel_preview_animation()
            except Exception:
                pass

            if hasattr(self, 'root') and self.root:
                try:
                    try:
                        self.root.after(0, self.root.quit)
                    except Exception:
                        pass
                    self.root.destroy()
                except Exception:
                    pass
        except Exception as e:
            try:
                rospy.logwarn(f"GUI: stop() encountered an error: {e}")
            except Exception:
                pass
    
    def _cancel_preview_animation(self):
        """Cancel any running preview animation."""
        self.preview_manager.cancel_preview_animation()
    
    def _request_current_position(self):
        """Request current position from NL Control."""
        try:
            # Publish a request for current position
            self.pub_position_reset.publish(String(data="request_current_position"))
            rospy.loginfo("GUI: Requested current position from NL Control")
        except Exception as e:
            rospy.logwarn(f"Failed to request current position: {e}")


def main():
    """Main entry point for GUI"""
    try:
        # Get world_id from parameter if available
        world_id = rospy.get_param('~world_id', None)
        
        gui = SpotSimulationGUI(world_id=world_id)
        gui.run()
    except KeyboardInterrupt:
        rospy.loginfo("GUI shutdown")
    except Exception as e:
        rospy.logerr(f"GUI error: {e}")


if __name__ == "__main__":
    main()
