#!/usr/bin/env python3
"""
Natural Language Control for Spot Robot (Standalone Version)
"""

import sys
import os
import threading
import time
import json
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Union, Optional

# Import our standalone modules
from state_machine import SpotStateMachine
from timing_utils import recorder
from world_manager import WorldManager
from message_bus import Publisher, Subscriber, String, Empty, rospy

from llm_router import LLMRouter
from plan_generator import PlanGenerationError, PlanGenerator
from plan_interpreter import generate_fallback_interpretation
from prompt_manager import get_available_prompts
from pose_tracker import GridFrameMapper, PoseTracker

load_dotenv()

class NaturalLanguageControl:
    def __init__(self, use_speech: bool = False, world_id: str | None = None):
        # Only initialize ROS node if not already initialized
        try:
            rospy.init_node('nl_control', anonymous=True)
        except rospy.exceptions.ROSException as e:
            if "already been called" in str(e):
                pass
            else:
                raise
        self.use_speech = use_speech

        if world_id is None:
            world_id = rospy.get_param('~world_id', 'custom_sim')

        self.grid_mapper = GridFrameMapper()
        self.pose_tracker = PoseTracker(self.grid_mapper)

        self.current_cell = list(self.pose_tracker.current_cell)
        self.current_facing = self.pose_tracker.current_facing  # N, S, E, W
        self.start_cell: list[int] | None = None
        self.start_position = None
        self.current_world_id = world_id
        self.default_start_pose = {"row": 4, "col": 4, "facing": "N"}

        self.last_expected_cell = [4.0, 4.0]
        
        # Accumulated position offsets (tracking errors across actions)
        self.accumulated_offset_row = 0.0  # Accumulated offset in grid cells (row)
        self.accumulated_offset_col = 0.0  # Accumulated offset in grid cells (col)

        self.grid_offset_row = self.grid_mapper.grid_offset_row
        self.grid_offset_col = self.grid_mapper.grid_offset_col
        self.yaw_origin = self.pose_tracker.yaw_origin
        self.current_yaw_rel = self.pose_tracker.current_yaw_rel
        self.real_robot_mode = False

        self.world_manager = WorldManager()

        self.llm_router = None
        self.plan_generator = None
        self.plan_interpreter = None
        self.interpretation_enabled = False
        self.wall_enforcement_enabled = False
        self.last_plan_wall_enforced = False
        self.last_plan_enforcement_applied = False
        self._max_parallel_requests = max(1, int(os.getenv("BATCH_MAX_PARALLEL_REQUESTS", "1")))
        self._batch_executor_lock = threading.Lock()
        self._batch_executor: Optional[ThreadPoolExecutor] = None
        self._batch_executor_workers = 0

        self.available_prompts = get_available_prompts()
        default_prompt = (
            "sim"
            if "sim" in self.available_prompts
            else ("base" if "base" in self.available_prompts else next(iter(self.available_prompts), "base"))
        )
        self.current_prompt_type = default_prompt
        
        # World model configuration - can be loaded from file or set programmatically
        rospy.loginfo("Loading world model...")
        try:
            if world_id:
                self.world_model = self.world_manager.get_world_config(world_id)
                if self.world_model:
                    world_name, world_desc = self.world_manager.get_world_info(world_id)
                    rospy.loginfo(f"Loaded world: {world_name} - {world_desc}")
                    # Pass full world data (including robot_start_position) to robot simulator
                    from robot_simulator import robot_simulator
                    world_data = self.world_manager.get_world_data(world_id)
                    robot_simulator.set_world_config(world_data)
                else:
                    rospy.logwarn(f"Invalid world ID: {world_id}, using default")
                    self.world_model = self.load_world_model()
            else:
                self.world_model = self.load_world_model()
            rospy.loginfo("World model loaded successfully")
            self._update_default_start_pose(world_id)
            
            # Initialize prompt template with current world model
            self._update_prompt_template()
        except Exception as e:
            rospy.logerr(f"Error loading world model: {e}")
            self.world_model = self.load_world_model()
            self._update_prompt_template()
        
        # Publisher for position updates
        self.pub_position = Publisher('/nl_control/robot_position')
        
        # Publisher for interpretation feedback
        self.pub_interpretation = Publisher('/llm_int/interpretation')

        # Publisher for structured plan previews consumed by the GUI
        self.pub_plan_preview = Publisher('/nl_control/plan_preview')
        # Publishers for batch preview mode used by the GUI input loader
        self.pub_batch_plan_preview = Publisher('/nl_control/batch_plan_preview')
        self.pub_batch_interpretation = Publisher('/nl_control/batch_plan_interpretation')
        
        # HoloLens stop signal handling
        self.stop_requested = False
        self.sub_hl_stop = Subscriber('/hl/stop', self.on_hl_stop)
        
        # World change handling
        self.sub_world_change = Subscriber('/gui/world_change', self.on_world_change)
        
        # Position reset handling
        self.sub_position_reset = Subscriber('/gui/position_reset', self.on_position_reset)
        
        # Position updates from simulator are configured after mode detection
        self.sub_robot_position = None

        # Track pose sync failures to auto-downgrade to dummy mode when appropriate
        self._pose_sync_fail_count = 0
        
        # Model selection handling
        self.sub_model_select = Subscriber('/nl_control/model_select', self.on_model_select)

        # Prompt type selection handling
        self.sub_prompt_select = Subscriber('/nl_control/prompt_select', self.on_prompt_select)

        # Batch preview request handling from GUI input loader
        self.sub_batch_preview_request = Subscriber('/gui/batch_preview_request', self.on_batch_preview_request)

        # Approval handling
        self.sub_approval = Subscriber('/hl/approval', self.on_approval)

        # Add subscriber for direct execution of cached plans
        self.sub_direct_execution = Subscriber('/nl_control/plan', self.on_direct_execution)
        self.sub_wall_enforcement = Subscriber('/nl_control/wall_enforcement', self.on_wall_enforcement)
        self.pending_plan = []
        self.awaiting_approval = False
        self._position_timer = None
        self._position_timer_lock = threading.Lock()
        self._has_position_update = False
        
        rospy.loginfo("Natural Language Control starting in STANDALONE mode")
        # Let the FSM auto-detect the mode based on available services
        self.spot_fsm = SpotStateMachine(dummy_mode=None, pose_tracker=self.pose_tracker)
        
        # Detect whether real robot is available; allow real mode in standalone
        self.real_robot_mode = False
        self._detect_real_robot_mode()
        mode_text = "REAL ROBOT" if self.real_robot_mode else "SIMULATION"
        rospy.loginfo(f"NL Control: Mode selected -> {mode_text}")
        self._configure_position_sources()
        
        # Setup pose synchronization if in real robot mode
        if self.real_robot_mode:
            self._setup_pose_synchronization()
        
        # Check current robot state and connect/power on if needed
        rospy.loginfo("Checking robot state...")
        rospy.sleep(1.0)  # Give FSM time to complete its startup sequence
        
        current_state = str(self.spot_fsm.current_state)
        rospy.loginfo(f"Current FSM state: {current_state}")
        
        # Only connect if not already connected
        if current_state in ["unknown", "disconnected"]:
            try:
                rospy.loginfo("Robot not connected, sending connect command...")
                recorder.publish_event('start_connect')
                self.spot_fsm.send("connect")
                recorder.publish_event('stop_connect')
                rospy.loginfo("Connect command completed successfully")
            except Exception as e:
                rospy.logerr(f"Connect command failed: {e}")
                raise
        else:
            rospy.loginfo("Robot already connected, skipping connect command")
        
        # Only power on if not already powered
        current_state = str(self.spot_fsm.current_state)
        if current_state in ["unknown", "disconnected", "connected"]:
            try:
                rospy.loginfo("Robot not powered, sending power_on command...")
                recorder.publish_event('start_power_on')
                self.spot_fsm.send("power_on")
                recorder.publish_event('stop_power_on')
                rospy.loginfo("Power_on command completed successfully")
            except Exception as e:
                rospy.logerr(f"Power_on command failed: {e}")
                raise
        else:
            rospy.loginfo("Robot already powered, skipping power_on command")
        
        # Stand up robot to get it ready for commands (only if not already standing)
        current_state = str(self.spot_fsm.current_state)
        if current_state not in ["Stand", "Moving", "Grasping"]:
            try:
                rospy.loginfo("Sending stand_up command to FSM...")
                recorder.publish_event('start_stand_up')
                self.spot_fsm.send("stand_up")
                recorder.publish_event('stop_stand_up')
                rospy.loginfo("Stand_up command completed successfully")
                
                rospy.loginfo(f"Current FSM state after stand_up: {self.spot_fsm.current_state}")
            except Exception as e:
                rospy.logerr(f"Stand_up command failed: {e}")
                raise
        else:
            rospy.loginfo(f"Robot already in {current_state} state, skipping stand_up command")
        
        rospy.loginfo("Robot ready for commands")
        
        # Publisher for simulation feedback
        self.pub_feedback = Publisher('/spot/execution_feedback')
        
        if use_speech:
            self.speech_sub = Subscriber('/hl/user_speech', self.speech_callback)
            self.speech_input = None
            self.speech_event = threading.Event()
            rospy.loginfo("Speech mode: listening on /hl/user_speech")
        else:
            rospy.loginfo("Terminal mode: type commands")
        
        rospy.loginfo("About to setup LLM...")
        try:
            self.setup_llm()
            self._refresh_batch_executor()
            rospy.loginfo("Ready")
        except Exception as e:
            rospy.logerr(f"LLM setup failed: {e}")
            rospy.logerr(f"LLM setup error details: {str(e)}")
            self.llm_router = None
            self.plan_generator = None
            self.plan_interpreter = None
            self._refresh_batch_executor()
            rospy.loginfo("Ready (LLM disabled)")

    def _detect_real_robot_mode(self):
        """Detect if real robot services are available."""
        try:
            from message_bus import is_ros_master_available
            if not is_ros_master_available():
                rospy.loginfo("NL Control: No ROS master - using simulation mode")
                return
            
            # First check if dummy_mode parameter is explicitly set
            try:
                # Wait for parameters to be set by launch file
                rospy.sleep(3.0)
                
                # Try multiple parameter names that might indicate dummy mode
                dummy_mode_param = False
                try:
                    dummy_mode_param = rospy.get_param('/spot_entrance/dummy_mode', False)
                    rospy.loginfo(f"NL Control: Checking /spot_entrance/dummy_mode: {dummy_mode_param}")
                except:
                    pass
                
                if not dummy_mode_param:
                    try:
                        dummy_mode_param = rospy.get_param('/estop_node/dummy_mode', False)
                        rospy.loginfo(f"NL Control: Checking /estop_node/dummy_mode: {dummy_mode_param}")
                    except:
                        pass
                
                if dummy_mode_param:
                    rospy.loginfo("NL Control: dummy_mode=True detected - using simulation mode")
                    return
                else:
                    rospy.loginfo("NL Control: dummy_mode=False - this is REAL ROBOT mode")
                    # If dummy_mode is explicitly False, this is real robot mode
                    self.real_robot_mode = True
                    return
            except Exception as e:
                rospy.loginfo(f"NL Control: Could not check dummy_mode parameter: {e}")
            
            # Check for key robot services (fallback if parameters not available)
            services_to_check = [
                '/spot_entrance/get_robot_pose',
                '/spot_entrance/connect',
                '/spot_entrance/stand'
            ]
            
            for service in services_to_check:
                try:
                    rospy.wait_for_service(service, timeout=1.0)
                except Exception:
                    rospy.loginfo(f"NL Control: Service {service} not available - using simulation mode")
                    return
            
            # If we get here, services are available but parameters weren't set
            # This means we're in real robot mode
            rospy.loginfo("NL Control: Real robot services detected - enabling real robot mode")
            self.real_robot_mode = True
            
        except Exception as e:
            rospy.loginfo(f"NL Control: Error detecting robot services: {e} - using simulation mode")

    def _setup_pose_synchronization(self):
        """Setup real robot pose synchronization.

        Uses env POSE_SYNC_PERIOD (seconds) for update period, default 0.10s.
        """
        try:
            # Only enable pose sync if real_robot_mode True
            if not self.real_robot_mode:
                return
            from spot_hololens_llm_interface.srv import GetRobotPose
            self.get_robot_pose_srv = rospy.ServiceProxy('/spot_entrance/get_robot_pose', GetRobotPose)
            # Restart timer with desired period
            try:
                if hasattr(self, 'pose_sync_timer') and self.pose_sync_timer is not None:
                    self.pose_sync_timer.shutdown()
            except Exception:
                pass
            period_s = float(os.getenv('POSE_SYNC_PERIOD', '0.10'))
            self.pose_sync_timer = rospy.Timer(rospy.Duration(period_s), self._sync_real_pose)
            rospy.loginfo("NL Control: Real robot pose synchronization enabled")
        except Exception as e:
            rospy.logerr(f"NL Control: Failed to setup pose synchronization: {e}")
            self.real_robot_mode = False
            self._configure_position_sources()

    def _configure_position_sources(self):
        """Configure position feeds based on real-robot availability."""
        if self.real_robot_mode:
            subscriber = getattr(self, 'sub_robot_position', None)
            if subscriber is not None:
                try:
                    subscriber.unregister()
                except Exception:
                    pass
            self.sub_robot_position = None
            rospy.loginfo("NL Control: Using real robot pose updates")
            # Ensure pose sync timer is running for continuous streaming
            try:
                if not hasattr(self, 'pose_sync_timer') or self.pose_sync_timer is None:
                    self._setup_pose_synchronization()
            except Exception as _e:
                rospy.logwarn(f"NL Control: Unable to start pose sync timer: {_e}")
        else:
            if getattr(self, 'sub_robot_position', None) is None:
                self.sub_robot_position = Subscriber('/robot_simulator/position', self.on_robot_position_update)
                rospy.loginfo("NL Control: Subscribed to simulator position updates")

    def _sync_real_pose(self, _event):
        """Periodically sync GUI position with the real robot pose (vision frame -> grid)."""
        if not self.real_robot_mode:
            return
            
        try:
            # If ROS indicates dummy mode at runtime, immediately downgrade
            try:
                if rospy.get_param('/spot_entrance/dummy_mode', False):
                    rospy.loginfo("NL Control: Runtime dummy_mode=True detected - switching to simulation")
                    self._switch_to_dummy_mode()
                    return
            except Exception:
                pass

            resp = self.get_robot_pose_srv()
            if not resp or not getattr(resp, 'success', False):
                self._pose_sync_fail_count += 1
                if self._pose_sync_fail_count >= 3:
                    rospy.logwarn("NL Control: Pose sync failing repeatedly - switching to simulation mode")
                    self._switch_to_dummy_mode()
                return
            robot_pose = getattr(resp, 'robot_pose', None)
            pose_msg = getattr(robot_pose, 'pose', None)
            if pose_msg is None:
                raise ValueError("Robot pose response missing geometry pose")

            pose_update = self.pose_tracker.update_from_pose(pose_msg)
            self._update_cached_pose_state(pose_update)

            if pose_update.get("offsets_initialized") and self.grid_mapper.has_offsets():
                row_offset_val = self.grid_mapper.grid_offset_row
                col_offset_val = self.grid_mapper.grid_offset_col
                if row_offset_val is None or col_offset_val is None:
                    raise ValueError("Grid offsets were expected but missing")
                row_offset = float(row_offset_val)
                col_offset = float(col_offset_val)
                try:
                    rospy.set_param('/nl_control/grid_offset_row', row_offset)
                    rospy.set_param('/nl_control/grid_offset_col', col_offset)
                    if self.pose_tracker.yaw_origin is not None:
                        rospy.set_param('/nl_control/yaw_origin', float(self.pose_tracker.yaw_origin))
                    rospy.loginfo(
                        f"NL Control: Set ROS parameters for grid offsets ({row_offset:.2f}, {col_offset:.2f})"
                    )
                except Exception as param_error:
                    rospy.logwarn(f"NL Control: Could not set ROS parameters for grid offsets: {param_error}")

            self.publish_position_update()
            self._pose_sync_fail_count = 0

        except Exception as e:
            rospy.logwarn(f"NL Control: Pose synchronization failed: {e}")
            self._pose_sync_fail_count += 1
            rospy.logwarn(f"NL Control: Pose sync failure count: {self._pose_sync_fail_count}/10")
            if self._pose_sync_fail_count >= 10:  # Increased threshold for more robust real robot connection
                rospy.logwarn("NL Control: Pose sync exception threshold reached - switching to simulation mode")
                self._switch_to_dummy_mode()

    def _switch_to_dummy_mode(self):
        """Disable real-robot pose sync and follow simulator updates for this run."""
        try:
            # Stop pose timer if present
            if hasattr(self, 'pose_sync_timer') and self.pose_sync_timer is not None:
                try:
                    self.pose_sync_timer.shutdown()
                except Exception:
                    pass
                self.pose_sync_timer = None
            self.real_robot_mode = False
            self.pose_tracker.reset_yaw_origin()
            self.grid_mapper.reset_offsets()
            self.grid_offset_row = None
            self.grid_offset_col = None
            self._configure_position_sources()
            self._pose_sync_fail_count = 0
            rospy.loginfo("NL Control: Switched to simulation position updates (dummy mode)")
        except Exception as _e:
            rospy.logwarn(f"NL Control: Failed switching to dummy mode: {_e}")
    
    def load_world_model(self):
        """Load world model configuration using WorldManager."""
        # Use the same world as the GUI if available, otherwise default to world 1
        try:
            if hasattr(self, 'world_manager') and self.world_manager:
                # Use the same world configuration as the GUI
                return self.world_manager.get_world_config("1")  # Default to world 1
            else:
                # Fallback to simple world if WorldManager not available
                return {
                    "waypoints": {
                        "Beverages": {"x": 2.0, "y": -1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None}
                    },
                    "zones": {
                        "DeliveryArea": {
                            "centroid": {"x": 2.0, "y": 0.5, "z": 0.0},
                            "yaw_hint": 1.57,
                            "tags": ["delivery", "drop-off", "destination"]
                        }
                    },
                    "synonyms": {
                        "beverages": "Beverages",
                        "drinks": "Beverages",
                        "delivery area": "DeliveryArea",
                        "drop-off area": "DeliveryArea"
                    }
                }
        except Exception as e:
            rospy.logwarn(f"Failed to load world model from WorldManager: {e}")
            # Fallback to simple world
            return {
                "waypoints": {
                    "Beverages": {"x": 2.0, "y": -1.0, "z": 0.0, "pick_yaw": 0.0, "drop_yaw": None}
                },
                "zones": {
                    "DeliveryArea": {
                        "centroid": {"x": 2.0, "y": 0.5, "z": 0.0},
                        "yaw_hint": 1.57,
                        "tags": ["delivery", "drop-off", "destination"]
                    }
                },
                "synonyms": {
                    "beverages": "Beverages",
                    "drinks": "Beverages",
                    "delivery area": "DeliveryArea",
                    "drop-off area": "DeliveryArea"
                }
            }
    
    def update_world_model(self, new_world_model):
        """Update the world model configuration."""
        self.world_model = new_world_model
        # Pass world configuration to robot simulator for wall checking
        if hasattr(self, 'spot_fsm') and hasattr(self.spot_fsm, 'robot_simulator'):
            self.spot_fsm.robot_simulator.set_world_config(new_world_model)
        rospy.loginfo("World model updated")
    
    def switch_world(self, world_id):
        """Switch to a different world configuration."""
        try:
            if world_id in self.world_manager.worlds or world_id in self.world_manager.custom_worlds:
                self.world_model = self.world_manager.get_world_config(world_id)
                world_name, world_desc = self.world_manager.get_world_info(world_id)
                rospy.loginfo(f"Switched to world: {world_name} - {world_desc}")
                self.current_world_id = world_id
                self._update_default_start_pose(world_id)
                
                # Pass full world data to robot simulator (not just config)
                from robot_simulator import robot_simulator
                print(f"DEBUG: NL Control calling robot_simulator.set_world_config()")
                if world_id in self.world_manager.worlds:
                    full_world_data = self.world_manager.worlds[world_id]
                else:
                    full_world_data = self.world_manager.custom_worlds[world_id]
                robot_simulator.set_world_config(full_world_data)
                if not full_world_data.get("robot_start_position"):
                    start_pose = self.default_start_pose
                    robot_simulator.set_pose(
                        start_pose["row"], start_pose["col"], start_pose["facing"]
                    )
                print(f"DEBUG: NL Control completed robot_simulator.set_world_config()")
                # Reset tracked start position so the next command uses the new world pose
                self.start_cell = None
                self._has_position_update = False
                
                return True
            else:
                rospy.logwarn(f"Invalid world ID: {world_id}")
                return False
        except Exception as e:
            rospy.logerr(f"Failed to switch world: {e}")
            return False
    
    def load_world_model_from_file(self, filepath):
        """Load world model from JSON file."""
        try:
            with open(filepath, 'r') as f:
                self.world_model = json.load(f)
            rospy.loginfo(f"World model loaded from {filepath}")
        except Exception as e:
            rospy.logerr(f"Failed to load world model from {filepath}: {e}")

    def _apply_start_pose_to_trackers(self, pose: Dict[str, object]) -> None:
        """Propagate the world's start pose into the pose tracker and cached state."""
        if not pose:
            return

        row = float(pose.get("row", self.current_cell[0] if self.current_cell else 4.0))
        col = float(pose.get("col", self.current_cell[1] if self.current_cell else 4.0))
        facing = pose.get("facing", self.current_facing)
        if not isinstance(facing, str):
            facing = self.current_facing

        self.pose_tracker.apply_start_pose({"row": row, "col": col, "facing": facing})
        self.current_cell = [row, col]
        self.current_facing = facing
        self.last_expected_cell = [row, col]
        
        # Reset accumulated offsets when starting pose changes
        self.accumulated_offset_row = 0.0
        self.accumulated_offset_col = 0.0

        # Cache mapper offsets/yaw so GUI sees cleared state until a live pose arrives.
        self.grid_offset_row = self.grid_mapper.grid_offset_row
        self.grid_offset_col = self.grid_mapper.grid_offset_col
        self.yaw_origin = self.pose_tracker.yaw_origin
        # Sync current_yaw_rel from pose_tracker (which now correctly sets it based on facing)
        self.current_yaw_rel = self.pose_tracker.current_yaw_rel

    def _update_default_start_pose(self, world_id: str | None):
        """Update the cached default start pose based on the active world."""
        pose = {"row": 4, "col": 4, "facing": "N"}
        try:
            if world_id:
                world_data = None
                if world_id in self.world_manager.custom_worlds:
                    world_data = self.world_manager.custom_worlds[world_id]
                elif world_id in self.world_manager.worlds:
                    world_data = self.world_manager.worlds[world_id]
                if world_data:
                    robot_start = world_data.get("robot_start_position")
                    if robot_start:
                        pose = {
                            "row": int(robot_start.get("row", pose["row"])),
                            "col": int(robot_start.get("col", pose["col"])),
                            "facing": robot_start.get("facing", pose["facing"]),
                        }
        except Exception as exc:
            rospy.logwarn(f"Failed to derive world start pose for {world_id}: {exc}")
        self.default_start_pose = pose
        self._apply_start_pose_to_trackers(self.default_start_pose)
    
    def get_actual_robot_position(self):
        """Get actual robot position in grid coordinates."""
        try:
            # In standalone mode, trust the simulator updates when available
            if hasattr(self.spot_fsm, 'dummy_mode') and self.spot_fsm.dummy_mode:
                if self.start_cell is None:
                    if self._has_position_update:
                        self.start_cell = list(self.current_cell)
                        rospy.loginfo(
                            "Standalone mode: Using simulator-reported start position "
                            f"at cell ({self.current_cell[0]}, {self.current_cell[1]}) facing {self.current_facing}"
                        )
                    else:
                        start_pose = self.default_start_pose
                        self.start_cell = [start_pose["row"], start_pose["col"]]
                        self.current_cell = [start_pose["row"], start_pose["col"]]
                        self.current_facing = start_pose["facing"]
                        self.last_expected_cell = [
                            float(start_pose["row"]),
                            float(start_pose["col"]),
                        ]
                        rospy.loginfo(
                            "Standalone mode: Simulator position unavailable, defaulting to world start cell "
                            f"({start_pose['row']}, {start_pose['col']}) facing {start_pose['facing']}"
                        )
                return True
            
            # Get robot pose from the service (returns vision frame position)
            response = self.spot_fsm.get_robot_pose()
            if isinstance(response, dict):
                success = bool(response.get('success', False))
                cell = response.get('robot_cell')
                facing = response.get('facing', self.current_facing)
                if cell:
                    self.current_cell = [float(cell[0]), float(cell[1])]
                    self.current_facing = facing
                return success

            if not response or not getattr(response, 'success', False):
                message = getattr(response, 'message', 'unknown error') if response else 'no response'
                rospy.logwarn(f"Failed to get robot pose: {message}")
                return False

            robot_pose = getattr(response, 'robot_pose', None)
            pose_msg = getattr(robot_pose, 'pose', None)
            if pose_msg is None:
                rospy.logwarn("Robot pose response missing pose data")
                return False

            pose_update = self.pose_tracker.update_from_pose(pose_msg)
            self._update_cached_pose_state(pose_update)
            rospy.loginfo(
                f"Current position (grid): ({self.current_cell[0]:.2f}, {self.current_cell[1]:.2f}) facing {self.current_facing}"
            )
            return True
        except Exception as e:
            rospy.logerr(f"Error getting robot position: {e}")
            return False
        
    def setup_llm(self):
        """Setup LLM router with support for 120B, Gemini Robotics, and Gemini Pro."""
        try:
            rospy.loginfo("Starting LLM setup with router...")
            self.llm_router = LLMRouter()
            # Set default to Gemini if available
            if self.llm_router.is_gemini_available():
                self.llm_router.set_model('gemini-2.0-flash-lite')
                rospy.loginfo("LLMRouter: Set default model to gemini-2.0-flash-lite")
            self.plan_generator = PlanGenerator(self.llm_router)
            self.plan_interpreter = None
            rospy.loginfo("LLM router setup completed successfully!")
        except Exception as e:
            rospy.logerr(f"Failed to setup LLM router: {e}")
            rospy.logerr(f"LLM will be disabled. Error details: {str(e)}")
            rospy.logerr(f"Exception type: {type(e).__name__}")
            self.llm_router = None
            self.plan_generator = None
            self.plan_interpreter = None
    
    def _supports_parallel_batch(self) -> bool:
        if not self.llm_router:
            return False
        model_id = getattr(self.llm_router, "current_model", "") or ""
        return isinstance(model_id, str) and model_id.startswith("gemini-")

    def _desired_batch_worker_count(self) -> int:
        if not self.llm_router:
            return 0
        if self._supports_parallel_batch():
            return max(1, self._max_parallel_requests)
        return 1

    def _refresh_batch_executor(self) -> None:
        workers = self._desired_batch_worker_count()
        with self._batch_executor_lock:
            if workers <= 0:
                if self._batch_executor:
                    self._batch_executor.shutdown(wait=False)
                self._batch_executor = None
                self._batch_executor_workers = 0
                return
            if self._batch_executor and self._batch_executor_workers == workers:
                return
            if self._batch_executor:
                self._batch_executor.shutdown(wait=False)
            self._batch_executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="batch-llm")
            self._batch_executor_workers = workers

    def _ensure_batch_executor(self) -> Optional[ThreadPoolExecutor]:
        self._refresh_batch_executor()
        return self._batch_executor
    
    def on_model_select(self, msg):
        """Handle model selection from GUI."""
        rospy.loginfo(f"DEBUG: on_model_select called with message: {msg.data}")
        if hasattr(self, 'llm_router') and self.llm_router:
            model_id = msg.data
            rospy.loginfo(f"DEBUG: Setting model in router to: {model_id}")
            self.llm_router.set_model(model_id)
            rospy.loginfo(f"DEBUG: Router current_model after set: {self.llm_router.current_model}")
            self._refresh_batch_executor()
            rospy.loginfo(f"Model selected: {model_id}")
        else:
            rospy.logwarn("LLM router not available, cannot change model")
            raise
    
    def on_prompt_select(self, msg):
        """Handle prompt type selection from GUI."""
        prompt_type = msg.data

        try:
            self.available_prompts = get_available_prompts()
            if not self.available_prompts:
                rospy.logwarn("Prompt selection requested but no prompts discovered")
                return

            if prompt_type in self.available_prompts:
                self.current_prompt_type = prompt_type
                rospy.loginfo(f"Prompt type selected: {prompt_type}")
                return

            lower = prompt_type.lower()
            for key in self.available_prompts.keys():
                if key.lower() == lower:
                    self.current_prompt_type = key
                    rospy.loginfo(f"Prompt type selected (matched case-insensitive): {key}")
                    return

            rospy.logwarn(f"Unknown prompt type: {prompt_type}, keeping current: {self.current_prompt_type}")
        except Exception as e:
            rospy.logwarn(f"Error handling prompt selection '{prompt_type}': {e}")

    def on_wall_enforcement(self, msg):
        """Handle wall enforcement toggle from GUI."""
        value = str(getattr(msg, "data", "")).strip().lower()
        enabled = value in ("1", "true", "yes", "on", "enable", "enabled")
        self.wall_enforcement_enabled = enabled
        self.last_plan_wall_enforced = enabled
        self.last_plan_enforcement_applied = False
        rospy.loginfo(
            f"Obstacle enforcement {'enabled' if enabled else 'disabled'} via GUI toggle"
        )

    def on_batch_preview_request(self, msg):
        """Handle batch preview requests from GUI input loader."""
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"Invalid JSON on /gui/batch_preview_request: {e}")
            return

        request_id = payload.get("request_id")
        command = (payload.get("command") or "").strip()

        if not command:
            rospy.logwarn("Batch preview request missing command text")
            start_cell = list(self.current_cell) if self.current_cell else None
            response = {
                "request_id": request_id,
                "command": command,
                "actions": [],
                "start_cell": start_cell,
                "start_facing": self.current_facing,
                "error": "missing_command",
                "timestamp": time.time()
            }
            self.pub_batch_plan_preview.publish(String(data=json.dumps(response)))
            return

        start_cell = list(self.current_cell) if self.current_cell else None
        start_facing = self.current_facing
        task_args = (request_id, command, start_cell, start_facing)
        executor = self._ensure_batch_executor()
        if executor:
            executor.submit(self._process_batch_preview_task, *task_args)
        else:
            self._process_batch_preview_task(*task_args)
    
    def _process_batch_preview_task(self, request_id, command, start_cell, start_facing):
        """Run batch preview generation in worker thread."""
        try:
            rospy.loginfo(f"Batch preview requested for command: '{command}' (request_id={request_id})")
            actions = self.parse_command(command)
        except Exception as exc:
            rospy.logerr(f"Batch preview task error for {request_id}: {exc}")
            actions = None

        current_cell = list(start_cell) if start_cell is not None else (list(self.current_cell) if self.current_cell else None)
        current_facing = start_facing or self.current_facing

        if not actions:
            rospy.logwarn(f"Batch preview failed to generate plan for request_id={request_id}")
            response = {
                "request_id": request_id,
                "command": command,
                "actions": [],
                "start_cell": current_cell,
                "start_facing": current_facing,
                "error": "plan_generation_failed",
                "timestamp": time.time()
            }
            self.pub_batch_plan_preview.publish(String(data=json.dumps(response)))
            if self.interpretation_enabled:
                interpretation_payload = {
                    "request_id": request_id,
                    "command": command,
                    "interpretation": "Plan generation failed",
                    "success": False,
                    "timestamp": time.time()
                }
                self.pub_batch_interpretation.publish(String(data=json.dumps(interpretation_payload)))
            return

        response = {
            "request_id": request_id,
            "command": command,
            "actions": actions,
            "start_cell": current_cell,
            "start_facing": current_facing,
            "timestamp": time.time()
        }
        self.pub_batch_plan_preview.publish(String(data=json.dumps(response)))

        if self.interpretation_enabled:
            interpretation = self._interpret_plan(actions)
            interpretation_payload = {
                "request_id": request_id,
                "command": command,
                "interpretation": interpretation,
                "success": True,
                "timestamp": time.time()
            }
            self.pub_batch_interpretation.publish(String(data=json.dumps(interpretation_payload)))
    
    def speech_callback(self, msg):
        """Callback for speech input."""
        self.speech_input = msg.data
        self.speech_event.set()
        rospy.loginfo(f"Speech: {msg.data}")
        
        # Note: received_hololens_input will be published in process_command to ensure correct order
    
    def parse_command(self, command):
        """Parse natural language command using LLM."""
        # Clear previous error
        self.last_plan_error = None
        
        if not self.plan_generator:
            error_msg = "Plan generator unavailable - cannot parse command"
            rospy.logwarn(error_msg)
            self.last_plan_error = error_msg
            return None

        try:
            # Get current robot state
            current_state = self.spot_fsm.current_state.name
            
            # Get actual robot position from vision frame
            if not self.get_actual_robot_position():
                rospy.logwarn("Using last known position")
            
            # Create current state JSON
            # Round robot_cell to integers for LLM (continuous position should map to nearest grid cell)
            rounded_cell = [int(round(self.current_cell[0])), int(round(self.current_cell[1]))]
            current_state_json = {
                "robot_state": current_state,
                "standing": current_state in ["Stand", "Moving", "Grasping"],
                "robot_cell": rounded_cell,
                "facing": self.current_facing,
                "arm": "stowed",  # Default - could be enhanced to track actual arm state
                "held_object": None,  # Default - could be enhanced to track held objects
                # "accumulated_offset_row": self.accumulated_offset_row,
                # "accumulated_offset_col": self.accumulated_offset_col
            }
            
            # Format prompt with current state, world model, and command
            rospy.loginfo(f"[LLM DEBUG] Sending position to LLM: continuous cell ({self.current_cell[0]:.2f}, {self.current_cell[1]:.2f}) -> rounded cell {rounded_cell} facing {self.current_facing}")
            rospy.loginfo(f"[LLM DEBUG] Robot state: {current_state}")
            rospy.loginfo(f"[LLM DEBUG] Current state JSON: {json.dumps(current_state_json, indent=2)}")
            rospy.loginfo(f"Using prompt: {self.current_prompt_type}")

            print("\n" + "="*80)
            print("LLM PROMPT DEBUG")
            print("="*80)
            print(f"CURRENT_STATE:")
            print(json.dumps(current_state_json, indent=2))
            # print(f"\nWORLD_MODEL:")
            # print(json.dumps(self.world_model, indent=2))
            print(f"\nTASK: {command}")
            print("="*80)
            
            result = self.plan_generator.generate_plan(
                command=command,
                current_state=current_state_json,
                world_model=self.world_model,
                prompt_name=self.current_prompt_type,
                enforce_wall_constraints=self.wall_enforcement_enabled,
            )

            if result.plan_text != result.raw_response:
                self.last_llm_full_response = result.raw_response
            else:
                self.last_llm_full_response = None

            self.last_plan_wall_enforced = self.wall_enforcement_enabled
            self.last_plan_enforcement_applied = result.enforcement_applied

            return result.actions

        except PlanGenerationError as err:
            error_msg = str(err)
            rospy.logerr(f"Plan generation failed: {err}")
            # Store error for GUI display
            if not hasattr(self, 'last_plan_error'):
                self.last_plan_error = None
            self.last_plan_error = error_msg
            return None
        except Exception as exc:  # pragma: no cover - defensive logging
            error_msg = f"Unexpected LLM failure: {exc}"
            rospy.logerr(error_msg)
            # Store error for GUI display
            if not hasattr(self, 'last_plan_error'):
                self.last_plan_error = None
            self.last_plan_error = error_msg
            return None
    
    def execute_actions(self, actions):
        """Execute FSM actions."""
        if not actions:
            return False
        
        for action in actions:
            # Check for stop signal before each action
            if self.stop_requested:
                rospy.loginfo("Stop signal received - finishing current task and stopping plan execution")
                # Reset stop flag for next plan
                self.stop_requested = False
                return False
            
            try:
                rospy.loginfo(f"Executing: {action}")
                # Publish feedback for simulation
                self.pub_feedback.publish(String(data=f"Step: {action}"))
                
                # Parse action name (remove parameters)
                if '=' in action:
                    action_name = action.split()[0]  # Get first word (action name)
                else:
                    action_name = action.strip()
                
                if '=' in action:
                    # Parse action with parameters
                    if ',' in action:
                        parts = action.split(',')
                        action_name = parts[0].strip()
                        param_parts = parts[1:]
                    else:
                        # Handle space-separated format: "move_to_cell row=2 col=3"
                        # Use regex to split on key= patterns while preserving multi-word values
                        import re
                        # Split by space only when followed by word=
                        words = re.split(r'\s+(?=\w+=)', action)
                        action_name = words[0].split()[0]  # Get first word only
                        param_parts = words[1:] if len(words) > 1 else []
                    
                    params = {}
                    for part in param_parts:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"\'')
                            
                            try:
                                params[key] = float(value) if '.' in value else int(value)
                            except ValueError:
                                params[key] = value
                    
                    # Execute the action and track position in standalone mode
                    # Special handling for start_drop_off: add internal flag
                    if action_name == 'start_drop_off':
                        params['_is_drop_off'] = True
                    
                    # Handle grid-based actions by transitioning to appropriate state first
                    if action_name == 'move_to_cell':
                        # Calculate movement with offset compensation (for real robot mode only)
                        target_row = float(params.get('row', self.current_cell[0]))
                        target_col = float(params.get('col', self.current_cell[1]))
                        
                        # For real robot mode, use accumulated offset compensation
                        if self.real_robot_mode:
                            # Calculate movement from actual current position (current_cell + accumulated offsets)
                            # This compensates for accumulated position errors from previous movements
                            actual_current_row = self.current_cell[0] + self.accumulated_offset_row
                            actual_current_col = self.current_cell[1] + self.accumulated_offset_col
                            
                            params['current_row'] = actual_current_row
                            params['current_col'] = actual_current_col
                            
                            # Store expected position for verification
                            self.last_expected_cell = [target_row, target_col]
                            
                            rospy.loginfo(f"NL: move_to_cell with offset compensation (real robot mode):")
                            rospy.loginfo(f"  Expected position: ({self.current_cell[0]:.2f},{self.current_cell[1]:.2f}) cells")
                            rospy.loginfo(f"  Actual position: ({actual_current_row:.2f},{actual_current_col:.2f}) cells")
                            rospy.loginfo(f"  Target: ({target_row:.2f},{target_col:.2f}) cells")
                            rospy.loginfo(f"  Accumulated offsets: row={self.accumulated_offset_row:.2f} cells ({self.accumulated_offset_row*57:.1f}cm), col={self.accumulated_offset_col:.2f} cells ({self.accumulated_offset_col*57:.1f}cm)")
                        else:
                            # For dummy mode, use simple position tracking (no offset compensation)
                            params['current_row'] = self.current_cell[0]
                            params['current_col'] = self.current_cell[1]
                            rospy.loginfo(f"NL: move_to_cell (dummy mode): ({self.current_cell[0]:.2f},{self.current_cell[1]:.2f}) -> ({target_row:.2f},{target_col:.2f})")
                        
                        # Transition to moving state first, then execute the movement
                        self.spot_fsm.send('start_moving', **params)
                    elif action_name == 'rotate_to':
                        # Transition to rotating state first, then execute the rotation
                        self.spot_fsm.send('start_rotating', **params)
                    else:
                        # For other actions, send directly
                        self.spot_fsm.send(action_name, **params)
                    
                    # Position tracking for both dummy and real robot modes
                    if action_name == 'move_to_cell':
                        # Store target position for later verification
                        target_row = params.get('row', self.current_cell[0])
                        target_col = params.get('col', self.current_cell[1])
                        
                        # For real robot mode, update expected position (actual position will be verified later)
                        # Keep current_cell as expected position, not actual position
                        if self.real_robot_mode:
                            # Update expected position (this will be compared with actual after movement)
                            self.current_cell = [target_row, target_col]
                            rospy.loginfo(f"[NL_CONTROL DEBUG] Moving to cell ({target_row}, {target_col}) - position will be verified after movement")
                        else:
                            # For dummy mode, update immediately
                            self.current_cell = [target_row, target_col]
                            rospy.loginfo(f"[NL_CONTROL DEBUG] Updated position to cell ({target_row}, {target_col})")
                    
                    elif action_name == 'rotate_to':
                        # Update facing direction
                        direction = params.get('direction', self.current_facing)
                        if direction in ["N", "S", "E", "W"]:
                            self.current_facing = direction
                            rospy.loginfo(f"[NL_CONTROL DEBUG] Rotated to face {direction}")
                        else:
                            rospy.logwarn(f"[NL_CONTROL DEBUG] Invalid direction: {direction}")
                else:
                    # Handle actions without parameters
                    action_name = action.strip()
                    if action_name == 'start_drop_off':
                        # Add internal flag for drop_off
                        self.spot_fsm.send(action_name, _is_drop_off=True)
                    else:
                        self.spot_fsm.send(action_name)
                
                # Publish completion feedback for simulation
                self.pub_feedback.publish(String(data=f"✓ {action}"))
                
                # Publish position update after each action completion
                self.publish_position_update()
                
                # Verify position after movement and update offsets (for real robot mode only)
                if action_name == 'move_to_cell' and self.real_robot_mode:
                    # Wait a moment for the robot to finish moving before verifying position
                    rospy.sleep(0.5)
                    self._verify_and_correct_position()
                
                # Check for stop signal after action completion
                if self.stop_requested:
                    rospy.loginfo("Stop signal received after action completion - stopping plan execution")
                    # Reset stop flag for next plan
                    self.stop_requested = False
                    return False
                
                rospy.sleep(0.1)  # Faster execution
                
            except Exception as e:
                rospy.logerr(f"Action failed '{action}': {e}")
                return False
        
        # Publish plan completion signal
        self.pub_feedback.publish(String(data="Plan completed successfully"))
        return True
    
    def _verify_and_correct_position(self):
        """Verify robot position after movement and update accumulated offsets."""
        try:
            # Only verify in real robot mode
            if not self.real_robot_mode:
                return
            
            # Get actual robot position
            response = self.spot_fsm.get_robot_pose()
            if not response:
                rospy.logwarn("Could not get robot pose for position verification - no response")
                return

            if isinstance(response, dict):
                rospy.logwarn("Position verification unavailable in simulator mode")
                return

            if hasattr(response, 'success') and not response.success:
                rospy.logwarn(
                    f"Could not get robot pose for position verification - service failed: "
                    f"{getattr(response, 'message', 'Unknown error')}"
                )
                return

            robot_pose = getattr(response, 'robot_pose', None)
            pose_msg = getattr(robot_pose, 'pose', None)
            if pose_msg is None:
                rospy.logwarn("Position verification failed - response missing pose data")
                return

            pose_update = self.pose_tracker.update_from_pose(pose_msg)
            actual_cell = pose_update.get('cell', self.current_cell)
            if not actual_cell:
                rospy.logwarn("Position verification failed - tracker did not return cell")
                return

            expected_row, expected_col = self.last_expected_cell
            actual_row = float(actual_cell[0])
            actual_col = float(actual_cell[1])
            offset_row = actual_row - expected_row
            offset_col = actual_col - expected_col
            
            # Update accumulated offsets (tracking errors across actions)
            self.accumulated_offset_row += offset_row
            self.accumulated_offset_col += offset_col

            rospy.loginfo("Position verification:")
            rospy.loginfo(f"  Expected: ({expected_row:.2f}, {expected_col:.2f}) cells")
            rospy.loginfo(f"  Actual: ({actual_row:.2f}, {actual_col:.2f}) cells")
            rospy.loginfo(
                f"  This movement's error: ({offset_row:.2f}, {offset_col:.2f}) cells = "
                f"({offset_row * 57:.1f}, {offset_col * 57:.1f}) cm"
            )

            # IMPORTANT: Do NOT update current_cell to actual position after verification
            # Keep current_cell as expected position, so accumulated offsets work correctly
            # The actual position for next movement = current_cell + accumulated_offset
            # Only update facing and yaw from pose_update, not position
            facing = pose_update.get("facing")
            if isinstance(facing, str):
                self.current_facing = facing
            yaw_rel = pose_update.get("yaw_rel")
            if isinstance(yaw_rel, (int, float)):
                self.current_yaw_rel = float(yaw_rel)
            self.yaw_origin = self.pose_tracker.yaw_origin
            self.grid_offset_row = self.grid_mapper.grid_offset_row
            self.grid_offset_col = self.grid_mapper.grid_offset_col

            if abs(offset_row) > 0.1 or abs(offset_col) > 0.1:
                rospy.logwarn(
                    f"Significant error in this movement: ({offset_row:.2f}, {offset_col:.2f}) cells = "
                    f"({offset_row * 57:.1f}, {offset_col * 57:.1f}) cm (exceeds 0.1 cell threshold)"
                )
            else:
                rospy.loginfo("Position verification: Movement accurate, no correction needed")
            
                
        except Exception as e:
            rospy.logwarn(f"Position verification failed: {e}")
    
    def process_command(self, command):
        """Process natural language command."""
        # Check for stop signal before processing new command
        if self.stop_requested:
            rospy.loginfo("Stop signal received - ignoring new command until stop is cleared")
            return False
        
        # Reset stop flag when processing new command
        self.stop_requested = False
            
        current_state = self.spot_fsm.current_state.name
        print(f"\nProcessing: {command}")
        print(f"Current robot state: {current_state}")
        
        # Publish timing event for HoloLens input received (when processing starts)
        recorder.publish_event('received_hololens_input')
        
        # Process with LLM immediately after receiving command
        actions = self.parse_command(command)
        if not actions:
            # Get error message if available, otherwise use default
            error_message = getattr(self, 'last_plan_error', None)
            if not error_message:
                error_message = "Plan generation failed - LLM backend unavailable or command unclear. Verify the language model service and try again."
            print("Parse failed - LLM backend unavailable or command unclear")
            print("Verify the language model service and try again")
            # Publish failure feedback
            self.pub_feedback.publish(String(data="Command parsing failed - LLM unavailable or command unclear"))
            try:
                self.pub_plan_preview.publish(String(data=json.dumps({
                    "actions": [],
                    "start_cell": None,
                    "start_facing": None,
                    "error": error_message
                })))
            except Exception as publish_error:
                rospy.logwarn(f"Failed to publish empty plan preview: {publish_error}")
            # Clear error after publishing
            self.last_plan_error = None
            return
        
        # Show full LLM response if reasoning was used
        if hasattr(self, 'last_llm_full_response') and self.last_llm_full_response:
            print("\n" + "="*80)
            print("LLM Generated Raw Plan")
            print("="*80)
            print(self.last_llm_full_response)
            print("="*80)
        
        plan_heading = "LLM Generated Plan"
        if getattr(self, "last_plan_wall_enforced", False):
            if getattr(self, "last_plan_enforcement_applied", False):
                plan_heading = "LLM Improved Plan (wall-safe adjustments applied)"
            else:
                plan_heading = "LLM Improved Plan (wall-safe paths enabled)"

        print(f"\n{plan_heading}:")
        for i, action in enumerate(actions, 1):
            print(f"  {i}. {action}")

        # Publish structured plan preview for GUI animation
        try:
            plan_payload = {
                "actions": actions,
                "start_cell": list(self.current_cell) if self.current_cell else None,
                "start_facing": self.current_facing,
                "timestamp": time.time(),
                "wall_enforcement_enabled": getattr(self, "last_plan_wall_enforced", False),
                "wall_enforcement_applied": getattr(self, "last_plan_enforcement_applied", False),
            }
            self.pub_plan_preview.publish(String(data=json.dumps(plan_payload)))
            rospy.loginfo("Published plan preview to GUI")
        except Exception as publish_error:
            rospy.logwarn(f"Failed to publish plan preview: {publish_error}")
        
        if self.interpretation_enabled:
            # Generate natural language interpretation
            natural_plan = self._interpret_plan(actions)
            # Publish interpretation to GUI
            self.pub_interpretation.publish(String(data=natural_plan))
        
        # Publish timing event for user confirmation start
        recorder.publish_event('start_user_confirmation')
        
        # In speech mode, wait for approval like orchestrator HA mode
        if self.use_speech:
            rospy.loginfo("Speech mode: waiting for approval")
            self.pending_plan = actions
            self.awaiting_approval = True
        else:
            # Terminal mode: ask for confirmation
            confirm = input("\nExecute this plan? (y/n): ").strip().lower()
            
            # Publish timing event for user confirmation end
            recorder.publish_event('stop_user_confirmation')
            
            if confirm in ['y', 'yes']:
                print("Executing...")
                self.execute_actions(actions)
            else:
                print("Cancelled - try a new command")
    
    def run(self):
        """Main loop."""
        print("\nNatural Language Control Ready!")
        if self.use_speech:
            print("Speech mode: listening on /hl/user_speech")
        else:
            print("Terminal mode: type commands or 'quit' to exit")
        print()
        
        while not rospy.is_shutdown():
            try:
                if self.use_speech:
                    if self.speech_event.wait(timeout=1.0):
                        command = self.speech_input
                        self.speech_event.clear()
                        if command:
                            self.process_command(command)
                else:
                    try:
                        command = input("Command: ").strip()
                        if command.lower() == 'quit':
                            print("Goodbye!")
                            break
                        if command:
                            self.process_command(command)
                    except (EOFError, KeyboardInterrupt):
                        print("\nGoodbye!")
                        break
                        
            except rospy.ROSInterruptException:
                break
            except Exception as e:
                print(f"Error: {e}")
                rospy.sleep(1.0)

    def stop(self):
        try:
            rospy.loginfo("NL Control: stop() called - initiating shutdown")
            self.stop_requested = True

            try:
                if hasattr(self, 'pose_sync_timer') and self.pose_sync_timer is not None:
                    if hasattr(self.pose_sync_timer, 'shutdown'):
                        self.pose_sync_timer.shutdown()
                    elif hasattr(self.pose_sync_timer, 'stop'):
                        self.pose_sync_timer.stop()
                    self.pose_sync_timer = None
            except Exception:
                pass

            try:
                rospy.signal_shutdown("NL Control stopping")
            except Exception:
                pass

        except Exception as e:
            try:
                rospy.logwarn(f"NL Control: stop() encountered an error: {e}")
            except Exception:
                pass
    
    def on_hl_stop(self, msg):
        """Handle HoloLens stop signal"""
        rospy.loginfo("HoloLens stop signal received - will finish current task and stop plan execution")
        self.stop_requested = True
    
    def on_world_change(self, msg):
        """Handle world change from GUI."""
        world_id = msg.data
        rospy.loginfo(f"World change received: {world_id}")
        print(f"DEBUG: NL Control received world change: {world_id}")
        if self.switch_world(world_id):
            rospy.loginfo(f"Successfully switched to world {world_id}")
            print(f"DEBUG: NL Control successfully switched to world {world_id}")
            # Update the prompt template with new world model
            self._update_prompt_template()
        else:
            rospy.logwarn(f"Failed to switch to world {world_id}")
            print(f"DEBUG: NL Control failed to switch to world {world_id}")
    
    def on_position_reset(self, msg):
        """Handle position reset from GUI."""
        reset_data = msg.data
        rospy.loginfo(f"Position reset received: {reset_data}")
        print(f"DEBUG: NL Control received position reset: {reset_data}")
        
        # Handle position request
        if reset_data == "request_current_position":
            rospy.loginfo("GUI requested current position - publishing current position")
            self.publish_position_update()
            return
        
        # Parse reset data: "reset_to_4_4_N"
        if reset_data.startswith("reset_to_"):
            try:
                parts = reset_data.split("_")
                if len(parts) >= 4:
                    row = int(parts[2])
                    col = int(parts[3])
                    facing = parts[4] if len(parts) > 4 else "N"
                    
                    # Reset internal position tracking
                    self.current_cell = [row, col]
                    self.current_facing = facing
                    self.pose_tracker.current_cell = [float(row), float(col)]
                    self.pose_tracker.current_facing = facing
                    self.pose_tracker.current_yaw_rel = 0.0
                    self.pose_tracker.yaw_origin = None
                    self._has_position_update = True
                    
                    rospy.loginfo(f"Reset position to: cell ({row}, {col}) facing {facing}")
                    print(f"DEBUG: NL Control reset position to: cell ({row}, {col}) facing {facing}")
                    
                    # Publish the reset position to update other components
                    self.publish_position_update()
                else:
                    rospy.logwarn(f"Invalid position reset format: {reset_data}")
            except (ValueError, IndexError) as e:
                rospy.logwarn(f"Failed to parse position reset: {reset_data}, error: {e}")
        else:
            rospy.logwarn(f"Invalid position reset message: {reset_data}")
    
    def on_approval(self, msg):
        """Handle approval from GUI."""
        try:
            import json
            payload = json.loads(msg.data)
        except Exception as e:
            rospy.logwarn(f"Invalid JSON on /hl/approval: {e}")
            return
        
        if not self.awaiting_approval:
            rospy.logwarn("Plan not awaiting approval, ignoring /hl/approval message")
            return
        
        decision = payload.get("approved")
        if decision:
            rospy.loginfo("Plan approved. Executing...")
            self.awaiting_approval = False
            plan = list(self.pending_plan)
            self.pending_plan = []
            # Execute actions in a separate thread to avoid blocking GUI on Windows
            execution_thread = threading.Thread(target=self.execute_actions, args=(plan,), daemon=True)
            execution_thread.start()
        else:
            rospy.loginfo("Plan rejected. Describe a new plan.")
            self.awaiting_approval = False
            self.pending_plan = []
    
    def on_direct_execution(self, msg):
        """Handle cached plans and regular plans for approval system."""
        rospy.loginfo(f"DEBUG: on_direct_execution called, received message: {type(msg)}")
        try:
            import json
            payload = json.loads(msg.data)
            rospy.loginfo(f"DEBUG: Parsed payload - direct_execution={payload.get('direct_execution')}, bypass_approval={payload.get('bypass_approval')}, actions_count={len(payload.get('actions', []))}")
        except Exception as e:
            rospy.logwarn(f"Invalid JSON on /nl_control/plan: {e}")
            import traceback
            rospy.logwarn(f"Traceback: {traceback.format_exc()}")
            return
        
        actions = payload.get("actions", [])
        if not actions:
            rospy.logwarn("No actions provided in plan message")
            return
        
        # Check if this is a direct execution request (bypass approval)
        direct_exec = payload.get("direct_execution", False)
        bypass_approval = payload.get("bypass_approval", False)
        
        rospy.loginfo(f"DEBUG: direct_execution={direct_exec}, bypass_approval={bypass_approval}")
        
        if direct_exec and bypass_approval:
            command = payload.get("command", "cached plan")
            
            rospy.loginfo(f"Direct execution of baseline/cached plan: {command}")
            rospy.loginfo(f"Actions: {actions}")
            
            # Execute actions directly in a separate thread
            execution_thread = threading.Thread(target=self.execute_actions, args=(actions,), daemon=True)
            execution_thread.start()
        else:
            # Regular plan or cached plan - set up for approval
            if payload.get("cached_plan"):
                command = payload.get("command", "cached plan")
                rospy.loginfo(f"Setting up cached plan for approval: {command}")
            else:
                # Regular plan from GUI - ensure approval state is set up
                rospy.loginfo(f"Setting up plan for approval (from GUI)")
            
            rospy.loginfo(f"Actions: {actions}")
            
            # Set up the approval state with the plan
            self.pending_plan = list(actions)
            self.awaiting_approval = True
            
            rospy.loginfo("Plan ready for approval")
    
    def _update_prompt_template(self):
        """Store reference to current world model for prompt formatting."""
        try:
            rospy.loginfo("Prompt template updated with new world model")
        except Exception as e:
            rospy.logwarn(f"Failed to update prompt template: {e}")
    
    def _update_cached_pose_state(self, pose_update: dict[str, object]) -> None:
        """Refresh cached position fields from a pose tracker update."""
        cell = pose_update.get("cell")
        if isinstance(cell, (list, tuple)) and len(cell) >= 2:
            self.current_cell = [float(cell[0]), float(cell[1])]

        facing = pose_update.get("facing")
        if isinstance(facing, str):
            self.current_facing = facing

        yaw_rel = pose_update.get("yaw_rel")
        if isinstance(yaw_rel, (int, float)):
            self.current_yaw_rel = float(yaw_rel)

        self.yaw_origin = self.pose_tracker.yaw_origin
        self.grid_offset_row = self.grid_mapper.grid_offset_row
        self.grid_offset_col = self.grid_mapper.grid_offset_col

    def _interpret_plan(self, plan):
        """Convert programmatic plan to natural language."""
        if not getattr(self, "interpretation_enabled", True):
            return ""

        if not plan:
            return "No plan generated"

        if not self.plan_interpreter:
            return generate_fallback_interpretation(plan)

        try:
            interpretation = self.plan_interpreter.interpret(list(plan))
            if interpretation:
                return interpretation
        except Exception as exc:  # pragma: no cover - defensive logging
            rospy.logwarn(f"Plan interpretation failed: {exc}")

        return generate_fallback_interpretation(plan)
    
    def on_robot_position_update(self, msg):
        """Handle robot position updates from robot simulator (dummy mode)."""
        try:
            rospy.loginfo(f"[NL_CONTROL DEBUG] Received position message: {msg}")
            rospy.loginfo(f"[NL_CONTROL DEBUG] Message type: {type(msg)}")
            rospy.loginfo(f"[NL_CONTROL DEBUG] Message data: {msg.data}")
            import json
            position_data = json.loads(msg.data)
            row = position_data.get('row', self.current_cell[0])
            col = position_data.get('col', self.current_cell[1])
            facing = position_data.get('facing', self.current_facing)
            
            # If we mistakenly are in real_robot_mode but receive simulator updates,
            # immediately switch to simulation so GUI follows simulated position.
            if getattr(self, 'real_robot_mode', False):
                rospy.loginfo("NL Control: Simulator position received while in real mode - switching to simulation")
                self._switch_to_dummy_mode()

            # Update internal position tracking
            import math
            self.current_cell = [row, col]
            self.current_facing = facing
            self.pose_tracker.current_cell = [float(row), float(col)]
            self.pose_tracker.current_facing = facing
            # Set current_yaw_rel based on facing direction relative to North
            facing_to_yaw = {
                'N': 0.0,
                'E': -math.pi / 2,
                'S': math.pi,
                'W': math.pi / 2,
            }
            self.pose_tracker.current_yaw_rel = facing_to_yaw.get(facing, 0.0)
            self.current_yaw_rel = self.pose_tracker.current_yaw_rel
            self.pose_tracker.yaw_origin = None
            self._has_position_update = True
            
            rospy.loginfo(f"[NL_CONTROL DEBUG] Received position update from robot simulator: ({row}, {col})")
            
            # Publish position update to GUI
            self.publish_position_update()
            
        except Exception as e:
            rospy.logwarn(f"Failed to handle robot position update: {e}")
            import traceback
            traceback.print_exc()

    def publish_position_update(self):
        """Publish current robot position to GUI with full position data."""
        try:
            import json
            position_data = self._build_position_payload()
            self.pub_position.publish(String(data=json.dumps(position_data)))
            
            # Re-publish once more after a delay using the freshest state.
            with self._position_timer_lock:
                if self._position_timer is not None:
                    self._position_timer.cancel()
                self._position_timer = threading.Timer(1.0, self._republish_latest_position)
                self._position_timer.daemon = True
                self._position_timer.start()
        except Exception as e:
            rospy.logwarn(f"Failed to publish position update: {e}")

    def _republish_latest_position(self):
        """Re-publish the latest position, ensuring timers do not queue stale data."""
        try:
            import json
            with self._position_timer_lock:
                self._position_timer = None
            position_data = self._build_position_payload()
            self.pub_position.publish(String(data=json.dumps(position_data)))
        except Exception as publish_error:
            rospy.logwarn(f"Failed to re-publish position update: {publish_error}")

    def _build_position_payload(self) -> Dict[str, Union[float, str]]:
        """Return a snapshot of the robot pose for GUI consumers."""
        import math
        gui_row = float(self.current_cell[0])
        gui_col = float(self.current_cell[1])
        yaw_deg = float(self.current_yaw_rel) * 180.0 / math.pi
        # Cell size is 0.57m (57cm) as defined in pose_tracker.py GridFrameMapper
        x_m = (gui_col + 0.5) * 0.57
        y_m = (gui_row + 0.5) * 0.57

        return {
            'row': gui_row,
            'col': gui_col,
            'facing': self.current_facing,
            'yaw_deg': yaw_deg,
            'x_m': x_m,
            'y_m': y_m,
        }


def main():
    # Configuration: True=speech, False=terminal
    USE_SPEECH = True  # Changed to True to listen to GUI commands
    
    try:
        # Check if we're running in launch mode (with parameters)
        try:
            rospy.init_node('nl_control', anonymous=True)
            world_id = rospy.get_param('~world_id', None)
            if world_id:
                # Running in launch mode - use parameter
                print(f"Launch mode: Using world {world_id}")
                controller = NaturalLanguageControl(use_speech=USE_SPEECH, world_id=world_id)
                controller.run()
                return
        except:
            # Not running in parameter mode, continue with interactive selection
            pass
        
        # Interactive world selection mode
        # Create world manager for selection without full controller
        wm = WorldManager()
        
        # Show world selection
        print("\n" + "="*60)
        print("SPOT ROBOT - WORLD SELECTION")
        print("="*60)
        print("Available Worlds:")
        print()
        
        world_list = wm.get_world_list()
        for world_id, name, description in world_list:
            print(f"{world_id}. {name}")
            print(f"   {description}")
            print()
        
        while True:
            try:
                choice = input("Select a world (1-10) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    print("No world selected. Exiting...")
                    return
                
                if choice in wm.worlds:
                    world_name, world_desc = wm.get_world_info(choice)
                    print(f"\nSelected: {world_name}")
                    print(f"Description: {world_desc}")
                    confirm = input("Continue with this world? (y/n): ").strip().lower()
                    if confirm in ['y', 'yes']:
                        selected_world = choice
                        break
                    else:
                        continue
                else:
                    print("Invalid selection. Please choose 1-10 or 'q' to quit.")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting...")
                return
        
        # Create the controller with selected world
        controller = NaturalLanguageControl(use_speech=USE_SPEECH, world_id=selected_world)
        controller.run()
    except KeyboardInterrupt:
        rospy.loginfo("Shutdown")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
    finally:
        rospy.loginfo("Natural Language Control shutdown complete")


if __name__ == "__main__":
    main()
