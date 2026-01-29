#!/usr/bin/env python3
"""
Robot Simulator - Dummy robot services for standalone simulation
"""

import time
import threading
import json
from typing import Dict, Any, Optional, Tuple
from message_bus import message_bus, Publisher, String, PoseStamped, Pose, Point, Quaternion, Header


class RobotSimulator:
    """Simulates Spot robot behavior in standalone mode."""
    
    def __init__(self):
        self.connected = False
        self.powered = False
        self.standing = False
        self.current_cell = [4, 4]  # [row, col] in grid - start in true middle
        self.facing = "N"  # N, S, E, W
        self.initial_cell = [4, 4]  # Store initial position
        self.arm_status = "stowed"
        self.gripper_status = "closed"
        self.has_object = False
        self.carried_object = None
        self.world_config = None  # Will be set by world manager
        
        # Publishers for state updates
        self.robot_state_pub = Publisher('/spot_entrance/robot_state')
        self.feedback_pub = Publisher('/spot/execution_feedback')
        self.position_pub = Publisher('/robot_simulator/position')
        
        # Start state publishing
        self.state_timer = threading.Timer(1.0, self._publish_state)
        self.state_timer.start()
        
        # Publish initial position after a delay to ensure subscribers are ready
        threading.Timer(2.0, self._publish_initial_position).start()
    
    def _publish_state(self):
        """Publish current robot state."""
        # Only publish state if we're in pure simulation mode (no real robot services)
        # In standalone mode with real robot services, let spot_entrance handle robot state
        try:
            from message_bus import is_ros_master_available
            if is_ros_master_available():
                # Check if real robot services are available
                import rospy
                try:
                    # If spot_entrance services are available, don't publish our own state
                    rospy.wait_for_service('/spot_entrance/get_robot_pose', timeout=0.1)
                    # Real robot services are available, skip publishing
                    self.state_timer = threading.Timer(1.0, self._publish_state)
                    self.state_timer.start()
                    return
                except rospy.ROSException:
                    # No real robot services, we can publish our state
                    pass
        except Exception:
            # Fallback: publish our state
            pass
            
        state_msg = f"connected:{self.connected},powered:{self.powered},standing:{self.standing}"
        self.robot_state_pub.publish(String(data=state_msg))
        
        # Schedule next state update
        self.state_timer = threading.Timer(1.0, self._publish_state)
        self.state_timer.start()
    
    def _publish_initial_position(self):
        """Publish initial robot position."""
        initial_position = {
            'row': self.current_cell[0],
            'col': self.current_cell[1],
            'facing': self.facing
        }
        print(f"Robot Simulator: Publishing initial position: {initial_position}")
        print(f"Robot Simulator: Publisher topic: {self.position_pub.topic}")
        print(f"Robot Simulator: Publisher type: {type(self.position_pub)}")
        self.position_pub.publish(String(data=json.dumps(initial_position)))
        print(f"Robot Simulator: Initial position published successfully")
    
    def connect(self) -> Dict[str, Any]:
        """Connect to robot."""
        print("Robot Simulator: Connecting...")
        time.sleep(0.5)  # Simulate connection delay
        self.connected = True
        return {"success": True, "message": "Connected to robot (simulation)"}
    
    def disconnect(self) -> Dict[str, Any]:
        """Disconnect from robot."""
        print("Robot Simulator: Disconnecting...")
        self.connected = False
        self.powered = False
        self.standing = False
        return {"success": True, "message": "Disconnected from robot (simulation)"}
    
    def power_on(self) -> Dict[str, Any]:
        """Power on robot."""
        if not self.connected:
            return {"success": False, "message": "Not connected"}
        
        print("Robot Simulator: Powering on...")
        time.sleep(1.0)  # Simulate power on delay
        self.powered = True
        return {"success": True, "message": "Robot powered on (simulation)"}
    
    def power_off(self) -> Dict[str, Any]:
        """Power off robot."""
        print("Robot Simulator: Powering off...")
        time.sleep(0.5)
        self.powered = False
        self.standing = False
        return {"success": True, "message": "Robot powered off (simulation)"}
    
    def stand(self, height: float = 0.0) -> Dict[str, Any]:
        """Stand up robot."""
        if not self.powered:
            return {"success": False, "message": "Robot not powered"}
        
        print(f"Robot Simulator: Standing up (height={height})...")
        
        # Simulate standing with position updates
        stand_time = 0.5  # Faster standing
        num_steps = 5  # Fewer steps for faster simulation
        step_time = stand_time / num_steps
        
        for i in range(num_steps):
            time.sleep(step_time)
            # Publish intermediate feedback
            if i == num_steps // 2:
                self.feedback_pub.publish(String(data="Step: robot rising to standing position"))
        
        self.standing = True
        
        # Publish final position after standing
        import json
        final_position = {
            'row': self.current_cell[0],
            'col': self.current_cell[1],
            'facing': self.facing
        }
        self.position_pub.publish(String(data=json.dumps(final_position)))
        
        # Give GUI time to process the update (critical for Windows)
        time.sleep(0.15)
        
        self.feedback_pub.publish(String(data="✓ stand_up"))
        return {"success": True, "message": "Robot standing (simulation)"}
    
    def sit(self) -> Dict[str, Any]:
        """Sit down robot."""
        print("Robot Simulator: Sitting down...")
        
        # Simulate sitting with position updates
        sit_time = 1.0
        num_steps = 10  # 10 steps per second
        step_time = sit_time / num_steps
        
        for i in range(num_steps):
            time.sleep(step_time)
            # Publish intermediate feedback
            if i == num_steps // 2:
                self.feedback_pub.publish(String(data="Step: robot lowering to sitting position"))
        
        self.standing = False
        
        # Publish final position after sitting
        import json
        final_position = {
            'row': self.current_cell[0],
            'col': self.current_cell[1],
            'facing': self.facing
        }
        self.position_pub.publish(String(data=json.dumps(final_position)))
        
        # Give GUI time to process the update (critical for Windows)
        time.sleep(0.15)
        
        self.feedback_pub.publish(String(data="✓ sit_down"))
        return {"success": True, "message": "Robot sitting (simulation)"}
    
    def move_to_cell(self, target_row: int, target_col: int) -> Dict[str, Any]:
        """Move robot to target cell (can be multiple cells away, but must be on same axis)."""
        if not self.standing:
            return {"success": False, "message": "Robot not standing"}
        
        current_row, current_col = self.current_cell
        
        # Check bounds
        if target_row < 0 or target_row >= 10 or target_col < 0 or target_col >= 10:
            return {"success": False, "message": f"Target cell ({target_row}, {target_col}) is out of bounds (0-9)"}
        
        # Check if movement is on one axis only (horizontal or vertical)
        row_diff = target_row - current_row
        col_diff = target_col - current_col
        
        if row_diff != 0 and col_diff != 0:
            return {"success": False, "message": f"Can only move on one axis. From ({current_row}, {current_col}) to ({target_row}, {target_col}) requires diagonal movement"}
        
        if row_diff == 0 and col_diff == 0:
            return {"success": False, "message": f"Already at target cell ({target_row}, {target_col})"}
        
        # Check for obstacles along the path (even though we don't block movement in simulation)
        path_obstacles = self._check_path_obstacles(current_row, current_col, target_row, target_col)
        if path_obstacles:
            print(f"Robot Simulator: WARNING - Path passes through obstacles: {path_obstacles}")
        
        print(f"Robot Simulator: Moving from ({current_row}, {current_col}) to ({target_row}, {target_col})...")
        
        # Simulate movement with position updates (slower for GUI refresh)
        move_time = 1.0  # total time for a move between cells
        num_steps = 10  # more steps for smoother, slower updates
        step_time = move_time / num_steps
        
        for i in range(num_steps + 1):
            # Small delay
            if i < num_steps:
                time.sleep(step_time)
        
        # Update position
        self.current_cell = [target_row, target_col]

        # Publish final position so the GUI reflects the completed move immediately
        final_position = {
            'row': self.current_cell[0],
            'col': self.current_cell[1],
            'facing': self.facing
        }
        print(f"Robot Simulator: Publishing position update: {final_position}")
        self.position_pub.publish(String(data=json.dumps(final_position)))
        
        # Give GUI time to process the update (critical for Windows)
        time.sleep(0.15)
        
        self.feedback_pub.publish(String(data=f"✓ move_to_cell row={target_row} col={target_col}"))
        return {"success": True, "message": f"Moved to cell ({target_row}, {target_col})"}
    
    def _check_path_obstacles(self, from_row: int, from_col: int, to_row: int, to_col: int) -> list:
        """Check for obstacles along the path from one cell to another."""
        if not self.world_config:
            return []  # No obstacles if no world config
        
        obstacle_cells = self.world_config.get("obstacle_cells", [])
        obstacles = []
        
        # Determine direction and step
        if from_row == to_row:  # Horizontal movement
            step = 1 if to_col > from_col else -1
            for col in range(from_col + step, to_col + step, step):
                if [from_row, col] in obstacle_cells:
                    obstacles.append([from_row, col])
        elif from_col == to_col:  # Vertical movement
            step = 1 if to_row > from_row else -1
            for row in range(from_row + step, to_row + step, step):
                if [row, from_col] in obstacle_cells:
                    obstacles.append([row, from_col])
        
        return obstacles
    
    def _can_move_to_cell(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        """Check if the target cell is a wall cell."""
        if not self.world_config:
            return True  # No walls if no world config
        
        obstacle_cells = self.world_config.get("obstacle_cells", [])
        
        # Check if target cell is a wall
        target_cell = [to_row, to_col]
        if target_cell in obstacle_cells:
            return False
        
        return True
    
    def set_world_config(self, world_config: Dict[str, Any]):
        """Set the world configuration for wall checking."""
        self.world_config = world_config
        
        # Get wall cells from config section
        config = world_config.get("config", {})
        obstacle_cells = config.get("obstacle_cells", [])
        print(f"Robot Simulator: Received {len(obstacle_cells)} wall cells")
        
        # Update robot position if specified in world config
        robot_start = world_config.get("robot_start_position")
        if robot_start:
            self.current_cell = [robot_start["row"], robot_start["col"]]
            self.initial_cell = [robot_start["row"], robot_start["col"]]
            self.facing = robot_start["facing"]
            print(f"Robot Simulator: Updated robot position to ({self.current_cell[0]}, {self.current_cell[1]}) facing {self.facing}")
            # Publish the updated position
            self._publish_initial_position()
    
    def rotate_to_direction(self, direction: str) -> Dict[str, Any]:
        """Rotate robot to face cardinal direction."""
        if not self.standing:
            return {"success": False, "message": "Robot not standing"}
        
        if direction not in ["N", "S", "E", "W"]:
            return {"success": False, "message": f"Invalid direction: {direction}. Must be N, S, E, or W"}
        
        if self.facing == direction:
            return {"success": True, "message": f"Already facing {direction}"}
        
        print(f"Robot Simulator: Rotating from {self.facing} to {direction}...")
        
        # Simulate rotation (slower for GUI refresh)
        rotate_time = 1.2
        num_steps = 6  # more steps for smoother, slower rotation
        step_time = rotate_time / num_steps
        
        for i in range(num_steps + 1):
            # Small delay
            if i < num_steps:
                time.sleep(step_time)
        
        # Update facing direction
        self.facing = direction

        final_orientation = {
            'row': self.current_cell[0],
            'col': self.current_cell[1],
            'facing': self.facing
        }
        self.position_pub.publish(String(data=json.dumps(final_orientation)))
        
        # Give GUI time to process the update (critical for Windows)
        time.sleep(0.15)
        
        self.feedback_pub.publish(String(data=f"✓ rotate_to_direction {direction}"))
        return {"success": True, "message": f"Now facing {direction}"}
    
    def get_robot_pose(self) -> Dict[str, Any]:
        """Get current robot pose in grid coordinates."""
        return {
            "success": True, 
            "robot_cell": self.current_cell,
            "facing": self.facing
        }
    
    def set_pose(self, row: int, col: int, facing: str) -> Dict[str, Any]:
        """Force the robot pose to a specific grid cell and orientation."""
        if facing not in ["N", "S", "E", "W"]:
            return {"success": False, "message": f"Invalid facing: {facing}"}

        self.current_cell = [row, col]
        self.facing = facing

        pose_data = {
            "row": row,
            "col": col,
            "facing": facing,
        }
        self.position_pub.publish(String(data=json.dumps(pose_data)))
        time.sleep(0.15)
        return {"success": True, "message": f"Pose set to ({row}, {col}) facing {facing}"}
    
    def get_initial_pose(self) -> Dict[str, Any]:
        """Get initial robot pose in grid coordinates."""
        if self.initial_cell is None:
            self.initial_cell = self.current_cell.copy()
        
        return {
            "success": True, 
            "initial_cell": self.initial_cell,
            "initial_facing": "N"
        }
    
    def arm_command(self, command_type: str) -> Dict[str, Any]:
        """Execute arm command."""
        if not self.standing:
            return {"success": False, "message": "Robot not standing"}
        
        print(f"Robot Simulator: Executing arm command: {command_type}")
        
        # Simulate arm movement with moderate updates
        arm_time = 0.5
        num_steps = 5  # 10 steps per second
        step_time = arm_time / num_steps
        
        for i in range(num_steps):
            time.sleep(step_time)
            # Publish intermediate feedback
            if i == num_steps // 2:
                self.feedback_pub.publish(String(data=f"Step: arm moving to {command_type} position"))
        
        if command_type == "stow":
            self.arm_status = "stowed"
        elif command_type == "ready":
            self.arm_status = "ready"
        elif command_type == "carry":
            self.arm_status = "carry"
        
        self.feedback_pub.publish(String(data=f"✓ arm_command {command_type}"))
        return {"success": True, "message": f"Arm command {command_type} executed"}
    
    def automated_grasp(self, object_type: str) -> Dict[str, Any]:
        """Execute automated grasp."""
        if not self.standing:
            return {"success": False, "message": "Robot not standing"}
        
        # Normalize object type: replace spaces with underscores to match GUI object keys
        normalized_object = object_type.replace(' ', '_')
        
        print(f"Robot Simulator: Grasping {object_type}...")
        
        # Simulate grasp sequence with arm movement and position updates
        grasp_time = 2.0
        num_steps = 20  # 10 steps per second
        step_time = grasp_time / num_steps
        
        for i in range(num_steps):
            time.sleep(step_time)
            
            # Simulate arm extending and gripper closing
            if i < num_steps // 3:  # First 1/3: extending arm
                self.arm_status = "extending"
                self.gripper_status = "open"
                if i == num_steps // 6:  # Halfway through extension
                    self.feedback_pub.publish(String(data=f"Step: extending arm to {object_type}"))
            elif i < 2 * num_steps // 3:  # Second 1/3: closing gripper
                self.arm_status = "grasping"
                self.gripper_status = "closing"
                if i == num_steps // 2:  # Start of closing
                    self.feedback_pub.publish(String(data=f"Step: closing gripper on {object_type}"))
            else:  # Final 1/3: retracting arm
                self.arm_status = "retracting"
                self.gripper_status = "closed"
                # Set object status when gripper closes (use normalized name)
                self.has_object = True
                self.carried_object = normalized_object
            
            
            # Publish arm/gripper status update with object status
            self.feedback_pub.publish(String(data=f"arm_status:{self.arm_status} gripper_status:{self.gripper_status} has_object:{self.has_object} carried_object:{self.carried_object}"))
        
        # Final state after grasping
        self.has_object = True
        # Keep normalized name to stay consistent with GUI object keys
        self.carried_object = normalized_object
        self.arm_status = "carry"
        self.gripper_status = "closed"
        
        # Publish final position after grasping
        import json
        final_position = {
            'row': self.current_cell[0],
            'col': self.current_cell[1],
            'facing': self.facing
        }
        self.position_pub.publish(String(data=json.dumps(final_position)))
        
        # Give GUI time to process the update (critical for Windows)
        time.sleep(0.15)
        
        self.feedback_pub.publish(String(data=f"✓ start_automated_grasp {object_type}"))
        return {"success": True, "message": f"Grasped {object_type}"}
    
    def drop_off(self) -> Dict[str, Any]:
        """Execute drop off sequence."""
        if not self.standing:
            return {"success": False, "message": "Robot not standing"}
        
        if not self.has_object:
            return {"success": False, "message": "No object to drop"}
        
        print(f"Robot Simulator: Dropping off {self.carried_object}...")
        
        # Simulate drop off sequence with arm movement and position updates
        drop_time = 2.0  # 2.0 seconds for more realistic drop-off
        num_steps = 20  # More steps for smoother drop-off
        step_time = drop_time / num_steps
        
        for i in range(num_steps):
            time.sleep(step_time)
            
            # Simulate arm extending and gripper opening
            if i < num_steps // 3:  # First 1/3: extending arm
                self.arm_status = "extending"
                self.gripper_status = "closed"
                if i == num_steps // 6:  # Halfway through extension
                    self.feedback_pub.publish(String(data=f"Step: extending arm to drop {self.carried_object}"))
            elif i < 2 * num_steps // 3:  # Second 1/3: opening gripper
                self.arm_status = "dropping"
                self.gripper_status = "opening"
                if i == num_steps // 2:  # Start of opening
                    self.feedback_pub.publish(String(data=f"Step: opening gripper to release {self.carried_object}"))
                    # Clear object status when gripper opens
                    self.has_object = False
                    self.carried_object = None
            else:  # Final 1/3: retracting arm
                self.arm_status = "retracting"
                self.gripper_status = "open"
            
            
            # Publish arm/gripper status update with object status
            self.feedback_pub.publish(String(data=f"arm_status:{self.arm_status} gripper_status:{self.gripper_status} has_object:{self.has_object} carried_object:{self.carried_object}"))
        
        # Final state after drop-off
        self.has_object = False
        self.carried_object = None
        self.arm_status = "stowed"
        self.gripper_status = "closed"
        
        # Publish final stowed state before completion
        self.feedback_pub.publish(String(data=f"[exec] arm_status:{self.arm_status} gripper_status:{self.gripper_status} has_object:{self.has_object} carried_object:{self.carried_object}"))
        
        # Publish final position after drop-off
        import json
        final_position = {
            'row': self.current_cell[0],
            'col': self.current_cell[1],
            'facing': self.facing
        }
        self.position_pub.publish(String(data=json.dumps(final_position)))
        
        # Give GUI time to process the update (critical for Windows)
        time.sleep(0.15)
        
        self.feedback_pub.publish(String(data="✓ start_drop_off"))
        return {"success": True, "message": "Object dropped off"}
    
    def get_image(self, image_source: str = "frontleft_fisheye_image") -> Dict[str, Any]:
        """Get image from camera."""
        print(f"Robot Simulator: Getting image from {image_source}...")
        time.sleep(0.2)  # Simulate image capture
        
        # Return a mock image (in real implementation, this would be actual image data)
        return {"success": True, "image": f"Mock image from {image_source}"}
    
    def get_current_state(self) -> str:
        """Get current robot state as string."""
        if not self.connected:
            return "disconnected"
        elif not self.powered:
            return "powered_off"
        elif self.standing:
            return "stand"
        else:
            return "sit"


# Global robot simulator instance - lazy initialization
_robot_simulator_instance = None

def get_robot_simulator():
    """Get the global robot simulator instance, creating it if necessary."""
    global _robot_simulator_instance
    if _robot_simulator_instance is None:
        _robot_simulator_instance = RobotSimulator()
    return _robot_simulator_instance

# For backward compatibility, create a proxy object
class RobotSimulatorProxy:
    def __getattr__(self, name):
        return getattr(get_robot_simulator(), name)

robot_simulator = RobotSimulatorProxy()
