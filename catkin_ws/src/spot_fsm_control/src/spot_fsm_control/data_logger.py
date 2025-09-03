#!/usr/bin/env python

import csv
import time
import json
import os
from datetime import datetime
from pathlib import Path


class DataLogger:
    """
    Comprehensive data logger for robot experiments.
    Logs robot states, actions, timing, and experiment data.
    """
    
    def __init__(self, participant_id=-1, condition="default", log_dir="/data/experiments"):
        """
        Initialize data logger.
        
        Args:
            participant_id: Participant number for experiment
            condition: Experimental condition name
            log_dir: Base directory for log files
        """
        self.participant_id = participant_id
        self.condition = condition
        self.timestamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
        
        # Create log directory
        self.log_dir = Path(log_dir) / f"P{participant_id:03d}" / condition / self.timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize CSV files
        self.action_log_file = self.log_dir / "actions.csv"
        self.state_log_file = self.log_dir / "states.csv"
        self.feedback_log_file = self.log_dir / "feedback.csv"
        self.error_log_file = self.log_dir / "errors.csv"
        
        # CSV headers
        self.action_headers = [
            "timestamp", "action_executed", "parameters", "success", 
            "execution_time", "battery_percentage", "robot_state"
        ]
        
        self.state_headers = [
            "timestamp", "state", "previous_state", "transition_reason"
        ]
        
        self.feedback_headers = [
            "timestamp", "action", "status", "message", "response_time"
        ]
        
        self.error_headers = [
            "timestamp", "error_type", "error_message", "action_context", "robot_state"
        ]
        
        # Initialize CSV files
        self._init_csv_files()
        
        # Performance tracking
        self.action_start_time = None
        self.last_battery_check = 0
        self.battery_percentage = 100.0
        
        print(f"Data Logger initialized: {self.log_dir}")
    
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        files_and_headers = [
            (self.action_log_file, self.action_headers),
            (self.state_log_file, self.state_headers),
            (self.feedback_log_file, self.feedback_headers),
            (self.error_log_file, self.error_headers)
        ]
        
        for file_path, headers in files_and_headers:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
    
    def log_action(self, action, parameters=None, success=True, execution_time=None, robot_state=None):
        """
        Log a robot action.
        
        Args:
            action: Action name (e.g., 'stand', 'move_relative')
            parameters: Action parameters as dict
            success: Whether action was successful
            execution_time: Time taken to execute action
            robot_state: Current robot state
        """
        timestamp = datetime.now().isoformat()
        params_str = json.dumps(parameters) if parameters else ""
        
        # Update battery if needed (simulate battery drain)
        if time.time() - self.last_battery_check > 60:  # Check every minute
            self.battery_percentage = max(0, self.battery_percentage - 0.1)
            self.last_battery_check = time.time()
        
        row = [
            timestamp,
            action,
            params_str,
            success,
            execution_time or 0.0,
            self.battery_percentage,
            robot_state or "unknown"
        ]
        
        with open(self.action_log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
        
        print(f"Action logged: {action} ({'success' if success else 'failed'})")
    
    def log_state_transition(self, new_state, previous_state=None, reason=""):
        """
        Log a state transition.
        
        Args:
            new_state: New robot state
            previous_state: Previous robot state
            reason: Reason for transition
        """
        timestamp = datetime.now().isoformat()
        
        row = [
            timestamp,
            new_state,
            previous_state or "unknown",
            reason
        ]
        
        with open(self.state_log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
        
        print(f"🔄 State transition: {previous_state} → {new_state} ({reason})")
    
    def log_feedback(self, action, status, message="", response_time=None):
        """
        Log FSM feedback.
        
        Args:
            action: Action that generated feedback
            status: Feedback status ('ok', 'error')
            message: Feedback message
            response_time: Time between action and feedback
        """
        timestamp = datetime.now().isoformat()
        
        row = [
            timestamp,
            action,
            status,
            message,
            response_time or 0.0
        ]
        
        with open(self.feedback_log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
        
        print(f"Feedback logged: {action} - {status}")
    
    def log_error(self, error_type, error_message, action_context=None, robot_state=None):
        """
        Log an error.
        
        Args:
            error_type: Type of error (e.g., 'safety_violation', 'execution_failed')
            error_message: Error message
            action_context: Action that caused the error
            robot_state: Robot state when error occurred
        """
        timestamp = datetime.now().isoformat()
        
        row = [
            timestamp,
            error_type,
            error_message,
            action_context or "unknown",
            robot_state or "unknown"
        ]
        
        with open(self.error_log_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)
        
        print(f"Error logged: {error_type} - {error_message}")
    
    def start_action_timer(self):
        """Start timing an action execution."""
        self.action_start_time = time.time()
    
    def end_action_timer(self):
        """End timing an action execution and return duration."""
        if self.action_start_time is None:
            return 0.0
        
        duration = time.time() - self.action_start_time
        self.action_start_time = None
        return duration
    
    def log_experiment_info(self, info_dict):
        """
        Log experiment information.
        
        Args:
            info_dict: Dictionary with experiment information
        """
        info_file = self.log_dir / "experiment_info.json"
        
        info_dict.update({
            "participant_id": self.participant_id,
            "condition": self.condition,
            "timestamp": self.timestamp,
            "log_directory": str(self.log_dir)
        })
        
        with open(info_file, 'w') as f:
            json.dump(info_dict, f, indent=2)
        
        print(f"Experiment info logged: {info_file}")
    
    def get_log_summary(self):
        """Get a summary of logged data."""
        summary = {
            "participant_id": self.participant_id,
            "condition": self.condition,
            "timestamp": self.timestamp,
            "log_directory": str(self.log_dir),
            "files": {}
        }
        
        # Count rows in each CSV file
        for file_path in [self.action_log_file, self.state_log_file, 
                         self.feedback_log_file, self.error_log_file]:
            if file_path.exists():
                with open(file_path, 'r') as f:
                    row_count = sum(1 for line in f) - 1  # Subtract header
                summary["files"][file_path.name] = row_count
        
        return summary
    
    def close(self):
        """Close the data logger and write summary."""
        summary = self.get_log_summary()
        summary_file = self.log_dir / "summary.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Data Logger closed. Summary: {summary}")


# Example usage
if __name__ == "__main__":
    # Create a test logger
    logger = DataLogger(participant_id=1, condition="test_condition")
    
    # Log some test data
    logger.log_action("stand", {"height": 0.0}, success=True, execution_time=2.5)
    logger.log_state_transition("stand", "sit", "action_completed")
    logger.log_feedback("stand", "ok", "Action completed successfully", 2.5)
    
    # Log experiment info
    logger.log_experiment_info({
        "description": "Test experiment",
        "robot_type": "Spot",
        "software_version": "V2"
    })
    
    # Close logger
    logger.close()
