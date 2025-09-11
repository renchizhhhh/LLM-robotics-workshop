#!/usr/bin/env python3
"""
Robot Performance Latency Analyzer

Analyzes timing events from robot operations to measure pipeline latencies.
Uses precise ROS timestamps from rosbag recordings for accurate measurements.
"""

import os
import glob
import subprocess
import rosbag
import rospy

import sys
sys.path.append(os.path.dirname(__file__))
from timing_utils import recorder


class LatencyAnalyzer:
    """Analyzes robot performance timing events with precise ROS timestamps."""
    
    # Pipeline phases for latency analysis
    PHASES = [
        ('dictation_start', 'dictation_stop', 'Dictation Processing'),
        ('dictation_stop', 'send_hololens_input', 'Dictation -> Send'),
        ('send_hololens_input', 'received_hololens_input', 'Send -> Received'),
        ('received_hololens_input', 'start_llm_processing', 'Received -> LLM'),
        ('start_llm_processing', 'stop_llm_processing', 'LLM Processing'),
        ('stop_llm_processing', 'start_user_confirmation', 'LLM -> Confirmation'),
        ('start_user_confirmation', 'stop_user_confirmation', 'User Confirmation'),
    ]
    
    def __init__(self):
        self.rosbag_dir = os.path.join(
            os.path.dirname(__file__), '..', '..', '..', '..', 'rosbag'
        )
        self.events = []
    
    def start_recording(self):
        """Begin data collection by clearing old files and starting recording."""
        self._clear_old_bags()
        recorder.start_recording()
        print("Recording started - run your robot commands now")
    
    def stop_recording(self):
        """End data collection and load events for analysis."""
        recorder.stop_recording()
        self._load_events()
        
        if self.events:
            print(f"Loaded {len(self.events)} events - use 'timeline' to view results")
        else:
            print("No events found - make sure to run robot commands first")
    
    def _clear_old_bags(self):
        """Remove old rosbag files to start fresh."""
        old_bags = glob.glob(os.path.join(self.rosbag_dir, "timing_events_*.bag"))
        for bag_file in old_bags:
            os.remove(bag_file)
    
    def _load_events(self):
        """Load timing events from the most recent rosbag with precise timestamps."""
        bags = glob.glob(os.path.join(self.rosbag_dir, "timing_events_*.bag"))
        if not bags:
            self.events = []
            return
        
        latest_bag = max(bags, key=os.path.getmtime)
        self.events = []
        
        try:
            with rosbag.Bag(latest_bag, 'r') as bag:
                for _, msg, timestamp in bag.read_messages('/timing_events'):
                    self.events.append({
                        'time': timestamp.to_sec(),
                        'type': msg.data.strip()
                    })
        except Exception as e:
            if "Unindexed" in str(e):
                subprocess.run(['rosbag', 'reindex', latest_bag], capture_output=True)
                with rosbag.Bag(latest_bag, 'r') as bag:
                    for _, msg, timestamp in bag.read_messages('/timing_events'):
                        self.events.append({
                            'time': timestamp.to_sec(),
                            'type': msg.data.strip()
                        })
            else:
                print(f"Error loading rosbag: {e}")
                self.events = []
    
    def _split_tasks(self):
        """Split events into complete task sequences."""
        if not self.events:
            return []
        
        sorted_events = sorted(self.events, key=lambda x: x['time'])
        tasks = []
        current_task = []
        
        for event in sorted_events:
            if event['type'] == 'dictation_start' and current_task:
                tasks.append(current_task)
                current_task = [event]
            elif (event['type'] == 'received_hololens_input' and current_task and 
                  not any(e['type'] == 'dictation_start' for e in current_task)):
                tasks.append(current_task)
                current_task = [event]
            else:
                current_task.append(event)
        
        if current_task:
            tasks.append(current_task)
        
        return self._filter_complete_tasks(tasks)
    
    def _filter_complete_tasks(self, tasks):
        """Filter out startup sequences and incomplete tasks."""
        complete_tasks = []
        startup_events = {
            'connect', 'power_on', 'disconnect', 
            'start_connect', 'stop_connect', 
            'start_power_on', 'stop_power_on', 
            'start_disconnect', 'stop_disconnect'
        }
        
        for task in tasks:
            event_types = [e['type'] for e in task]
            
            if all(t in startup_events for t in event_types):
                continue
            
            if (any(t not in ['dictation_start', 'dictation_stop'] for t in event_types) and 
                len(task) > 2):
                complete_tasks.append(task)
        
        return complete_tasks
    
    def _get_event_times(self, events):
        """Create event type -> timestamp mapping."""
        return {e['type']: e['time'] for e in events}
    
    def _extract_robot_actions(self, events):
        """Extract robot action durations from start/stop event pairs."""
        times = self._get_event_times(events)
        actions = []
        processed = set()
        
        for event in events:
            if event['type'].startswith('start_'):
                action_name = event['type'][6:]
                stop_event = f'stop_{action_name}'
                
                if (action_name not in processed and 
                    action_name not in ['llm_processing', 'user_confirmation'] and 
                    stop_event in times):
                    
                    actions.append({
                        'name': action_name,
                        'duration': times[stop_event] - event['time']
                    })
                    processed.add(action_name)
        
        return actions
    
    def _show_task(self, events):
        """Display timeline for one task."""
        times = self._get_event_times(events)
        actions = self._extract_robot_actions(events)
        
        for start, end, desc in self.PHASES:
            if start in times and end in times:
                duration = times[end] - times[start]
                print(f"({duration:.3f}s): {desc}")
        
        for i, action in enumerate(actions):
            if i == 0 and 'stop_user_confirmation' in times:
                start_time = times.get(f'start_{action["name"]}')
                if start_time:
                    transition = start_time - times['stop_user_confirmation']
                    if transition > 0.001:
                        print(f"({transition:.3f}s): Confirmation -> {action['name']}")
            
            print(f"({action['duration']:.3f}s): {action['name']}")
            
            if i < len(actions) - 1:
                next_action = actions[i + 1]
                current_stop = times.get(f'stop_{action["name"]}')
                next_start = times.get(f'start_{next_action["name"]}')
                
                if current_stop and next_start:
                    transition = next_start - current_stop
                    if transition > 0.001:
                        print(f"({transition:.3f}s): {action['name']} -> {next_action['name']}")
    
    def show_timeline(self):
        """Display complete timeline analysis."""
        if not self.events:
            print("No events loaded. Use 'stop' to load events first.")
            return
        
        print("\n=== TIMELINE ===")
        tasks = self._split_tasks()
        
        for i, task in enumerate(tasks, 1):
            if i > 1:
                print(f"\n==== End of Task {i-1} ====\n")
            self._show_task(task)
        
        if tasks:
            print(f"\n==== End of Task {len(tasks)} ====")

def main():
    # Initialize ROS node
    rospy.init_node('latency_analyzer', anonymous=True)
    
    analyzer = LatencyAnalyzer()
    print("Latency Analyzer\nCommands: start, stop, timeline")
    
    while True:
        try:
            cmd = input("\n> ").strip().lower()
            if cmd == 'start':
                analyzer.start_recording()
            elif cmd == 'stop':
                analyzer.stop_recording()
            elif cmd == 'timeline':
                analyzer.show_timeline()
            else:
                print("Commands: start, stop, timeline")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

if __name__ == '__main__':
    main()