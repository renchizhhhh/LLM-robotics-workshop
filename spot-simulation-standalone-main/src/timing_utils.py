#!/usr/bin/env python3
"""
Timing utilities for robot performance analysis (standalone version).
"""

import time
from message_bus import message_bus, Publisher, String


class TimingRecorder:
    """Manages timing event recording for standalone simulation."""
    
    def __init__(self):
        self.publisher = Publisher('/timing_events')
        self.events = []  # Store events in memory for debugging
    
    def publish_event(self, event_type: str):
        """Publish timing event."""
        timestamp = time.time()
        event_data = f"{timestamp}:{event_type}"
        
        # Store event for debugging
        self.events.append({
            'timestamp': timestamp,
            'event_type': event_type,
            'formatted': event_data
        })
        
        # Publish to message bus
        self.publisher.publish(String(data=event_data))
        
        print(f"[TIMING] {event_type} at {timestamp:.3f}")
    
    def get_events(self):
        """Get all recorded events."""
        return self.events.copy()
    
    def clear_events(self):
        """Clear all recorded events."""
        self.events.clear()
    
    def get_event_summary(self):
        """Get a summary of timing events."""
        if not self.events:
            return "No timing events recorded"
        
        summary = []
        for event in self.events:
            summary.append(f"{event['event_type']}: {event['timestamp']:.3f}")
        
        return "\n".join(summary)


# Global instance for direct access
recorder = TimingRecorder()
