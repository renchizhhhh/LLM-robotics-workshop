#!/usr/bin/env python3
"""
Message Bus - Simple pub/sub system to replace ROS topics
Supports both real ROS and mock implementation
"""

import threading
from typing import Dict, List, Callable, Any
import time

# Try to import ROS modules
try:
    import rospy
    import rosgraph
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


class MessageBus:
    """Simple message bus to replace ROS pub/sub system."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.lock = threading.Lock()
    
    def publish(self, topic: str, data: Any):
        """Publish data to a topic."""
        with self.lock:
            callbacks = self.subscribers.get(topic, [])
        
        # Call callbacks outside of lock to avoid deadlocks
        for callback in callbacks:
            try:
                callback(data)
            except Exception as e:
                print(f"Error in message bus callback for topic {topic}: {e}")
    
    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic with a callback function."""
        with self.lock:
            self.subscribers.setdefault(topic, []).append(callback)
    
    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe from a topic."""
        with self.lock:
            if topic in self.subscribers:
                try:
                    self.subscribers[topic].remove(callback)
                except ValueError:
                    pass  # Callback not found
    
    def get_subscriber_count(self, topic: str) -> int:
        """Get number of subscribers for a topic."""
        with self.lock:
            return len(self.subscribers.get(topic, []))


# Global message bus instance
message_bus = MessageBus()


def is_ros_master_available():
    """Check if ROS master is available."""
    if not ROS_AVAILABLE:
        return False
    try:
        return rosgraph.is_master_online()
    except Exception:
        return False


class Publisher:
    """Publisher class that uses real ROS when available, mock otherwise."""
    
    def __init__(self, topic: str, queue_size: int = 10):
        self.topic = topic
        self.queue_size = queue_size
        self.ros_publisher = None
        
        # If ROS is available and master is online, use real ROS
        if ROS_AVAILABLE and is_ros_master_available():
            try:
                # Import ROS message types
                from std_msgs.msg import String
                import rospy as real_rospy
                self.ros_publisher = real_rospy.Publisher(topic, String, queue_size=queue_size)
            except Exception as e:
                print(f"Failed to create ROS publisher for {topic}: {e}")
                self.ros_publisher = None
    
    def publish(self, data: Any):
        """Publish data to the topic."""
        if self.ros_publisher is not None:
            # Use real ROS publisher
            from std_msgs.msg import String as RosString
            
            # Check if data is already a ROS message (has _md5sum attribute)
            if hasattr(data, '_md5sum'):
                # Already a ROS message, publish directly
                self.ros_publisher.publish(data)
            elif hasattr(data, 'data'):
                # Convert message_bus.String to std_msgs.msg.String
                msg = RosString()
                msg.data = str(data.data)
                self.ros_publisher.publish(msg)
            else:
                # Convert raw data to String message
                msg = RosString()
                msg.data = str(data)
                self.ros_publisher.publish(msg)
        else:
            # Use mock message bus
            message_bus.publish(self.topic, data)


class Subscriber:
    """Subscriber class that uses real ROS when available, mock otherwise."""
    
    def __init__(self, topic: str, callback: Callable, queue_size: int = 10):
        self.topic = topic
        self.callback = callback
        self.queue_size = queue_size
        self.ros_subscriber = None
        
        # If ROS is available and master is online, use real ROS
        if ROS_AVAILABLE and is_ros_master_available():
            try:
                import rospy as real_rospy
                from std_msgs.msg import String
                print(f"[MessageBus] Creating ROS subscriber for {topic}")
                self.ros_subscriber = real_rospy.Subscriber(topic, String, callback)
                print(f"[MessageBus] Successfully created ROS subscriber for {topic}: {self.ros_subscriber}")
            except Exception as e:
                print(f"Failed to create ROS subscriber for {topic}: {e}")
                import traceback
                traceback.print_exc()
                self.ros_subscriber = None
        
        # Fallback to mock message bus
        if self.ros_subscriber is None:
            print(f"[MessageBus] Using mock message bus subscriber for {topic}")
            message_bus.subscribe(topic, callback)
        else:
            print(f"[MessageBus] ROS subscriber active for {topic}")
    
    def unregister(self):
        """Unsubscribe from the topic."""
        if self.ros_subscriber is not None:
            self.ros_subscriber.unregister()
        else:
            message_bus.unsubscribe(self.topic, self.callback)


class ServiceProxy:
    """Service proxy that uses real ROS when available, mock otherwise."""
    
    def __init__(self, service_name: str, service_type=None):
        self.service_name = service_name
        self.service_type = service_type
        self.ros_service_proxy = None
        
        # If ROS is available and master is online, use real ROS
        if ROS_AVAILABLE and is_ros_master_available():
            try:
                import rospy as real_rospy
                real_rospy.wait_for_service(service_name, timeout=2.0)
                self.ros_service_proxy = real_rospy.ServiceProxy(service_name, service_type)
            except Exception as e:
                print(f"Failed to create ROS service proxy for {service_name}: {e}")
                self.ros_service_proxy = None
    
    def __call__(self, request=None):
        """Call the service."""
        if self.ros_service_proxy is not None:
            # Use real ROS service
            try:
                if request is None:
                    # Call service without arguments
                    return self.ros_service_proxy()
                else:
                    # Call service with request
                    return self.ros_service_proxy(request)
            except Exception as e:
                print(f"ROS service call failed for {self.service_name}: {e}")
                # Return mock response on failure
                return self._create_mock_response()
        else:
            # Use mock service
            return self._create_mock_response()
    
    def _create_mock_response(self):
        """Create a mock response for standalone mode."""
        class Response:
            def __init__(self):
                self.success = True
                self.message = "Service call successful (standalone mode)"
        
        return Response()


class Timer:
    """Timer class that uses real ROS when available, mock otherwise."""
    
    def __init__(self, duration: float, callback: Callable, oneshot: bool = False):
        self.duration = duration
        self.callback = callback
        self.oneshot = oneshot
        self.running = True
        self.ros_timer = None
        
        # If ROS is available and master is online, use real ROS
        if ROS_AVAILABLE and is_ros_master_available():
            try:
                self.ros_timer = rospy.Timer(rospy.Duration(duration), callback, oneshot=oneshot)
            except Exception as e:
                print(f"Failed to create ROS timer: {e}")
                self.ros_timer = None
        
        # Fallback to mock timer
        if self.ros_timer is None:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
    
    def _run(self):
        while self.running:
            time.sleep(self.duration)
            if self.running:
                try:
                    self.callback(None)  # Pass None as event (no ROS event object)
                except Exception as e:
                    print(f"Error in timer callback: {e}")
                
                if self.oneshot:
                    break
    
    def stop(self):
        """Stop the timer."""
        self.running = False
        if self.ros_timer is not None:
            self.ros_timer.shutdown()


# Mock ROS message classes for compatibility
class String:
    def __init__(self, data: str = ""):
        self.data = data


class Empty:
    def __init__(self):
        pass


class PoseStamped:
    def __init__(self):
        self.pose = Pose()
        self.header = Header()


class Pose:
    def __init__(self):
        self.position = Point()
        self.orientation = Quaternion()


class Point:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class Quaternion:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.w = 1.0


class Header:
    def __init__(self):
        self.stamp = None
        self.frame_id = ""


# ROS node functionality
class Node:
    """ROS node that uses real ROS when available, mock otherwise."""
    
    def __init__(self, name: str, anonymous: bool = False):
        self.name = name
        self.anonymous = anonymous
        self._is_shutdown = False
        self._ros_initialized = False
        
        # Don't try to initialize ROS node in __init__ to avoid recursion
        # This will be done lazily when needed
    
    def init_node(self, name: str, anonymous: bool = False):
        """Initialize the node."""
        self.name = name
        self.anonymous = anonymous
        self._is_shutdown = False
        
        if ROS_AVAILABLE and is_ros_master_available():
            try:
                # Use the real rospy module directly to avoid recursion
                import rospy as real_rospy
                real_rospy.init_node(name, anonymous=anonymous)
                self._ros_initialized = True
                return True
            except Exception as e:
                print(f"Failed to initialize ROS node: {e}")
                self._ros_initialized = False
                return False
        else:
            self._ros_initialized = False
            return True  # Mock mode always succeeds
    
    def is_shutdown(self) -> bool:
        if self._ros_initialized:
            import rospy as real_rospy
            return real_rospy.is_shutdown()
        return self._is_shutdown
    
    def loginfo(self, msg: str):
        if self._ros_initialized:
            import rospy as real_rospy
            real_rospy.loginfo(msg)
        else:
            print(f"[INFO] {msg}")
    
    def logwarn(self, msg: str):
        if self._ros_initialized:
            import rospy as real_rospy
            real_rospy.logwarn(msg)
        else:
            print(f"[WARN] {msg}")
    
    def logerr(self, msg: str):
        if self._ros_initialized:
            import rospy as real_rospy
            real_rospy.logerr(msg)
        else:
            print(f"[ERROR] {msg}")
    
    def sleep(self, duration: float):
        if self._ros_initialized:
            import rospy as real_rospy
            real_rospy.sleep(duration)
        else:
            time.sleep(duration)
    
    def get_param(self, param_name: str, default=None):
        """Get parameter value."""
        if self._ros_initialized:
            import rospy as real_rospy
            return real_rospy.get_param(param_name, default)
        else:
            return default
    
    def set_param(self, param_name: str, value):
        """Set parameter value."""
        if self._ros_initialized:
            import rospy as real_rospy
            real_rospy.set_param(param_name, value)
    
    def wait_for_service(self, service_name: str, timeout: float = 5.0):
        """Wait for service to be available."""
        if self._ros_initialized:
            try:
                import rospy as real_rospy
                real_rospy.wait_for_service(service_name, timeout=timeout)
                return True
            except Exception:
                return False
        else:
            return True  # Mock mode always succeeds
    
    def wait_for_message(self, topic: str, message_type, timeout: float = 5.0):
        """Wait for message."""
        if self._ros_initialized:
            try:
                import rospy as real_rospy
                return real_rospy.wait_for_message(topic, message_type, timeout=timeout)
            except Exception:
                return None
        else:
            # Return a default message
            if message_type == String:
                return String("")
            elif message_type == Empty:
                return Empty()
            return None
    
    def Subscriber(self, topic: str, message_type, callback):
        """Create a subscriber."""
        return Subscriber(topic, callback)
    
    def Publisher(self, topic: str, message_type, queue_size: int = 10):
        """Create a publisher."""
        return Publisher(topic)
    
    def ServiceProxy(self, service_name: str, service_type):
        """Create a service proxy."""
        return ServiceProxy(service_name, service_type)
    
    def Timer(self, duration, callback, oneshot=False):
        """Create a timer."""
        return Timer(duration, callback, oneshot)
    
    def Duration(self, duration):
        """Create a duration object."""
        return duration
    
    def signal_shutdown(self, reason: str = ""):
        """Signal shutdown."""
        if self._ros_initialized:
            import rospy as real_rospy
            real_rospy.signal_shutdown(reason)
        else:
            self.loginfo(f"Shutting down node: {reason}")
        self._is_shutdown = True
    
    # Backwards-compatible alias
    def shutdown(self, reason: str = ""):
        self.signal_shutdown(reason)


# Global mock ROS node - lazy initialization
_rospy_instance = None

def get_rospy():
    """Get the global rospy instance, creating it if necessary."""
    global _rospy_instance
    if _rospy_instance is None:
        _rospy_instance = Node("standalone_simulation")
    return _rospy_instance

# For backward compatibility, create a proxy object
class RospyProxy:
    def __getattr__(self, name):
        return getattr(get_rospy(), name)

rospy = RospyProxy()
