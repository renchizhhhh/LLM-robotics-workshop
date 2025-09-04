import time
import math
from spot_hololens_llm_interface.spot_entrance import SpotRobotManager
from spot_hololens_llm_interface.spot_shared_services import SpotSharedServices

# Minimal fake message classes to mimic geometry_msgs/Pose
class SimplePosition:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x; self.y = y; self.z = z

class SimpleOrientation:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = x; self.y = y; self.z = z; self.w = w

class SimplePose:
    def __init__(self, x=0.0, y=0.0, yaw=0.0):
        self.position = SimplePosition(x, y, 0.0)
        # convert yaw to quaternion (z,w)
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        self.orientation = SimpleOrientation(0.0, 0.0, sy, cy)

# Fake command client to capture calls
class FakeCommandClient:
    def __init__(self):
        self.calls = []

    def robot_command(self, cmd, end_time_secs=None):
        # record a tuple for assertions; cmd object type irrelevant for test
        self.calls.append((cmd, end_time_secs))
        # return a fake cmd id
        return "fake-cmd-id"

# Minimal FakeRobotManager
class FakeRobotManager:
    def __init__(self, command_client):
        self._clients = {'command': command_client}

    def get_clients(self):
        return self._clients

# Build request-like object
class Req:
    def __init__(self, pose, frame_name):
        self.target_pose = pose
        self.frame_name = frame_name

def test_move_to_position_body_frame():
    # Create fake manager and services instance
    fake_cmd = FakeCommandClient()
    manager = FakeRobotManager(fake_cmd)
    services = SpotSharedServices(manager)

    # Target: 1.0 m forward in body frame, no yaw change
    pose = SimplePose(x=1.0, y=0.0, yaw=0.0)
    req = Req(pose, "body")

    resp = services.handle_move_to_position(req)

    assert resp.success, f"Expected success: {resp.message}"
    # Expect at least one forward command recorded
    assert len(fake_cmd.calls) >= 1, "No command calls recorded"
    # The last command should have an end_time_secs indicating a forward command duration
    last_call = fake_cmd.calls[-1]
    assert last_call[1] is not None, "Expected end_time_secs in robot_command call"

if __name__ == "__main__":
    # Run test directly
    test_move_to_position_body_frame()
    print("test_move_to_position_body_frame passed")
