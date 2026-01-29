#!/usr/bin/env python3
"""
Manual smoke exercise for PoseTracker + GUI.

It spawns the standalone controller (which opens the real GUI), then
publishes a short pose trajectory that should walk the robot from the
centre to (6, 4) while rotating west. Watch the GUI grid to confirm.
"""

import json
import subprocess
import time
from pathlib import Path
from threading import Thread

import sys
# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


from message_bus import Publisher, String, rospy
from pose_tracker import GridFrameMapper, PoseTracker

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = REPO_ROOT / "main.py"


def _publish_simulator_pose(row, col, facing):
    payload = json.dumps({"row": row, "col": col, "facing": facing})
    Publisher("/robot_simulator/position").publish(String(data=payload))


def drive_sequence():
    mapper = GridFrameMapper()
    tracker = PoseTracker(mapper)

    # Mirror the path the GUI should show.
    steps = [
        (4.0, 4.0, "N"),
        (5.0, 4.0, "N"),
        (6.0, 4.0, "N"),
        (6.0, 4.0, "W"),
    ]

    rospy.sleep(2.0)  # allow GUI to finish booting
    for row, col, facing in steps:
        _publish_simulator_pose(row, col, facing)
        tracker.current_cell = [row, col]
        tracker.current_facing = facing
        print(f"Tracker view -> cell=({row:.1f},{col:.1f}), facing={facing}")
        time.sleep(1.0)

    print("Done. Close the GUI window to exit the demo.")


def main():
    controller_proc = subprocess.Popen(
        ["python", str(CONTROLLER)],
        cwd=str(REPO_ROOT),
        env=None,
    )

    try:
        Thread(target=drive_sequence, daemon=True).start()
        controller_proc.wait()
    finally:
        if controller_proc.poll() is None:
            controller_proc.terminate()
            controller_proc.wait(timeout=5)


if __name__ == "__main__":
    main()