#!/usr/bin/env python3
"""
Main entry point for Standalone Spot Simulation
"""

import sys
import os
import threading
import time
import signal
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Add catkin workspace to Python path for real robot support
catkin_ws_path = Path(__file__).parent.parent / "catkin_ws" / "devel" / "lib" / "python3" / "dist-packages"
if catkin_ws_path.exists():
    sys.path.insert(0, str(catkin_ws_path))
    print(f"✓ Added catkin workspace to Python path: {catkin_ws_path}")
    
    # Set up ROS environment variables if not already set
    if 'ROS_PACKAGE_PATH' not in os.environ:
        catkin_ws_root = Path(__file__).parent.parent / "catkin_ws"
        os.environ['ROS_PACKAGE_PATH'] = str(catkin_ws_root / "src")
        print(f"✓ Set ROS_PACKAGE_PATH: {os.environ['ROS_PACKAGE_PATH']}")
else:
    print(f"⚠️  Catkin workspace not found at: {catkin_ws_path}")
    print("   Real robot services will not be available")

# Import our standalone modules
from gui_core import SpotSimulationGUI
from nl_control import NaturalLanguageControl
from message_bus import rospy


class SpotSimulation:
    """Main simulation coordinator."""
    
    def __init__(self, world_id=None, use_speech=True, ros_mode=False):
        self.world_id = world_id
        self.use_speech = use_speech
        self.ros_mode = ros_mode
        self.gui = None
        self.nl_control = None
        self.running = False
        self.threads = []
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Windows-specific signal handling
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self._signal_handler) # type: ignore
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, shutting down...")
        self.shutdown()
        # Force exit after a short delay to allow cleanup
        import threading
        def force_exit():
            time.sleep(3)
            print("Force exiting...")
            os._exit(0)
        threading.Thread(target=force_exit, daemon=True).start()
    
    def start_gui(self):
        """Start the GUI in the main thread to avoid Tkinter threading issues."""
        try:
            print("Starting GUI...")
            self.gui = SpotSimulationGUI(world_id=self.world_id)
            
            # Don't start GUI in a separate thread - it will be run in main thread
            print("GUI initialized successfully")
            return True
        except Exception as e:
            print(f"Failed to start GUI: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def start_nl_control(self):
        """Start the NL control in a separate thread."""
        try:
            print("Starting Natural Language Control...")
            self.nl_control = NaturalLanguageControl(
                use_speech=self.use_speech, 
                world_id=self.world_id,
            )
            
            # Start NL control in a separate thread
            nl_thread = threading.Thread(target=self.nl_control.run, daemon=True)
            nl_thread.start()
            self.threads.append(nl_thread)
            
            print("Natural Language Control started successfully")
            return True
        except Exception as e:
            print(f"Failed to start NL Control: {e}")
            return False
    
    def run(self):
        """Run the complete simulation."""
        print("="*60)
        if self.ros_mode:
            print("SPOT ROBOT SIMULATION - ROS MODE")
        else:
            print("SPOT ROBOT SIMULATION - STANDALONE MODE")
        print("="*60)
        
        # Check for API keys
        self._check_api_keys()
        
        # Check ROS mode and initialize if needed
        self._check_ros_mode()
        
        # Start components
        self.running = True
        
        # Start NL Control first (initializes ROS node)
        if not self.start_nl_control():
            print("Error: NL Control failed to start, simulation cannot continue")
            return
        
        # Start GUI after NL control (so ROS node is already initialized)
        if not self.start_gui():
            print("Warning: GUI failed to start, continuing in headless mode")
        
        print("\nSimulation is running!")
        print("Commands:")
        print("  - Type natural language commands in the GUI")
        print("  - Use 'quit' to exit")
        print("  - Press Ctrl+C to stop")
        print()
        
        try:
            # Run GUI in main thread if available, otherwise keep main thread alive
            if self.gui:
                print("Running GUI in main thread...")
                # Run GUI mainloop in main thread
                self.gui.run()
            else:
                # Keep main thread alive with shorter sleep for better responsiveness
                while self.running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nReceived interrupt signal")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            self.shutdown()
    
    def _check_api_keys(self):
        """Check for required API keys."""
        gemini_key = os.getenv('GOOGLE_API_KEY')
        
        print("Checking API keys...")

        if not gemini_key:
            print("WARNING: GOOGLE_API_KEY not found!")
            print("  - Plan interpretation will use fallback mode")
            print("  - Set GOOGLE_API_KEY environment variable for better plan descriptions")
        else:
            print("✓ GOOGLE_API_KEY found")

        print()

    def _check_ros_mode(self):
        """Check and initialize ROS mode if requested."""
        if not self.ros_mode:
            return
        
        try:
            # Try to import ROS modules
            import rospy
            import rosgraph
            
            # Check if ROS master is available
            if not rosgraph.is_master_online():
                print("WARNING: ROS mode requested but ROS master not available")
                print("  - Falling back to standalone simulation mode")
                print("  - To use ROS mode, ensure ROS master is running")
                self.ros_mode = False
                return
            
            print("✓ ROS master detected - attempting to connect to robot services")
            
            # Check for robot services
            services_to_check = [
                '/spot_entrance/connect',
                '/spot_entrance/stand',
                '/spot_entrance/move_to_position'
            ]
            
            missing_services = []
            for service in services_to_check:
                try:
                    rospy.wait_for_service(service, timeout=2.0)
                except rospy.ROSException:
                    missing_services.append(service)
            
            if missing_services:
                print(f"WARNING: Missing robot services: {missing_services}")
                print("  - Falling back to standalone simulation mode")
                print("  - To use ROS mode, ensure catkin_ws is built and spot_dummy_test.launch is running")
                self.ros_mode = False
            else:
                print("✓ Robot services detected - ROS mode enabled")
                
        except ImportError:
            print("WARNING: ROS modules not available")
            print("  - Falling back to standalone simulation mode")
            print("  - To use ROS mode, install ROS and source the workspace")
            self.ros_mode = False
        except Exception as e:
            print(f"WARNING: ROS mode check failed: {e}")
            print("  - Falling back to standalone simulation mode")
            self.ros_mode = False
    
    def shutdown(self):
        """Shutdown the simulation."""
        if not self.running:
            return
        
        print("Shutting down simulation...")
        self.running = False
        
        # Shutdown components
        if self.nl_control:
            print("Stopping NL Control...")
            try:
                if hasattr(self.nl_control, 'stop'):
                    self.nl_control.stop()
            except Exception as e:
                print(f"Error stopping NL Control: {e}")
        
        if self.gui:
            print("Stopping GUI...")
            try:
                if hasattr(self.gui, 'stop'):
                    self.gui.stop()
                elif hasattr(self.gui, 'root') and self.gui.root:
                    self.gui.root.quit()
                    self.gui.root.destroy()
            except Exception as e:
                print(f"Error stopping GUI: {e}")
        
        # Wait for threads to finish (with timeout)
        print("Waiting for threads to finish...")
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
                if thread.is_alive():
                    print(f"Thread {thread.name} did not stop gracefully")
        
        print("Simulation stopped")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Standalone Spot Robot Simulation")
    parser.add_argument("--world", type=str, default=None, 
                       help="World ID to use (1-11, default: interactive selection)")
    parser.add_argument("--no-speech", action="store_true", 
                       help="Disable speech mode (use terminal input)")
    parser.add_argument("--headless", action="store_true", 
                       help="Run without GUI (headless mode)")
    parser.add_argument("--ros-mode", action="store_true", 
                       help="Force ROS mode (attempt to connect to real robot services)")
    
    args = parser.parse_args()
    
    # Override speech mode if headless
    use_speech = not args.no_speech and not args.headless
    
    try:
        simulation = SpotSimulation(
            world_id=args.world,
            use_speech=use_speech,
            ros_mode=args.ros_mode
        )
        
        if args.headless:
            print("Running in headless mode (GUI disabled)")
            simulation.start_nl_control()
            simulation.run()
        else:
            simulation.run()
            
    except Exception as e:
        print(f"Simulation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    main()
