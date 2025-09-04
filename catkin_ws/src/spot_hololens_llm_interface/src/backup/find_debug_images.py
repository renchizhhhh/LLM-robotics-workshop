#!/usr/bin/env python3
"""
Helper script to find and display debug images from the Spot FSM control system.
"""

import os
import glob
import subprocess
import sys

def find_debug_images():
    """Find all debug images in the system."""
    # Common locations where debug images might be saved
    search_paths = [
        "/catkin_ws/debug_images",
        "./debug_images",
        "debug_images",
        os.path.expanduser("~/debug_images"),
        "/tmp/debug_images"
    ]
    
    found_images = []
    
    for path in search_paths:
        if os.path.exists(path):
            print(f"Checking directory: {path}")
            # Look for debug images with timestamps
            pattern = os.path.join(path, "debug_detection_*.jpg")
            images = glob.glob(pattern)
            if images:
                found_images.extend(images)
                print(f"  Found {len(images)} debug images")
            else:
                print(f"  No debug images found")
        else:
            print(f"Directory does not exist: {path}")
    
    return found_images

def display_image_info(images):
    """Display information about found debug images."""
    if not images:
        print("\nNo debug images found!")
        return
    
    print(f"\nFound {len(images)} debug images:")
    print("-" * 80)
    
    for i, image_path in enumerate(sorted(images, reverse=True)):  # Sort by newest first
        try:
            stat = os.stat(image_path)
            size_mb = stat.st_size / (1024 * 1024)
            print(f"{i+1:2d}. {image_path}")
            print(f"     Size: {size_mb:.2f} MB")
            print(f"     Modified: {stat.st_mtime}")
            print()
        except Exception as e:
            print(f"{i+1:2d}. {image_path} (Error getting info: {e})")
            print()

def open_image_viewer(image_path):
    """Try to open an image in the default viewer."""
    try:
        if sys.platform.startswith('linux'):
            # Try different image viewers on Linux
            viewers = ['xdg-open', 'eog', 'gthumb', 'gimp', 'display']
            for viewer in viewers:
                try:
                    subprocess.run([viewer, image_path], check=True)
                    print(f"Opened {image_path} with {viewer}")
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            print(f"Could not open image with any viewer. Try opening manually: {image_path}")
        elif sys.platform.startswith('darwin'):  # macOS
            subprocess.run(['open', image_path], check=True)
            print(f"Opened {image_path} with default viewer")
        elif sys.platform.startswith('win'):  # Windows
            os.startfile(image_path)
            print(f"Opened {image_path} with default viewer")
        else:
            print(f"Unknown platform, cannot open image automatically")
            print(f"Image path: {image_path}")
    except Exception as e:
        print(f"Error opening image: {e}")

def main():
    """Main function."""
    print("=== Spot FSM Control Debug Image Finder ===")
    print("This script helps you find debug images saved during object detection.")
    print()
    
    # Find debug images
    images = find_debug_images()
    
    # Display information
    display_image_info(images)
    
    if images:
        print("Options:")
        print("1. Enter image number to open in viewer")
        print("2. Enter 'all' to open all images")
        print("3. Enter 'path' to show absolute paths")
        print("4. Enter 'quit' to exit")
        print()
        
        while True:
            try:
                choice = input("Enter choice (1-4, or image number): ").strip().lower()
                
                if choice == 'quit':
                    break
                elif choice == 'all':
                    print("Opening all images...")
                    for image in images:
                        open_image_viewer(image)
                        input("Press Enter to continue to next image...")
                elif choice == 'path':
                    print("\nAbsolute paths to all debug images:")
                    for image in images:
                        print(f"  {os.path.abspath(image)}")
                elif choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(images):
                        open_image_viewer(images[idx])
                    else:
                        print(f"Invalid image number. Choose 1-{len(images)}")
                else:
                    print("Invalid choice. Please enter a number, 'all', 'path', or 'quit'")
                    
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    print("\nDebug image finder completed.")

if __name__ == "__main__":
    main()
