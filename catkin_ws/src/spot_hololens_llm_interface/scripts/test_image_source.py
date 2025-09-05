#!/usr/bin/env python3
import os
import sys
import argparse
import logging

try:
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.image import ImageClient
except Exception as e:
    create_standard_sdk = None
    ImageClient = None

def list_image_sources_for_robot(robot):
    # Use the robot object (from bosdyn SDK) to list image sources.
    image_client = robot.ensure_client(ImageClient.default_service_name)
    sources = image_client.list_image_sources()
    print("Available image sources:")
    for src in sources:
        # src is an image_pb2.ImageSource
        print(f" name: {src.name}")
        print(f"  type: {src.image_type}")
        print(f"  formats: {list(src.image_formats)}")
        print(f"  pixels: {list(src.pixel_formats)}")
        print("")

def get_robot_from_credentials(host, username, password):
    sdk = create_standard_sdk('test_image_source')
    robot = sdk.create_robot(host)
    robot.authenticate(username, password)
    return robot

def main():
    parser = argparse.ArgumentParser(
        description="List Boston Dynamics Spot image sources. Credentials via args or env vars."
    )
    parser.add_argument("--host", help="Spot hostname/IP (or SPOT_HOST env)", default=os.environ.get("BOSDYN_IP"))
    parser.add_argument("--username", help="Spot username (or SPOT_USERNAME env)", default=os.environ.get("BOSDYN_CLIENT_USERNAME"))
    parser.add_argument("--password", help="Spot password (or SPOT_PASSWORD env)", default=os.environ.get("BOSDYN_CLIENT_PASSWORD"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if create_standard_sdk is None or ImageClient is None:
        print("bosdyn SDK not available in this environment. Install the Boston Dynamics SDK to use this script.")
        sys.exit(1)

    if args.host and args.username and args.password:
        try:
            robot = get_robot_from_credentials(args.host, args.username, args.password)
        except Exception as e:
            print(f"Failed to authenticate to Spot: {e}")
            sys.exit(1)
        try:
            list_image_sources_for_robot(robot)
        except Exception as e:
            print(f"Failed to list image sources: {e}")
            sys.exit(1)
    else:
        print("Missing credentials. Provide --host, --username, --password or set SPOT_HOST, SPOT_USERNAME, SPOT_PASSWORD environment variables.")
        sys.exit(2)

if __name__ == "__main__":
    main()