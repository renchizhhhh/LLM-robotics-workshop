import numpy as np
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
import bosdyn.geometry
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand, blocking_sit
from bosdyn.choreography.client.choreography import ChoreographyClient
from dotenv import load_dotenv
import os
import io
from PIL import Image
import cv2
import time
from datetime import datetime


# TODO: Maybe figure out how exactly it is going to stand before we run it (so we don't break something)
# TODO: The methods are synchronous now, maybe make them async somehow
def stand_low(command_client):
    cmd = RobotCommandBuilder.synchro_stand_command(body_height=0.0)
    command_client.robot_command(cmd)

def stand_high(command_client):
    cmd = RobotCommandBuilder.synchro_stand_command(body_height=0.5)
    command_client.robot_command(cmd)

def move_command(command_client, fw, st, ro):
    cmd = RobotCommandBuilder.synchro_velocity_command(fw, st, ro)
    command_client.robot_command(cmd, end_time_secs=time.time() + 1)


def sit_down(command_client):
    cmd = RobotCommandBuilder.sit_command()
    command_client.robot_command(cmd)

def stop(command_client):
    cmd = RobotCommandBuilder.stop_command()
    command_client.robot_command(cmd)

def recognize_gestures(robot):
    # Initialize an ImageClient
    image_client = robot.ensure_client(ImageClient.default_service_name)
    sources = image_client.list_image_sources()

    # Create a command client to be able to command the robot
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    camera_images = [
        'hand_color_image', 
        'frontleft_depth', 
        'frontleft_depth_in_visual_frame', 
        'frontleft_fisheye_image', 
        'frontright_depth', 
        'frontright_depth_in_visual_frame', 
        'frontright_fisheye_image'
    ]

    while True:
        # Retrieve image from the arm camera
        # TODO: Replace to match arm camera
        image_response = image_client.get_image_from_sources([camera_images[0]])[0]
        image = np.array(Image.open(io.BytesIO(image_response.shot.image.data)))[:, :, ::-1]
        cv2.imshow('Image', cv2.flip(image, 1))

        forward, strafe, rotate = 0, 0, 0

        if cv2.waitKey(1) & 0xFF == ord('p'):
            break
        if cv2.waitKey(0) == ord('a'):
            strafe += 0.4
        if cv2.waitKey(0) == ord('d'):
            strafe -= 0.4
        if cv2.waitKey(0) == ord('i'):
            stand_high(command_client)
        if cv2.waitKey(0) == ord('k'):
            stand_low(command_client)
        if cv2.waitKey(0) == ord('q'):
            stop(command_client)
        if cv2.waitKey(0) == ord('j'):
            rotate += 0.2
        if cv2.waitKey(0) == ord('l'):
            rotate -= 0.2
        if cv2.waitKey(0) == ord('w'):
            forward += 0.4
        if cv2.waitKey(0) == ord('s'):
            forward -= 0.4
        if cv2.waitKey(0) == ord("b"):
            sit_down(command_client)
        if cv2.waitKey(0) == ord("t"):
            time.sleep(10)
            image_response = image_client.get_image_from_sources([camera_images[0]])[0]
            image = np.array(Image.open(io.BytesIO(image_response.shot.image.data)))[:, :, ::-1]
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.imwrite(f"./image_captures/spot_image_{now}.jpg", image)

        move_command(command_client, forward, strafe, rotate)

    blocking_sit(command_client, duration_sec=5.0, use_initial_posture=True)
    cv2.destroyAllWindows()


def main():
    # Load the environment variables from .env
    load_dotenv()

    # Set up the boston dynamics SPOT (based on hello_spot.py)
    # bosdyn.client.util.setup_logging()
    sdk = bosdyn.client.create_standard_sdk('HelloSpotClient')
    robot = sdk.create_robot("192.168.80.3")
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                    "such as the estop SDK example, to configure E-Stop."

    # Power on and start controlling the robot
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."

        recognize_gestures(robot)

        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), "Robot power off failed."


if __name__ == '__main__':
    main()
