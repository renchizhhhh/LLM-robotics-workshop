#!/bin/bash

PARTICIPANT = "$1"
CONDITION = "$2"

echo PARTICIPANT
echo CONDITION

export BOSDYN_CLIENT_USERNAME=user
export BOSDYN_CLIENT_PASSWORD=corspotuser1

cd ./catkin_ws
catkin_make
source devel/setup.bash
dos2unix /catkin_ws/src/ros_tcp_endpoint/src/ros_tcp_endpoint/default_server_endpoint.py
python3.8 -m pip install -r src/spot_hololens_llm_interface/requirements.txt
roslaunch spot_hololens_llm_interface spot_hololens_llm_interface.launch &
rosbag record data_collection eye_gaze_in_pixel gaze_hit_object chatter
