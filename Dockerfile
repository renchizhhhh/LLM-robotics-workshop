# syntax=docker/dockerfile:1
FROM osrf/ros:noetic-desktop
EXPOSE 10000
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

COPY . .

RUN apt-get update && apt-get install -y \
    python3-pip \ 
    dos2unix \
    python-is-python3

RUN source ./ros_entrypoint.sh
RUN source /opt/ros/noetic/setup.bash
RUN chmod +x /catkin_ws/src/ros_tcp_endpoint/src/ros_tcp_endpoint/*.py
RUN dos2unix /catkin_ws/src/ros_tcp_endpoint/src/ros_tcp_endpoint/default_server_endpoint.py

# RUN python3.8 -m pip install -r /catkin_ws/src/spot_fsm_control/requirements.txt

# WORKDIR /catkin_ws
# RUN catkin_make
# RUN source devel/setup.bash
