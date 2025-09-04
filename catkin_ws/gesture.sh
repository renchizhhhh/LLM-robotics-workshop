echo "docker CP works! #2"
cd /catkin_ws
catkin_make
source devel/setup.bash

dos2unix /catkin_ws/src/ros_tcp_endpoint/src/ros_tcp_endpoint/default_server_endpoint.py

python3.8 -m pip install -r src/spot_hololens_llm_interface/requirements.txt

roslaunch gesture_classification gesture.launch