catkin_make
cd devel
call setup.bat 
cd ..
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

roslaunch spot_hololens_llm_interface spot_hololens_llm_interface.launch
rosbag record data_collection eye_gaze_in_pixel gaze_hit_object chatter
