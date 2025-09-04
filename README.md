Use the image
Please make sure docker is installed and running on your system. Then use the following command to build the image and run the container

```shell
docker build -t llm_spot_docker_image .
# Open the port connection to connect Unity ROS-TCP-Connector with the Docker image
docker run -it -p 10000:10000 llm_spot_docker_image
# Or with local file sync
docker run -it -p 10000:10000 -v ${PWD}/catkin_ws:/catkin_ws -w /catkin_ws llm_spot_docker_image
```
