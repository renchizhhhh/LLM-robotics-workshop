# spot_hololens_llm_interface (Docker + ROS Noetic)

This repo contains a ROS Noetic catkin workspace (`/Spot/catkin_ws`) with the `spot_hololens_llm_interface` package and a main “dummy mode” launch file:

- `/Spot/catkin_ws/src/spot_hololens_llm_interface/launch/spot_dummy_test.launch`

## Docker (recommended)

### Build the image

From the repo root:

```bash
docker build -t llm_spot_docker_image .
```

### Run the container

Expose the Unity/HoloLens ROS-TCP endpoint (default port `10000`):

```bash
docker run --rm -it -p 10000:10000 llm_spot_docker_image
```

For local code edits on the host (bind-mount the repo into `/Spot`):

```bash
docker run --rm -it -p 10000:10000 -v "$PWD":/Spot -w /Spot llm_spot_docker_image
```

### Inside the container: activate `spotenv`, build, and launch

The Docker image creates a conda env named `spotenv` and adds an `activate_env` alias to `/root/.bashrc`.

```bash
activate_env
cd /Spot/catkin_ws
catkin_make
source devel/setup.bash

# Dummy-mode launch (no real robot required)
roslaunch spot_hololens_llm_interface spot_dummy_test.launch dummy_mode:=true tcp_port:=10000 record_bag:=false
```

## Python environment (native `spotenv`)

If you’re not using Docker, you’ll need ROS Noetic installed on the host and a Python env named `spotenv` matching what the Docker image uses.

1) Create and activate the env:

```bash
conda create -n spotenv python=3.11 -y
conda activate spotenv
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

2) Make ROS Python packages visible to `spotenv` (matches the Dockerfile behavior):

```bash
python - <<'PY'
import site, pathlib
pth = pathlib.Path(site.getsitepackages()[0]) / "ros.pth"
pth.write_text("/usr/lib/python3/dist-packages\n")
print("Wrote", pth)
PY
```

3) Build and source the catkin workspace:

```bash
source /opt/ros/noetic/setup.bash
cd catkin_ws
catkin_make
source devel/setup.bash
```

## Running the main launch file (`spot_dummy_test.launch`)

Run (after sourcing ROS + the catkin workspace as shown above):

```bash
roslaunch spot_hololens_llm_interface spot_dummy_test.launch dummy_mode:=true
```

Useful args (from `spot_dummy_test.launch`):

- `dummy_mode` (`true` for dummy services; `false` will try to use a real robot at `hostname`)
- `hostname` (default: `192.168.1.109`)
- `verbose` (default: `false`)
- `tcp_ip` (default: `0.0.0.0`)
- `tcp_port` (default: `10000`)
- `record_bag` (default: `true`)
- `bag_file` (default: `$HOME/.ros/spot_dummy_topics`)

## Notes / troubleshooting

- If `roslaunch` can’t find `spot_hololens_llm_interface`, you likely forgot `source devel/setup.bash`.
- If `spotenv` can’t `import rospy`, add the correct ROS dist-packages path(s) to `ros.pth` (your system may use `/opt/ros/noetic/lib/python3/dist-packages`).
