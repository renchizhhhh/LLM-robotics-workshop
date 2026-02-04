# LLM robotics workshop

This repo contains the software used in an LLM robotics workshop:

- A **Python/Tkinter GUI** for visualizing a grid world, running LLM planners, previewing trajectories, and (optionally) executing plans on a real Spot robot.
- A **host/client web interface** for collecting participants’ prompts and showing results/leaderboards on devices over a local Wi‑Fi network.
- A **ROS Noetic catkin workspace** that bridges the GUI to **Spot SDK 5.0.1** services for real robot execution.

## Repo layout

- `spot-simulation-standalone-main/`: GUI + web interface + evaluation scripts (workshop laptop).
  - GUI entrypoint: `spot-simulation-standalone-main/main.py`
  - Web server entrypoint: `spot-simulation-standalone-main/vevox/survey_server.py`
- `catkin_ws/`: ROS Noetic workspace for real robot control.
  - Main dummy launch: `catkin_ws/src/spot_hololens_llm_interface/launch/spot_dummy_test.launch`

## Install

### Option A: Docker (recommended)

Build:

```bash
docker build -t llm_spot_docker_image .
```

Run (ROS-TCP endpoint default port is `10000`):

```bash
docker run --rm -it -p 10000:10000 llm_spot_docker_image
```

Dev loop (edit on host, run in container):

```bash
docker run --rm -it -p 10000:10000 -v "$PWD":/Spot -w /Spot llm_spot_docker_image
```

Inside the container:

```bash
activate_env
cd /Spot/catkin_ws
catkin_make
source devel/setup.bash
```

### Option B: Native

You need ROS Noetic installed and a Python env named `spotenv`.

```bash
conda create -n spotenv python=3.11 -y
conda activate spotenv
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If `spotenv` can’t `import rospy`, add ROS’ dist-packages to the env (same approach as the Docker image):

```bash
python - <<'PY'
import site, pathlib
pth = pathlib.Path(site.getsitepackages()[0]) / "ros.pth"
pth.write_text("/usr/lib/python3/dist-packages\n")
print("Wrote", pth)
PY
```

Build the ROS workspace:

```bash
source /opt/ros/noetic/setup.bash
cd catkin_ws
catkin_make
source devel/setup.bash
```

## Run

### 1) Start the workshop web interface (host + mobile clients)

The survey server is a Flask-SocketIO app:

```bash
cd spot-simulation-standalone-main/vevox
export DASHBOARD_PASSWORD='change-me'
export PORT=5000
python survey_server.py
```

Notes:
- `PORT` defaults to `10000` in `survey_server.py`, which can conflict with the ROS-TCP endpoint; using `5000` avoids that.
- For mobile devices, set `EXTERNAL_IP=<laptop_ip>` so QR codes point to a reachable address.

### 2) Start the GUI (simulation mode)

```bash
python spot-simulation-standalone-main/main.py
```

### 3) Start ROS services (dummy mode or real robot)

Dummy mode (no robot required):

```bash
roslaunch spot_hololens_llm_interface spot_dummy_test.launch dummy_mode:=true tcp_port:=10000 record_bag:=false
```

If you want the GUI to attempt to connect to ROS services, start it with:

```bash
python spot-simulation-standalone-main/main.py --ros-mode
```

### `spot_dummy_test.launch` args

- `dummy_mode` (`true` for dummy services; `false` will try to use a real robot at `hostname`)
- `hostname` (default: `192.168.1.109`)
- `verbose` (default: `false`)
- `tcp_ip` (default: `0.0.0.0`)
- `tcp_port` (default: `10000`)
- `record_bag` (default: `true`)
- `bag_file` (default: `$HOME/.ros/spot_dummy_topics`)

## Configuration

- `spot-simulation-standalone-main/.env` is loaded automatically (and is ignored by git); put keys like `GOOGLE_API_KEY=...` there.
- Spot credentials/robot access depend on your local setup (Spot SDK + network + auth).

## Troubleshooting

- If `roslaunch` can’t find `spot_hololens_llm_interface`, you likely forgot `source catkin_ws/devel/setup.bash`.
- Tkinter GUI needs a display. If you try to run the GUI inside Docker, you’ll need X11/Wayland forwarding (running it natively is simplest).
