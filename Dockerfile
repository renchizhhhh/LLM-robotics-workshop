# syntax=docker/dockerfile:1
FROM osrf/ros:noetic-desktop
EXPOSE 10000
RUN rm /bin/sh && ln -s /bin/bash /bin/sh && \
    mkdir -p /Spot
COPY . /Spot/

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    python-is-python3 \
    wget \
    ca-certificates \
    lsb-release \
    gnupg2 \
    dos2unix \
    curl \
    git

RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh && \
    export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes && \
    export PATH="/root/miniconda3/bin:$PATH" && \
    /root/miniconda3/bin/conda create -n spotenv python=3.11 --yes && \
    /root/miniconda3/envs/spotenv/bin/python -m pip install --upgrade pip setuptools wheel && \
    /root/miniconda3/envs/spotenv/bin/pip install -r /Spot/catkin_ws/requirements.txt && \
    /root/miniconda3/bin/conda clean --all --yes

RUN source ./ros_entrypoint.sh || true
RUN chmod +x /Spot/catkin_ws/src/ros_tcp_endpoint/src/ros_tcp_endpoint/*.py || true
RUN dos2unix /Spot/catkin_ws/src/ros_tcp_endpoint/src/ros_tcp_endpoint/default_server_endpoint.py || true

# add to the bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    echo "alias activate_env='source /root/miniconda3/bin/activate && conda activate spotenv'" >> /root/.bashrc && \
    echo "alias deactivate_env='conda deactivate && conda deactivate'" >> /root/.bashrc

# Then add the pth file to the conda env: 
RUN echo "/usr/lib/python3/dist-packages" > /root/miniconda3/envs/spotenv/lib/python3.11/site-packages/ros.pth

# First make 
# WORKDIR /Spot/catkin_ws
# RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make"
# RUN /bin/bash -c "source devel/setup.bash || true"
# Then activate the conda env
