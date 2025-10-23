# syntax=docker/dockerfile:1
FROM osrf/ros:noetic-desktop
EXPOSE 10000
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

COPY . .

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     software-properties-common \
#     wget \
#     ca-certificates \
#     lsb-release \
#     gnupg2 \
#     dos2unix \
#     curl \
#     && add-apt-repository ppa:deadsnakes/ppa -y \
#     && apt-get update \
#     && apt-get install -y --no-install-recommends \
#        python3.9 \
#        python3.9-venv \
#        python3.9-distutils \
#        python3.9-dev \
#        python3-pip \
#     && rm -rf /var/lib/apt/lists/*

# NEED TO TEST CONDA INSTALLATION WHILE BUILDING IMAGE
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
    source /root/miniconda3/bin/activate && \
    /root/miniconda3/bin/conda init bash && \
    export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes && \
    export PATH="/root/miniconda3/bin:$PATH" && \
    conda create -n spotenv --file /catkin_ws/requirements.txt --yes && \
    conda clean --all --yes

RUN source ./ros_entrypoint.sh
RUN source /opt/ros/noetic/setup.bash
RUN chmod +x /catkin_ws/src/ros_tcp_endpoint/src/ros_tcp_endpoint/*.py
RUN dos2unix /catkin_ws/src/ros_tcp_endpoint/src/ros_tcp_endpoint/default_server_endpoint.py

# RUN python3.9 -m venv /root/spotenv \
#  && /root/spotenv/bin/python -m pip install --upgrade pip setuptools wheel \
#  && /root/spotenv/bin/pip install -r /catkin_ws/requirements.txt

# ENV VIRTUAL_ENV=/root/spotenv

WORKDIR /catkin_ws
RUN catkin_make
RUN source devel/setup.bash
