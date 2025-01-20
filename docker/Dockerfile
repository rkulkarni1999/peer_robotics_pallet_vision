# First Pull base docker image. 
# RUN docker pull nvidia/cuda:12.2.0-base-ubuntu22.04 || true

# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set up environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH="/root/miniconda3/bin:$PATH"

# Update and install required tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    lsb-release \
    software-properties-common \
    curl \
    wget \
    git \
    bash && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8

# Install Miniconda
WORKDIR /root
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm miniconda.sh && \
    /root/miniconda3/bin/conda init bash

# Set SHELL to enable Conda initialization
SHELL ["/bin/bash", "--login", "-c"]

# Install ROS2 Humble
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    ros-dev-tools \
    python3-rosdep && \
    rosdep init && \
    rosdep fix-permissions && \
    rosdep update

# Set up ROS2 environment variables
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Create Conda YOLO environment and install dependencies
RUN conda create --name yolo python=3.10.14 -y
    # conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics -y 

# RUN bash -c "source /root/miniconda3/bin/activate yolo && python --version"

# Clone your YOLO repository
WORKDIR /root/peer_ws
RUN git clone https://github.com/rkulkarni1999/peer_robotics_pallet_vision.git src/peer_robotics_pallet_vision

# Install dependencies using rosdep
RUN source /opt/ros/humble/setup.bash && \
    rosdep install --from-paths src --ignore-src -r -y

# Build the package
RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install

# Source the workspace in .bashrc
# RUN echo "source /root/peer_ws/install/setup.bash" >> ~/.bashrc

# Default shell
# CMD ["/bin/bash"]

RUN source install/setup.bash

COPY setup_env.sh /root/setup_environment.sh

# Make the script executable
RUN chmod +x /root/setup_environment.sh

CMD ["/bin/bash", "-c", "/root/setup_environment.sh && /bin/bash"]