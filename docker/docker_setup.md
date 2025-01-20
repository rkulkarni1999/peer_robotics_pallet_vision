<!-- # Building and Running the docker container

```bash
cd docker
```

```bash
docker build -t peer2_ws:latest .
```

```bash
docker run --rm -it --gpus all peer2_ws
```

# Running the ros2 nodes. 

```bash
cd ~/peer_ws

conda activate yolo

source install/setup.bash

cd src/peer_robotics_pallet_vision/

# running the ros2 bag
ros2 bag play rosbags/internship_assignment_sample_bag --loop

# open a new interactive terminal
conda activate yolo && cd peer_ws && source install/setup.bash && cd src/peer_robotics_pallet_vision/

# running the detector node
python peer_robotics_pallet_vision/nodes/pallet_detector.py 
```



# Pruning the container after use

```bash
docker rm $(docker ps -aq)  

docker rmi peer2_ws

docker system prune -a     
``` -->

# Usage Guide: Building and Running the Docker Container

This guide explains how to build, run, and use the Docker container for the Peer Robotics Pallet Vision project, along with instructions for pruning the container after use.

---

## **Step 1: Build the Docker Container**

Navigate to the `docker` directory:
```bash
cd docker
```

Build the Docker image:

```bash
docker build -t peer2_ws:latest .
```

## Step 2: Run the Docker Container

Run the container with GPU support enabled (assuming that gpus are setup on the laptop):

```bash
docker run --rm -it --gpus all peer2_ws
```

## Step 3: Running the ROS 2 Nodes

1. Setting Up the Workspace

```bash
cd ~/peer_ws

# Activate the Conda environment:

conda activate yolo

source install/setup.bash

cd src/peer_robotics_pallet_vision/
```

2. Running the Rosbag
```bash
ros2 bag play rosbags/internship_assignment_sample_bag --loop
```

3. Open a New Terminal

Start a new interactive terminal, then set up the environment:

```bash
conda activate yolo && cd peer_ws && source install/setup.bash && cd src/peer_robotics_pallet_vision/
```

4. Running the Detector Node

```bash
python peer_robotics_pallet_vision/nodes/pallet_detector.py
```

Step 4: Pruning the Container After Use
```bash
docker rm $(docker ps -aq)
docker rmi peer_ws
docker system prune -a
```