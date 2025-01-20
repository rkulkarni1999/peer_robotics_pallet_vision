# Building and Running the docker container

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
```