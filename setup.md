```bash
mkdir -p ros2ws/src && cd ros2ws/src

git clone https://github.com/rkulkarni1999/peer_robotics_pallet_vision.git

cd peer_robotics_pallet_vision

chmod +x setup_env.sh

./setup_env.sh

conda activate yolo_env

export PYTHONPATH=$(python3 -c "import site; print(site.getsitepackages()[0])"):$PYTHONPATH
export PATH=$(python3 -c "import sys; print(':'.join(sys.path))"):$PATH
```

- Navigate back to your <ros2ws>  

```bash
source install/setup.bash

cd src/peer_robotics_pallet_vision/
```

```bash

```