# Command for setting up the environment to run package

```bash
mkdir -p <ros2_ws>/src && cd <ros2_ws>/src  

git clone https://github.com/rkulkarni1999/peer_robotics_pallet_vision.git

cd peer_robotics_pallet_vision

chmod +x setup_env.sh

./setup_env.sh

conda activate yolo_env

# Navigate back to your <ros2ws>  

source install/setup.bash

export PYTHONPATH=$(python3 -c "import site; print(site.getsitepackages()[0])"):$PYTHONPATH
export PATH=$(python3 -c "import sys; print(':'.join(sys.path))"):$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd src/peer_robotics_pallet_vision/
```

