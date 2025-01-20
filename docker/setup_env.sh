#!/bin/bash

# Source ROS 2 setup
source /opt/ros/humble/setup.bash || { echo "Failed to source ROS 2 setup!"; exit 1; }

# Activate Conda environment
if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate yolo || { echo "Failed to activate Conda environment!"; exit 1; }
else
    echo "Conda initialization script not found!"
    exit 1
fi

# Check if ultralytics is installed, and install if missing
if ! python -c "from ultralytics import YOLO" 2>/dev/null; then
    echo "Installing PyTorch, CUDA, and Ultralytics..."
    conda install -n yolo -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics -y || { echo "Failed to install Ultralytics and dependencies!"; exit 1; }
    conda install -c conda-forge gdal libtiff || { echo "Failed to install GDAL and libtiff!"; exit 1; }
else
    echo "Ultralytics and dependencies already installed."
fi

# Source the ROS 2 workspace
if [ -f "/root/peer_ws/install/setup.bash" ]; then
    source /root/peer_ws/install/setup.bash || { echo "Failed to source ROS 2 workspace setup!"; exit 1; }
else
    echo "ROS 2 workspace setup script not found!"
    exit 1
fi

# Ensure `gdown` is installed
if ! command -v gdown &>/dev/null; then
    echo "Installing gdown..."
    pip install gdown || { echo "Failed to install gdown!"; exit 1; }
else
    echo "gdown is already installed."
fi

# Install unzip if not already installed
if ! command -v unzip &>/dev/null; then
    echo "Installing unzip..."
    sudo apt-get update && sudo apt-get install -y unzip || { echo "Failed to install unzip!"; exit 1; }
else
    echo "unzip is already installed."
fi

# Create the target directory if it doesn't exist
TARGET_DIR="/root/peer_ws/src/peer_robotics_pallet_vision/rosbags"
mkdir -p "${TARGET_DIR}" || { echo "Failed to create target directory!"; exit 1; }

# Download the file into the target directory
echo "Downloading file into ${TARGET_DIR}..."
gdown "https://drive.google.com/uc?id=1BvhP653G3PqfUq96L18gDBIi-5oOYqcr" --fuzzy -O "${TARGET_DIR}/internship_assignment_sample_bag.zip" || { echo "File download failed!"; exit 1; }

# Unzip the downloaded file
echo "Unzipping file in ${TARGET_DIR}..."
cd "${TARGET_DIR}" || { echo "Failed to change to target directory!"; exit 1; }
unzip internship_assignment_sample_bag.zip || { echo "Failed to unzip the file!"; exit 1; }
rm internship_assignment_sample_bag.zip  # Clean up the zip file after extraction
cd /root/peer_ws || { echo "Failed to return to the base workspace directory!"; exit 1; }

echo "Setup complete!"

