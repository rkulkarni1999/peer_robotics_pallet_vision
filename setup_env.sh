#!/bin/bash

# Ensure the script is executed from the correct directory
if [[ $(basename "$PWD") != "peer_robotics_pallet_vision" ]]; then
    echo "You must run this script from '<ros2_ws>/src/peer_robotics_pallet_vision'!"
    exit 1
fi

# Navigate to the workspace root
cd ../../ || { echo "Failed to navigate to the workspace root!"; exit 1; }

# Install ROS 2 dependencies using rosdep
echo "Installing ROS 2 dependencies..."
if ! command -v rosdep &>/dev/null; then
    echo "rosdep is not installed. Installing it now..."
    sudo apt-get update && sudo apt-get install -y python3-rosdep || { echo "Failed to install rosdep!"; exit 1; }
fi

# Initialize rosdep if not already initialized
if [ ! -f "/etc/ros/rosdep/sources.list.d/20-default.list" ]; then
    echo "Initializing rosdep..."
    sudo rosdep init || { echo "Failed to initialize rosdep!"; exit 1; }
fi

rosdep update || { echo "Failed to update rosdep!"; exit 1; }

# Install package dependencies
rosdep install --from-paths src --ignore-src -r -y || { echo "Failed to install ROS 2 package dependencies!"; exit 1; }

# Build the workspace
echo "Building the ROS 2 workspace..."
colcon build || { echo "Failed to build the workspace!"; exit 1; }

# Source ROS 2 Humble setup
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash || { echo "Failed to source ROS 2 setup!"; exit 1; }
else
    echo "ROS 2 Humble is not installed or not properly sourced!"
    exit 1
fi

# Source the workspace
if [ -f "install/setup.bash" ]; then
    source install/setup.bash || { echo "Failed to source the workspace setup!"; exit 1; }
else
    echo "Workspace setup script not found! Make sure the workspace has been built successfully."
    exit 1
fi

# Check if Conda is initialized
if ! command -v conda &>/dev/null; then
    echo "Conda is not installed or not initialized! Please install Conda and set up its base environment."
    exit 1
fi

# Create and activate Conda environment from environment.yaml
ENV_NAME="yolo_env"
ENV_YAML="src/peer_robotics_pallet_vision/environment.yaml"  # Adjust path as it's relative to the workspace root

if conda env list | grep -q "${ENV_NAME}"; then
    echo "Conda environment '${ENV_NAME}' already exists. Activating it..."
else
    echo "Creating Conda environment '${ENV_NAME}' from ${ENV_YAML}..."
    conda env create -f "${ENV_YAML}" || { echo "Failed to create Conda environment!"; exit 1; }
fi

# Ensure gdown is installed
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

# Create target directory for the downloaded file
TARGET_DIR="src/peer_robotics_pallet_vision/rosbags"
mkdir -p "${TARGET_DIR}" || { echo "Failed to create target directory!"; exit 1; }

# Download and unzip the file from Google Drive
FILE_ID="1BvhP653G3PqfUq96L18gDBIi-5oOYqcr"
echo "Downloading the sample rosbag into ${TARGET_DIR}..."
gdown "https://drive.google.com/uc?id=${FILE_ID}" --fuzzy -O "${TARGET_DIR}/internship_assignment_sample_bag.zip" || { echo "File download failed!"; exit 1; }

echo "Unzipping the sample rosbag in ${TARGET_DIR}..."
cd "${TARGET_DIR}" || { echo "Failed to change to target directory!"; exit 1; }
unzip internship_assignment_sample_bag.zip || { echo "Failed to unzip the file!"; exit 1; }
rm internship_assignment_sample_bag.zip  

cd ../../../../ || { echo "Failed to return to the workspace root directory!"; exit 1; }

conda init

conda activate "${ENV_NAME}" || { echo "Failed to activate Conda environment!"; exit 1; }

echo "Setup complete! The ROS 2 dependencies are installed, the workspace is built, and the Conda environment is ready."



