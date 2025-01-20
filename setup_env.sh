#!/bin/bash

# Ensure the script is executed from the correct directory
if [[ $(basename "$PWD") != "peer_robotics_pallet_vision" ]]; then
    echo "You must run this script from '<ros2_ws>/src/peer_robotics_pallet_vision'!"
    exit 1
fi

# Navigate to the workspace root
cd ../../ || { echo "Failed to navigate to the workspace root!"; exit 1; }

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

conda activate "${ENV_NAME}" || { echo "Failed to activate Conda environment!"; exit 1; }

echo "Setup complete! The ROS 2 workspace is built, and the Conda environment is ready."
