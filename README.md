# Pallet & Ground Detection-Segmentation Application

## Objective
Solution for pallet detection and pallet-ground segmentation in a manufacturing or warehousing environment. The solution is optimized for real-time deployment on edge devices like the NVIDIA Jetson AGX Orin and is suitable for mobile robotics applications.

<table>
  <tr>
    <td>
      <img src="videos/gifs/banner.gif" alt="Banner Segmentation" width="400"/>
    </td>
    <td>
      <img src="videos/gifs/banner_detection.gif" alt="Banner Detection" width="400"/>
    </td>
  </tr>
</table>

---

## Summary
### **1. Dataset Acquisition and Preparation**
- [x] **Dataset**: Used the recommended dataset (as per assignment).
- [x] **Annotation**: Initially used Grounded SAM, but switched to manual annotation for 100 images. Fine-tuned YOLOv11 to annotate the rest.
- [x] **Organization**: Organized in YOLOv11 format with train, validation, and test splits.
- [x] **Augmentation**: Applied techniques like flipping, rotation, Gaussian noise, blur, shear, grayscale, and saturation adjustments.

---

### **2. Object Detection and Semantic Segmentation**
- [x] **Object Detection Model**: Fine-tuned YOLOv11 for pallet detection.
- [x] **Semantic Segmentation Model**: Fine-tuned YOLOv11-seg for pallet-ground segmentation. 
- [x] **Training**: Trained using the recommended dataset. Trained using full (FP32) and half (FP16) precision.
- [x] **Performance Metrics**: Evaluated models using:
  - **mAP** for object detection.
  - **IoU** for semantic segmentation.
---

### **3. ROS 2 Node Development**
- [x] **ROS 2 Package**: Created a ROS 2 package compatible with Humble.
- [x] **Node Implementation**:
  - Subscribes to image and depth topics from a camera. 
  - Performs detection (pallets) and segmentation (pallet-ground).
  - Publishes detection and segmentation results.
  - Real-time inference and visualization using cv2. 
- [x] **Off-the-Shelf Usability**: Included a README to ensure nodes run without additional setup issues.

---

### **4. Edge Deployment Optimization (Optional)**
- [x] **Model Conversion**: Converted models to ONNX and TensorRT formats for optimized edge deployment. 
- [x] **Optimization Techniques**: Trained using Half precision, Applied quantization (INT8) for improved performance on the NVIDIA Jetson Orin.
- [ ] **Tested on Orin Nano**: Current unavailability of hardware. 
---

### **5. Dockerized Deployment**
- [ ] **Dockerization**: Coming soon... 
---

### **6. Evaluation Criteria**
- [x] **Live Camera Feed**: Full integration and testing with ros2 bags obtained from real warehouse [Link] the ZED 2i camera feed provided in the assignment. 

- [x] **Detection Accuracy**: Initial testing performed; detection accuracy under varying conditions to be further evaluated.

---

---

## Table of Contents
1. [Dataset Preparation](#dataset-preparation)
2. [Model Development](#model-development)
3. [ROS 2 Node Development](#ros-2-node-development)
4. [Edge Deployment Optimization](#edge-deployment-optimization)
5. [How to Run](#how-to-run)
6. [Results and Evaluation](#results-and-evaluation)
7. [Future Work](#future-work)

---

## Dataset Preparation

1. **Dataset Acquisition**:
   
   - Link for recommended dataset : [Pallets](https://drive.google.com/drive/folders/1xSqKa55QrNGufLRQZAbp0KFGYr9ecqgT)
    
2. **Data Annotation**:

   - Annotated Dataset Link (detection and segmentation): [Pallets-Annotated](https://drive.google.com/file/d/1MsLv1pdn9zk5YqzmE9aiHjZcGETamikd/view?usp=sharing)
   - For Training : store in the `dataset` folder in the package.

4. **Dataset Organization**:
   - Split into:
     - Training: 80%
     - Validation: 10%
     - Test: 10%

5. ROS2 Bags for real-world Deployment. 

    - Feed from RealSense D455 : [r2b_storage](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/resources/r2bdataset2023)
    
    - Feed from zed 2i : [camera_data](https://drive.usercontent.google.com/download?id=1BvhP653G3PqfUq96L18gDBIi-5oOYqcr&export=download&authuser=0)
    
    - For real-time inference, store in the `rosbag` folder of the package. 

---

## Model Development

### Object Detection

- **Model**: YOLOv11 nano. 

- **Training Framework**: PyTorch (trained in FP32 and FP16 precision).

- **Losses, Mean Average Precision (MAP), Precision and Recall** 

![Training and Validation Losses](videos/images/detect_train_plot.png)
  
- **Performance Metrics**:
  - For FP32 model: 
    - mAP (on test set): 
      - Final mAP@50: 0.801
      - Final mAP@50:95: 0.575
    - Inference Speed: 3.4 ms
  
  - For Optimized model (trained FP16, then Quantized to Int8): 

    - Inference Speed: 2.04 ms

### Semantic Segmentation

- **Model**: YOLOv11 nano seg.

- **Training Framework**: PyTorch (trained in FP32 precision).

  - **Losses, Mean Average Precision (MAP), Precision and Recall**

![Training and Validation Losses](videos/images/segment_train_plot.png)

- **Performance Metrics**:
- mAP
  - IOU: 0.78
  - Inference Speed: 4.2 ms 

### Results

#### Inference on Test Set

#### Real World Deployment

- Feed from RealSense D455
  - Detection (Pallets)
  
- ![Detection](videos/gifs/segmentation.gif)

  - Segmentation (Pallets, Ground)
  
- ![Detection](videos/gifs/detection.gif)

- Inferences from Zed2i Camera. 

  - Detection (Pallets)
  
- ![Detection](videos/images/zed2i_detection.jpg)

  - Segmentation (Pallets, Ground)
  
- ![Detection](videos/images/zed2i_segmenation.png)
  

---

## ROS 2 Node Development

### ROS 2 Package
- Developed using ROS 2 Humble.

- Package Name: `peer_robotics_pallet_vision`

- Nodes:

  - `detector_node`:
    - Subscribes to camera image and depth topics.
    - Performs object detection on live camera feeds.
    - Publishes original images overlayed with predictions as visualization.

  - `segmentation_node`:
    - Subscribes to camera image topics.
    - Performs semantic segmentation on live camera feed. 
    - Publishes original images overlayed with segmentation masks. 

### Topics
- **Subscribed Topics**:
  
  - `/robot1/zed2i/left/image_rect_color` (sensor_msgs/Image)
  - `/camera/depth/image_raw` (sensor_msgs/Image)

- **Published Topics**:
  
  - `/pallet_detection` (sensor_msgs/Image with bounding boxes)
  - `/pallet_segmentation` (sensor_msgs/Image with segmented regions)

### Visualization
- Bounding boxes and segmentation masks are overlaid on the input image and then published. 

---

## Installation and Setup

### Prerequisites

#### Install ROS 2 Humble
Ensure you have **ROS 2 Humble** installed on **Ubuntu 22.04**. Follow the official ROS 2 documentation for installation:  
[ROS 2 Humble Installation Guide](https://docs.ros.org/en/humble/Installation.html)

---

#### Install Miniconda

Install **Miniconda** to manage the Python environment. Follow the official installation guide:  
[Miniconda Installation Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

---

### Environment Setup

0. Create Conda Environment
This package includes an `environment.yml` file to set up the required Python environment. Run the following commands:

1. Clone this repository into your workspace:
   ```bash
   $ mkdir ~/ros2_ws/src && cd ros2_ws/src
   
   $ git clone https://github.com/your-repo/peer_robotics_pallet_vision.git


2. navigate to package 

```bash
cd ~/ros2_ws/src/peer_robotics_pallet_vision
```

3. create conda environment

```bash
conda env create -f environment.yml
```

If the environment.yml file does not work, manually set up the environment:

```bash
conda create -n yolo python=3.10.14
conda activate yolo
conda install -c conda-forge ultralytics
```

```bash
cd ~/ros2_ws
```

```bash
colcon build
```
```bash
source install/setup.bash
```

```bash
conda activate yolo
```

```bash
cd src/peer_robotics_pallet_vision
```

### Usage

- First Run the rosbag for image data publishing. 

```bash
ros2 bag play rosbags/internship_assignment_sample_bag/ --loop
```
 
- In a different terminal: 

```bash
ros2 topic list
```

Based on this identity rgb and depth topics and then: 

- For Detection Node:

```bash
ros2 launch peer_robotics_pallet_vision pallet_detector.launch.py rgb_topic:=/d455_1_rgb_image depth_topic:=/d455_1_depth_image rosbag:=False
```

- For Segmentation Node:

```bash
ros2 launch peer_robotics_pallet_vision pallet_segmentor.launch.py rgb_topic:=/d455_1_rgb_image depth_topic:=/d455_1_depth_image rosbag:=False
```

- Ensure that you have placed: 

```bash
yolo/models/final/detection/detection_final.pt
yolo/models/final/segmentation/segmentation_final.pt
```

These models are part of the repo. 

#### Training Pipeline

1. Prepare dataset: 

  ```bash
  dataset/
  ├── train/
  │   ├── images/
  │   ├── labels/
  ├── val/
  │   ├── images/
  │   ├── labels/
  └── data.yaml
  ```

  Ensure right paths are set in 'data.yaml'


2. Both detection and segmenation models are trained in the same way :

  ```bash
  python train_yolo_detection.py --model_path yolo/models/yolo11n.pt --data_path ./datasets/detection_dataset/data.yaml --image_size 640 --batch_size 8 --epochs 200 --device 0 --save_period 50
  ```

3. To run inference : 

  ```bash
  python test_yolo_detection.py
  ```


## Edge Deployment Optimization

### Model Optimization

- Converted models to:

  - **ONNX**: For interoperability. 

  - **TensorRT**: For high-performance inference on AGX Orin. Note that implementation using TensorRT API is assumed out of scope for this assignment. 

  - Files for the above can be found in the `yolo/models/final/optimized_models_detection`:

- Applied:

  - **Quantization**: INT8 precision for reduced memory and faster inference.

  - **Pruning**: Not Applied because the model yolov11n and yolo11n-seg are already small. 



### Dockerized Deployment {TODO}
