from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
import os

def generate_launch_description():

    rosbag_path = os.path.join(
        os.getcwd(),
        "rosbags/r2b_storage/"
    )

    # Path to your RViz configuration file
    rviz_config_path = os.path.join(
        os.getcwd(),
        "rviz/rviz.rviz/"  
    )

    # Rosbag playback process
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', rosbag_path, '--loop'],
        output='screen'
    )

    # Segmentation node
    detection_node = Node(
        package='peer_robotics_pallet_vision',
        executable='detection_node', 
        name='detection_node',
        parameters=[{
            'rgb_topic': '/d455_1_rgb_image',
            'depth_topic': '/d455_1_depth_image',
            'output_topic': '/detection_inference/overlay_image'
        }],
        output='screen'
    )

    detection_display_node = Node(
        package='peer_robotics_pallet_vision',
        executable='detection_display_node',
        name='detection_display_node',
        parameters=[
                {'input_topic': '/detection_inference/overlay_image'}  # Input topic parameter
                # {'input_topic': '/d455_1_rgb_image'}
            ],
            output='screen',
    )

    return LaunchDescription([
        # bag_play,
        detection_node,
        detection_display_node,
    ])
