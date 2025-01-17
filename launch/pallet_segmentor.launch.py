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
    segmentation_node = Node(
        package='peer_robotics_pallet_vision',
        executable='segmentation_node', 
        name='segmentation_node',
        parameters=[{
            'rgb_topic': '/d455_1_rgb_image',
            'depth_topic': '/d455_1_depth_image',
            'output_topic': '/segmentation_inference/overlay_image'
        }],
        output='screen'
    )

    # display node
    segmentation_display_node = Node(
        package='peer_robotics_pallet_vision',
        executable='segmentation_display_node',
        name='segmentation_display_node',
        parameters=[
                {'input_topic': '/segmentation_inference/overlay_image'}  # Input topic parameter
            ],
            output='screen',
    )

    return LaunchDescription([
        bag_play,
        segmentation_node,
        segmentation_display_node,
    ])
