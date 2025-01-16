from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
import os


def generate_launch_description():

    rosbag_path = os.path.join(
        os.getcwd,
        "ros_dev/peer_ws/src/rosbag_package/rosbag_internship/rosbag_warehouse_data/r2b_storage/"
    )

    # Path to your RViz configuration file
    rviz_config_path = os.path.join(
        os.getcwd,
        "ros_dev/peer_ws/rviz_config/segmentation_config.rviz"  # Replace with the actual path to your .rviz2 file
    )

    # Rosbag playback process
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', rosbag_path, '--loop'],
        output='screen'
    )

    # Segmentation node
    segmentation_node = Node(
        package='your_package_name',  # Replace with your package name
        executable='segmentation_node',  # Replace with your node executable name
        name='segmentation_node',
        parameters=[{
            'rgb_topic': '/d455_1_rgb_image',
            'depth_topic': '/d455_1_depth_image',
            'output_topic': '/segmentation_inference/overlay_image'
        }],
        output='screen'
    )

    # RViz2 node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_path],
        name='rviz2',
        output='screen'
    )

    return LaunchDescription([
        bag_play,
        segmentation_node,
        rviz_node
    ])
