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

    # RViz2 node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        # arguments=['-d', rviz_config_path],
        name='rviz2',
        output='screen'
    )
    
    # # Image view node
    # image_view_node = Node(
    #     package='image_view',
    #     executable='image_view',
    #     name='image_view',
    #     parameters=[{
    #         'image_transport': 'raw'
    #     }],
    #     arguments=[
    #         '--width', '640', 
    #         '--height', '640'  # Set the desired height
    #     ],
    #     remappings=[
    #         ('/image', '/segmentation_inference/overlay_image')
    #     ],
    #     output='screen'
    # )

    return LaunchDescription([
        bag_play,
        segmentation_node,
        # image_view_node,
        rviz_node,
    ])
