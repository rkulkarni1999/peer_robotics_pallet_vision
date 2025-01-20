from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():

    # Args
    rgb_topic_arg = DeclareLaunchArgument(
        'rgb_topic',
        default_value='/robot1/zed2i/left/image_rect_color',  # Default RGB topic
        description='Input RGB topic for the detection node'
    )

    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='/d455_1_depth_image',  
        description='Input Depth topic for the detection node'
    )
    
    rosbag_arg = DeclareLaunchArgument(
        'rosbag',
        default_value='False', 
        description='Set to True to play the rosbag'
    )

    # Paths
    rosbag_path = os.path.join(
        os.getcwd(),
        "rosbags/internship_assignment_sample_bag"
        # "rosbags/r2b_storage"
    )

    rviz_config_path = os.path.join(
        os.getcwd(),
        "rviz/rviz.rviz/"  
    )

    # Nodes and processes
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', rosbag_path, '--loop'],
        output='screen',
        condition=IfCondition(LaunchConfiguration('rosbag'))  
    )

    # Segmentation node
    segmentation_node = Node(
        package='peer_robotics_pallet_vision',
        executable='segmentation_node', 
        name='segmentation_node',
        parameters=[
            {'rgb_topic': LaunchConfiguration('rgb_topic')},
            {'depth_topic': LaunchConfiguration('depth_topic')},
            {'output_topic': '/segmentation_inference/overlay_image'}
        ],
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
        rgb_topic_arg,
        depth_topic_arg,
        rosbag_arg,
        segmentation_node,
        segmentation_display_node,
        bag_play,
    ])
