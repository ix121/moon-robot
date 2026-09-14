import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('eco65_visual_grasp')
    default_params = os.path.join(share, 'config', 'eco65_visual_grasp.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('execute_motion', default_value='false'),
        Node(
            package='eco65_visual_grasp',
            executable='eco65_visual_grasp_node',
            name='eco65_visual_grasp',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {'execute_motion': LaunchConfiguration('execute_motion')},
            ],
        ),
    ])
