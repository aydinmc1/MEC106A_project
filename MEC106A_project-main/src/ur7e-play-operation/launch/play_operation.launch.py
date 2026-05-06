"""Launch the play operation state machine."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("ur7e_play_operation")
    config = os.path.join(pkg_dir, "config", "operation_params.yaml")

    auto_start = LaunchConfiguration("auto_start")
    declare_auto_start = DeclareLaunchArgument(
        "auto_start", default_value="false",
        description="Automatically start operation on launch",
    )

    operation_node = Node(
        package="ur7e_play_operation",
        executable="operation_node",
        name="operation_node",
        parameters=[
            config,
            {"auto_start": auto_start},
        ],
        output="screen",
    )

    return LaunchDescription([
        declare_auto_start,
        operation_node,
    ])
