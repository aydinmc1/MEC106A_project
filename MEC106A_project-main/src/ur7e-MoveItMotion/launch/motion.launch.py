"""Launch the motion server node."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("ur7e_motion")
    config = os.path.join(pkg_dir, "config", "named_poses.yaml")

    motion_server = Node(
        package="ur7e_motion",
        executable="motion_server_node",
        name="motion_server_node",
        parameters=[
            {"named_poses_file": config},
            {"planning_group": "ur_manipulator"},
            {"default_velocity_scaling": 0.3},
            {"default_acceleration_scaling": 0.2},
        ],
        output="screen",
    )

    return LaunchDescription([motion_server])
