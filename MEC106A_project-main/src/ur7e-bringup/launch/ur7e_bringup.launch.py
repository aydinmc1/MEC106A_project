"""
Main bringup launch file for the UR7e operation system.

Launches:
  - UR robot driver (real robot or mock)
  - RealSense camera driver
  - Robot state publisher (URDF)
  - MoveIt2 move_group
  - ur7e_camera_vision nodes
  - ur7e_play_operation state machine
  - RViz2

Usage:
  ros2 launch ur7e_bringup ur7e_bringup.launch.py
  ros2 launch ur7e_bringup ur7e_bringup.launch.py use_mock_hardware:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    GroupAction,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # ── Package directories ───────────────────────────────────────────────────
    bringup_dir = get_package_share_directory("ur7e_bringup")

    # ── Launch arguments ──────────────────────────────────────────────────────
    use_mock_hardware = LaunchConfiguration("use_mock_hardware")
    robot_ip = LaunchConfiguration("robot_ip")
    use_rviz = LaunchConfiguration("use_rviz")
    use_camera = LaunchConfiguration("use_camera")

    declare_mock = DeclareLaunchArgument(
        "use_mock_hardware",
        default_value="false",
        description="Start with mock hardware (no physical robot needed)",
    )
    declare_robot_ip = DeclareLaunchArgument(
        "robot_ip",
        default_value="192.168.1.102",
        description="IP address of the UR7e robot",
    )
    declare_rviz = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Launch RViz2",
    )
    declare_camera = DeclareLaunchArgument(
        "use_camera",
        default_value="true",
        description="Launch the RealSense camera driver",
    )

    # ── UR Robot driver ───────────────────────────────────────────────────────
    ur_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ur_robot_driver"),
                "launch",
                "ur_control.launch.py",
            ])
        ),
        launch_arguments={
            "ur_type": "ur7e",
            "robot_ip": robot_ip,
            "use_mock_hardware": use_mock_hardware,
            "launch_rviz": "false",
        }.items(),
    )

    # ── Camera driver ─────────────────────────────────────────────────────────
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("realsense2_camera"),
                "launch",
                "rs_launch.py",
            ])
        ),
        launch_arguments={
            "enable_color": "true",
            "enable_depth": "true",
            "enable_infra1": "false",
            "enable_infra2": "false",
            # Lab 5 style RealSense setup: aligned depth and point cloud enabled.
            "align_depth.enable": "true",
            "pointcloud.enable": "true",
            "rgb_camera.color_profile": "1920x1080x30",
            "camera_name": "camera",
        }.items(),
        condition=IfCondition(use_camera),
    )

    # ── Vision nodes ──────────────────────────────────────────────────────────
    vision_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ur7e_camera_vision"),
                "launch",
                "camera_vision.launch.py",
            ])
        ),
        condition=IfCondition(use_camera),
    )

    # ── Motion / MoveIt2 ─────────────────────────────────────────────────────
    motion_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("ur7e_motion"),
                "launch",
                "motion.launch.py",
            ])
        ),
    )

    # ── Operation state machine ───────────────────────────────────────────────
    # Small delay so robot driver and MoveIt2 are ready before we start
    operation_launch = TimerAction(
        period=5.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([
                        FindPackageShare("ur7e_play_operation"),
                        "launch",
                        "play_operation.launch.py",
                    ])
                ),
            )
        ],
    )

    # ── RViz2 ─────────────────────────────────────────────────────────────────
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(bringup_dir, "rviz", "ur7e_operation.rviz")],
        condition=IfCondition(use_rviz),
        output="log",
    )

    return LaunchDescription([
        declare_mock,
        declare_robot_ip,
        declare_rviz,
        declare_camera,
        ur_driver_launch,
        camera_launch,
        vision_launch,
        motion_launch,
        operation_launch,
        rviz_node,
    ])
