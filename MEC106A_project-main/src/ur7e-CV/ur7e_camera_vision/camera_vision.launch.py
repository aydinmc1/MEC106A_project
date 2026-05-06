"""
Launch the RealSense-backed Operation CV stack.

This launch file does not start the RealSense driver; ur7e_bringup.launch.py does
that. It subscribes to RealSense color/aligned-depth streams and publishes the
Operation target/pose topics.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("ur7e_camera_vision")
    config_file = os.path.join(pkg_dir, "config", "camera_vision_params.yaml")
    hsv_config_file = os.path.join(pkg_dir, "config", "operation_hsv_config.json")

    color_image_topic = LaunchConfiguration("color_image_topic")
    depth_image_topic = LaunchConfiguration("depth_image_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    camera_frame = LaunchConfiguration("camera_frame")
    base_frame = LaunchConfiguration("base_frame")
    publish_static_camera_tf = LaunchConfiguration("publish_static_camera_tf")
    static_parent_frame = LaunchConfiguration("static_parent_frame")
    static_child_frame = LaunchConfiguration("static_child_frame")

    declare_color_image_topic = DeclareLaunchArgument(
        "color_image_topic",
        default_value="/camera/camera/color/image_raw",
        description="Aligned RealSense color image topic",
    )
    declare_depth_image_topic = DeclareLaunchArgument(
        "depth_image_topic",
        default_value="/camera/camera/aligned_depth_to_color/image_raw",
        description="RealSense depth image aligned to the color image",
    )
    declare_camera_info_topic = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/camera/camera/color/camera_info",
        description="CameraInfo for the color stream",
    )
    declare_camera_frame = DeclareLaunchArgument(
        "camera_frame",
        default_value="camera_color_optical_frame",
        description="Optical camera frame for detections",
    )
    declare_base_frame = DeclareLaunchArgument(
        "base_frame",
        default_value="base_link",
        description="UR7e base frame to transform detections into",
    )
    declare_publish_static_camera_tf = DeclareLaunchArgument(
        "publish_static_camera_tf",
        default_value="false",
        description="Publish a measured wrist_3_link -> camera optical static TF",
    )
    declare_static_parent_frame = DeclareLaunchArgument(
        "static_parent_frame",
        default_value="wrist_3_link",
        description="Parent frame for the optional camera mount transform",
    )
    declare_static_child_frame = DeclareLaunchArgument(
        "static_child_frame",
        default_value="camera_color_optical_frame",
        description="Child camera optical frame for the optional camera mount transform",
    )

    color_mapper_node = Node(
        package="ur7e_camera_vision",
        executable="color_mapper_node",
        name="color_mapper_node",
        parameters=[
            config_file,
            {
                "input_topic": color_image_topic,
                "output_topic": "/camera/colormap/image",
                "normalize_input": False,
            },
        ],
        output="screen",
    )

    camera_info_node = Node(
        package="ur7e_camera_vision",
        executable="camera_info_node",
        name="camera_info_node",
        parameters=[
            config_file,
            {
                "camera_info_topic": camera_info_topic,
                "image_topic": color_image_topic,
            },
        ],
        output="screen",
    )

    board_detector_node = Node(
        package="ur7e_camera_vision",
        executable="board_detector_node",
        name="board_detector_node",
        parameters=[
            config_file,
            {
                "camera_image_topic": color_image_topic,
                "camera_depth_topic": depth_image_topic,
                "camera_info_topic": camera_info_topic,
                "camera_frame": camera_frame,
                "base_frame": base_frame,
                "use_depth_for_3d": True,
                "config_path": hsv_config_file,
            },
        ],
        output="screen",
    )

    static_camera_tf_node = Node(
        package="ur7e_camera_vision",
        executable="static_camera_tf_node",
        name="static_camera_tf_node",
        parameters=[
            {
                "parent_frame": static_parent_frame,
                "child_frame": static_child_frame,
                # Fill these from your measured RealSense mount calibration.
                "translation_xyz_m": [0.0, 0.0, 0.0],
                "rotation_rpy_deg": [0.0, 0.0, 0.0],
            }
        ],
        condition=IfCondition(publish_static_camera_tf),
        output="screen",
    )

    return LaunchDescription([
        declare_color_image_topic,
        declare_depth_image_topic,
        declare_camera_info_topic,
        declare_camera_frame,
        declare_base_frame,
        declare_publish_static_camera_tf,
        declare_static_parent_frame,
        declare_static_child_frame,
        color_mapper_node,
        camera_info_node,
        board_detector_node,
        static_camera_tf_node,
    ])
