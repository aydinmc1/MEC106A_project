"""
helper utilities for converting pixel detections to 3d poses and transforming between frames.
used by board_detector_node and move_above_board_node.
"""

import numpy as np
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from sensor_msgs.msg import CameraInfo
import tf2_ros
import tf2_geometry_msgs
from rclpy.time import Time


def pixel_to_camera_3d(pixel_x, pixel_y, depth_z, camera_info_msg):
    """
    convert a 2d pixel coordinate + depth to a 3d point in the camera frame.
    uses camera intrinsics from CameraInfo message.
    
    args:
        pixel_x: x coordinate in image (pixels)
        pixel_y: y coordinate in image (pixels)
        depth_z: depth value (meters), assumed z-distance from camera
        camera_info_msg: sensor_msgs/CameraInfo message with K matrix
    
    returns:
        tuple (x, y, z) in camera frame (meters)
    """
    # camera intrinsics from K matrix
    # K = [fx  0 cx]
    #     [ 0 fy cy]
    #     [ 0  0  1]
    K = camera_info_msg.K
    fx = K[0]
    fy = K[4]
    cx = K[2]
    cy = K[5]
    
    # unprojection formula
    x_cam = (pixel_x - cx) * depth_z / fx
    y_cam = (pixel_y - cy) * depth_z / fy
    z_cam = depth_z
    
    return x_cam, y_cam, z_cam


def make_point_stamped(x, y, z, frame_id, timestamp=None):
    """
    create a PointStamped message in the given frame.
    
    args:
        x, y, z: coordinates (float)
        frame_id: string name of frame
        timestamp: rclpy.time.Time or None (defaults to now)
    
    returns:
        geometry_msgs/PointStamped
    """
    point = PointStamped()
    point.header.frame_id = frame_id
    if timestamp:
        point.header.stamp = timestamp.to_msg()
    else:
        point.header.stamp.sec = 0
        point.header.stamp.nanosec = 0
    point.point.x = x
    point.point.y = y
    point.point.z = z
    return point


def make_pose_stamped(x, y, z, frame_id, orientation_quat=None):
    """
    create a PoseStamped message.
    
    args:
        x, y, z: position (float)
        frame_id: string name of frame
        orientation_quat: geometry_msgs/Quaternion or None (defaults to identity)
    
    returns:
        geometry_msgs/PoseStamped
    """
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp.sec = 0
    pose.header.stamp.nanosec = 0
    
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z
    
    if orientation_quat:
        pose.pose.orientation = orientation_quat
    else:
        # identity quaternion
        pose.pose.orientation.w = 1.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
    
    return pose


def identity_quaternion():
    """
    return an identity quaternion (no rotation).
    """
    q = Quaternion()
    q.w = 1.0
    q.x = 0.0
    q.y = 0.0
    q.z = 0.0
    return q


def transform_pose(pose_stamped, target_frame, tf_buffer, timeout=1.0):
    """
    transform a PoseStamped from its frame to target_frame using tf2.
    
    args:
        pose_stamped: geometry_msgs/PoseStamped in source frame
        target_frame: string name of target frame
        tf_buffer: tf2_ros.Buffer object for lookups
        timeout: seconds to wait for transform
    
    returns:
        geometry_msgs/PoseStamped in target_frame, or None if transform fails
    """
    try:
        # lookup transform from source frame to target frame
        transform = tf_buffer.lookup_transform(
            target_frame,
            pose_stamped.header.frame_id,
            Time(seconds=0.0),  # get latest
            timeout=tf2_ros.Duration(seconds=timeout)
        )
        # apply transform using tf2_geometry_msgs
        pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
        return pose_transformed
    except tf2_ros.TransformException as ex:
        # print(f"transform failed: {ex}")
        return None


def transform_point(point_stamped, target_frame, tf_buffer, timeout=1.0):
    """
    transform a PointStamped from its frame to target_frame using tf2.
    
    args:
        point_stamped: geometry_msgs/PointStamped in source frame
        target_frame: string name of target frame
        tf_buffer: tf2_ros.Buffer object
        timeout: seconds to wait
    
    returns:
        geometry_msgs/PointStamped in target_frame, or None if transform fails
    """
    try:
        transform = tf_buffer.lookup_transform(
            target_frame,
            point_stamped.header.frame_id,
            Time(seconds=0.0),
            timeout=tf2_ros.Duration(seconds=timeout)
        )
        point_transformed = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
        return point_transformed
    except tf2_ros.TransformException as ex:
        # print(f"transform failed: {ex}")
        return None
