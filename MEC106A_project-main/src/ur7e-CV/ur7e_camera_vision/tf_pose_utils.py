"""
Utilities for converting camera pixels to 3D points and creating/translating ROS2 poses.
"""

from __future__ import annotations

from typing import Optional, Tuple

from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo
import tf2_geometry_msgs
import tf2_ros


def _camera_k(camera_info_msg: CameraInfo):
    """Return the CameraInfo intrinsic matrix in a way that works across ROS versions."""
    k = getattr(camera_info_msg, "k", None)
    if k is None or len(k) < 6:
        k = getattr(camera_info_msg, "K", None)
    if k is None or len(k) < 6:
        raise ValueError("CameraInfo is missing intrinsic matrix k/K")
    return k


def _assign_stamp(header, timestamp=None) -> None:
    """Assign header.stamp from rclpy Time, builtin_interfaces/Time, or leave zero."""
    if timestamp is None:
        header.stamp.sec = 0
        header.stamp.nanosec = 0
    elif hasattr(timestamp, "to_msg"):
        header.stamp = timestamp.to_msg()
    elif isinstance(timestamp, TimeMsg) or (hasattr(timestamp, "sec") and hasattr(timestamp, "nanosec")):
        header.stamp = timestamp
    else:
        header.stamp.sec = 0
        header.stamp.nanosec = 0


def pixel_to_camera_3d(pixel_x, pixel_y, depth_z, camera_info_msg: CameraInfo) -> Tuple[float, float, float]:
    """
    Convert a 2D pixel coordinate plus depth to a 3D point in the optical camera frame.

    depth_z is in meters. The result is also in meters.
    """
    k = _camera_k(camera_info_msg)
    fx = float(k[0])
    fy = float(k[4])
    cx = float(k[2])
    cy = float(k[5])

    x_cam = (float(pixel_x) - cx) * float(depth_z) / fx
    y_cam = (float(pixel_y) - cy) * float(depth_z) / fy
    z_cam = float(depth_z)
    return x_cam, y_cam, z_cam


def make_point_stamped(x, y, z, frame_id: str, timestamp=None) -> PointStamped:
    """Create a PointStamped in the given frame."""
    point = PointStamped()
    point.header.frame_id = frame_id
    _assign_stamp(point.header, timestamp)
    point.point.x = float(x)
    point.point.y = float(y)
    point.point.z = float(z)
    return point


def make_pose_stamped(
    x,
    y,
    z,
    frame_id: str,
    orientation_quat: Optional[Quaternion] = None,
    timestamp=None,
) -> PoseStamped:
    """Create a PoseStamped in the given frame."""
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    _assign_stamp(pose.header, timestamp)
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = float(z)
    pose.pose.orientation = orientation_quat if orientation_quat is not None else identity_quaternion()
    return pose


def identity_quaternion() -> Quaternion:
    q = Quaternion()
    q.w = 1.0
    q.x = 0.0
    q.y = 0.0
    q.z = 0.0
    return q


def transform_pose(pose_stamped: PoseStamped, target_frame: str, tf_buffer: tf2_ros.Buffer, timeout=1.0):
    """Transform a PoseStamped to target_frame using tf2, or return None."""
    try:
        transform = tf_buffer.lookup_transform(
            target_frame,
            pose_stamped.header.frame_id,
            Time(),
            timeout=Duration(seconds=float(timeout)),
        )
        return tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
    except Exception:
        return None


def transform_point(point_stamped: PointStamped, target_frame: str, tf_buffer: tf2_ros.Buffer, timeout=1.0):
    """Transform a PointStamped to target_frame using tf2, or return None."""
    try:
        transform = tf_buffer.lookup_transform(
            target_frame,
            point_stamped.header.frame_id,
            Time(),
            timeout=Duration(seconds=float(timeout)),
        )
        return tf2_geometry_msgs.do_transform_point(point_stamped, transform)
    except Exception:
        return None
