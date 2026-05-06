"""
Static camera-to-UR7e TF broadcaster.

Use this only after measuring/calibrating the RealSense mount. The lab's TF
concept is that a fixed transform connects the camera optical frame to the UR7e
wrist frame; once that transform is in TF, detected camera-frame points can be
transformed into base_link.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Convert roll/pitch/yaw in radians to quaternion x,y,z,w."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


def _as_float_list(value: Iterable, length: int, default: float = 0.0):
    out = [float(v) for v in list(value)[:length]]
    while len(out) < length:
        out.append(default)
    return out


class StaticCameraTfNode(Node):
    def __init__(self) -> None:
        super().__init__("static_camera_tf_node")

        self.declare_parameter("parent_frame", "wrist_3_link")
        self.declare_parameter("child_frame", "camera_color_optical_frame")
        self.declare_parameter("translation_xyz_m", [0.0, 0.0, 0.0])
        self.declare_parameter("rotation_rpy_deg", [0.0, 0.0, 0.0])
        self.declare_parameter("warn_if_identity", True)

        parent_frame = str(self.get_parameter("parent_frame").value)
        child_frame = str(self.get_parameter("child_frame").value)
        xyz = _as_float_list(self.get_parameter("translation_xyz_m").value, 3)
        rpy_deg = _as_float_list(self.get_parameter("rotation_rpy_deg").value, 3)
        rpy_rad = [math.radians(v) for v in rpy_deg]
        qx, qy, qz, qw = quaternion_from_rpy(*rpy_rad)

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = xyz[0]
        transform.transform.translation.y = xyz[1]
        transform.transform.translation.z = xyz[2]
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw

        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcaster.sendTransform(transform)

        if bool(self.get_parameter("warn_if_identity").value) and all(abs(v) < 1e-9 for v in xyz + rpy_deg):
            self.get_logger().warn(
                "Publishing an identity camera TF. Replace translation_xyz_m and "
                "rotation_rpy_deg with the measured RealSense mount transform before robot motion."
            )

        self.get_logger().info(
            f"published static TF {parent_frame} -> {child_frame}: "
            f"xyz(m)={xyz}, rpy(deg)={rpy_deg}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StaticCameraTfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
