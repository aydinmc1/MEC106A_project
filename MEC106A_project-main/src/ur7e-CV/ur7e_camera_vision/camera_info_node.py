"""
camera_info_node.py
===================
Subscribes to sensor_msgs/CameraInfo and:
  1. Logs a formatted summary of intrinsics on first reception.
  2. Publishes a ur7e_interfaces/CameraMetrics topic with runtime stats.
  3. Watches for camera health (drops, timeouts).

Topics
------
Sub: /camera/camera/color/camera_info    (sensor_msgs/CameraInfo)
     /camera/camera/color/image_raw      (sensor_msgs/Image)  — for FPS tracking
Pub: /camera/info_display         (ur7e_interfaces/CameraMetrics)
"""

import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image
from ur7e_interfaces.msg import CameraMetrics


class CameraInfoNode(Node):
    """Monitors camera health and broadcasts intrinsic/runtime metrics."""

    def __init__(self):
        super().__init__("camera_info_node")

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("output_topic", "/camera/info_display")
        self.declare_parameter("publish_interval_sec", 1.0)
        self.declare_parameter("health_timeout_sec", 2.0)

        self._info_topic = self.get_parameter("camera_info_topic").value
        self._image_topic = self.get_parameter("image_topic").value
        self._output_topic = self.get_parameter("output_topic").value
        self._publish_interval = self.get_parameter("publish_interval_sec").value
        self._health_timeout = self.get_parameter("health_timeout_sec").value

        # ── State ─────────────────────────────────────────────────────────────
        self._intrinsics_logged = False
        self._last_camera_info: CameraInfo | None = None
        self._frame_count = 0
        self._fps_window_frames = 0
        self._fps_window_start = time.monotonic()
        self._fps_measured = 0.0
        self._last_image_time = time.monotonic()

        # ── QoS ───────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Pub / Sub ─────────────────────────────────────────────────────────
        self._info_sub = self.create_subscription(
            CameraInfo, self._info_topic, self._info_callback, sensor_qos
        )
        self._image_sub = self.create_subscription(
            Image, self._image_topic, self._image_callback, sensor_qos
        )
        self._pub = self.create_publisher(CameraMetrics, self._output_topic, 10)

        # Timer to publish metrics and check health periodically
        self._timer = self.create_timer(self._publish_interval, self._publish_metrics)

        self.get_logger().info(
            f"camera_info_node started\n"
            f"  monitoring → {self._info_topic}\n"
            f"  publishing → {self._output_topic}"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _info_callback(self, msg: CameraInfo) -> None:
        self._last_camera_info = msg
        if not self._intrinsics_logged:
            self._log_intrinsics(msg)
            self._intrinsics_logged = True

    def _image_callback(self, msg: Image) -> None:
        """Track incoming frame rate."""
        self._frame_count += 1
        self._fps_window_frames += 1
        self._last_image_time = time.monotonic()

        # Recalculate FPS every second of wall time
        elapsed = time.monotonic() - self._fps_window_start
        if elapsed >= 1.0:
            self._fps_measured = self._fps_window_frames / elapsed
            self._fps_window_frames = 0
            self._fps_window_start = time.monotonic()

    def _publish_metrics(self) -> None:
        """Publish CameraMetrics; check for camera health timeout."""
        age = time.monotonic() - self._last_image_time
        is_healthy = (self._frame_count > 0) and (age < self._health_timeout)

        msg = CameraMetrics()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._last_camera_info.header.frame_id if self._last_camera_info is not None else "camera_color_optical_frame"
        msg.fps_measured = float(self._fps_measured)
        msg.frame_count = self._frame_count
        msg.is_healthy = is_healthy

        if not is_healthy and self._frame_count > 0:
            msg.status_message = f"Camera stale: no frame for {age:.1f}s"
            self.get_logger().warn(msg.status_message, throttle_duration_sec=5.0)
        elif self._frame_count == 0:
            msg.status_message = "Waiting for first frame..."
        else:
            msg.status_message = f"OK  {self._fps_measured:.1f} fps"

        if self._last_camera_info is not None:
            k = self._last_camera_info.k  # 3x3 row-major intrinsic matrix
            msg.fx = k[0]
            msg.fy = k[4]
            msg.cx = k[2]
            msg.cy = k[5]
            msg.distortion_coeffs = list(self._last_camera_info.d[:5])

        self._pub.publish(msg)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_intrinsics(self, msg: CameraInfo) -> None:
        k = msg.k
        self.get_logger().info(
            f"\n{'─'*50}\n"
            f"  Camera intrinsics received\n"
            f"  Resolution : {msg.width} × {msg.height}\n"
            f"  Frame ID   : {msg.header.frame_id}\n"
            f"  fx={k[0]:.2f}  fy={k[4]:.2f}\n"
            f"  cx={k[2]:.2f}  cy={k[5]:.2f}\n"
            f"  Distortion : {list(msg.d)}\n"
            f"  Model      : {msg.distortion_model}\n"
            f"{'─'*50}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CameraInfoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
