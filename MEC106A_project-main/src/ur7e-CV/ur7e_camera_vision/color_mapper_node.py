"""
color_mapper_node - Convert RealSense images to false-color representations.

Subscribes to RealSense D435i streams and applies OpenCV colormaps for visualization.
Useful for displaying depth maps or highlighting specific color ranges in the image.

RealSense Topics (configurable via launch):
    Input:  /camera/camera/color/image_raw        (RGB) or /camera/camera/depth/image_rect_raw (depth)
    Output: /camera/colormap/image         (BGR8, false-color)

Parameters:
    input_topic    - RealSense topic to colormap (default: /camera/camera/color/image_raw)
    output_topic   - Where to publish colored result
    colormap_id    - OpenCV colormap constant (2=JET, 11=HOT, 14=INFERNO, 16=VIRIDIS)
    normalize_input - Normalize image before colormapping (good for depth)
    publish_rate_hz - Rate limit in Hz (0=unlimited)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


# Map of human-readable names to OpenCV constants for easy parameter docs
COLORMAP_NAMES = {
    0: "AUTUMN", 1: "BONE", 2: "JET", 3: "WINTER", 4: "RAINBOW",
    5: "OCEAN", 6: "SUMMER", 7: "SPRING", 8: "COOL", 9: "HSV",
    10: "PINK", 11: "HOT", 12: "PARULA", 13: "MAGMA", 14: "INFERNO",
    15: "PLASMA", 16: "VIRIDIS", 17: "CIVIDIS", 18: "TWILIGHT",
}


class ColorMapperNode(Node):
    """Converts raw/depth images to false-color representations."""

    def __init__(self):
        super().__init__("color_mapper_node")

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("input_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("output_topic", "/camera/colormap/image")
        self.declare_parameter("colormap_id", 2)          # COLORMAP_JET
        self.declare_parameter("normalize_input", False)   # set True for depth
        self.declare_parameter("publish_rate_hz", 0.0)     # 0 = unlimited

        self._input_topic = self.get_parameter("input_topic").value
        self._output_topic = self.get_parameter("output_topic").value
        self._colormap_id = self.get_parameter("colormap_id").value
        self._normalize = self.get_parameter("normalize_input").value
        self._rate_hz = self.get_parameter("publish_rate_hz").value

        # ── State ─────────────────────────────────────────────────────────────
        self._bridge = CvBridge()
        self._frame_count = 0
        self._last_publish_time = self.get_clock().now()

        # ── QoS: best-effort matches camera drivers ────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Pub / Sub ─────────────────────────────────────────────────────────
        self._sub = self.create_subscription(
            Image,
            self._input_topic,
            self._image_callback,
            sensor_qos,
        )
        self._pub = self.create_publisher(Image, self._output_topic, 10)

        # ── Parameter change callback ──────────────────────────────────────
        self.add_on_set_parameters_callback(self._on_parameter_change)

        colormap_name = COLORMAP_NAMES.get(self._colormap_id, "UNKNOWN")
        self.get_logger().info(
            f"color_mapper_node started\n"
            f"  input  → {self._input_topic}\n"
            f"  output → {self._output_topic}\n"
            f"  colormap: {colormap_name} (id={self._colormap_id})\n"
            f"  normalize: {self._normalize}"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _image_callback(self, msg: Image) -> None:
        """Receive a raw image, apply colormap, and republish."""
        # Rate limiting
        if self._rate_hz > 0.0:
            now = self.get_clock().now()
            dt = (now - self._last_publish_time).nanoseconds * 1e-9
            if dt < (1.0 / self._rate_hz):
                return
            self._last_publish_time = now

        try:
            # Convert ROS Image → OpenCV
            # Use passthrough to handle 16-bit depth images correctly
            cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as e:
            self.get_logger().error(f"cv_bridge error: {e}", throttle_duration_sec=5.0)
            return

        try:
            colormap_image = self._apply_colormap(cv_image)
        except Exception as e:
            self.get_logger().error(f"Colormap error: {e}", throttle_duration_sec=5.0)
            return

        try:
            out_msg = self._bridge.cv2_to_imgmsg(colormap_image, encoding="bgr8")
            out_msg.header = msg.header   # preserve original timestamp and frame_id
            self._pub.publish(out_msg)
            self._frame_count += 1
        except CvBridgeError as e:
            self.get_logger().error(f"Publish error: {e}", throttle_duration_sec=5.0)

    def _apply_colormap(self, image: np.ndarray) -> np.ndarray:
        """
        Convert any image to an 8-bit grayscale and apply the chosen colormap.
        Handles: uint8 RGB/BGR, uint16 depth (millimetres), float32 depth.
        """
        # Convert to single-channel for colormap
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Normalize to 0-255 uint8
        if gray.dtype == np.uint16:
            # Depth images: clip at 4m, scale to 8-bit
            if self._normalize:
                cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
                gray = gray.astype(np.uint8)
            else:
                gray = np.clip(gray / 16, 0, 255).astype(np.uint8)
        elif gray.dtype == np.float32 or gray.dtype == np.float64:
            cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
            gray = gray.astype(np.uint8)
        elif gray.dtype != np.uint8:
            cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)
            gray = gray.astype(np.uint8)

        if self._normalize and gray.dtype == np.uint8:
            cv2.normalize(gray, gray, 0, 255, cv2.NORM_MINMAX)

        return cv2.applyColorMap(gray, self._colormap_id)

    def _on_parameter_change(self, params):
        """Allow live parameter updates via ros2 param set."""
        from rcl_interfaces.msg import SetParametersResult
        for param in params:
            if param.name == "colormap_id":
                self._colormap_id = param.value
                name = COLORMAP_NAMES.get(self._colormap_id, "UNKNOWN")
                self.get_logger().info(f"Colormap changed to {name} (id={self._colormap_id})")
            elif param.name == "normalize_input":
                self._normalize = param.value
                self.get_logger().info(f"normalize_input set to {self._normalize}")
            elif param.name == "publish_rate_hz":
                self._rate_hz = param.value
        return SetParametersResult(successful=True)


def main(args=None):
    rclpy.init(args=args)
    node = ColorMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
