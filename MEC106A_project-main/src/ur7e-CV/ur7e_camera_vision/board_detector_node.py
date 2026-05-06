"""
board_detector_node - RealSense-backed Operation board/piece detection for UR7e.

This node replaces the old placeholder CV with the reusable features from the
standalone/non-experiment computer-vision file, but runs them as ROS2 callbacks:
  - RealSense color + aligned depth + CameraInfo subscriptions
  - HSV yellow object detection
  - bilateral denoising, circularity filtering, solidity filtering
  - optional silver cavity detection/matching
  - RealSense depth sampling at detections
  - pixel/depth to 3D camera-frame projection
  - Kalman tracking/smoothing
  - TF transform to base_link when the camera-to-robot TF chain exists

Service compatibility is preserved through ur7e_interfaces/srv/DetectBoard.
The response fields are still named board_center_x/y/z, but the returned point is
now the selected Operation target/piece center in the camera optical frame.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, Pose, PoseArray, PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
import tf2_geometry_msgs
import tf2_ros

from ur7e_interfaces.srv import DetectBoard
from ur7e_camera_vision.operation_vision_core import (
    CameraProjection,
    OperationVisionProcessor,
    PieceTracker,
    TrackedDetection,
    average_camera_points,
    depth_image_to_meters,
    load_config,
)
from ur7e_camera_vision.tf_pose_utils import (
    identity_quaternion,
    make_point_stamped,
    make_pose_stamped,
)


class BoardDetectorNode(Node):
    """Detect and publish Operation-game target positions from RealSense data."""

    def __init__(self) -> None:
        super().__init__("board_detector_node")

        # ── RealSense topics ────────────────────────────────────────────────
        # Defaults match the RealSense ROS2 launch style used in the lab when
        # camera_name:=camera is set. Override these if your camera publishes
        # the shorter /camera/color/... topic names.
        self.declare_parameter("camera_image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("camera_color_topic", "/camera/camera/color/image_raw")  # legacy alias
        self.declare_parameter("camera_depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("base_frame", "base_link")

        # ── Detection/config parameters ─────────────────────────────────────
        self.declare_parameter("config_path", "operation_hsv_config.json")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("use_depth_for_3d", True)
        self.declare_parameter("assumed_board_depth", 0.50)  # fallback, metres
        self.declare_parameter("depth_window_px", 9)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 2.00)
        self.declare_parameter("use_cavity_filter", False)
        self.declare_parameter("track_max_distance_m", 0.06)
        self.declare_parameter("track_max_age_frames", 30)
        self.declare_parameter("target_policy", "highest_confidence")  # highest_confidence | mean | nearest

        # Optional live overrides for common HSV values. If left negative, JSON/default is used.
        self.declare_parameter("h_low", -1)
        self.declare_parameter("h_high", -1)
        self.declare_parameter("s_low", -1)
        self.declare_parameter("s_high", -1)
        self.declare_parameter("v_low", -1)
        self.declare_parameter("v_high", -1)
        self.declare_parameter("min_area", -1)
        self.declare_parameter("min_circularity", -1.0)
        self.declare_parameter("min_solidity", -1.0)

        # ── Read parameters ─────────────────────────────────────────────────
        image_topic = self.get_parameter("camera_image_topic").value
        legacy_color_topic = self.get_parameter("camera_color_topic").value
        # Prefer camera_image_topic when it has been explicitly passed; keep the
        # camera_color_topic alias for older launch files.
        self.color_topic = image_topic or legacy_color_topic
        self.depth_topic = self.get_parameter("camera_depth_topic").value
        self.info_topic = self.get_parameter("camera_info_topic").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.use_depth = bool(self.get_parameter("use_depth_for_3d").value)
        self.assumed_depth_m = float(self.get_parameter("assumed_board_depth").value)
        self.depth_window_px = int(self.get_parameter("depth_window_px").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.use_cavity_filter = bool(self.get_parameter("use_cavity_filter").value)
        self.target_policy = str(self.get_parameter("target_policy").value)

        config = load_config(str(self.get_parameter("config_path").value))
        self._apply_parameter_overrides(config)

        # ── CV state ────────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.processor = OperationVisionProcessor(config)
        self.projection = CameraProjection()
        self.tracker = PieceTracker(
            max_distance_m=float(self.get_parameter("track_max_distance_m").value),
            max_age_frames=int(self.get_parameter("track_max_age_frames").value),
        )

        self.latest_color_msg: Optional[Image] = None
        self.latest_depth_msg: Optional[Image] = None
        self.latest_bgr_image: Optional[np.ndarray] = None
        self.latest_depth_m: Optional[np.ndarray] = None
        self.latest_camera_info: Optional[CameraInfo] = None
        self.latest_tracked: List[TrackedDetection] = []
        self.latest_target_camera: Optional[Tuple[float, float, float]] = None
        self.latest_target_stamp = None

        self._got_color = False
        self._got_depth = False
        self._got_info = False

        # ── TF setup ────────────────────────────────────────────────────────
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── QoS: RealSense sensor streams are usually best-effort ───────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ── Subscribers ─────────────────────────────────────────────────────
        self.create_subscription(Image, self.color_topic, self._on_color_image, sensor_qos)
        self.create_subscription(Image, self.depth_topic, self._on_depth_image, sensor_qos)
        self.create_subscription(CameraInfo, self.info_topic, self._on_camera_info, sensor_qos)

        # ── Service compatibility ───────────────────────────────────────────
        self.create_service(DetectBoard, "detect_board", self._on_detect_board_request)

        # ── Publishers ──────────────────────────────────────────────────────
        # Keep older names for compatibility, and add clearer Operation names.
        self.board_center_pub = self.create_publisher(PointStamped, "board_center", 10)
        self.detected_board_pose_pub = self.create_publisher(PoseStamped, "detected_board_pose", 10)
        self.board_pose_camera_pub = self.create_publisher(PoseStamped, "/board_pose_camera", 10)
        self.board_pose_base_pub = self.create_publisher(PoseStamped, "/board_pose_base", 10)
        self.target_camera_pub = self.create_publisher(PointStamped, "/operation/target_camera", 10)
        self.target_base_pub = self.create_publisher(PointStamped, "/operation/target_base", 10)
        self.pieces_camera_pub = self.create_publisher(PoseArray, "/operation/pieces_camera", 10)
        self.pieces_base_pub = self.create_publisher(PoseArray, "/operation/pieces_base", 10)
        self.debug_image_pub = self.create_publisher(Image, "/operation/debug_image", 10)

        publish_rate_hz = max(0.1, float(self.get_parameter("publish_rate_hz").value))
        self.timer = self.create_timer(1.0 / publish_rate_hz, self._process_latest_frame)

        self.get_logger().info(
            "\nboard_detector_node refactored for RealSense + Operation CV\n"
            f"  color:       {self.color_topic}\n"
            f"  depth:       {self.depth_topic}\n"
            f"  camera_info: {self.info_topic}\n"
            f"  camera_frame:{self.camera_frame}\n"
            f"  base_frame:  {self.base_frame}\n"
            f"  use_depth:   {self.use_depth}\n"
            f"  cavity_filter:{self.use_cavity_filter}\n"
            f"  target_policy:{self.target_policy}"
        )

    # ────────────────────────────────────────────────────────────────────────
    # ROS callbacks
    # ────────────────────────────────────────────────────────────────────────

    def _apply_parameter_overrides(self, config: dict) -> None:
        """Let launch/YAML override the JSON config without requiring a new file."""
        overrides = {
            "h_low": self.get_parameter("h_low").value,
            "h_high": self.get_parameter("h_high").value,
            "s_low": self.get_parameter("s_low").value,
            "s_high": self.get_parameter("s_high").value,
            "v_low": self.get_parameter("v_low").value,
            "v_high": self.get_parameter("v_high").value,
            "min_area": self.get_parameter("min_area").value,
            "min_circularity": self.get_parameter("min_circularity").value,
            "min_solidity": self.get_parameter("min_solidity").value,
        }
        for key, value in overrides.items():
            if isinstance(value, (int, float)) and value >= 0:
                config[key] = value

    def _on_color_image(self, msg: Image) -> None:
        self.latest_color_msg = msg
        try:
            self.latest_bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"color image conversion failed: {exc}", throttle_duration_sec=5.0)
            return

        if not self._got_color:
            self._got_color = True
            self.get_logger().info(f"✓ receiving RealSense color images on {self.color_topic}")

    def _on_depth_image(self, msg: Image) -> None:
        self.latest_depth_msg = msg
        try:
            raw_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.latest_depth_m = depth_image_to_meters(raw_depth)
        except CvBridgeError as exc:
            self.get_logger().error(f"depth image conversion failed: {exc}", throttle_duration_sec=5.0)
            return

        if not self._got_depth:
            self._got_depth = True
            self.get_logger().info(f"✓ receiving RealSense depth images on {self.depth_topic}")

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg
        try:
            self.projection.update_from_camera_info(msg)
        except ValueError as exc:
            self.get_logger().error(str(exc), throttle_duration_sec=5.0)
            return

        if not self._got_info:
            self._got_info = True
            k = getattr(msg, "k", [0.0] * 9)
            self.get_logger().info(
                f"✓ camera intrinsics: {msg.width}x{msg.height}, "
                f"fx={k[0]:.1f}, fy={k[4]:.1f}, cx={k[2]:.1f}, cy={k[5]:.1f}"
            )

    def _on_detect_board_request(
        self, request: DetectBoard.Request, response: DetectBoard.Response
    ) -> DetectBoard.Response:
        """Return the latest selected target point in the camera frame."""
        if not bool(request.trigger):
            response.success = False
            response.message = "request.trigger was false"
            return response

        # Process once immediately so service calls do not depend only on timer timing.
        self._process_latest_frame()

        if self.latest_target_camera is None:
            response.success = False
            response.message = "no Operation target detected yet"
            return response

        x, y, z = self.latest_target_camera
        response.success = True
        response.board_center_x = float(x)
        response.board_center_y = float(y)
        response.board_center_z = float(z)
        response.message = (
            f"target detected in {self.camera_frame}: "
            f"x={x:.3f}, y={y:.3f}, z={z:.3f}; tracks={len(self.latest_tracked)}"
        )
        return response

    # ────────────────────────────────────────────────────────────────────────
    # Processing and publishing
    # ────────────────────────────────────────────────────────────────────────

    def _process_latest_frame(self) -> None:
        if self.latest_bgr_image is None or self.latest_color_msg is None:
            return
        if not self.projection.has_intrinsics:
            self.get_logger().warn("waiting for RealSense CameraInfo", throttle_duration_sec=2.0)
            return

        depth_image_m = self.latest_depth_m if self.use_depth else None
        if self.use_depth and depth_image_m is None:
            self.get_logger().warn(
                "waiting for aligned RealSense depth; using fallback depth until it arrives",
                throttle_duration_sec=5.0,
            )

        cavities = self.processor.detect_cavities(self.latest_bgr_image)
        detections_2d, debug_image = self.processor.detect_pieces(
            self.latest_bgr_image,
            cavities=cavities,
            use_cavity_filter=self.use_cavity_filter,
        )
        detections_3d = self.processor.project_detections(
            detections_2d,
            projection=self.projection,
            depth_image_m=depth_image_m,
            assumed_depth_m=self.assumed_depth_m,
            depth_window_px=self.depth_window_px,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
        )
        tracked = self.tracker.update(detections_3d)
        self.latest_tracked = tracked

        # Overlay 3D track labels on the debug image.
        for tracked_det in tracked:
            x_m, y_m, z_m = tracked_det.point_camera_m
            if z_m <= 1e-6:
                continue
            # Reproject for display.
            u = int(round((x_m / z_m) * self.projection.fx + self.projection.cx))  # type: ignore[operator]
            v = int(round((y_m / z_m) * self.projection.fy + self.projection.cy))  # type: ignore[operator]
            h, w = debug_image.shape[:2]
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(debug_image, (u, v), 16, (0, 255, 0), 2)
                cv2.putText(
                    debug_image,
                    f"ID:{tracked_det.track_id} z:{z_m:.2f}m c:{tracked_det.confidence:.1f}",
                    (u - 35, v - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (0, 255, 0),
                    1,
                )

        self._publish_debug_image(debug_image)
        self._publish_piece_arrays(tracked)

        target_camera = self._select_target_camera(tracked)
        self.latest_target_camera = target_camera
        self.latest_target_stamp = self.latest_color_msg.header.stamp
        if target_camera is not None:
            self._publish_target(target_camera)

    def _select_target_camera(self, tracked: List[TrackedDetection]) -> Optional[Tuple[float, float, float]]:
        if not tracked:
            return None

        if self.target_policy == "mean":
            return average_camera_points([t.point_camera_m for t in tracked])

        if self.target_policy == "nearest":
            nearest = min(tracked, key=lambda t: t.point_camera_m[2])
            return nearest.point_camera_m

        # Default: highest confidence, then lowest track ID.
        best = sorted(tracked, key=lambda t: (-t.confidence, t.track_id))[0]
        return best.point_camera_m

    def _publish_debug_image(self, debug_image: np.ndarray) -> None:
        try:
            msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
            msg.header = self.latest_color_msg.header  # type: ignore[union-attr]
            self.debug_image_pub.publish(msg)
        except CvBridgeError as exc:
            self.get_logger().error(f"debug image publish failed: {exc}", throttle_duration_sec=5.0)

    def _publish_piece_arrays(self, tracked: List[TrackedDetection]) -> None:
        if self.latest_color_msg is None:
            return
        camera_array = PoseArray()
        camera_array.header.frame_id = self.camera_frame
        camera_array.header.stamp = self.latest_color_msg.header.stamp

        for tracked_det in tracked:
            pose = Pose()
            pose.position.x = float(tracked_det.point_camera_m[0])
            pose.position.y = float(tracked_det.point_camera_m[1])
            pose.position.z = float(tracked_det.point_camera_m[2])
            pose.orientation = identity_quaternion()
            camera_array.poses.append(pose)

        self.pieces_camera_pub.publish(camera_array)

        base_array = self._try_transform_pose_array(camera_array, self.base_frame)
        if base_array is not None:
            self.pieces_base_pub.publish(base_array)

    def _publish_target(self, target_camera: Tuple[float, float, float]) -> None:
        if self.latest_color_msg is None:
            return
        x, y, z = target_camera
        stamp = self.latest_color_msg.header.stamp

        point_camera = make_point_stamped(x, y, z, self.camera_frame, timestamp=stamp)
        pose_camera = make_pose_stamped(
            x, y, z, self.camera_frame, orientation_quat=identity_quaternion(), timestamp=stamp
        )
        self.board_center_pub.publish(point_camera)
        self.detected_board_pose_pub.publish(pose_camera)
        self.board_pose_camera_pub.publish(pose_camera)
        self.target_camera_pub.publish(point_camera)

        point_base = self._try_transform_point(point_camera, self.base_frame)
        if point_base is not None:
            self.target_base_pub.publish(point_base)

        pose_base = self._try_transform_pose(pose_camera, self.base_frame)
        if pose_base is not None:
            self.board_pose_base_pub.publish(pose_base)

    def _try_transform_point(self, point: PointStamped, target_frame: str) -> Optional[PointStamped]:
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                point.header.frame_id,
                Time(),
                timeout=Duration(seconds=0.15),
            )
            return tf2_geometry_msgs.do_transform_point(point, transform)
        except Exception as exc:
            self.get_logger().warn(f"point TF unavailable {point.header.frame_id}->{target_frame}: {exc}", throttle_duration_sec=3.0)
            return None

    def _try_transform_pose(self, pose: PoseStamped, target_frame: str) -> Optional[PoseStamped]:
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                pose.header.frame_id,
                Time(),
                timeout=Duration(seconds=0.15),
            )
            return tf2_geometry_msgs.do_transform_pose(pose, transform)
        except Exception as exc:
            self.get_logger().warn(f"pose TF unavailable {pose.header.frame_id}->{target_frame}: {exc}", throttle_duration_sec=3.0)
            return None

    def _try_transform_pose_array(self, pose_array: PoseArray, target_frame: str) -> Optional[PoseArray]:
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                pose_array.header.frame_id,
                Time(),
                timeout=Duration(seconds=0.15),
            )
        except Exception:
            return None

        out = PoseArray()
        out.header.frame_id = target_frame
        out.header.stamp = pose_array.header.stamp
        for pose in pose_array.poses:
            stamped = PoseStamped()
            stamped.header = pose_array.header
            stamped.pose = pose
            transformed = tf2_geometry_msgs.do_transform_pose(stamped, transform)
            out.poses.append(transformed.pose)
        return out


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BoardDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
