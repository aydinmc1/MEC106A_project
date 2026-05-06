"""
board_detector_node: ROS2 wrapper around PieceDetector.

All CV logic lives in piece_detector.py (zero ROS2 imports there).
This node only handles ROS2 plumbing: subscriptions, publishers, service.

Subscriptions:
  /camera/color/image_raw                  sensor_msgs/Image (BGR8)
  /camera/aligned_depth_to_color/image_raw sensor_msgs/Image (uint16 depth)
  /camera/color/camera_info                sensor_msgs/CameraInfo

Publishers:
  /piece_detections       std_msgs/String   — JSON array of all detections
  /detection_debug_image  sensor_msgs/Image — annotated colour frame
  board_center            geometry_msgs/PointStamped  (highest-confidence piece)
  detected_board_pose     geometry_msgs/PoseStamped   (same point as a pose)

Services:
  detect_board  ur_board_detection/DetectBoard
               Runs detection on demand; returns the highest-confidence piece.
               The response fields board_center_{x,y,z} hold the 3-D grasp
               point in the camera frame (meters).  piece_name and grasp
               metadata are embedded in the message string.
"""

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

import numpy as np
from cv_bridge import CvBridge

from geometry_msgs.msg import PointStamped, PoseStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from ur_board_detection.srv import DetectBoard
from ur_board_detection.tf_pose_utils import (
    identity_quaternion,
    make_point_stamped,
    make_pose_stamped,
)
from ur_board_detection.piece_detector import PieceDetection, PieceDetector


class BoardDetectorNode(Node):
    def __init__(self):
        super().__init__("board_detector_node")

        # --- Parameters ---
        self.declare_parameter("camera_image_topic",
                               "/camera/color/image_raw")
        self.declare_parameter("camera_depth_topic",
                               "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic",
                               "/camera/color/camera_info")
        self.declare_parameter("camera_frame",
                               "camera_color_optical_frame")
        self.declare_parameter("assumed_board_depth", 0.5)
        self.declare_parameter("config_path",         "operation_hsv_config.json")
        self.declare_parameter("publish_rate_hz",     10.0)

        color_topic   = self.get_parameter("camera_image_topic").value
        depth_topic   = self.get_parameter("camera_depth_topic").value
        info_topic    = self.get_parameter("camera_info_topic").value
        self.camera_frame = self.get_parameter("camera_frame").value
        config_path   = self.get_parameter("config_path").value

        # --- State ---
        self.latest_bgr:       np.ndarray | None = None
        self.latest_depth:     np.ndarray | None = None  # uint16 aligned depth
        self.latest_image_msg: Image | None      = None
        self.last_detections:  list[PieceDetection] = []
        self.cv_bridge = CvBridge()

        # --- Detector (pure CV, no ROS2) ---
        self.detector = PieceDetector(config_path=config_path)

        # --- QoS: best-effort / depth-1 for image streams ---
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # --- Subscriptions ---
        self.create_subscription(Image, color_topic,
                                 self._on_color_image, qos_profile=img_qos)
        self.create_subscription(Image, depth_topic,
                                 self._on_depth_image, qos_profile=img_qos)
        self.create_subscription(CameraInfo, info_topic,
                                 self._on_camera_info, qos_profile=10)

        # --- Publishers ---
        self.piece_det_pub    = self.create_publisher(String,       "/piece_detections",      10)
        self.debug_image_pub  = self.create_publisher(Image,        "/detection_debug_image", 10)
        self.board_center_pub = self.create_publisher(PointStamped, "board_center",           10)
        self.board_pose_pub   = self.create_publisher(PoseStamped,  "detected_board_pose",    10)

        # --- Service ---
        self.create_service(DetectBoard, "detect_board",
                            self._handle_detect_board_service)

        # --- Timer ---
        period = 1.0 / float(self.get_parameter("publish_rate_hz").value)
        self.create_timer(period, self._process_latest_frame)

        self.get_logger().info(
            f"board_detector_node ready  "
            f"color={color_topic}  depth={depth_topic}"
        )

    # ------------------------------------------------------------------
    # Subscription callbacks
    # ------------------------------------------------------------------

    def _on_color_image(self, msg: Image) -> None:
        self.latest_image_msg = msg
        try:
            self.latest_bgr = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"colour image conversion failed: {exc}")

    def _on_depth_image(self, msg: Image) -> None:
        try:
            # 'passthrough' preserves the uint16 pixel values exactly
            self.latest_depth = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as exc:
            self.get_logger().error(f"depth image conversion failed: {exc}")

    def _on_camera_info(self, msg: CameraInfo) -> None:
        K = msg.k
        self.detector.update_intrinsics(fx=K[0], fy=K[4], cx=K[2], cy=K[5])

    # ------------------------------------------------------------------
    # Timer: run detection every tick and publish results
    # ------------------------------------------------------------------

    def _process_latest_frame(self) -> None:
        if self.latest_bgr is None:
            return

        bgr   = self.latest_bgr
        depth = self.latest_depth  # None until the depth topic is received

        detections = self.detector.process_frame(bgr, depth)
        self.last_detections = detections

        # 1. Publish JSON detection array for downstream nodes
        self._publish_detection_json(detections)

        # 2. Publish annotated debug image
        # last_cavities / last_stubs are cached by process_frame()
        self._publish_debug_image(bgr, detections)

        # 3. Publish board-centre markers for the highest-confidence piece
        if detections:
            best = max(detections, key=lambda d: d.confidence)
            self._publish_board_markers(best)

    def _publish_detection_json(self, detections: list[PieceDetection]) -> None:
        payload = []
        for det in detections:
            x3, y3, z3 = det.grasp_point_3d
            payload.append({
                "piece_name":         det.piece_name,
                "grasp_px":           list(det.grasp_point_2d),
                "grasp_3d_m":         [round(x3, 4), round(y3, 4), round(z3, 4)],
                "grasp_depth_offset": det.grasp_depth_offset,
                "approach_axis":      det.approach_axis,
                "confidence":         round(det.confidence, 3),
            })
        self.piece_det_pub.publish(String(data=json.dumps(payload)))

    def _publish_debug_image(self, bgr, detections: list[PieceDetection]) -> None:
        # draw_debug() uses self.detector.last_cavities / last_stubs automatically
        debug_img = self.detector.draw_debug(bgr, detections)
        try:
            debug_msg = self.cv_bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
            if self.latest_image_msg is not None:
                debug_msg.header = self.latest_image_msg.header
            self.debug_image_pub.publish(debug_msg)
        except Exception as exc:
            self.get_logger().error(f"debug image publish failed: {exc}")

    def _publish_board_markers(self, det: PieceDetection) -> None:
        x3, y3, z3 = det.grasp_point_3d
        pt = make_point_stamped(x3, y3, z3, self.camera_frame)
        self.board_center_pub.publish(pt)
        pose = make_pose_stamped(x3, y3, z3, self.camera_frame,
                                 orientation_quat=identity_quaternion())
        self.board_pose_pub.publish(pose)

    # ------------------------------------------------------------------
    # DetectBoard service
    # ------------------------------------------------------------------

    def _handle_detect_board_service(self, request, response):
        """
        Run detection on the latest frame and return the highest-confidence
        piece's 3-D grasp point.  Piece name and grasp metadata are included
        in the response message string.
        """
        if not request.trigger:
            response.success = False
            response.message = "trigger flag not set"
            return response

        if self.latest_bgr is None:
            response.success = False
            response.message = "no image received yet"
            self.get_logger().warn("detect_board called but no image available")
            return response

        # Fresh detection on the current frame
        detections = self.detector.process_frame(self.latest_bgr, self.latest_depth)
        self.last_detections = detections

        if not detections:
            response.success = False
            response.message = "no pieces detected"
            return response

        best = max(detections, key=lambda d: d.confidence)
        x3, y3, z3 = best.grasp_point_3d

        response.success = True
        response.message = (
            f"piece={best.piece_name} "
            f"conf={best.confidence:.2f} "
            f"ax={best.approach_axis} "
            f"dz={best.grasp_depth_offset:.3f}m"
        )
        response.board_center_x = float(x3)
        response.board_center_y = float(y3)
        response.board_center_z = float(z3)

        self.get_logger().info(
            f"detect_board → {best.piece_name} at "
            f"({x3:.3f},{y3:.3f},{z3:.3f})m  conf={best.confidence:.2f}"
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = BoardDetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
