#!/usr/bin/env python3
"""
ros2 prototype scaffold for board / piece detection + tf2 + moveit-style motion.

this file is meant to replace the standalone cv2 loop with ros2 nodes while keeping
most of the useful detection ideas from the original prototype:
- hsv thresholding
- circularity / solidity filtering
- rough pixel-to-3d projection
- kalman smoothing

notes:
- this is prototype code, not final robot-safe code.
- moveit2 control is intentionally wrapped as a placeholder so the file can still
  be read/run in a simple ros2 python package before your exact moveit python api
  is finalized.
- for real robot motion, test in simulation first, then add safety limits,
  collision objects, speed limits, and an e-stop workflow.
"""

import json
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.time import Time

from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped, Quaternion
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from std_srvs.srv import Trigger

import tf2_ros
from tf2_geometry_msgs import do_transform_pose


CONFIG_FILE = "operation_hsv_config.json"
SNAPSHOT_DIR = "snapshots"


# -----------------------------------------------------------------------------
# config helpers
# -----------------------------------------------------------------------------

def get_default_config() -> dict:
    """default hsv range for bright yellow operation-game pieces."""
    return {
        "h_low": 15,
        "h_high": 35,
        "s_low": 100,
        "s_high": 255,
        "v_low": 100,
        "v_high": 255,
        "min_area": 100,
        "min_circularity": 0.5,
        "min_solidity": 0.7,
        "safe_height_above_board_m": 0.15,
        "assumed_board_z_m": 0.50,
    }


def load_config(config_path: str = CONFIG_FILE) -> dict:
    """load hsv / detection config from json, or use defaults."""
    if not os.path.exists(config_path):
        return get_default_config()

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        config = get_default_config()
        config.update(loaded)
        return config
    except Exception as exc:
        print(f"error loading config, using defaults: {exc}")
        return get_default_config()


# -----------------------------------------------------------------------------
# detection and tracking helpers
# -----------------------------------------------------------------------------

@dataclass
class Detection2D:
    """one raw image-space detection."""
    center_px: Tuple[int, int]
    area_px: float
    circularity: float
    solidity: float


@dataclass
class Detection3D:
    """one detection projected into the camera frame."""
    center_px: Tuple[int, int]
    point_camera_m: Tuple[float, float, float]
    area_px: float


class KalmanFilter1D:
    """small 1d kalman filter for smoothing noisy measurements."""

    def __init__(self, process_variance=1e-4, measurement_variance=1e-1, initial_value=0.0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = initial_value
        self.posteri_error_estimate = 1.0
        self.priori_estimate = 0.0
        self.priori_error_estimate = 1.0

    def update(self, measurement: float) -> float:
        # predict
        self.priori_estimate = self.posteri_estimate
        self.priori_error_estimate = self.posteri_error_estimate + self.process_variance

        # update
        kalman_gain = self.priori_error_estimate / (
            self.priori_error_estimate + self.measurement_variance
        )
        self.posteri_estimate = self.priori_estimate + kalman_gain * (
            measurement - self.priori_estimate
        )
        self.posteri_error_estimate = (1.0 - kalman_gain) * self.priori_error_estimate
        return self.posteri_estimate


class PieceTracker:
    """track projected 3d detections across frames."""

    def __init__(self, max_distance_m=0.05, history_length=10, max_age_frames=30):
        self.tracks = {}
        self.next_id = 0
        self.max_distance_m = max_distance_m
        self.history_length = history_length
        self.max_age_frames = max_age_frames

    def update(self, detections: List[Detection3D]) -> List[Tuple[int, float, float, float, float]]:
        """
        return list of (track_id, filtered_x_m, filtered_y_m, filtered_z_m, confidence).
        """
        if not detections:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]["age"] += 1
                if self.tracks[track_id]["age"] > self.max_age_frames:
                    del self.tracks[track_id]
            return []

        matched_tracks = set()
        results = []

        for det in detections:
            det_x, det_y, det_z = det.point_camera_m
            best_track_id = None
            best_distance = self.max_distance_m

            for track_id, track_data in list(self.tracks.items()):
                if track_id in matched_tracks:
                    continue

                last_x = track_data["kf_x"].posteri_estimate
                last_y = track_data["kf_y"].posteri_estimate
                last_z = track_data["kf_z"].posteri_estimate
                distance = math.sqrt(
                    (det_x - last_x) ** 2 + (det_y - last_y) ** 2 + (det_z - last_z) ** 2
                )

                if distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id

            if best_track_id is None:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = {
                    "kf_x": KalmanFilter1D(initial_value=det_x),
                    "kf_y": KalmanFilter1D(initial_value=det_y),
                    "kf_z": KalmanFilter1D(initial_value=det_z),
                    "positions": deque([(det_x, det_y, det_z)], maxlen=self.history_length),
                    "age": 0,
                    "hits": 1,
                }
                results.append((track_id, det_x, det_y, det_z, 0.0))
                continue

            matched_tracks.add(best_track_id)
            track = self.tracks[best_track_id]
            track["age"] = 0
            track["hits"] += 1

            filtered_x = track["kf_x"].update(det_x)
            filtered_y = track["kf_y"].update(det_y)
            filtered_z = track["kf_z"].update(det_z)
            track["positions"].append((filtered_x, filtered_y, filtered_z))

            confidence = min(track["hits"] / 10.0, 1.0)
            results.append((best_track_id, filtered_x, filtered_y, filtered_z, confidence))

        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks and self.tracks[track_id]["age"] > 0:
                self.tracks[track_id]["age"] += 1
                if self.tracks[track_id]["age"] > self.max_age_frames:
                    del self.tracks[track_id]

        return results


class BoardDetector:
    """hsv-based board/piece detector pulled out of the old while-loop."""

    def __init__(self, config: dict):
        self.config = config

    @staticmethod
    def calculate_circularity(contour) -> float:
        area = cv2.contourArea(contour)
        if area <= 0:
            return 0.0
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return 0.0
        return 4.0 * np.pi * area / (perimeter ** 2)

    @staticmethod
    def calculate_solidity(contour) -> float:
        area = cv2.contourArea(contour)
        if area <= 0:
            return 0.0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return 0.0
        return area / hull_area

    def detect(self, bgr_image: np.ndarray) -> Tuple[List[Detection2D], np.ndarray]:
        """run hsv detection and return detections plus debug image."""
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        lower = np.array([
            self.config["h_low"],
            self.config["s_low"],
            self.config["v_low"],
        ])
        upper = np.array([
            self.config["h_high"],
            self.config["s_high"],
            self.config["v_high"],
        ])

        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.bilateralFilter(mask, 9, 75, 75)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        debug_image = bgr_image.copy()

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config["min_area"]:
                continue

            circularity = self.calculate_circularity(contour)
            if circularity < self.config["min_circularity"]:
                continue

            solidity = self.calculate_solidity(contour)
            if solidity < self.config["min_solidity"]:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
            else:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

            detections.append(
                Detection2D(
                    center_px=(center_x, center_y),
                    area_px=area,
                    circularity=circularity,
                    solidity=solidity,
                )
            )

            cv2.drawContours(debug_image, [contour], -1, (0, 255, 255), 2)
            cv2.circle(debug_image, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(
                debug_image,
                f"area={area:.0f}",
                (center_x + 8, center_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )

        return detections, debug_image


# -----------------------------------------------------------------------------
# tf / camera projection helpers
# -----------------------------------------------------------------------------

class CameraProjection:
    """convert pixels into 3d camera-frame points using camera intrinsics."""

    def __init__(self):
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

    def update_from_camera_info(self, camera_info: CameraInfo) -> None:
        self.fx = camera_info.k[0]
        self.fy = camera_info.k[4]
        self.cx = camera_info.k[2]
        self.cy = camera_info.k[5]

    @property
    def has_intrinsics(self) -> bool:
        return all(v is not None and v > 0 for v in [self.fx, self.fy, self.cx, self.cy])

    def pixel_to_3d(self, u_px: float, v_px: float, z_m: float) -> Tuple[float, float, float]:
        """project one pixel to camera coordinates at assumed depth z_m."""
        if not self.has_intrinsics:
            raise RuntimeError("camera intrinsics are not available yet")

        x_m = (u_px - self.cx) * z_m / self.fx
        y_m = (v_px - self.cy) * z_m / self.fy
        return x_m, y_m, z_m


def make_pose_stamped(
    frame_id: str,
    stamp,
    xyz: Tuple[float, float, float],
    quat_xyzw: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> PoseStamped:
    pose = PoseStamped()
    pose.header = Header()
    pose.header.frame_id = frame_id
    pose.header.stamp = stamp
    pose.pose.position = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
    pose.pose.orientation = Quaternion(
        x=float(quat_xyzw[0]),
        y=float(quat_xyzw[1]),
        z=float(quat_xyzw[2]),
        w=float(quat_xyzw[3]),
    )
    return pose


def average_points(points: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
    arr = np.array(points, dtype=float)
    return tuple(np.mean(arr, axis=0).tolist())


# -----------------------------------------------------------------------------
# ros2 node 1: detect board / pieces from camera
# -----------------------------------------------------------------------------

class BoardDetectorNode(Node):
    """
    subscribes to camera image + camera info, detects pieces, and publishes board pose.

    service:
    - /capture_board_snapshot: saves latest image and runs detection once

    publishers:
    - /board_pose_camera: approximate board pose in camera frame
    - /board_pose_base: board pose transformed into base frame when tf is available
    - /board_debug_image: debug image with detections drawn on top
    """

    def __init__(self):
        super().__init__("board_detector_node")

        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("config_path", CONFIG_FILE)
        self.declare_parameter("snapshot_dir", SNAPSHOT_DIR)
        self.declare_parameter("publish_rate_hz", 5.0)

        self.image_topic = self.get_parameter("image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.snapshot_dir = self.get_parameter("snapshot_dir").value

        config_path = self.get_parameter("config_path").value
        self.config = load_config(config_path)

        self.bridge = CvBridge()
        self.detector = BoardDetector(self.config)
        self.projection = CameraProjection()
        self.tracker = PieceTracker(max_distance_m=0.05)

        self.latest_image_msg: Optional[Image] = None
        self.latest_bgr_image: Optional[np.ndarray] = None
        self.latest_camera_info: Optional[CameraInfo] = None

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.board_pose_camera_pub = self.create_publisher(PoseStamped, "/board_pose_camera", 10)
        self.board_pose_base_pub = self.create_publisher(PoseStamped, "/board_pose_base", 10)
        self.debug_image_pub = self.create_publisher(Image, "/board_debug_image", 10)

        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.camera_info_callback,
            10,
        )

        self.capture_service = self.create_service(
            Trigger,
            "/capture_board_snapshot",
            self.capture_snapshot_callback,
        )

        timer_period = 1.0 / float(self.get_parameter("publish_rate_hz").value)
        self.timer = self.create_timer(timer_period, self.process_latest_frame)

        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.get_logger().info("board detector node started")

    def camera_info_callback(self, msg: CameraInfo) -> None:
        self.latest_camera_info = msg
        self.projection.update_from_camera_info(msg)

    def image_callback(self, msg: Image) -> None:
        self.latest_image_msg = msg
        try:
            self.latest_bgr_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"failed to convert image: {exc}")

    def capture_snapshot_callback(self, request, response):
        del request

        if self.latest_bgr_image is None:
            response.success = False
            response.message = "no image received yet"
            return response

        stamp = self.get_clock().now().nanoseconds
        snapshot_path = os.path.join(self.snapshot_dir, f"board_snapshot_{stamp}.png")
        cv2.imwrite(snapshot_path, self.latest_bgr_image)

        detections_2d, _ = self.detector.detect(self.latest_bgr_image)
        response.success = True
        response.message = f"saved {snapshot_path}; found {len(detections_2d)} detections"
        return response

    def process_latest_frame(self) -> None:
        if self.latest_bgr_image is None or self.latest_image_msg is None:
            return

        if not self.projection.has_intrinsics:
            self.get_logger().warn("waiting for camera_info intrinsics", throttle_duration_sec=2.0)
            return

        detections_2d, debug_image = self.detector.detect(self.latest_bgr_image)
        detections_3d = self.project_detections(detections_2d)
        tracked = self.tracker.update(detections_3d)

        if tracked:
            tracked_points = [(x, y, z) for _, x, y, z, _ in tracked]
            board_center_camera = average_points(tracked_points)

            pose_camera = make_pose_stamped(
                frame_id=self.camera_frame,
                stamp=self.latest_image_msg.header.stamp,
                xyz=board_center_camera,
            )
            self.board_pose_camera_pub.publish(pose_camera)

            pose_base = self.try_transform_pose(pose_camera, self.base_frame)
            if pose_base is not None:
                self.board_pose_base_pub.publish(pose_base)

        debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding="bgr8")
        debug_msg.header = self.latest_image_msg.header
        self.debug_image_pub.publish(debug_msg)

    def project_detections(self, detections_2d: List[Detection2D]) -> List[Detection3D]:
        detections_3d = []
        assumed_z_m = float(self.config["assumed_board_z_m"])

        for det in detections_2d:
            u_px, v_px = det.center_px

            # todo: replace assumed z with depth camera, calibrated board plane, or pnp.
            x_m, y_m, z_m = self.projection.pixel_to_3d(u_px, v_px, assumed_z_m)

            detections_3d.append(
                Detection3D(
                    center_px=det.center_px,
                    point_camera_m=(x_m, y_m, z_m),
                    area_px=det.area_px,
                )
            )

        return detections_3d

    def try_transform_pose(self, pose: PoseStamped, target_frame: str) -> Optional[PoseStamped]:
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                pose.header.frame_id,
                Time(),
                timeout=Duration(seconds=0.25),
            )
            return do_transform_pose(pose, transform)
        except Exception as exc:
            self.get_logger().warn(f"tf transform unavailable: {exc}", throttle_duration_sec=2.0)
            return None


# -----------------------------------------------------------------------------
# ros2 node 2: move above detected board pose
# -----------------------------------------------------------------------------

class MoveIt2PrototypeClient:
    """
    placeholder for your real moveit2 interface.

    depending on your stack, this could become:
    - pymoveit2 MoveIt2 object
    - moveit_py planning component
    - action client to /move_action
    - service wrapper around a c++ move_group node
    """

    def __init__(self, node: Node, planning_group: str, end_effector_frame: str):
        self.node = node
        self.planning_group = planning_group
        self.end_effector_frame = end_effector_frame

    def move_to_pose(self, target_pose_base: PoseStamped) -> bool:
        self.node.get_logger().info(
            "moveit placeholder: would plan/execute to "
            f"x={target_pose_base.pose.position.x:.3f}, "
            f"y={target_pose_base.pose.position.y:.3f}, "
            f"z={target_pose_base.pose.position.z:.3f} "
            f"in frame={target_pose_base.header.frame_id}"
        )

        # todo: replace with real moveit2 planning + execution call.
        # example shape, not guaranteed api:
        # self.moveit2.move_to_pose(
        #     position=[x, y, z],
        #     quat_xyzw=[qx, qy, qz, qw],
        #     cartesian=False,
        # )
        # self.moveit2.wait_until_executed()
        return True


class MoveAboveBoardNode(Node):
    """
    listens for board pose in base frame and exposes a service to move above it.

    service:
    - /move_above_board
    """

    def __init__(self):
        super().__init__("move_above_board_node")

        self.declare_parameter("board_pose_topic", "/board_pose_base")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("end_effector_frame", "tool0")
        self.declare_parameter("planning_group", "ur_manipulator")
        self.declare_parameter("safe_height_above_board_m", 0.15)

        self.board_pose_topic = self.get_parameter("board_pose_topic").value
        self.base_frame = self.get_parameter("base_frame").value
        self.end_effector_frame = self.get_parameter("end_effector_frame").value
        self.safe_height = float(self.get_parameter("safe_height_above_board_m").value)

        self.latest_board_pose_base: Optional[PoseStamped] = None

        planning_group = self.get_parameter("planning_group").value
        self.moveit_client = MoveIt2PrototypeClient(
            node=self,
            planning_group=planning_group,
            end_effector_frame=self.end_effector_frame,
        )

        self.board_pose_sub = self.create_subscription(
            PoseStamped,
            self.board_pose_topic,
            self.board_pose_callback,
            10,
        )

        self.move_service = self.create_service(
            Trigger,
            "/move_above_board",
            self.move_above_board_callback,
        )

        self.get_logger().info("move above board node started")

    def board_pose_callback(self, msg: PoseStamped) -> None:
        self.latest_board_pose_base = msg

    def move_above_board_callback(self, request, response):
        del request

        if self.latest_board_pose_base is None:
            response.success = False
            response.message = "no board pose received yet"
            return response

        target_pose_base = self.make_target_pose_above_board(self.latest_board_pose_base)
        success = self.moveit_client.move_to_pose(target_pose_base)

        response.success = bool(success)
        response.message = "sent move-above-board command" if success else "move command failed"
        return response

    def make_target_pose_above_board(self, board_pose_base: PoseStamped) -> PoseStamped:
        target_pose = PoseStamped()
        target_pose.header.frame_id = self.base_frame
        target_pose.header.stamp = self.get_clock().now().to_msg()

        target_pose.pose.position.x = board_pose_base.pose.position.x
        target_pose.pose.position.y = board_pose_base.pose.position.y
        target_pose.pose.position.z = board_pose_base.pose.position.z + self.safe_height

        # todo: set orientation to point the gripper/camera down at the board based on your ur tool frame.
        target_pose.pose.orientation.x = 0.0
        target_pose.pose.orientation.y = 0.0
        target_pose.pose.orientation.z = 0.0
        target_pose.pose.orientation.w = 1.0
        return target_pose


# -----------------------------------------------------------------------------
# simple single-file entrypoint
# -----------------------------------------------------------------------------

def main(args=None):
    rclpy.init(args=args)

    # run both nodes in one process for a simple prototype.
    # later, split this file into board_detector_node.py, tf_pose_utils.py,
    # move_above_board_node.py, and launch them separately.
    detector_node = BoardDetectorNode()
    motion_node = MoveAboveBoardNode()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(detector_node)
    executor.add_node(motion_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        detector_node.destroy_node()
        motion_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
