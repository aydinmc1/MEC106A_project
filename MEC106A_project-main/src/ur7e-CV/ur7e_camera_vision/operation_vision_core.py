"""
Reusable computer-vision helpers for the UR7e Operation project.

This module pulls the useful logic out of the standalone/non-experiment CV file
and makes it safe to call from ROS2 callbacks:
  - HSV thresholding for yellow pieces/board regions
  - circularity and solidity filtering
  - silver cavity detection and optional cavity matching
  - RealSense depth sampling around each detected pixel
  - pixel + depth -> 3D camera-frame projection
  - simple Kalman-based temporal smoothing/tracking
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from sensor_msgs.msg import CameraInfo

CONFIG_FILE = "operation_hsv_config.json"


def get_default_config() -> dict:
    """Default thresholds based on the standalone Operation CV prototype."""
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
        "silver_s_max": 80,
        "silver_v_min": 105,
        "cavity_min_area": 250,
        "cavity_max_area": 25000,
        "cavity_min_circularity": 0.08,
        "cavity_match_padding_px": 30,
    }


def resolve_config_path(config_path: str = CONFIG_FILE) -> Optional[str]:
    """Find a config file from cwd, this package, or the workspace src root."""
    if os.path.isabs(config_path) and os.path.exists(config_path):
        return config_path

    module_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(module_dir)
    workspace_src_dir = os.path.dirname(package_dir)

    candidates = [
        config_path,
        os.path.join(module_dir, config_path),
        os.path.join(package_dir, config_path),
        os.path.join(package_dir, "config", config_path),
        os.path.join(workspace_src_dir, config_path),
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def load_config(config_path: str = CONFIG_FILE) -> dict:
    """Load HSV/detection config; missing keys fall back to defaults."""
    config = get_default_config()
    resolved = resolve_config_path(config_path)
    if resolved is None:
        return config

    try:
        with open(resolved, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            config.update(loaded)
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return config


@dataclass
class Detection2D:
    """One image-space detection before depth projection."""

    center_px: Tuple[int, int]
    area_px: float
    circularity: float
    solidity: float
    bbox_xywh: Tuple[int, int, int, int]
    contour: np.ndarray


@dataclass
class Cavity2D:
    """One detected silver cavity/target outline in image space."""

    center_px: Tuple[int, int]
    radius_px: float
    area_px: float
    circularity: float
    contour: np.ndarray


@dataclass
class Detection3D:
    """One detection projected using RealSense depth and CameraInfo intrinsics."""

    center_px: Tuple[int, int]
    point_camera_m: Tuple[float, float, float]
    depth_m: float
    area_px: float
    circularity: float
    solidity: float


@dataclass
class TrackedDetection:
    """One temporally-smoothed 3D detection."""

    track_id: int
    point_camera_m: Tuple[float, float, float]
    confidence: float
    age: int


class CameraProjection:
    """Project aligned color pixels into the optical camera frame."""

    def __init__(self) -> None:
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

    @property
    def has_intrinsics(self) -> bool:
        return None not in (self.fx, self.fy, self.cx, self.cy)

    def update_from_camera_info(self, camera_info: CameraInfo) -> None:
        k = getattr(camera_info, "k", None)
        if k is None or len(k) < 6:
            k = getattr(camera_info, "K", None)
        if k is None or len(k) < 6:
            raise ValueError("CameraInfo is missing intrinsic matrix k/K")
        self.fx = float(k[0])
        self.fy = float(k[4])
        self.cx = float(k[2])
        self.cy = float(k[5])

    def pixel_to_3d(self, u_px: int, v_px: int, depth_m: float) -> Tuple[float, float, float]:
        if not self.has_intrinsics:
            raise RuntimeError("Camera intrinsics are not available yet")
        assert self.fx is not None and self.fy is not None
        assert self.cx is not None and self.cy is not None
        x_m = (float(u_px) - self.cx) * depth_m / self.fx
        y_m = (float(v_px) - self.cy) * depth_m / self.fy
        z_m = float(depth_m)
        return x_m, y_m, z_m


def depth_image_to_meters(depth_image: np.ndarray) -> np.ndarray:
    """
    Convert a RealSense depth image to meters.

    RealSense commonly publishes uint16 millimeters. Some pipelines publish float
    meters. This handles both.
    """
    if depth_image.dtype == np.uint16:
        return depth_image.astype(np.float32) * 0.001
    if depth_image.dtype == np.float32 or depth_image.dtype == np.float64:
        return depth_image.astype(np.float32)
    # Fallback: assume millimetres for integer-like images.
    return depth_image.astype(np.float32) * 0.001


def sample_depth_at_pixel(
    depth_image_m: np.ndarray,
    u_px: int,
    v_px: int,
    window_px: int = 7,
    min_depth_m: float = 0.05,
    max_depth_m: float = 2.0,
) -> Optional[float]:
    """Return median valid depth around one pixel, or None if invalid."""
    if depth_image_m is None or depth_image_m.size == 0:
        return None

    h, w = depth_image_m.shape[:2]
    u_px = int(np.clip(u_px, 0, w - 1))
    v_px = int(np.clip(v_px, 0, h - 1))

    half = max(0, int(window_px) // 2)
    x1 = max(0, u_px - half)
    x2 = min(w, u_px + half + 1)
    y1 = max(0, v_px - half)
    y2 = min(h, v_px + half + 1)

    patch = np.asarray(depth_image_m[y1:y2, x1:x2], dtype=np.float32).reshape(-1)
    valid = patch[np.isfinite(patch)]
    valid = valid[(valid > min_depth_m) & (valid < max_depth_m)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def calculate_circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter ** 2))


def calculate_solidity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0.0
    return float(area / hull_area)


class OperationVisionProcessor:
    """Stateless-ish image processor for pieces and cavities."""

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = get_default_config()
        if config:
            self.config.update(config)

    def update_config(self, config: dict) -> None:
        self.config.update(config)

    def detect_cavities(self, bgr_image: np.ndarray) -> List[Cavity2D]:
        """Detect likely silver-lined Operation board cavities."""
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        lower_silver = np.array([0, 0, int(self.config["silver_v_min"])])
        upper_silver = np.array([179, int(self.config["silver_s_max"]), 255])
        silver_mask = cv2.inRange(hsv, lower_silver, upper_silver)

        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)

        mask = cv2.bitwise_or(silver_mask, edges)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cavities: List[Cavity2D] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config["cavity_min_area"] or area > self.config["cavity_max_area"]:
                continue

            circularity = calculate_circularity(contour)
            if circularity < self.config["cavity_min_circularity"]:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
            else:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

            (_, _), radius = cv2.minEnclosingCircle(contour)
            cavities.append(
                Cavity2D(
                    center_px=(center_x, center_y),
                    radius_px=float(radius),
                    area_px=float(area),
                    circularity=float(circularity),
                    contour=contour,
                )
            )

        cavities.sort(key=lambda c: (c.center_px[1], c.center_px[0]))
        return cavities

    @staticmethod
    def point_matches_cavity(point_px: Tuple[int, int], cavities: Iterable[Cavity2D], padding_px: int) -> bool:
        """Return True if a point is inside or near one detected cavity."""
        px, py = point_px
        for cavity in cavities:
            contour_distance = cv2.pointPolygonTest(cavity.contour, (float(px), float(py)), True)
            if contour_distance >= -padding_px:
                return True
            cx, cy = cavity.center_px
            center_distance = math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
            if center_distance <= cavity.radius_px + padding_px:
                return True
        return False

    def detect_pieces(
        self,
        bgr_image: np.ndarray,
        cavities: Optional[List[Cavity2D]] = None,
        use_cavity_filter: bool = False,
    ) -> Tuple[List[Detection2D], np.ndarray]:
        """Detect yellow pieces/board regions and return a debug image."""
        debug_image = bgr_image.copy()
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([
            int(self.config["h_low"]),
            int(self.config["s_low"]),
            int(self.config["v_low"]),
        ])
        upper_yellow = np.array([
            int(self.config["h_high"]),
            int(self.config["s_high"]),
            int(self.config["v_high"]),
        ])

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask = cv2.bilateralFilter(mask, 9, 75, 75)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[Detection2D] = []
        rejected_by_cavity = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config["min_area"]:
                continue

            circularity = calculate_circularity(contour)
            if circularity < self.config["min_circularity"]:
                continue

            solidity = calculate_solidity(contour)
            if solidity < self.config["min_solidity"]:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            if moments["m00"] > 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
            else:
                center_x = x + w // 2
                center_y = y + h // 2

            if use_cavity_filter and cavities:
                if not self.point_matches_cavity(
                    (center_x, center_y), cavities, int(self.config["cavity_match_padding_px"])
                ):
                    rejected_by_cavity += 1
                    cv2.circle(debug_image, (center_x, center_y), 12, (80, 80, 80), 1)
                    continue

            detections.append(
                Detection2D(
                    center_px=(center_x, center_y),
                    area_px=float(area),
                    circularity=float(circularity),
                    solidity=float(solidity),
                    bbox_xywh=(x, y, w, h),
                    contour=contour,
                )
            )

        # Draw cavities first, then accepted piece detections.
        if cavities:
            for idx, cavity in enumerate(cavities):
                cv2.drawContours(debug_image, [cavity.contour], -1, (255, 255, 255), 2)
                cv2.circle(debug_image, cavity.center_px, 4, (255, 0, 255), -1)
                cv2.putText(
                    debug_image,
                    f"C{idx}",
                    (cavity.center_px[0] + 6, cavity.center_px[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 0, 255),
                    1,
                )

        for idx, det in enumerate(detections):
            x, y, w, h = det.bbox_xywh
            cv2.drawContours(debug_image, [det.contour], -1, (0, 255, 255), 2)
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 200, 255), 1)
            cv2.circle(debug_image, det.center_px, 5, (0, 0, 255), -1)
            cv2.putText(
                debug_image,
                f"P{idx} A:{int(det.area_px)} C:{det.circularity:.2f}",
                (det.center_px[0] + 8, det.center_px[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
            )

        cv2.putText(
            debug_image,
            f"pieces:{len(detections)} cavities:{len(cavities or [])} rejected:{rejected_by_cavity}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            debug_image,
            "HSV + bilateral + circularity + solidity + Kalman + RealSense depth",
            (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (230, 230, 230),
            1,
        )

        return detections, debug_image

    def project_detections(
        self,
        detections_2d: Iterable[Detection2D],
        projection: CameraProjection,
        depth_image_m: Optional[np.ndarray],
        assumed_depth_m: float,
        depth_window_px: int,
        min_depth_m: float,
        max_depth_m: float,
    ) -> List[Detection3D]:
        detections_3d: List[Detection3D] = []
        for det in detections_2d:
            u_px, v_px = det.center_px
            depth_m: Optional[float] = None
            if depth_image_m is not None:
                depth_m = sample_depth_at_pixel(
                    depth_image_m,
                    u_px,
                    v_px,
                    window_px=depth_window_px,
                    min_depth_m=min_depth_m,
                    max_depth_m=max_depth_m,
                )
            if depth_m is None:
                depth_m = assumed_depth_m

            try:
                point_camera = projection.pixel_to_3d(u_px, v_px, depth_m)
            except RuntimeError:
                continue

            detections_3d.append(
                Detection3D(
                    center_px=det.center_px,
                    point_camera_m=point_camera,
                    depth_m=float(depth_m),
                    area_px=det.area_px,
                    circularity=det.circularity,
                    solidity=det.solidity,
                )
            )
        return detections_3d


class KalmanFilter1D:
    """Small 1D Kalman filter for smoothing noisy measurements."""

    def __init__(
        self,
        process_variance: float = 1e-4,
        measurement_variance: float = 1e-1,
        initial_value: float = 0.0,
    ) -> None:
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = initial_value
        self.posteri_error_estimate = 1.0
        self.priori_estimate = 0.0
        self.priori_error_estimate = 1.0

    def update(self, measurement: float) -> float:
        self.priori_estimate = self.posteri_estimate
        self.priori_error_estimate = self.posteri_error_estimate + self.process_variance
        kalman_gain = self.priori_error_estimate / (
            self.priori_error_estimate + self.measurement_variance
        )
        self.posteri_estimate = self.priori_estimate + kalman_gain * (
            measurement - self.priori_estimate
        )
        self.posteri_error_estimate = (1.0 - kalman_gain) * self.priori_error_estimate
        return float(self.posteri_estimate)


class PieceTracker:
    """Nearest-neighbor 3D tracker with per-axis Kalman filtering."""

    def __init__(self, max_distance_m: float = 0.05, history_length: int = 10, max_age_frames: int = 30):
        self.tracks: Dict[int, dict] = {}
        self.next_id = 0
        self.max_distance_m = float(max_distance_m)
        self.history_length = int(history_length)
        self.max_age_frames = int(max_age_frames)

    def update(self, detections: List[Detection3D]) -> List[TrackedDetection]:
        if not detections:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]["age"] += 1
                if self.tracks[track_id]["age"] > self.max_age_frames:
                    del self.tracks[track_id]
            return []

        matched_tracks = set()
        results: List[TrackedDetection] = []

        # Prioritize larger/cleaner detections so a stable target gets an ID first.
        sorted_detections = sorted(detections, key=lambda d: d.area_px, reverse=True)

        for det in sorted_detections:
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
                matched_tracks.add(track_id)
                results.append(
                    TrackedDetection(
                        track_id=track_id,
                        point_camera_m=(float(det_x), float(det_y), float(det_z)),
                        confidence=0.0,
                        age=0,
                    )
                )
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
            results.append(
                TrackedDetection(
                    track_id=best_track_id,
                    point_camera_m=(filtered_x, filtered_y, filtered_z),
                    confidence=float(confidence),
                    age=0,
                )
            )

        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]["age"] += 1
                if self.tracks[track_id]["age"] > self.max_age_frames:
                    del self.tracks[track_id]

        return sorted(results, key=lambda t: (-t.confidence, t.track_id))


def average_camera_points(points: Iterable[Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
    arr = np.asarray(list(points), dtype=np.float64)
    if arr.size == 0:
        return None
    return tuple(np.mean(arr, axis=0).astype(float).tolist())
