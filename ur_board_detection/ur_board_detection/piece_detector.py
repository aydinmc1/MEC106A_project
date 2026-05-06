"""
piece_detector.py — pure-CV detection module for the Operation board game.

No ROS2 imports anywhere in this file.  Import it from board_detector_node.py
or run it directly:

    python piece_detector.py          # uses RealSense D435i
    python piece_detector.py --webcam # fallback webcam (no depth)

Detection pipeline per frame:
  1. Detect white board cavities (adaptive threshold, area-only filter)
  2. Detect yellow piece blobs
  3. Detect green cylinder stubs inside yellow regions (grasp targets)
  4. Match each green stub to its enclosing cavity
  5. Classify piece type from cavity shape (Hu moments → geometric fallback)
  6. Unproject green stub centroid to 3-D using aligned D435i depth
  7. Return list of PieceDetection objects
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Piece catalog
# ---------------------------------------------------------------------------

@dataclass
class PieceSpec:
    """Static description of one Operation piece type."""
    name: str

    # Extra Z (meters) to push the tweezer past the board surface.
    # Deeper for chunky pieces like spare_ribs; shallow for tiny pins.
    grasp_depth_offset: float

    # 'vertical': gripper descends straight down (most pieces).
    # 'angled': gripper tilts to match the long axis of the cavity
    #           (rubber-band / ankle_bone).
    approach_axis: str

    # Expected cavity bounding-box aspect ratio (width ÷ height).
    # Used when Hu moments are not yet calibrated.
    aspect_ratio_range: Tuple[float, float]

    # Expected cavity area in pixels² at a nominal viewing distance
    # (~50 cm camera-to-board).  Coarse sanity check only.
    area_range: Tuple[float, float]

    # Hu moments of a representative cavity contour (log10-scaled, 7 values).
    # Leave as None until you capture a template during calibration.
    cavity_hu_moments: Optional[List[float]] = None


PIECE_CATALOG: List[PieceSpec] = [
    PieceSpec(
        name="frog_in_throat",
        grasp_depth_offset=0.005,
        approach_axis="vertical",
        aspect_ratio_range=(0.8, 1.2),
        area_range=(400, 2000),
    ),
    PieceSpec(
        name="teddy_bear",
        grasp_depth_offset=0.005,
        approach_axis="vertical",
        aspect_ratio_range=(0.7, 1.3),
        area_range=(600, 3000),
    ),
    PieceSpec(
        name="shoulder_bone",
        grasp_depth_offset=0.004,
        approach_axis="vertical",
        aspect_ratio_range=(0.6, 1.4),
        area_range=(200, 1200),
    ),
    PieceSpec(
        name="broken_heart",
        grasp_depth_offset=0.005,
        approach_axis="vertical",
        aspect_ratio_range=(0.8, 1.4),
        area_range=(600, 3000),
    ),
    PieceSpec(
        name="wrist_piece",
        grasp_depth_offset=0.005,
        approach_axis="vertical",
        aspect_ratio_range=(1.5, 4.0),
        area_range=(400, 2500),
    ),
    PieceSpec(
        name="butterfly",
        grasp_depth_offset=0.005,
        approach_axis="vertical",
        aspect_ratio_range=(1.5, 3.5),
        area_range=(800, 4000),
    ),
    PieceSpec(
        name="duck",
        grasp_depth_offset=0.005,
        approach_axis="vertical",
        aspect_ratio_range=(0.8, 1.6),
        area_range=(400, 2500),
    ),
    PieceSpec(
        name="mushroom",
        grasp_depth_offset=0.004,
        approach_axis="vertical",
        aspect_ratio_range=(0.6, 1.3),
        area_range=(300, 1800),
    ),
    PieceSpec(
        name="tongue_jar",
        grasp_depth_offset=0.004,
        approach_axis="vertical",
        aspect_ratio_range=(0.7, 1.3),
        area_range=(200, 1500),
    ),
    PieceSpec(
        # Very elongated cavity — tweezer tilts along the long axis to clear
        # the rim on both sides before descending.
        name="rubber_band",
        grasp_depth_offset=0.010,
        approach_axis="angled",
        aspect_ratio_range=(3.0, 8.0),
        area_range=(800, 5000),
    ),
    PieceSpec(
        name="pulling_rope",
        grasp_depth_offset=0.007,
        approach_axis="angled",
        aspect_ratio_range=(2.0, 6.0),
        area_range=(600, 4000),
    ),
    PieceSpec(
        name="ice_cube_tray",
        grasp_depth_offset=0.006,
        approach_axis="vertical",
        aspect_ratio_range=(1.8, 4.5),
        area_range=(800, 4500),
    ),
]

_CATALOG_BY_NAME: Dict[str, PieceSpec] = {p.name: p for p in PIECE_CATALOG}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PieceDetection:
    piece_name: str
    grasp_point_2d: Tuple[int, int]             # (px, py) in color image
    grasp_point_3d: Tuple[float, float, float]  # (x, y, z) camera frame, meters
    grasp_depth_offset: float                   # extra Z for tweezer descent, meters
    approach_axis: str                          # 'vertical' | 'angled'
    confidence: float                           # 0.0 – 1.0

    # Cavity contour kept for debug drawing; excluded from repr to stay readable.
    cavity_contour: Optional[np.ndarray] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_CONFIG_FILE = "operation_hsv_config.json"


def _resolve_config(path: str) -> Optional[str]:
    if os.path.isabs(path) and os.path.exists(path):
        return path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for candidate in [
        path,
        os.path.join(script_dir, path),
        os.path.join(os.path.dirname(script_dir), path),
    ]:
        if os.path.exists(candidate):
            return candidate
    return None


def _default_config() -> dict:
    return {
        # Yellow piece HSV (tuned values from operation_hsv_config.json)
        "h_low": 20,    "h_high": 31,
        "s_low": 87,    "s_high": 255,
        "v_low": 99,    "v_high": 255,
        "min_area": 37,

        # Green cylinder stub HSV — may need field tuning
        "green_h_low": 40,  "green_h_high": 80,
        "green_s_low": 80,  "green_s_high": 255,
        "green_v_low": 80,  "green_v_high": 255,
        "green_min_area": 15,

        # Cavity detection (white board surface → dark cavities)
        "cavity_min_area": 1019,
        "cavity_max_area": 13636,
        # Adaptive threshold parameters: block size (odd int) and C constant
        "cavity_thresh_block": 31,
        "cavity_thresh_c": 8,
        # Morphological cleanup kernel size for cavities
        "cavity_morph_k": 5,

        # Hu moment match distance threshold (lower = stricter)
        "hu_match_threshold": 0.30,

        # Fallback assumed depth when depth image unavailable (meters)
        "assumed_depth_m": 0.50,
        # D435i depth scale: meters per raw uint16 unit (standard = 0.001)
        "depth_scale": 0.001,

        # When False (default): if no green stub is found inside a cavity,
        # fall back to the yellow region centroid as the grasp point with
        # confidence=0.3.  Useful for testing classification before stubs
        # are printed on all pieces.
        # When True: only report pieces where a green stub is confirmed.
        "require_green_stub": False,
    }


def load_config(config_path: str = _CONFIG_FILE) -> dict:
    """Load tuned HSV values from JSON; fill missing keys with defaults."""
    cfg = _default_config()
    resolved = _resolve_config(config_path)
    if resolved is None:
        return cfg
    try:
        with open(resolved, "r", encoding="utf-8") as f:
            cfg.update(json.load(f))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[piece_detector] config load failed ({exc}), using defaults")
    return cfg


# ---------------------------------------------------------------------------
# PieceDetector
# ---------------------------------------------------------------------------

class PieceDetector:
    """
    All CV logic for finding Operation pieces, cavities, and green stubs.

    Intrinsics are optional at construction time; call update_intrinsics()
    whenever a CameraInfo message arrives.  Without intrinsics the 3-D x/y
    components are returned as 0.0; only z (depth) is reported.

    After each process_frame() call, the intermediate results are available
    as attributes:
        self.last_cavities  — list of cavity dicts from detect_cavities()
        self.last_stubs     — list of stub dicts from detect_green_stubs()
    These are used by draw_debug() so callers don't have to rerun detection.
    """

    def __init__(self, config_path: str = _CONFIG_FILE):
        self.cfg = load_config(config_path)

        # Camera intrinsics — populated by update_intrinsics()
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        # Cached intermediates from the most recent process_frame() call
        self.last_cavities: List[dict] = []
        self.last_stubs:    List[dict] = []

        self._build_hsv_bounds()

    def update_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None:
        """Supply camera intrinsics (from ROS CameraInfo or pyrealsense2)."""
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def _build_hsv_bounds(self) -> None:
        c = self.cfg
        self._yellow_lo = np.array([c["h_low"],       c["s_low"],       c["v_low"]])
        self._yellow_hi = np.array([c["h_high"],      c["s_high"],      c["v_high"]])
        self._green_lo  = np.array([c["green_h_low"], c["green_s_low"], c["green_v_low"]])
        self._green_hi  = np.array([c["green_h_high"],c["green_s_high"],c["green_v_high"]])

    # ------------------------------------------------------------------
    # Step 1: Cavity detection (adaptive threshold on white board)
    # ------------------------------------------------------------------

    def detect_cavities(self, bgr: np.ndarray) -> List[dict]:
        """
        Find board cavities using adaptive thresholding on the white board
        surface.  Dark cavities appear as white foreground in the inverted
        threshold image.

        Filtering by area only — no circularity filter, because cavities
        are irregular (heart, frog, rubber-band, small round pins, etc.).

        Returns:
            List of dicts:
                'contour': np.ndarray (OpenCV contour)
                'center':  (px, py) int tuple
                'area':    float
                'bbox':    (x, y, w, h) int tuple
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        block = int(self.cfg["cavity_thresh_block"])
        block = max(3, block | 1)  # ensure odd and >= 3

        # THRESH_BINARY_INV: white board → black, dark cavities → white
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            int(self.cfg["cavity_thresh_c"]),
        )

        k_size = max(3, int(self.cfg["cavity_morph_k"]))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  k, iterations=1)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cavities = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.cfg["cavity_min_area"] or area > self.cfg["cavity_max_area"]:
                continue
            x, y, w, h = cv2.boundingRect(c)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            cavities.append({
                "contour": c,
                "center":  (cx, cy),
                "area":    area,
                "bbox":    (x, y, w, h),
            })
        return cavities

    # ------------------------------------------------------------------
    # Step 2: Yellow region detection
    # ------------------------------------------------------------------

    def detect_yellow_regions(self, bgr: np.ndarray) -> Tuple[List[dict], np.ndarray]:
        """
        Find yellow blobs (pieces sitting in cavities).

        Returns:
            regions:     list of dicts with 'contour', 'bbox', 'center', 'mask'
            yellow_mask: full-frame binary mask (uint8)
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._yellow_lo, self._yellow_hi)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.cfg["min_area"]:
                continue
            x, y, w, h = cv2.boundingRect(c)
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + h // 2
            sub = np.zeros(bgr.shape[:2], dtype=np.uint8)
            cv2.drawContours(sub, [c], -1, 255, -1)
            regions.append({
                "contour": c,
                "bbox":    (x, y, w, h),
                "center":  (cx, cy),
                "mask":    sub,
            })
        return regions, mask

    # ------------------------------------------------------------------
    # Step 3: Green stub detection (primary grasp target)
    # ------------------------------------------------------------------

    def detect_green_stubs(
        self,
        bgr: np.ndarray,
        yellow_mask: np.ndarray,
    ) -> List[dict]:
        """
        Find green blobs that sit inside yellow regions.
        The green cylinder stub centroid is the exact tweezer target point.

        Args:
            bgr:         color frame
            yellow_mask: output of detect_yellow_regions() — restricts search
                         to pixels that are already on a yellow piece

        Returns:
            List of dicts:
                'center':  (px, py)
                'area':    float
                'contour': np.ndarray
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self._green_lo, self._green_hi)

        # Only keep green pixels that also belong to a yellow region
        green_mask = cv2.bitwise_and(green_mask, yellow_mask)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN,  k)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, k)

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stubs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.cfg["green_min_area"]:
                continue
            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                bx, by, bw, bh = cv2.boundingRect(c)
                cx, cy = bx + bw // 2, by + bh // 2
            stubs.append({"center": (cx, cy), "area": area, "contour": c})
        return stubs

    # ------------------------------------------------------------------
    # Step 4: Cavity classification
    # ------------------------------------------------------------------

    def _classify_cavity(self, cavity: dict) -> Tuple[str, float]:
        """
        Match a cavity to a piece type.

        Priority:
          1. Hu moment matching (if any PieceSpec has cavity_hu_moments set).
          2. Geometric fallback: aspect ratio + area scoring.

        Returns:
            (piece_name, confidence)  where confidence ∈ [0, 1]
        """
        contour = cavity["contour"]
        area    = cavity["area"]
        x, y, w, h = cavity["bbox"]
        aspect = (w / h) if h > 0 else 1.0

        # --- Hu moment matching ---
        calibrated = [p for p in PIECE_CATALOG if p.cavity_hu_moments is not None]
        if calibrated:
            hu_raw = cv2.HuMoments(cv2.moments(contour)).flatten()
            # Log-scale to normalise the wide dynamic range of Hu moments
            with np.errstate(divide="ignore", invalid="ignore"):
                hu_log = np.where(
                    hu_raw != 0,
                    np.sign(hu_raw) * np.log10(np.abs(hu_raw) + 1e-10),
                    0.0,
                )
            best_name = "unknown"
            best_dist = float("inf")
            for spec in calibrated:
                ref = np.array(spec.cavity_hu_moments)
                dist = float(np.sum(np.abs(hu_log - ref)))
                if dist < best_dist:
                    best_dist = dist
                    best_name = spec.name

            threshold = float(self.cfg["hu_match_threshold"])
            if best_dist < threshold:
                conf = max(0.0, 1.0 - best_dist / threshold)
                return best_name, conf

        # --- Geometric fallback ---
        # Normalise aspect so both landscape and portrait are handled uniformly
        norm_ar = aspect if aspect >= 1.0 else (1.0 / aspect)

        best_name  = "unknown"
        best_score = -1.0
        for spec in PIECE_CATALOG:
            ar_lo, ar_hi = spec.aspect_ratio_range
            a_lo,  a_hi  = spec.area_range

            ar_ok   = ar_lo <= norm_ar <= ar_hi
            area_ok = a_lo  <= area   <= a_hi

            # AR match is worth 1.0; area match adds 0.5
            score = (1.0 if ar_ok else 0.0) + (0.5 if area_ok else 0.0)
            if score > best_score:
                best_score = score
                best_name  = spec.name

        # Max geometric score = 1.5 → cap confidence at 1.0
        confidence = min(best_score / 1.5, 1.0)
        return best_name, confidence

    # ------------------------------------------------------------------
    # Step 5: Match stub to its enclosing cavity
    # ------------------------------------------------------------------

    def _match_stub_to_cavity(
        self,
        stub_center: Tuple[int, int],
        cavities: List[dict],
    ) -> Optional[dict]:
        """
        Return the cavity whose contour encloses the stub centre.
        If no cavity strictly contains the stub, return the nearest one
        provided it is within 40 pixels (handles slight detection offset).
        """
        px, py = float(stub_center[0]), float(stub_center[1])
        best_cavity = None
        best_dist   = float("inf")

        for cav in cavities:
            # pointPolygonTest: positive = inside, negative = outside
            d = cv2.pointPolygonTest(cav["contour"], (px, py), measureDist=True)
            if d >= 0:
                return cav  # definitive containment
            if abs(d) < best_dist:
                best_dist   = abs(d)
                best_cavity = cav

        if best_dist <= 40.0:
            return best_cavity
        return None

    # ------------------------------------------------------------------
    # Step 6: 3-D unprojection
    # ------------------------------------------------------------------

    def _deproject_to_3d(
        self,
        px: int,
        py: int,
        depth_image: Optional[np.ndarray],
    ) -> Tuple[float, float, float]:
        """
        Convert a pixel + aligned depth image to a camera-frame 3-D point
        in meters.

        depth_image: uint16 aligned depth from D435i
                     (each unit = cfg['depth_scale'] meters, typically 1 mm).
        When depth is unavailable or zero, falls back to cfg['assumed_depth_m'].
        When intrinsics are unavailable, x/y are returned as 0.0.
        """
        z = float(self.cfg["assumed_depth_m"])

        if depth_image is not None:
            h, w = depth_image.shape[:2]
            if 0 <= py < h and 0 <= px < w:
                raw = int(depth_image[py, px])
                if raw > 0:
                    z = raw * float(self.cfg["depth_scale"])

        if self.fx is None:
            # No intrinsics yet — can only report depth
            return 0.0, 0.0, z

        x3 = (px - self.cx) * z / self.fx
        y3 = (py - self.cy) * z / self.fy
        return float(x3), float(y3), float(z)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process_frame(
        self,
        bgr: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
    ) -> List[PieceDetection]:
        """
        Run the complete detection pipeline on one BGR frame.

        Args:
            bgr:         color image  (H×W×3, uint8, BGR)
            depth_image: aligned depth (H×W, uint16) or None

        Returns:
            List[PieceDetection] — one entry per detected piece.

            Primary path  (confidence from classifier):
              green stub found → stub centroid is the grasp point.

            Fallback path  (confidence = 0.3, fixed):
              cfg["require_green_stub"] is False AND a yellow region centroid
              sits inside a cavity that has no matched stub → yellow centroid
              is used as the grasp point.  Allows cavity detection and piece
              classification to be tested before green stubs are printed.

            When cfg["require_green_stub"] is True the fallback is suppressed
            and only stub-confirmed pieces are returned.

        Side-effects:
            self.last_cavities and self.last_stubs are updated so
            draw_debug() can reuse them without re-running detection.
        """
        # Steps 1–3: detect scene elements
        cavities = self.detect_cavities(bgr)
        yellow_regions, yellow_mask = self.detect_yellow_regions(bgr)
        stubs = self.detect_green_stubs(bgr, yellow_mask)

        # Cache intermediates for draw_debug()
        self.last_cavities = cavities
        self.last_stubs    = stubs

        results: List[PieceDetection] = []

        # Track which cavity dicts have already been claimed by a stub so the
        # fallback path does not double-report the same cavity.
        # id() of the dict is stable for the lifetime of this call.
        claimed_cavity_ids: set = set()

        # ------------------------------------------------------------------
        # Primary path: green stub → cavity → classification
        # ------------------------------------------------------------------
        for stub in stubs:
            cx, cy = stub["center"]

            # Step 4: find which cavity this stub lives in
            cavity = self._match_stub_to_cavity((cx, cy), cavities)

            # Step 5: classify piece from cavity shape
            if cavity is not None:
                claimed_cavity_ids.add(id(cavity))
                piece_name, confidence = self._classify_cavity(cavity)
                cavity_contour = cavity["contour"]
            else:
                # Stub visible but no matching cavity — report as unknown
                piece_name     = "unknown"
                confidence     = 0.1
                cavity_contour = None

            # Step 6: 3-D unprojection at green stub centroid
            x3, y3, z3 = self._deproject_to_3d(cx, cy, depth_image)

            spec = _CATALOG_BY_NAME.get(piece_name)
            results.append(PieceDetection(
                piece_name=piece_name,
                grasp_point_2d=(cx, cy),
                grasp_point_3d=(x3, y3, z3),
                grasp_depth_offset=spec.grasp_depth_offset if spec else 0.005,
                approach_axis=spec.approach_axis if spec else "vertical",
                confidence=confidence,
                cavity_contour=cavity_contour,
            ))

        # ------------------------------------------------------------------
        # Fallback path: yellow centroid for cavities with no stub
        # ------------------------------------------------------------------
        if not self.cfg.get("require_green_stub", False):
            for region in yellow_regions:
                rx, ry = region["center"]
                cavity = self._match_stub_to_cavity((rx, ry), cavities)

                if cavity is None:
                    continue  # yellow blob not inside any cavity
                if id(cavity) in claimed_cavity_ids:
                    continue  # a stub already owns this cavity

                claimed_cavity_ids.add(id(cavity))
                piece_name, _ = self._classify_cavity(cavity)
                x3, y3, z3 = self._deproject_to_3d(rx, ry, depth_image)

                spec = _CATALOG_BY_NAME.get(piece_name)
                results.append(PieceDetection(
                    piece_name=piece_name,
                    grasp_point_2d=(rx, ry),
                    grasp_point_3d=(x3, y3, z3),
                    grasp_depth_offset=spec.grasp_depth_offset if spec else 0.005,
                    approach_axis=spec.approach_axis if spec else "vertical",
                    # Fixed lower confidence: cavity shape matched but no
                    # green stub confirmed, so grasp point accuracy is lower.
                    confidence=0.3,
                    cavity_contour=cavity["contour"],
                ))

        return results

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    def draw_debug(
        self,
        bgr: np.ndarray,
        detections: List[PieceDetection],
        cavities: Optional[List[dict]] = None,
        stubs:    Optional[List[dict]] = None,
    ) -> np.ndarray:
        """
        Annotated copy of bgr with:
          - Cyan outlines for all detected cavities
          - Bright-green circles for green stubs
          - Yellow crosshair + labels for each matched detection

        cavities / stubs default to self.last_cavities / self.last_stubs
        so callers that have already called process_frame() need not pass them.
        """
        if cavities is None:
            cavities = self.last_cavities
        if stubs is None:
            stubs = self.last_stubs

        out = bgr.copy()

        # All cavities — cyan outline
        for cav in cavities:
            cv2.drawContours(out, [cav["contour"]], -1, (255, 255, 0), 2)
            cx, cy = cav["center"]
            cv2.circle(out, (cx, cy), 3, (255, 255, 0), -1)

        # Green stubs — bright green circles
        for stub in stubs:
            cx, cy = stub["center"]
            cv2.circle(out, (cx, cy), 7, (0, 255, 0), 2)
            cv2.circle(out, (cx, cy), 2, (0, 255, 0), -1)

        # Matched detections — yellow crosshair + label
        for det in detections:
            px, py = det.grasp_point_2d
            x3, y3, z3 = det.grasp_point_3d

            # Crosshair at exact grasp point
            cv2.drawMarker(out, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 22, 2)

            # Matched cavity outline in orange
            if det.cavity_contour is not None:
                cv2.drawContours(out, [det.cavity_contour], -1, (0, 165, 255), 2)

            # Label box
            lines = [
                det.piece_name,
                f"conf={det.confidence:.2f}",
                f"3D=({x3:.3f},{y3:.3f},{z3:.3f})",
                f"ax={det.approach_axis}  dz={det.grasp_depth_offset:.3f}m",
            ]
            tx, ty = px + 14, py - 10
            for i, line in enumerate(lines):
                cv2.putText(out, line, (tx, ty + i * 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # HUD
        cv2.putText(
            out,
            f"Pieces: {len(detections)}  Cavities: {len(cavities)}  Stubs: {len(stubs)}",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA,
        )
        return out

    # ------------------------------------------------------------------
    # Standalone test mode — RealSense D435i
    # ------------------------------------------------------------------

    def run_standalone(self, use_webcam: bool = False) -> None:
        """
        Live annotated feed for tuning CV parameters without running ROS2.

        Tries pyrealsense2 first; falls back to webcam if it is absent or
        --webcam flag is set.

        Keys:
          q / ESC  quit
          s        print current HSV config to stdout
        """
        if not use_webcam:
            try:
                import pyrealsense2 as rs  # type: ignore
                self._run_realsense(rs)
                return
            except ImportError:
                print("[piece_detector] pyrealsense2 not found — using webcam fallback")

        self._run_webcam()

    def _run_realsense(self, rs) -> None:
        """Inner loop for RealSense D435i."""
        pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)

        profile = pipeline.start(cfg)

        # Actual depth scale from the sensor (usually 0.001 m/unit)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.cfg["depth_scale"] = depth_sensor.get_depth_scale()

        # Real intrinsics from the colour stream
        color_profile = profile.get_stream(rs.stream.color)
        intr = color_profile.as_video_stream_profile().get_intrinsics()
        self.update_intrinsics(intr.fx, intr.fy, intr.ppx, intr.ppy)

        align = rs.align(rs.stream.color)

        print("[piece_detector] RealSense running — press 'q' or ESC to quit, 's' to dump config")
        try:
            while True:
                frames   = pipeline.wait_for_frames()
                aligned  = align.process(frames)

                color_f = aligned.get_color_frame()
                depth_f = aligned.get_depth_frame()
                if not color_f or not depth_f:
                    continue

                bgr         = np.asanyarray(color_f.get_data())
                depth_image = np.asanyarray(depth_f.get_data())  # uint16

                detections = self.process_frame(bgr, depth_image)
                debug      = self.draw_debug(bgr, detections)

                cv2.imshow("Operation Piece Detector  [RealSense]", debug)
                self._print_detections(detections)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    print(json.dumps(self.cfg, indent=2))
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

    def _run_webcam(self) -> None:
        """Fallback inner loop using a regular webcam (no depth)."""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("[piece_detector] could not open camera 0")
            return

        print("[piece_detector] Webcam fallback (no depth) — press 'q' or ESC to quit, 's' to dump config")
        try:
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break

                detections = self.process_frame(bgr, depth_image=None)
                debug      = self.draw_debug(bgr, detections)

                cv2.imshow("Operation Piece Detector  [webcam, no depth]", debug)
                self._print_detections(detections)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    print(json.dumps(self.cfg, indent=2))
        finally:
            cap.release()
            cv2.destroyAllWindows()

    @staticmethod
    def _print_detections(detections: List[PieceDetection]) -> None:
        for det in detections:
            x3, y3, z3 = det.grasp_point_3d
            print(
                f"  [{det.piece_name:20s}] "
                f"2D={det.grasp_point_2d}  "
                f"3D=({x3:+.3f},{y3:+.3f},{z3:.3f})m  "
                f"conf={det.confidence:.2f}  "
                f"ax={det.approach_axis}  "
                f"dz={det.grasp_depth_offset:.3f}m"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    webcam_flag = "--webcam" in sys.argv
    detector = PieceDetector()
    detector.run_standalone(use_webcam=webcam_flag)
