import cv2
import numpy as np
import json
import os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import deque

# Configuration file for HSV values
CONFIG_FILE = 'operation_hsv_config.json'

def load_config():
    """Load HSV configuration from file"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            return config
        except:
            print("Error loading config, using defaults")
            return get_default_config()
    else:
        print("No config found, using defaults")
        return get_default_config()

def get_default_config():
    """Default HSV range for bright yellow"""
    return {
        'h_low': 15,
        'h_high': 35,
        's_low': 100,
        's_high': 255,
        'v_low': 100,
        'v_high': 255,
        'min_area': 100
    }

class KalmanFilter1D:
    """1D Kalman filter for smoothing noisy measurements"""
    
    def __init__(self, process_variance=1e-4, measurement_variance=1e-1, initial_value=0.0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.posteri_estimate = initial_value
        self.posteri_error_estimate = 1.0
        self.priori_estimate = 0.0
        self.priori_error_estimate = 1.0
    
    def update(self, measurement):
        """Update filter with new measurement"""
        # Predict
        self.priori_estimate = self.posteri_estimate
        self.priori_error_estimate = self.posteri_error_estimate + self.process_variance
        
        # Update
        kalman_gain = self.priori_error_estimate / (self.priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = self.priori_estimate + kalman_gain * (measurement - self.priori_estimate)
        self.posteri_error_estimate = (1 - kalman_gain) * self.priori_error_estimate
        
        return self.posteri_estimate

class PieceTracker:
    """Track detected pieces across frames using Kalman filtering"""
    
    def __init__(self, max_distance=50, history_length=10):
        self.tracks = {}  # id -> track data
        self.next_id = 0
        self.max_distance = max_distance
        self.history_length = history_length
    
    def update(self, detections):
        """Update tracks with new detections
        
        Args:
            detections: list of (x, y, z, area) tuples
        
        Returns:
            list of (id, filtered_x, filtered_y, filtered_z, confidence)
        """
        if len(detections) == 0:
            # Age existing tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > 30:  # Remove tracks older than 30 frames
                    del self.tracks[track_id]
            return []
        
        # Match detections to existing tracks using nearest neighbor
        matched_tracks = set()
        results = []
        
        for det_idx, (det_x, det_y, det_z, det_area) in enumerate(detections):
            best_track_id = None
            best_distance = self.max_distance
            
            # Find closest track
            for track_id, track_data in list(self.tracks.items()):
                if track_id in matched_tracks:
                    continue
                
                last_x = track_data['kf_x'].posteri_estimate
                last_y = track_data['kf_y'].posteri_estimate
                
                distance = np.sqrt((det_x - last_x)**2 + (det_y - last_y)**2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_track_id = track_id
            
            # Update or create track
            if best_track_id is not None:
                matched_tracks.add(best_track_id)
                track = self.tracks[best_track_id]
                track['age'] = 0
                
                # Update Kalman filters
                filtered_x = track['kf_x'].update(det_x)
                filtered_y = track['kf_y'].update(det_y)
                filtered_z = track['kf_z'].update(det_z)
                
                # Update history
                track['positions'].append((filtered_x, filtered_y, filtered_z))
                
                # Calculate confidence (0-1) based on how long we've tracked it
                confidence = min(track['age'] / 10.0, 1.0)
                
                results.append((best_track_id, filtered_x, filtered_y, filtered_z, confidence))
            else:
                # Create new track
                track_id = self.next_id
                self.next_id += 1
                
                self.tracks[track_id] = {
                    'kf_x': KalmanFilter1D(initial_value=det_x),
                    'kf_y': KalmanFilter1D(initial_value=det_y),
                    'kf_z': KalmanFilter1D(initial_value=det_z),
                    'positions': deque([(det_x, det_y, det_z)], maxlen=self.history_length),
                    'age': 0
                }
                results.append((track_id, det_x, det_y, det_z, 0.0))
        
        # Age unmatched tracks - use list() to avoid dictionary size change errors
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > 30:
                    del self.tracks[track_id]
        
        return results

class DepthEstimator:
    """Estimate depth and 3D coordinates from 2D camera view"""
    
    def __init__(self, frame_width=640, frame_height=480):
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Camera intrinsic parameters (estimated for typical webcam)
        self.fx = frame_width  # focal length in x (pixels)
        self.fy = frame_height  # focal length in y (pixels)
        self.cx = frame_width / 2  # principal point x
        self.cy = frame_height / 2  # principal point y
        
        # Assume average game piece size (roughly 15-20mm diameter)
        self.piece_diameter_mm = 17.5
        
        # Reference: at 300mm distance, piece occupies ~50 pixels
        self.reference_distance = 300  # mm
        self.reference_pixel_size = 50  # pixels
        
    def estimate_distance_from_area(self, contour_area):
        """
        Estimate distance based on contour area.
        Assumes pieces of similar size - larger area means closer.
        Uses inverse square law approximation.
        """
        if contour_area < 10:
            return 1000  # Very far
        
        # Simple inverse relationship: distance ~ reference_area / area
        reference_area = np.pi * (self.reference_pixel_size / 2) ** 2
        estimated_distance = self.reference_distance * np.sqrt(reference_area / contour_area)
        
        # Clamp to reasonable range (100mm to 1500mm)
        return np.clip(estimated_distance, 100, 1500)
    
    def pixel_to_3d(self, x_pixel, y_pixel, distance_mm):
        """
        Convert pixel coordinates to 3D world coordinates.
        Assumes camera is at origin looking in +Z direction.
        Returns (x_mm, y_mm, z_mm)
        """
        # Normalize pixel coordinates to camera frame
        x_norm = (x_pixel - self.cx) / self.fx
        y_norm = (y_pixel - self.cy) / self.fy
        
        # Project to world coordinates at estimated distance
        x_world = x_norm * distance_mm
        y_world = y_norm * distance_mm
        z_world = distance_mm
        
        return x_world, y_world, z_world
    
    def draw_coordinate_frame(self, image, x_pixel, y_pixel, size=20, thickness=2):
        """Draw 3D coordinate frame (X-red, Y-green, Z-blue) at pixel position"""
        origin = (int(x_pixel), int(y_pixel))
        
        # X-axis (red)
        cv2.arrowedLine(image, origin, (int(x_pixel + size), int(y_pixel)), 
                       (0, 0, 255), thickness, tipLength=0.3)
        cv2.putText(image, 'X', (int(x_pixel + size + 5), int(y_pixel)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Y-axis (green)
        cv2.arrowedLine(image, origin, (int(x_pixel), int(y_pixel + size)), 
                       (0, 255, 0), thickness, tipLength=0.3)
        cv2.putText(image, 'Y', (int(x_pixel - 15), int(y_pixel + size + 10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Z-axis (blue) - drawn smaller since it points away from camera
        cv2.circle(image, origin, 3, (255, 0, 0), -1)
        cv2.putText(image, 'Z', (int(x_pixel - 10), int(y_pixel - 10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

def calculate_circularity(contour):
    """Calculate circularity of contour (1.0 = perfect circle)"""
    area = cv2.contourArea(contour)
    if area == 0:
        return 0
    
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    
    circularity = 4 * np.pi * area / (perimeter ** 2)
    return circularity

def calculate_solidity(contour):
    """Calculate solidity of contour (area / convex hull area)"""
    area = cv2.contourArea(contour)
    if area == 0:
        return 0
    
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    
    if hull_area == 0:
        return 0
    
    solidity = area / hull_area
    return solidity

# Load configuration
config = load_config()
h_low = config['h_low']
h_high = config['h_high']
s_low = config['s_low']
s_high = config['s_high']
v_low = config['v_low']
v_high = config['v_high']
min_area = config['min_area']

# Initialize camera and depth estimator
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
depth_estimator = DepthEstimator()
tracker = PieceTracker(max_distance=100, history_length=10)

# Setup 3D plotting
plt.ion()  # Enable interactive mode
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.show(block=False)

print("Operation Detection with Depth Perception + Noise Filtering")
print("=" * 60)
print("Filters Applied:")
print("  - Bilateral filtering (edge-preserving denoising)")
print("  - Circularity filtering (reject non-circular shapes)")
print("  - Kalman filtering (smooth tracking across frames)")
print("=" * 60)
print("Press 'q' to quit (ESC key in CV window)")
print("3D visualization opens in separate window")
print("=" * 60)

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame")
        break
    
    frame_count += 1
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for yellow
    lower_yellow = np.array([h_low, s_low, v_low])
    upper_yellow = np.array([h_high, s_high, v_high])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply bilateral filtering to mask for noise reduction
    mask = cv2.bilateralFilter(mask, 9, 75, 75)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process detected pieces with filtering
    detections = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < min_area:
            continue
        
        # Filter by circularity (game pieces are roughly circular)
        circularity = calculate_circularity(contour)
        if circularity < 0.5:  # Reject non-circular shapes
            continue
        
        # Filter by solidity (reject fragmented blobs)
        solidity = calculate_solidity(contour)
        if solidity < 0.7:
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = x + w // 2, y + h // 2
        
        # Estimate distance from contour area
        distance = depth_estimator.estimate_distance_from_area(area)
        
        # Convert to 3D coordinates
        x_3d, y_3d, z_3d = depth_estimator.pixel_to_3d(cx, cy, distance)
        
        detections.append((x_3d, y_3d, z_3d, area))
    
    # Update tracker with filtered detections (applies Kalman filtering)
    tracked_pieces = tracker.update(detections)
    
    pieces_3d_frame = []
    
    # Draw tracked pieces on frame
    for track_id, x_3d, y_3d, z_3d, confidence in tracked_pieces:
        # Only display high-confidence tracks
        if confidence > 0.3 or True:  # Show all for now
            pieces_3d_frame.append([x_3d, y_3d, z_3d])
            
            # Project 3D to 2D for visualization
            # Simple inverse of pixel_to_3d
            x_pixel = (x_3d / z_3d) * depth_estimator.fx + depth_estimator.cx
            y_pixel = (y_3d / z_3d) * depth_estimator.fy + depth_estimator.cy
            
            x_pixel = int(np.clip(x_pixel, 0, frame.shape[1] - 1))
            y_pixel = int(np.clip(y_pixel, 0, frame.shape[0] - 1))
            
            # Color based on confidence
            color_intensity = int(255 * confidence)
            color = (0, color_intensity, 255 - color_intensity)  # Red to green gradient
            
            # Draw circle at detected position
            cv2.circle(frame, (x_pixel, y_pixel), 15, color, 2)
            
            # Draw coordinate frame
            depth_estimator.draw_coordinate_frame(frame, x_pixel, y_pixel, size=15, thickness=2)
            
            # Add label with track ID, distance, and confidence
            label = f'ID:{track_id} Z:{int(z_3d)}mm Conf:{confidence:.1f}'
            cv2.putText(frame, label, (x_pixel - 30, y_pixel - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Update 3D plot - always update (every frame)
    ax.clear()
    
    # Plot detected points if any
    if len(pieces_3d_frame) > 0:
        points = np.array(pieces_3d_frame)
        
        # Plot points with yellow color (like Operation pieces)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                 c='yellow', marker='o', s=200, edgecolors='black', linewidth=2)
        
        # Add labels for each point
        for i, (x, y, z) in enumerate(points):
            ax.text(x, y, z, f'  P{i}', fontsize=9)
    
    # Labels and limits - always set these
    ax.set_xlabel('X (mm)', fontsize=10)
    ax.set_ylabel('Y (mm)', fontsize=10)
    ax.set_zlabel('Z (mm) - Distance', fontsize=10)
    ax.set_title(f'Operation Pieces in 3D Space ({len(pieces_3d_frame)} detected)', fontsize=12)
    
    # Set reasonable limits
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    ax.set_zlim(100, 1500)
    
    # Draw coordinate frame at origin
    ax.quiver(0, 0, 0, 100, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, label='X')
    ax.quiver(0, 0, 0, 0, 100, 0, color='green', arrow_length_ratio=0.1, linewidth=2, label='Y')
    ax.quiver(0, 0, 0, 0, 0, 200, color='blue', arrow_length_ratio=0.1, linewidth=2, label='Z')
    
    ax.text(100, 0, 0, 'X', color='red', fontsize=10)
    ax.text(0, 100, 0, 'Y', color='green', fontsize=10)
    ax.text(0, 0, 200, 'Z', color='blue', fontsize=10)
    
    ax.legend(loc='upper right', fontsize=9)
    
    # Refresh the figure - use canvas drawing
    try:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
    except:
        pass
    
    # Add info text
    info_text = f'Tracked: {len(tracked_pieces)} pieces | Frame: {frame_count}'
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, 'Filters: Bilateral + Circularity + Kalman', (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    cv2.imshow('Operation Detection with Depth & Noise Filtering', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.close('all')
print("Done!")
