import cv2
import numpy as np
import json
import os

# Configuration file for saving HSV values
CONFIG_FILE = 'operation_hsv_config.json'

# Default HSV range for bright yellow
DEFAULT_CONFIG = {
    'h_low': 15,
    'h_high': 35,
    's_low': 100,
    's_high': 255,
    'v_low': 100,
    'v_high': 255,
    'min_area': 100
}

def load_config():
    """Load HSV configuration from file or use defaults"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
            print(f"Loaded saved configuration from {CONFIG_FILE}")
            return config
        except Exception as e:
            print(f"Error loading config file: {e}, using defaults")
            return DEFAULT_CONFIG.copy()
    else:
        print("No saved configuration found, using default values")
        return DEFAULT_CONFIG.copy()

def save_config(config):
    """Save HSV configuration to file"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved configuration to {CONFIG_FILE}")
    except Exception as e:
        print(f"Error saving config: {e}")

# Load or create configuration
config = load_config()
h_low = config['h_low']
h_high = config['h_high']
s_low = config['s_low']
s_high = config['s_high']
v_low = config['v_low']
v_high = config['v_high']
min_area = config['min_area']

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Operation Game Piece Detection")
print("=" * 50)
print("Press 's' to save current HSV range values")
print("Press 'r' to reset to default values")
print("Press 'q' to quit")
print("=" * 50)

def create_trackbars(window_name):
    """Create trackbars for HSV adjustment"""
    cv2.createTrackbar('H_Low', window_name, h_low, 180, lambda x: None)
    cv2.createTrackbar('H_High', window_name, h_high, 180, lambda x: None)
    cv2.createTrackbar('S_Low', window_name, s_low, 255, lambda x: None)
    cv2.createTrackbar('S_High', window_name, s_high, 255, lambda x: None)
    cv2.createTrackbar('V_Low', window_name, v_low, 255, lambda x: None)
    cv2.createTrackbar('V_High', window_name, v_high, 255, lambda x: None)
    cv2.createTrackbar('Min Area', window_name, min_area, 1000, lambda x: None)

def get_trackbar_values(window_name):
    """Read current trackbar values"""
    h_low = cv2.getTrackbarPos('H_Low', window_name)
    h_high = cv2.getTrackbarPos('H_High', window_name)
    s_low = cv2.getTrackbarPos('S_Low', window_name)
    s_high = cv2.getTrackbarPos('S_High', window_name)
    v_low = cv2.getTrackbarPos('V_Low', window_name)
    v_high = cv2.getTrackbarPos('V_High', window_name)
    min_area = cv2.getTrackbarPos('Min Area', window_name)
    return h_low, h_high, s_low, s_high, v_low, v_high, min_area

# Create window and trackbars
window_name = 'Operation Piece Detection'
cv2.namedWindow(window_name)
create_trackbars(window_name)

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame")
        break
    
    frame_count += 1
    
    # Get current HSV range from trackbars
    h_low, h_high, s_low, s_high, v_low, v_high, min_area = get_trackbar_values(window_name)
    
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for yellow color
    lower_yellow = np.array([h_low, s_low, v_low])
    upper_yellow = np.array([h_high, s_high, v_high])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around detected pieces
    detected_pieces = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            detected_pieces += 1
            
            # Draw rectangle (green for detected)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add label with area
            label = f'Piece {detected_pieces} (A:{int(area)})'
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add info text
    info_text = f'Detected: {detected_pieces} pieces | Frame: {frame_count}'
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    hsv_text = f'HSV Range: H({h_low}-{h_high}) S({s_low}-{s_high}) V({v_low}-{v_high})'
    cv2.putText(frame, hsv_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Show original and mask side by side
    display = np.hstack((frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)))
    cv2.imshow(window_name, display)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save current values to config file
        config = {
            'h_low': h_low,
            'h_high': h_high,
            's_low': s_low,
            's_high': s_high,
            'v_low': v_low,
            'v_high': v_high,
            'min_area': min_area
        }
        save_config(config)
        print(f"Saved HSV Range:")
        print(f"H: {h_low} - {h_high}")
        print(f"S: {s_low} - {s_high}")
        print(f"V: {v_low} - {v_high}")
        print(f"Min Area: {min_area}\n")
    elif key == ord('r'):
        cv2.setTrackbarPos('H_Low', window_name, DEFAULT_CONFIG['h_low'])
        cv2.setTrackbarPos('H_High', window_name, DEFAULT_CONFIG['h_high'])
        cv2.setTrackbarPos('S_Low', window_name, DEFAULT_CONFIG['s_low'])
        cv2.setTrackbarPos('S_High', window_name, DEFAULT_CONFIG['s_high'])
        cv2.setTrackbarPos('V_Low', window_name, DEFAULT_CONFIG['v_low'])
        cv2.setTrackbarPos('V_High', window_name, DEFAULT_CONFIG['v_high'])
        cv2.setTrackbarPos('Min Area', window_name, DEFAULT_CONFIG['min_area'])
        print("Reset to default values\n")

cap.release()
cv2.destroyAllWindows()
