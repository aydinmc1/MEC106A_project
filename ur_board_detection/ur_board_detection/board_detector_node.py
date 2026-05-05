"""
board_detector_node: subscribes to camera feed, captures snapshots, runs placeholder detection.
publishes detected board info and exposes a DetectBoard service.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, PoseStamped
from ur_board_detection.srv import DetectBoard

from ur_board_detection.tf_pose_utils import (
    pixel_to_camera_3d,
    make_point_stamped,
    make_pose_stamped,
    identity_quaternion,
)


class BoardDetectorNode(Node):
    def __init__(self):
        super().__init__("board_detector_node")
        
        # parameters
        self.declare_parameter("camera_image_topic", "/camera/color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("assumed_board_depth", 0.5)  # meters, distance to board from camera
        
        self.camera_image_topic = self.get_parameter("camera_image_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.assumed_board_depth = self.get_parameter("assumed_board_depth").value
        
        # state
        self.latest_image = None
        self.latest_camera_info = None
        self.cv_bridge = CvBridge()
        
        # qos for image subscriptions (best effort, smaller history)
        img_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # subscriptions
        self.image_sub = self.create_subscription(
            Image,
            self.camera_image_topic,
            self.on_image,
            qos_profile=img_qos
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self.on_camera_info,
            qos_profile=10
        )
        
        # service
        self.detect_service = self.create_service(
            DetectBoard,
            "detect_board",
            self.handle_detect_board_service
        )
        
        # publishers (for debugging/visualization)
        self.board_center_pub = self.create_publisher(PointStamped, "board_center", 10)
        self.detected_board_pose_pub = self.create_publisher(PoseStamped, "detected_board_pose", 10)
        
        self.get_logger().info(f"board_detector_node started. camera image: {self.camera_image_topic}")
    
    def on_image(self, msg):
        """callback for camera image subscription."""
        self.latest_image = msg
    
    def on_camera_info(self, msg):
        """callback for camera info subscription."""
        self.latest_camera_info = msg
    
    def detect_board_placeholder(self, cv_image):
        """
        placeholder board detection function.
        
        in a real scenario, this would run a trained detector (YOLO, etc.) to find board
        edges and features. for now, we mock it:
        - find bright yellow regions (mimics the operation board)
        - estimate board center and rough size
        
        args:
            cv_image: opencv image (bgr)
        
        returns:
            dict with keys:
                'success': bool
                'board_center_px': (x, y) tuple in pixels
                'board_corners_px': list of (x, y) tuples for board corners
                'board_size_px': (width, height) estimate
                'message': string description
        """
        # TODO: replace this with real detector when ready.
        
        # convert to hsv for color-based detection
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # mock board detection: look for bright yellow regions
        # yellow hue range in opencv: ~15-35 (out of 0-180)
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return {
                'success': False,
                'board_center_px': None,
                'board_corners_px': [],
                'board_size_px': None,
                'message': 'no yellow board detected'
            }
        
        # find largest contour (assume it's the board)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < 1000:  # filter out small noise
            return {
                'success': False,
                'board_center_px': None,
                'board_corners_px': [],
                'board_size_px': None,
                'message': f'largest contour too small: {area} px^2'
            }
        
        # get bounding rect and centroid
        x, y, w, h = cv2.boundingRect(largest_contour)
        M = cv2.moments(largest_contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx = x + w // 2
            cy = y + h // 2
        
        # approximate board corners (bounding box corners for now)
        board_corners = [
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h)
        ]
        
        return {
            'success': True,
            'board_center_px': (cx, cy),
            'board_corners_px': board_corners,
            'board_size_px': (w, h),
            'message': f'detected board at ({cx}, {cy}), area={area:.0f} px^2'
        }
    
    def handle_detect_board_service(self, request, response):
        """
        service handler for DetectBoard service.
        captures latest image, runs detection, and returns results.
        """
        if self.latest_image is None:
            response.success = False
            response.message = "no image received yet"
            self.get_logger().warn("detect_board service called but no image available")
            return response
        
        if self.latest_camera_info is None:
            response.success = False
            response.message = "no camera info received yet"
            self.get_logger().warn("detect_board service called but no camera info available")
            return response
        
        # convert ros image to opencv
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(self.latest_image, desired_encoding="bgr8")
        except Exception as e:
            response.success = False
            response.message = f"failed to convert image: {str(e)}"
            return response
        
        # run detection
        detection_result = self.detect_board_placeholder(cv_image)
        
        if not detection_result['success']:
            response.success = False
            response.message = detection_result['message']
            return response
        
        # convert pixel coords to 3d camera coords
        board_center_px = detection_result['board_center_px']
        x_cam, y_cam, z_cam = pixel_to_camera_3d(
            board_center_px[0],
            board_center_px[1],
            self.assumed_board_depth,
            self.latest_camera_info
        )
        
        # populate response
        response.success = True
        response.message = detection_result['message']
        response.board_center_x = x_cam
        response.board_center_y = y_cam
        response.board_center_z = z_cam
        
        # also publish for visualization/debugging
        board_center_point = make_point_stamped(
            x_cam, y_cam, z_cam,
            self.camera_frame,
            timestamp=self.latest_image.header.stamp
        )
        self.board_center_pub.publish(board_center_point)
        
        board_pose = make_pose_stamped(
            x_cam, y_cam, z_cam,
            self.camera_frame,
            orientation_quat=identity_quaternion()
        )
        self.detected_board_pose_pub.publish(board_pose)
        
        self.get_logger().info(f"board detected at camera frame: ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})")
        
        return response


def main(args=None):
    rclpy.init(args=args)
    node = BoardDetectorNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
