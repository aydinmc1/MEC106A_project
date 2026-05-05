"""
move_above_board_node: listens for board detection, constructs a target pose above the board,
and uses moveit2 (or placeholder) to move the ur robot.
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from ur_board_detection.srv import DetectBoard
from ur_board_detection.tf_pose_utils import (
    make_pose_stamped,
    transform_pose,
    identity_quaternion,
)

import tf2_ros
import tf2_geometry_msgs


class MoveAboveBoardNode(Node):
    def __init__(self):
        super().__init__("move_above_board_node")
        
        # parameters
        self.declare_parameter("base_link_frame", "base_link")
        self.declare_parameter("ee_frame", "tool0")
        self.declare_parameter("z_offset_above_board", 0.1)  # meters
        self.declare_parameter("detector_service", "detect_board")
        self.declare_parameter("use_moveit", True)
        
        self.base_link_frame = self.get_parameter("base_link_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.z_offset_above_board = self.get_parameter("z_offset_above_board").value
        self.detector_service_name = self.get_parameter("detector_service").value
        self.use_moveit = self.get_parameter("use_moveit").value
        
        # tf2 buffer for transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # client for board detection service
        self.detect_board_client = self.create_client(DetectBoard, self.detector_service_name)
        
        # TODO: initialize moveit2 python interface here
        # from moveit2_python import MoveIt2, MoveIt2Gripper
        # self.moveit2 = MoveIt2(self)
        # self.moveit2_gripper = MoveIt2Gripper(self)
        
        self.get_logger().info(f"move_above_board_node started. base frame: {self.base_link_frame}")
    
    def call_detect_board(self):
        """
        call the detect_board service and return the detected board position.
        
        returns:
            dict with 'success': bool, 'x', 'y', 'z': board center in camera frame
            or None if service call fails
        """
        if not self.detect_board_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("detect_board service not available")
            return None
        
        request = DetectBoard.Request()
        request.trigger = True
        
        future = self.detect_board_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is None:
            self.get_logger().error("detect_board service call failed")
            return None
        
        response = future.result()
        
        if not response.success:
            self.get_logger().warn(f"board detection failed: {response.message}")
            return None
        
        self.get_logger().info(f"board detected: {response.message}")
        
        return {
            'success': True,
            'x': response.board_center_x,
            'y': response.board_center_y,
            'z': response.board_center_z,
            'camera_frame': 'camera_color_optical_frame',  # TODO: get from detector node
        }
    
    def move_above_board(self):
        """
        main workflow:
        1. call detect_board service
        2. get detected board position in camera frame
        3. transform to base_link frame
        4. create target pose above board (add z_offset)
        5. call moveit2 to plan and execute
        """
        self.get_logger().info("=== starting move_above_board workflow ===")
        
        # step 1: detect board
        board_result = self.call_detect_board()
        if board_result is None:
            self.get_logger().error("board detection failed, aborting")
            return False
        
        # step 2: create pose in camera frame
        board_pose_camera = make_pose_stamped(
            board_result['x'],
            board_result['y'],
            board_result['z'],
            board_result['camera_frame'],
            orientation_quat=identity_quaternion()
        )
        
        self.get_logger().info(f"board pose in camera frame: {board_pose_camera.pose.position}")
        
        # step 3: transform to base_link
        board_pose_base = transform_pose(
            board_pose_camera,
            self.base_link_frame,
            self.tf_buffer,
            timeout=2.0
        )
        
        if board_pose_base is None:
            self.get_logger().error(f"failed to transform board pose to {self.base_link_frame}")
            return False
        
        self.get_logger().info(f"board pose in {self.base_link_frame}: {board_pose_base.pose.position}")
        
        # step 4: create target pose above board
        target_pose_base = make_pose_stamped(
            board_pose_base.pose.position.x,
            board_pose_base.pose.position.y,
            board_pose_base.pose.position.z + self.z_offset_above_board,
            self.base_link_frame,
            orientation_quat=identity_quaternion()  # TODO: use proper end-effector orientation
        )
        
        self.get_logger().info(f"target pose above board: {target_pose_base.pose.position}")
        
        # step 5: plan and execute motion (moveit2 placeholder)
        if self.use_moveit:
            success = self.plan_and_move_moveit2(target_pose_base)
        else:
            success = self.plan_and_move_placeholder(target_pose_base)
        
        return success
    
    def plan_and_move_moveit2(self, target_pose):
        """
        use moveit2 python interface to plan and execute motion.
        
        TODO: this is a placeholder. real implementation requires:
        - moveit2_python or similar interface
        - group name, ee link, base link configuration
        - collision checking and planning options
        
        args:
            target_pose: geometry_msgs/PoseStamped in base_link frame
        
        returns:
            bool success
        """
        self.get_logger().info("[MoveIt2 PLACEHOLDER] planning motion to target pose...")
        self.get_logger().info(f"target position: ({target_pose.pose.position.x:.3f}, "
                               f"{target_pose.pose.position.y:.3f}, "
                               f"{target_pose.pose.position.z:.3f})")
        
        # TODO: replace with actual moveit2 calls
        # example placeholder code:
        # try:
        #     self.moveit2.set_pose_goal(target_pose)
        #     self.moveit2.plan_and_execute()
        #     self.get_logger().info("motion executed successfully")
        #     return True
        # except Exception as e:
        #     self.get_logger().error(f"moveit2 error: {e}")
        #     return False
        
        self.get_logger().info("[MoveIt2 PLACEHOLDER] motion would be executed here")
        return True
    
    def plan_and_move_placeholder(self, target_pose):
        """
        placeholder for motion planning when moveit2 is not available.
        just logs the target pose for now.
        
        args:
            target_pose: geometry_msgs/PoseStamped
        
        returns:
            bool success
        """
        self.get_logger().info("[PLACEHOLDER] would move robot to:")
        self.get_logger().info(f"  position: ({target_pose.pose.position.x:.3f}, "
                               f"{target_pose.pose.position.y:.3f}, "
                               f"{target_pose.pose.position.z:.3f})")
        self.get_logger().info(f"  orientation: (w={target_pose.pose.orientation.w:.3f}, "
                               f"x={target_pose.pose.orientation.x:.3f}, "
                               f"y={target_pose.pose.orientation.y:.3f}, "
                               f"z={target_pose.pose.orientation.z:.3f})")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = MoveAboveBoardNode()
    
    # run the workflow once and exit (or could be triggered by a service/action)
    success = node.move_above_board()
    
    if success:
        node.get_logger().info("=== workflow completed successfully ===")
    else:
        node.get_logger().error("=== workflow failed ===")
    
    rclpy.shutdown()


if __name__ == "__main__":
    main()
