"""
motion_server_node.py
=====================
Wraps MoveIt2 (via moveit_commander / pymoveit2) and exposes:

  Action servers
  --------------
  /move_to_named_pose   (ur7e_interfaces/action/MoveToNamedPose)

  Service servers
  ---------------
  /get_named_pose       (ur7e_interfaces/srv/GetNamedPose)

  Publishers
  ----------
  /motion/joint_states  (sensor_msgs/JointState) — current joints at 10 Hz

Named poses are defined in config/named_poses.yaml and loaded on startup.
Add new poses there without changing code.
"""

import yaml
import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Header

from ur7e_interfaces.action import MoveToNamedPose
from ur7e_interfaces.srv import GetNamedPose

# NOTE: Replace this stub with real pymoveit2 or moveit_commander bindings
# e.g.: from pymoveit2 import MoveIt2
# We stub it here so the node starts without a running MoveIt2 for development.


class MoveItStub:
    """
    Stub that mimics the pymoveit2 MoveIt2 interface.
    Replace with: from pymoveit2 import MoveIt2
    """
    def move_to_named(self, name: str) -> bool:
        return True   # pretend success

    def get_current_joint_values(self) -> list[float]:
        return [0.0] * 6

    def get_current_pose(self) -> PoseStamped:
        p = PoseStamped()
        p.header.frame_id = "base_link"
        p.pose.orientation.w = 1.0
        return p


class MotionServerNode(Node):
    """Named-pose motion server built on top of MoveIt2."""

    JOINT_NAMES = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    def __init__(self):
        super().__init__("motion_server_node")

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("named_poses_file", "")
        self.declare_parameter("planning_group", "ur_manipulator")
        self.declare_parameter("default_velocity_scaling", 0.3)
        self.declare_parameter("default_acceleration_scaling", 0.2)

        poses_file = self.get_parameter("named_poses_file").value
        self._planning_group = self.get_parameter("planning_group").value
        self._default_vel = self.get_parameter("default_velocity_scaling").value
        self._default_acc = self.get_parameter("default_acceleration_scaling").value

        # ── Load named poses ──────────────────────────────────────────────────
        self._named_poses: dict = {}
        if poses_file:
            self._load_named_poses(poses_file)
        else:
            self.get_logger().warn(
                "named_poses_file parameter not set — no named poses loaded. "
                "Set it in config/named_poses.yaml."
            )

        # ── MoveIt2 interface (swap stub for real) ────────────────────────────
        # For real usage:
        #   from pymoveit2 import MoveIt2
        #   self._moveit = MoveIt2(node=self, joint_names=self.JOINT_NAMES,
        #                          base_link_name="base_link",
        #                          end_effector_name="tool0",
        #                          group_name=self._planning_group)
        self._moveit = MoveItStub()
        self.get_logger().warn(
            "Using MoveIt2 STUB — replace MoveItStub with real pymoveit2 binding."
        )

        # ── Callback group for concurrent action + service ────────────────────
        self._cb_group = ReentrantCallbackGroup()

        # ── Action server ─────────────────────────────────────────────────────
        self._move_action_server = ActionServer(
            self,
            MoveToNamedPose,
            "/move_to_named_pose",
            execute_callback=self._execute_move,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb_group,
        )

        # ── Service servers ───────────────────────────────────────────────────
        self._get_pose_srv = self.create_service(
            GetNamedPose,
            "/get_named_pose",
            self._get_named_pose_callback,
            callback_group=self._cb_group,
        )

        # ── Joint state publisher ─────────────────────────────────────────────
        self._js_pub = self.create_publisher(JointState, "/motion/joint_states", 10)
        self._js_timer = self.create_timer(0.1, self._publish_joint_states)

        self.get_logger().info(
            f"motion_server_node ready\n"
            f"  Named poses: {list(self._named_poses.keys())}\n"
            f"  Planning group: {self._planning_group}"
        )

    # ── Named pose loading ─────────────────────────────────────────────────────

    def _load_named_poses(self, filepath: str) -> None:
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
            self._named_poses = data.get("named_poses", {})
            self.get_logger().info(
                f"Loaded {len(self._named_poses)} named poses from {filepath}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to load named poses: {e}")

    # ── Action server callbacks ────────────────────────────────────────────────

    def _goal_callback(self, goal_request):
        pose_name = goal_request.pose_name
        if pose_name not in self._named_poses:
            self.get_logger().warn(f"Rejecting goal: unknown pose '{pose_name}'")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.get_logger().info("Motion cancel requested")
        return CancelResponse.ACCEPT

    async def _execute_move(self, goal_handle):
        """Execute a MoveToNamedPose action goal."""
        import time
        goal = goal_handle.request
        pose_name = goal.pose_name
        vel_scale = goal.velocity_scaling or self._default_vel
        acc_scale = goal.acceleration_scaling or self._default_acc

        self.get_logger().info(
            f"Executing move to '{pose_name}' "
            f"(vel={vel_scale:.2f}, acc={acc_scale:.2f})"
        )

        feedback = MoveToNamedPose.Feedback()
        result = MoveToNamedPose.Result()

        start_time = time.monotonic()

        # Publish initial feedback
        feedback.progress_percent = 0.0
        feedback.current_pose = self._moveit.get_current_pose()
        goal_handle.publish_feedback(feedback)

        # Execute motion (replace stub with real moveit call)
        success = self._moveit.move_to_named(pose_name)

        # Simulate progress feedback for stub
        for pct in [25.0, 50.0, 75.0, 100.0]:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.message = "Cancelled"
                return result
            feedback.progress_percent = pct
            feedback.current_pose = self._moveit.get_current_pose()
            goal_handle.publish_feedback(feedback)

        result.execution_time_seconds = float(time.monotonic() - start_time)
        result.success = success
        result.message = f"Reached '{pose_name}'" if success else "Motion failed"

        if success:
            goal_handle.succeed()
        else:
            goal_handle.abort()

        return result

    # ── Service callbacks ──────────────────────────────────────────────────────

    def _get_named_pose_callback(self, request, response):
        pose_name = request.pose_name
        if pose_name not in self._named_poses:
            response.success = False
            response.message = f"Unknown pose: '{pose_name}'"
            return response

        pose_data = self._named_poses[pose_name]
        response.success = True
        response.message = f"Pose '{pose_name}' found"
        response.joint_positions = pose_data.get("joints", [])

        # Build PoseStamped from config if cartesian data present
        if "position" in pose_data:
            p = pose_data["position"]
            o = pose_data.get("orientation", {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})
            response.pose.header.frame_id = "base_link"
            response.pose.pose.position.x = p.get("x", 0.0)
            response.pose.pose.position.y = p.get("y", 0.0)
            response.pose.pose.position.z = p.get("z", 0.0)
            response.pose.pose.orientation.x = o.get("x", 0.0)
            response.pose.pose.orientation.y = o.get("y", 0.0)
            response.pose.pose.orientation.z = o.get("z", 0.0)
            response.pose.pose.orientation.w = o.get("w", 1.0)
        return response

    # ── Joint state publisher ──────────────────────────────────────────────────

    def _publish_joint_states(self):
        joints = self._moveit.get_current_joint_values()
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.name = self.JOINT_NAMES
        msg.position = joints
        self._js_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MotionServerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
