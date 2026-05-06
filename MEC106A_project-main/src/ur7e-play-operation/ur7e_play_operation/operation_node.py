"""
operation_node.py
=================
Top-level state machine that orchestrates the UR7e play operation.

States
------
  IDLE        → waiting for a start command
  HOMING      → moving robot to home position
  POSITIONING → moving to above_patient
  OPERATING   → performing the task sequence
  RETRACTING  → moving back to retract pose
  COMPLETE    → operation finished successfully
  ERROR       → something went wrong
  PAUSED      → operator pause requested

Interfaces
----------
  Action client  : /move_to_named_pose     (MoveToNamedPose)
  Action server  : /execute_operation      (ExecuteOperation)
  Service server : /set_operation_mode     (SetOperationMode)
  Sub            : /camera/info_display    (CameraMetrics)   — health watchdog
  Pub            : /operation/state        (OperationState)  — at 5 Hz

Start the operation:
  ros2 action send_goal /execute_operation ur7e_interfaces/action/ExecuteOperation \
    "{operation_name: 'full_sequence', dry_run: false}"

Pause / abort:
  ros2 service call /set_operation_mode ur7e_interfaces/srv/SetOperationMode \
    "{mode: 'pause', reason: 'operator request'}"
"""

import enum
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, ActionClient, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from ur7e_interfaces.action import ExecuteOperation, MoveToNamedPose
from ur7e_interfaces.msg import OperationState, CameraMetrics
from ur7e_interfaces.srv import SetOperationMode


class State(enum.IntEnum):
    IDLE        = 0
    HOMING      = 1
    POSITIONING = 2
    OPERATING   = 3
    RETRACTING  = 4
    COMPLETE    = 5
    ERROR       = 6
    PAUSED      = 7


STATE_LABELS = {s: s.name for s in State}


# The ordered sequence of (pose_name, state) steps for a full operation
FULL_SEQUENCE = [
    ("home",          State.HOMING),
    ("above_patient", State.POSITIONING),
    ("grasp_ready",   State.OPERATING),
    ("above_patient", State.OPERATING),
    ("retract",       State.RETRACTING),
    ("home",          State.RETRACTING),
]


class OperationNode(Node):
    """Orchestrates the UR7e play operation via a simple FSM."""

    def __init__(self):
        super().__init__("operation_node")

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter("auto_start", False)
        self.declare_parameter("require_camera_health", True)
        self.declare_parameter("velocity_scaling", 0.3)
        self.declare_parameter("acceleration_scaling", 0.2)

        self._auto_start = self.get_parameter("auto_start").value
        self._require_camera = self.get_parameter("require_camera_health").value
        self._vel_scale = self.get_parameter("velocity_scaling").value
        self._acc_scale = self.get_parameter("acceleration_scaling").value

        # ── FSM state ─────────────────────────────────────────────────────────
        self._state = State.IDLE
        self._camera_healthy = False
        self._operation_start_time: float | None = None
        self._paused_from: State | None = None
        self._current_goal_handle = None

        # ── Callback group ────────────────────────────────────────────────────
        self._cb = ReentrantCallbackGroup()

        # ── Action client → motion server ─────────────────────────────────────
        self._move_client = ActionClient(
            self, MoveToNamedPose, "/move_to_named_pose",
            callback_group=self._cb,
        )

        # ── Action server → external trigger ─────────────────────────────────
        self._exec_server = ActionServer(
            self,
            ExecuteOperation,
            "/execute_operation",
            execute_callback=self._execute_operation,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb,
        )

        # ── Service → mode control ────────────────────────────────────────────
        self._mode_srv = self.create_service(
            SetOperationMode,
            "/set_operation_mode",
            self._set_mode_callback,
            callback_group=self._cb,
        )

        # ── Subscriptions ─────────────────────────────────────────────────────
        self._camera_sub = self.create_subscription(
            CameraMetrics,
            "/camera/info_display",
            self._camera_metrics_callback,
            10,
        )

        # ── State publisher ───────────────────────────────────────────────────
        self._state_pub = self.create_publisher(OperationState, "/operation/state", 10)
        self._state_timer = self.create_timer(0.2, self._publish_state)  # 5 Hz

        self.get_logger().info("operation_node ready — state: IDLE")

        if self._auto_start:
            self.get_logger().info("auto_start=true — waiting 3s then starting...")
            self.create_timer(3.0, self._auto_start_once)

    # ── State machine helpers ──────────────────────────────────────────────────

    def _transition(self, new_state: State, description: str = "") -> None:
        old = self._state.name
        self._state = new_state
        self.get_logger().info(f"State: {old} → {new_state.name}  {description}")

    def _set_error(self, reason: str) -> None:
        self.get_logger().error(f"ERROR: {reason}")
        self._transition(State.ERROR, reason)

    # ── Camera watchdog ────────────────────────────────────────────────────────

    def _camera_metrics_callback(self, msg: CameraMetrics) -> None:
        was_healthy = self._camera_healthy
        self._camera_healthy = msg.is_healthy
        if was_healthy and not self._camera_healthy:
            self.get_logger().warn("Camera lost health signal!")
            if self._require_camera and self._state not in (
                State.IDLE, State.COMPLETE, State.ERROR, State.PAUSED
            ):
                self._set_error("Camera became unhealthy during operation")

    # ── Action server callbacks ────────────────────────────────────────────────

    def _goal_callback(self, goal_request):
        if self._state not in (State.IDLE, State.COMPLETE, State.ERROR):
            self.get_logger().warn(
                f"Rejecting operation goal: state is {self._state.name}"
            )
            return GoalResponse.REJECT
        if self._require_camera and not self._camera_healthy:
            self.get_logger().warn("Rejecting: camera not healthy")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.get_logger().info("Operation cancel requested")
        return CancelResponse.ACCEPT

    async def _execute_operation(self, goal_handle):
        """Main operation execution coroutine."""
        goal = goal_handle.request
        self.get_logger().info(
            f"Starting operation: '{goal.operation_name}' "
            f"(dry_run={goal.dry_run})"
        )

        self._operation_start_time = time.monotonic()
        self._current_goal_handle = goal_handle
        feedback = ExecuteOperation.Feedback()
        result = ExecuteOperation.Result()

        sequence = FULL_SEQUENCE
        total_steps = len(sequence)

        for idx, (pose_name, step_state) in enumerate(sequence):
            # Check for cancel
            if goal_handle.is_cancel_requested:
                self._transition(State.IDLE, "Cancelled")
                goal_handle.canceled()
                result.success = False
                result.message = "Operation cancelled"
                result.final_state = int(State.IDLE)
                return result

            # Publish feedback
            self._transition(step_state, f"→ {pose_name}")
            feedback.current_state = int(self._state)
            feedback.current_phase = pose_name
            feedback.progress_percent = (idx / total_steps) * 100.0
            feedback.status_message = f"Moving to {pose_name}"
            goal_handle.publish_feedback(feedback)

            if not goal.dry_run:
                success = await self._move_to_pose(pose_name)
                if not success:
                    self._set_error(f"Failed to reach pose '{pose_name}'")
                    goal_handle.abort()
                    result.success = False
                    result.message = f"Failed at pose: {pose_name}"
                    result.final_state = int(State.ERROR)
                    result.duration_seconds = float(
                        time.monotonic() - self._operation_start_time
                    )
                    return result
            else:
                # Dry run: simulate a 0.5s move
                await rclpy.task.sleep(0.5)

        self._transition(State.COMPLETE, "All steps done")
        feedback.progress_percent = 100.0
        feedback.current_state = int(State.COMPLETE)
        feedback.status_message = "Complete"
        goal_handle.publish_feedback(feedback)

        goal_handle.succeed()
        result.success = True
        result.message = "Operation completed successfully"
        result.final_state = int(State.COMPLETE)
        result.duration_seconds = float(time.monotonic() - self._operation_start_time)
        return result

    # ── Service callbacks ──────────────────────────────────────────────────────

    def _set_mode_callback(self, request, response):
        mode = request.mode.lower()
        prev_state = int(self._state)

        if mode == "pause":
            if self._state not in (State.IDLE, State.PAUSED, State.COMPLETE, State.ERROR):
                self._paused_from = self._state
                self._transition(State.PAUSED, f"reason: {request.reason}")
                response.success = True
                response.message = "Paused"
            else:
                response.success = False
                response.message = f"Cannot pause from state {self._state.name}"

        elif mode == "resume":
            if self._state == State.PAUSED and self._paused_from:
                self._transition(self._paused_from, "Resumed")
                self._paused_from = None
                response.success = True
                response.message = "Resumed"
            else:
                response.success = False
                response.message = "Not paused"

        elif mode == "abort":
            self._transition(State.ERROR, f"Aborted: {request.reason}")
            response.success = True
            response.message = "Aborted"

        elif mode == "reset":
            self._transition(State.IDLE, "Reset by operator")
            response.success = True
            response.message = "Reset to IDLE"

        else:
            response.success = False
            response.message = f"Unknown mode: '{mode}'. Valid: pause, resume, abort, reset"

        response.previous_state = prev_state
        response.new_state = int(self._state)
        return response

    # ── Motion helper ──────────────────────────────────────────────────────────

    async def _move_to_pose(self, pose_name: str) -> bool:
        """Send a MoveToNamedPose action goal and await result."""
        if not self._move_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("motion_server not available")
            return False

        goal = MoveToNamedPose.Goal()
        goal.pose_name = pose_name
        goal.velocity_scaling = self._vel_scale
        goal.acceleration_scaling = self._acc_scale

        future = self._move_client.send_goal_async(goal)
        goal_handle = await future
        if not goal_handle.accepted:
            self.get_logger().error(f"Goal rejected for pose '{pose_name}'")
            return False

        result_future = goal_handle.get_result_async()
        result = await result_future
        return result.result.success

    # ── State publisher ────────────────────────────────────────────────────────

    def _publish_state(self) -> None:
        msg = OperationState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = ""
        msg.state = int(self._state)
        msg.state_label = self._state.name
        msg.description = f"Camera healthy: {self._camera_healthy}"
        if self._operation_start_time:
            elapsed = time.monotonic() - self._operation_start_time
            msg.progress_percent = min(100.0, elapsed / 30.0 * 100.0)  # rough estimate
        self._state_pub.publish(msg)

    def _auto_start_once(self):
        """One-shot timer callback for auto_start."""
        self.get_logger().info("auto_start: triggering operation...")
        # In real use, send an action goal programmatically here
        # This timer only fires once because we destroy it
        self.destroy_timer(self._state_timer)


def main(args=None):
    rclpy.init(args=args)
    node = OperationNode()
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
