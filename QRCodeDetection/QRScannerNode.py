#!/usr/bin/env python3
"""
qr_scanner_node.py
──────────────────
ROS2 node implementing the AeroTHON 2026 QR scanner pipeline.

Works identically on:
  - Real hardware (RPi5 + RPi Camera Module 3)
  - Gazebo simulation (camera plugin publishing sensor_msgs/Image)

The node operates in two modes, switched by the FSM via /qr_scanner/mode:
  - PHASE2: downward camera, single QR, store Delivery ID
  - PHASE5: downward camera, multi-QR, match against Delivery ID,
            project target to NED, trigger confidence-based movement stop

Subscriptions:
    /down_frame             (sensor_msgs/Image)     — camera frames
    /vehicle_state          (geometry_msgs/PoseStamped) — current NED position + altitude
    /qr_scanner/mode        (std_msgs/String)        — 'PHASE2' or 'PHASE5'
    /qr_scanner/delivery_id (std_msgs/String)        — inject stored ID (Phase 5)

Publications:
    /delivery_id            (std_msgs/String)        — confirmed delivery ID (Phase 2)
    /qr_count               (std_msgs/Int32)         — number of QR codes detected this frame
    /target_waypoint        (geometry_msgs/Point)    — target NED position (Phase 5)
    /qr_scanner/stop_move   (std_msgs/Bool)          — True when confidence > stop threshold
    /qr_scanner/status      (std_msgs/String)        — human-readable status for debugging

Parameters (set via ROS2 param file or command line):
    gate_required       int   (default 3)    — frames for confirmation gate
    min_confidence      float (default 0.5)  — minimum confidence to enter gate
    stop_confidence     float (default 0.7)  — confidence to trigger movement stop (Phase 5)
    delivery_altitude   float (default 10.0) — hover altitude in metres for projection
    camera_fx           float                — focal length x (from calibration)
    camera_fy           float                — focal length y (from calibration)
    camera_cx           float                — principal point x (from calibration)
    camera_cy           float                — principal point y (from calibration)

ROS2 dependencies:
    rclpy, sensor_msgs, std_msgs, geometry_msgs, cv_bridge

Run:
    ros2 run aerothon qr_scanner_node
    ros2 run aerothon qr_scanner_node --ros-args -p gate_required:=5
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg   import Image, CameraInfo
from std_msgs.msg      import String, Int32, Bool
from geometry_msgs.msg import Point, PoseStamped

from cv_bridge import CvBridge
from loguru   import logger
import sys

# ── Import pure pipeline logic ────────────────────────────────────────────────
from QRPipeline import (
    preprocess,
    detect_qr,
    quad_confidence,
    store_delivery_id,
    find_matching_qr,
    project_to_ned,
    quad_centroid,
    ConfirmationGate,
    DETECTOR_NAME,
)

# ── Configure loguru to play nicely with ROS2 console ────────────────────────
logger.remove()
logger.add(sys.stdout,
           format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | {message}")


class QRScannerNode(Node):
    """
    ROS2 node for QR code detection in both Phase 2 and Phase 5.

    Mode switching:
        The node starts in IDLE mode and does nothing.
        The FSM publishes 'PHASE2' or 'PHASE5' to /qr_scanner/mode
        to activate the appropriate pipeline.
        Publishing 'IDLE' or 'RESET' stops processing and clears state.
    """

    # Valid mode strings
    MODE_IDLE   = "IDLE"
    MODE_PHASE2 = "PHASE2"
    MODE_PHASE5 = "PHASE5"

    def __init__(self) -> None:
        super().__init__("qr_scanner_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("gate_required",     3)
        self.declare_parameter("min_confidence",    0.5)
        self.declare_parameter("stop_confidence",   0.7)
        self.declare_parameter("delivery_altitude", 10.0)
        # Camera intrinsics — must be set from calibration or /camera_info
        # Defaults are approximate for RPi Camera Module 3 at 1280x720
        self.declare_parameter("camera_fx", 920.0)
        self.declare_parameter("camera_fy", 920.0)
        self.declare_parameter("camera_cx", 640.0)
        self.declare_parameter("camera_cy", 360.0)

        self._gate_required     = self.get_parameter("gate_required").value
        self._min_confidence    = self.get_parameter("min_confidence").value
        self._stop_confidence   = self.get_parameter("stop_confidence").value
        self._delivery_altitude = self.get_parameter("delivery_altitude").value
        self._fx = self.get_parameter("camera_fx").value
        self._fy = self.get_parameter("camera_fy").value
        self._cx = self.get_parameter("camera_cx").value
        self._cy = self.get_parameter("camera_cy").value

        # ── State ──────────────────────────────────────────────────────────────
        self._mode          = self.MODE_IDLE
        self._gate          = ConfirmationGate(required=self._gate_required)
        self._delivery_id   : str | None = None   # set after Phase 2 confirms
        self._stop_sent     : bool       = False   # Phase 5 stop trigger (one-shot)
        self._current_alt   : float      = self._delivery_altitude
        self._bridge        = CvBridge()

        # ── QoS ───────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            Image, "/down_frame", self._frame_cb, sensor_qos)

        self.create_subscription(
            String, "/qr_scanner/mode", self._mode_cb, reliable_qos)

        self.create_subscription(
            String, "/qr_scanner/delivery_id", self._delivery_id_cb, reliable_qos)

        self.create_subscription(
            PoseStamped, "/vehicle_state", self._vehicle_state_cb, sensor_qos)

        self.create_subscription(
            CameraInfo, "/camera_info", self._camera_info_cb, reliable_qos)

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_delivery_id   = self.create_publisher(String,  "/delivery_id",          reliable_qos)
        self._pub_qr_count      = self.create_publisher(Int32,   "/qr_count",              sensor_qos)
        self._pub_target        = self.create_publisher(Point,   "/target_waypoint",       reliable_qos)
        self._pub_stop_move     = self.create_publisher(Bool,    "/qr_scanner/stop_move",  reliable_qos)
        self._pub_status        = self.create_publisher(String,  "/qr_scanner/status",     reliable_qos)

        logger.info(f"qr_scanner_node started | detector: {DETECTOR_NAME}")
        logger.info(f"gate_required={self._gate_required} | "
                    f"min_conf={self._min_confidence} | "
                    f"stop_conf={self._stop_confidence}")
        logger.info("Waiting for mode command on /qr_scanner/mode ...")

    # ═════════════════════════════════════════════════════════════════════════
    # Subscriber callbacks
    # ═════════════════════════════════════════════════════════════════════════

    def _mode_cb(self, msg: String) -> None:
        """Switch operating mode. Published by the mission FSM."""
        new_mode = msg.data.strip().upper()
        if new_mode not in (self.MODE_IDLE, self.MODE_PHASE2, self.MODE_PHASE5, "RESET"):
            logger.warning(f"Unknown mode '{new_mode}' — ignoring.")
            return

        if new_mode == "RESET":
            self._reset()
            return

        if new_mode != self._mode:
            logger.info(f"Mode change: {self._mode} -> {new_mode}")
            self._mode = new_mode
            self._gate.reset()
            self._stop_sent = False
            self._publish_status(f"Mode={self._mode}")

    def _delivery_id_cb(self, msg: String) -> None:
        """
        Allow the FSM to inject the Delivery ID directly for Phase 5.
        In normal operation, Phase 2 publishes to /delivery_id which the
        FSM stores and re-publishes here when activating Phase 5.
        """
        if self._delivery_id is None and msg.data.strip():
            self._delivery_id = msg.data.strip()
            logger.info(f"Delivery ID injected for Phase 5: '{self._delivery_id}'")

    def _vehicle_state_cb(self, msg: PoseStamped) -> None:
        """Update current altitude for NED projection."""
        # PoseStamped z is altitude in NED convention (positive up)
        self._current_alt = abs(msg.pose.position.z)

    def _camera_info_cb(self, msg: CameraInfo) -> None:
        """Update camera intrinsics from /camera_info if available."""
        if len(msg.k) == 9:
            self._fx = msg.k[0]
            self._fy = msg.k[4]
            self._cx = msg.k[2]
            self._cy = msg.k[5]

    def _frame_cb(self, msg: Image) -> None:
        """
        Main pipeline callback — called on every incoming camera frame.
        Dispatches to Phase 2 or Phase 5 handler based on current mode.
        """
        if self._mode == self.MODE_IDLE:
            return

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            logger.warning(f"CvBridge conversion failed: {e}")
            return

        # ── Pre-processing (same for both phases) ─────────────────────────────
        processed  = preprocess(frame)
        detections = detect_qr(processed, min_confidence=self._min_confidence)

        # ── Publish QR count (both phases) ────────────────────────────────────
        count_msg      = Int32()
        count_msg.data = len(detections)
        self._pub_qr_count.publish(count_msg)

        # ── Dispatch to phase handler ─────────────────────────────────────────
        if self._mode == self.MODE_PHASE2:
            self._handle_phase2(detections)
        elif self._mode == self.MODE_PHASE5:
            self._handle_phase5(detections)

    # ═════════════════════════════════════════════════════════════════════════
    # Phase 2 — Initial QR scan
    # ═════════════════════════════════════════════════════════════════════════

    def _handle_phase2(self, detections: list[dict]) -> None:
        """
        Phase 2: drone is hovering at 5 m above start zone QR.
        Feed best detection into gate. On confirmation, store and publish
        Delivery ID, switch to IDLE (FSM will activate PHASE5 later).

        The pipeline also runs during the forward approach movement —
        the gate starts filling as soon as the QR enters the frame.
        """
        if self._delivery_id is not None:
            # Already confirmed — nothing more to do
            return

        best = detections[0]["data"] if detections else None

        if self._gate.update(best):
            # Gate confirmed
            self._delivery_id = store_delivery_id(self._gate.confirmed)
            logger.success(f"[Phase 2] DELIVERY ID CONFIRMED: '{self._delivery_id}'")
            logger.info(f"[Phase 2] Gate buffer: {list(self._gate._buffer)}")
            logger.info(f"[Phase 2] Confidence at confirm: "
                        f"{detections[0]['confidence'] if detections else 'n/a'}")

            # Publish to /delivery_id — FSM uses this as Phase 2 exit condition
            out = String()
            out.data = self._delivery_id
            self._pub_delivery_id.publish(out)
            logger.info(f"[Phase 2] Published to /delivery_id: '{self._delivery_id}'")

            self._publish_status(f"PHASE2_CONFIRMED:{self._delivery_id}")
            self._mode = self.MODE_IDLE   # FSM takes over

        else:
            self._publish_status(
                f"PHASE2_SCANNING gate={self._gate.progress} "
                f"best={best or 'none'}"
            )

    # ═════════════════════════════════════════════════════════════════════════
    # Phase 5 — Delivery zone multi-QR scan
    # ═════════════════════════════════════════════════════════════════════════

    def _handle_phase5(self, detections: list[dict]) -> None:
        """
        Phase 5: drone navigating delivery zone at 10 m altitude.
        Pipeline runs while moving. When any detection exceeds stop_confidence,
        publish stop command. Gate fills on detections matching delivery_id.
        On confirmation, project target to NED and publish /target_waypoint.
        """
        if self._delivery_id is None:
            logger.warning("[Phase 5] No Delivery ID set — cannot match. "
                           "Publish ID to /qr_scanner/delivery_id first.")
            return

        # ── Confidence-triggered movement stop ────────────────────────────────
        if detections and not self._stop_sent:
            best_conf = max(d["confidence"] for d in detections)
            if best_conf >= self._stop_confidence:
                stop_msg      = Bool()
                stop_msg.data = True
                self._pub_stop_move.publish(stop_msg)
                self._stop_sent = True
                logger.info(f"[Phase 5] Confidence {best_conf:.2f} >= "
                             f"{self._stop_confidence} — STOP MOVEMENT published")

        # ── Match against Delivery ID ─────────────────────────────────────────
        # Only pass detections that match the delivery ID into the gate.
        # Non-matching codes are counted (/qr_count) but don't affect gate.
        matching = find_matching_qr(detections, self._delivery_id)
        gate_value = matching["data"] if matching else None

        if self._gate.update(gate_value):
            # Gate confirmed on matching QR
            logger.success(f"[Phase 5] TARGET CONFIRMED: '{self._gate.confirmed}'")

            # ── Project to NED ────────────────────────────────────────────────
            pts = matching["points"]
            u, v = quad_centroid(pts)
            altitude = self._current_alt if self._current_alt > 1.0 \
                       else self._delivery_altitude

            dx, dy = project_to_ned(
                pixel_u=u, pixel_v=v,
                altitude_m=altitude,
                cx=self._cx, cy=self._cy,
                fx=self._fx, fy=self._fy,
            )

            target = Point()
            target.x = dx   # FSM adds current NED x to get absolute position
            target.y = dy   # FSM adds current NED y to get absolute position
            target.z = 0.0  # Altitude handled by FSM descent logic

            self._pub_target.publish(target)
            logger.info(f"[Phase 5] Target NED offset: dx={dx:.3f} m, dy={dy:.3f} m")
            logger.info(f"[Phase 5] Projected at altitude={altitude:.1f} m, "
                        f"pixel=({u:.0f}, {v:.0f})")
            logger.info(f"[Phase 5] Published to /target_waypoint")

            self._publish_status(
                f"PHASE5_CONFIRMED target_dx={dx:.3f} target_dy={dy:.3f}"
            )
            self._mode = self.MODE_IDLE   # FSM takes over for descent

        else:
            status_parts = [f"PHASE5_SCANNING gate={self._gate.progress}"]
            if detections:
                status_parts.append(
                    f"best_conf={max(d['confidence'] for d in detections):.2f}"
                )
            status_parts.append(f"match={'yes' if matching else 'no'}")
            self._publish_status(" ".join(status_parts))

    # ═════════════════════════════════════════════════════════════════════════
    # Helpers
    # ═════════════════════════════════════════════════════════════════════════

    def _reset(self) -> None:
        """Full state reset — call between mission runs."""
        self._mode        = self.MODE_IDLE
        self._delivery_id = None
        self._stop_sent   = False
        self._gate.reset()
        logger.info("QR scanner node reset — all state cleared.")
        self._publish_status("RESET")

    def _publish_status(self, text: str) -> None:
        msg      = String()
        msg.data = text
        self._pub_status.publish(msg)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main(args=None) -> None:
    rclpy.init(args=args)
    node = QRScannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()