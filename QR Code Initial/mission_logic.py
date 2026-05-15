import cv2
import time
from qr_detector import detect_qr_codes
from qr_manager import QRManager
from qr_scanner_moving import MovingScanner
from qr_fip import find_fip_candidates, find_qr_regions, draw_fips
from qr_preprocess import preprocess

# ── Camera ───────────────────────────────────────────────────────────
# On Windows/Linux for testing:
cap = cv2.VideoCapture(0)
# On RPi with Camera Module 3, replace above with:
# from pi_camera import PiCamera
# cap = PiCamera()

# ── Config ───────────────────────────────────────────────────────────
ALIGN_THRESHOLD_X = 40
ALIGN_THRESHOLD_Y = 40

PHASE_MOVING   = "moving"
PHASE_HALTED   = "halted"
PHASE_DELIVERY = "delivery"

# ── Helpers ──────────────────────────────────────────────────────────
def draw_detections(frame, detections):
    for qr in detections:
        pts = qr['points']
        if pts:
            for i in range(len(pts)):
                cv2.line(frame, pts[i], pts[(i+1) % len(pts)],
                         (0,255,0), 2)
        cv2.circle(frame, qr['center'], 5, (0,0,255), -1)
        cv2.putText(frame, f"{qr['data']} [{qr['method']}]",
                    (qr['center'][0]+8, qr['center'][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)


def draw_alignment(frame, center, frame_cx, frame_cy):
    cx, cy  = center
    error_x = cx - frame_cx
    error_y = cy - frame_cy

    h_cmd = ("MOVE LEFT"    if error_x < -ALIGN_THRESHOLD_X else
             "MOVE RIGHT"   if error_x >  ALIGN_THRESHOLD_X else
             "ALIGNED X")
    v_cmd = ("MOVE UP"      if error_y < -ALIGN_THRESHOLD_Y else
             "MOVE DOWN"    if error_y >  ALIGN_THRESHOLD_Y else
             "ALIGNED Y")

    cv2.line(frame, (frame_cx, frame_cy), (cx, cy), (255,0,255), 2)
    cv2.putText(frame, f"err_x:{error_x} err_y:{error_y}",
                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.putText(frame, h_cmd, (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(frame, v_cmd, (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    if abs(error_x) <= ALIGN_THRESHOLD_X and abs(error_y) <= ALIGN_THRESHOLD_Y:
        cv2.putText(frame, "*** ALIGNED — DROP PAYLOAD ***", (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    # Return errors so mission logic can use them for MAVLink later
    return error_x, error_y


# ── Main loop ────────────────────────────────────────────────────────
def run():
    if not cap.isOpened():
        print("Could not open camera")
        return

    manager      = QRManager()
    scanner      = MovingScanner()
    phase        = PHASE_MOVING
    altitude_sim = "low"
    prev_time    = time.time()

    print("Controls:")
    print("  'h' — toggle altitude sim (low=5m / high=10m)")
    print("  'm' — manually toggle moving/halted")
    print("  't' — manually promote Phase 1 result to delivery target")
    print("  'r' — reset everything")
    print("  'q' — quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw    = frame.shape[:2]
        frame_cx  = fw // 2
        frame_cy  = fh // 2

        # ── MOVING: scan-on-the-fly ───────────────────────────────────
        if phase == PHASE_MOVING:
            confidence, action, decoded_so_far = scanner.process_frame(frame)
            scanner.draw_overlay(frame)

            cv2.putText(frame, f"PHASE: MOVING | {action.upper()}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200,200,0), 2)

            # Opportunistic: feed any moving decode into Phase 1 history
            if decoded_so_far:
                manager.start_history.append(decoded_so_far)

            # Auto-halt when confidence threshold reached
            if action == 'halt':
                phase = PHASE_HALTED
                scanner.reset()
                print("[Mission] Confidence met — halting for scan")

            # Show FIPs on frame for debug
            gray    = preprocess(frame, altitude_sim)
            fips    = find_fip_candidates(gray)
            regions = find_qr_regions(fips) if len(fips) >= 3 else []
            draw_fips(frame, fips, regions)

        # ── HALTED: full decode, Phase 1 confirmation ─────────────────
        elif phase == PHASE_HALTED:
            detections = detect_qr_codes(frame, altitude_sim)
            draw_detections(frame, detections)

            cv2.putText(frame, "PHASE: HALTED — SCANNING",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0,200,200), 2)

            confirmed = manager.process_start_scan(detections)
            if confirmed:
                cv2.putText(frame, f"TARGET LOCKED: {confirmed}",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0,200,0), 2)
                # Auto-promote and move to delivery phase
                manager.set_delivery_target(confirmed)
                phase = PHASE_DELIVERY

        # ── DELIVERY: match target among multiple QRs ─────────────────
        elif phase == PHASE_DELIVERY:
            detections = detect_qr_codes(frame, altitude_sim)
            draw_detections(frame, detections)

            cv2.putText(frame,
                        f"PHASE: DELIVERY | TARGET: {manager.delivery_target}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255,100,0), 2)

            matched = manager.process_delivery_scan(detections)
            if matched:
                cv2.putText(frame, f"MATCH: {manager.delivery_target}",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0,200,0), 2)
                error_x, error_y = draw_alignment(
                    frame, matched['center'], frame_cx, frame_cy
                )
                # error_x, error_y available here for MAVLink velocity commands

        # ── HUD ───────────────────────────────────────────────────────
        curr_time = time.time()
        fps       = 1.0 / max(curr_time - prev_time, 1e-9)
        prev_time = curr_time

        cv2.circle(frame, (frame_cx, frame_cy), 6, (255,255,0), -1)
        cv2.putText(frame, f"FPS:{fps:.1f}",
                    (fw-110, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0,255,255), 2)
        cv2.putText(frame, f"ALT:{altitude_sim.upper()}",
                    (fw-110, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200,200,0), 2)
        cv2.putText(frame, f"TARGET:{manager.delivery_target or 'NONE'}",
                    (20, fh-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,255), 1)

        cv2.imshow("QR Detection — Downward Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            altitude_sim = "high" if altitude_sim == "low" else "low"
            print(f"Altitude sim: {altitude_sim}")
        elif key == ord('m'):
            phase = PHASE_HALTED if phase == PHASE_MOVING else PHASE_MOVING
            scanner.reset()
            print(f"Phase manually set: {phase}")
        elif key == ord('t'):
            from qr_validator import confirm
            result = confirm(manager.start_history)
            if result:
                manager.set_delivery_target(result)
                phase = PHASE_DELIVERY
            else:
                print("Nothing confirmed in Phase 1 yet")
        elif key == ord('r'):
            manager.reset()
            scanner.reset()
            phase = PHASE_MOVING
            print("Full reset")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()