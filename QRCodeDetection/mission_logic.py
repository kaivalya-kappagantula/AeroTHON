import cv2
import time
from qr_detector import detect_qr_codes
from qr_manager import QRManager

ALIGN_THRESHOLD_X = 40
ALIGN_THRESHOLD_Y = 40

def draw_detections(frame, detections):
    for qr in detections:
        pts = qr['points']
        for i in range(4):
            cv2.line(frame, pts[i], pts[(i+1)%4], (0, 255, 0), 2)
        cv2.circle(frame, qr['center'], 5, (0, 0, 255), -1)
        cv2.putText(frame, qr['data'], (qr['center'][0]+8, qr['center'][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

def draw_alignment(frame, center, frame_cx, frame_cy):
    cx, cy = center
    error_x = cx - frame_cx
    error_y = cy - frame_cy

    h_cmd = "ALIGNED X"
    v_cmd = "ALIGNED Y"
    if error_x < -ALIGN_THRESHOLD_X: h_cmd = "MOVE LEFT"
    elif error_x > ALIGN_THRESHOLD_X: h_cmd = "MOVE RIGHT"
    if error_y < -ALIGN_THRESHOLD_Y: v_cmd = "MOVE UP"
    elif error_y > ALIGN_THRESHOLD_Y: v_cmd = "MOVE DOWN"

    cv2.putText(frame, f"err_x: {error_x}  err_y: {error_y}", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, h_cmd, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    cv2.putText(frame, v_cmd, (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    if abs(error_x) <= ALIGN_THRESHOLD_X and abs(error_y) <= ALIGN_THRESHOLD_Y:
        cv2.putText(frame, "*** ALIGNED — DROP PAYLOAD ***", (20, 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

def run():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera")
        return

    manager = QRManager()
    altitude_sim = "low"
    prev_time = time.time()

    print("Controls: 'h' = toggle altitude | 't' = set delivery target | 'q' = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame_cx, frame_cy = w // 2, h // 2
        detections = detect_qr_codes(frame, altitude_sim)

        # Phase 1
        if manager.delivery_target is None:
            confirmed = manager.process_start_scan(detections)
            if confirmed:
                cv2.putText(frame, f"TARGET LOCKED: {confirmed}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,0), 2)

        # Phase 2
        else:
            matched = manager.process_delivery_scan(detections)
            if matched:
                cv2.putText(frame, f"MATCH: {manager.delivery_target}", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,200,0), 2)
                draw_alignment(frame, matched['center'], frame_cx, frame_cy)

        draw_detections(frame, detections)

        # HUD
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.circle(frame, (frame_cx, frame_cy), 6, (255,255,0), -1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"MODE: {'HIGH' if altitude_sim=='high' else 'LOW'}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)
        cv2.putText(frame, f"TARGET: {manager.delivery_target or 'NOT SET'}", (20, h-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.imshow("QR Test Rig", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('h'):
            altitude_sim = "high" if altitude_sim == "low" else "low"
        elif key == ord('t'):
            confirmed = manager.process_start_scan([])
            from qr_validator import confirm as val_confirm
            result = val_confirm(manager.start_history)
            if result:
                manager.set_delivery_target(result)
            else:
                print("Nothing confirmed in Phase 1 yet")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()