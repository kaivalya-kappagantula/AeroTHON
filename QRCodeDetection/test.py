import cv2
from collections import deque, Counter

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera")
    exit()

detector = cv2.QRCodeDetector()

history = deque(maxlen=5)
last_confirmed = None

VALID_CODES = None 

ALIGN_THRESHOLD_X = 40
ALIGN_THRESHOLD_Y = 40

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    h, w = frame.shape[:2]
    frame_center_x = w // 2
    frame_center_y = h // 2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    data, points, _ = detector.detectAndDecode(gray)

    current_read = None
    qr_center = None

    if points is not None and data:
        data = data.strip()

        if VALID_CODES is None or data in VALID_CODES:
            current_read = data
            points = points[0]

            for i in range(4):
                pt1 = (int(points[i][0]), int(points[i][1]))
                pt2 = (int(points[(i + 1) % 4][0]), int(points[(i + 1) % 4][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            cx = int(points[:, 0].mean())
            cy = int(points[:, 1].mean())
            qr_center = (cx, cy)

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Read: {data}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    history.append(current_read)

    filtered = [x for x in history if x is not None]
    confirmed = None

    if filtered:
        counts = Counter(filtered)
        text, count = counts.most_common(1)[0]
        if count >= 3:
            confirmed = text
            last_confirmed = confirmed

    cv2.circle(frame, (frame_center_x, frame_center_y), 6, (255, 255, 0), -1)
    cv2.putText(frame, "Frame Center", (frame_center_x + 10, frame_center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    if confirmed:
        cv2.putText(frame, f"CONFIRMED: {confirmed}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
    elif last_confirmed:
        cv2.putText(frame, f"LAST: {last_confirmed}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

    if qr_center is not None:
        cx, cy = qr_center
        error_x = cx - frame_center_x
        error_y = cy - frame_center_y

        cv2.line(frame, (frame_center_x, frame_center_y), (cx, cy), (255, 0, 255), 2)

        cv2.putText(frame, f"error_x: {error_x}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"error_y: {error_y}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        horizontal_cmd = "ALIGNED X"
        vertical_cmd = "ALIGNED Y"

        if error_x < -ALIGN_THRESHOLD_X:
            horizontal_cmd = "MOVE LEFT"
        elif error_x > ALIGN_THRESHOLD_X:
            horizontal_cmd = "MOVE RIGHT"

        if error_y < -ALIGN_THRESHOLD_Y:
            vertical_cmd = "MOVE UP"
        elif error_y > ALIGN_THRESHOLD_Y:
            vertical_cmd = "MOVE DOWN"

        cv2.putText(frame, horizontal_cmd, (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, vertical_cmd, (20, 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if abs(error_x) <= ALIGN_THRESHOLD_X and abs(error_y) <= ALIGN_THRESHOLD_Y:
            cv2.putText(frame, "QR ALIGNED", (20, 265),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("QR Alignment", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()