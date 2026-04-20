import cv2
from pyzbar.pyzbar import decode as pyzbar_decode
from qr_preprocess import preprocess

def detect_qr_codes(frame, altitude_sim="low"):
    gray = preprocess(frame, altitude_sim)
    results = []
    for obj in pyzbar_decode(gray):
        data = obj.data.decode('utf-8').strip()
        pts = [(p.x, p.y) for p in obj.polygon]

        if altitude_sim == "high":
            pts = [(x // 2, y // 2) for x, y in pts]

        cx = sum(p[0] for p in pts) // len(pts)
        cy = sum(p[1] for p in pts) // len(pts)
        results.append({'data': data, 'center': (cx, cy), 'points': pts})
    return results