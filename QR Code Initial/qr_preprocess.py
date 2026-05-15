import cv2
import numpy as np

def preprocess(frame, altitude_sim="low"):
    """
    altitude_sim:
        'low'  — 5m, standard contrast enhancement
        'high' — 10m, upscale + sharpen for small QRs
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if altitude_sim == "high":
        gray = cv2.resize(gray, None, fx=2, fy=2,
                          interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        gray = cv2.filter2D(gray, -1, kernel)
    else:
        gray = cv2.equalizeHist(gray)
    return gray


def preprocess_small(frame):
    """Low-res version for use while drone is moving — cheaper FIP detection."""
    small = cv2.resize(frame, (320, 240))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return gray