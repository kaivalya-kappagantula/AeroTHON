import cv2
import numpy as np

def preprocess(frame, altitude_sim="low"):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if altitude_sim == "high":
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        gray = cv2.filter2D(gray, -1, kernel)
    else:
        gray = cv2.equalizeHist(gray)
    return gray