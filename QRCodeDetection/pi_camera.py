import cv2
from picamera2 import Picamera2

class PiCamera:
    def __init__(self, width=640, height=480):
        self.picam2 = Picamera2()
        self.picam2.configure(
            self.picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (width, height)}
            )
        )
        self.picam2.start()

    def read(self):
        frame = self.picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame

    def isOpened(self):
        return True

    def release(self):
        self.picam2.stop()