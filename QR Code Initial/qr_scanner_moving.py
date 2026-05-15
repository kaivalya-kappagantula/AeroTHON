import cv2
import numpy as np
from collections import deque
from pyzbar.pyzbar import decode as pyzbar_decode
from qr_preprocess import preprocess_small
from qr_fip import find_fip_candidates

CONFIDENCE_HALT    = 0.80
CONFIDENCE_ATTEMPT = 0.50
MAX_HISTORY        = 10
FIP_STRIDE         = 3   # run FIP every Nth frame to save CPU

class MovingScanner:
    """
    Runs while the drone is moving.
    Builds a confidence score from FIP detection + decode attempts.
    Tells mission logic when to halt for a proper scan.
    """

    def __init__(self):
        self._fip_history    = deque(maxlen=MAX_HISTORY)
        self._decode_history = deque(maxlen=MAX_HISTORY)
        self.confidence      = 0.0
        self.best_decode     = None
        self._frame_count    = 0
        self._last_fips      = []

    def reset(self):
        self._fip_history.clear()
        self._decode_history.clear()
        self.confidence   = 0.0
        self.best_decode  = None
        self._frame_count = 0
        self._last_fips   = []

    # ---------------------------------------------------------------- #
    #  Internal scoring                                                  #
    # ---------------------------------------------------------------- #
    def _fip_score(self, fips, frame_area):
        if not fips:
            return 0.0
        count_score = min(len(fips) / 3.0, 1.0)
        size_scores = []
        for f in fips:
            ratio = f['area'] / frame_area
            if 0.001 < ratio < 0.05:    size_scores.append(1.0)
            elif ratio >= 0.05:          size_scores.append(0.5)
            else:                        size_scores.append(0.2)
        return 0.6 * count_score + 0.4 * float(np.mean(size_scores))

    def _stability_score(self):
        recent = [f for f in self._fip_history if f]
        if len(recent) < 3:
            return 0.0
        centroids = [
            (np.mean([f['center'][0] for f in fs]),
             np.mean([f['center'][1] for f in fs]))
            for fs in recent
        ]
        variance = (np.std([c[0] for c in centroids]) +
                    np.std([c[1] for c in centroids]))
        return max(0.0, 1.0 - variance / 80.0)

    def _decode_score(self):
        if not self._decode_history:
            return 0.0
        return sum(1 for d in self._decode_history if d) / len(self._decode_history)

    def _attempt_decode(self, gray):
        results = pyzbar_decode(gray)
        if results:
            return results[0].data.decode('utf-8').strip()
        up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        results = pyzbar_decode(up)
        if results:
            return results[0].data.decode('utf-8').strip()
        return None

    # ---------------------------------------------------------------- #
    #  Main — call every frame while moving                             #
    # ---------------------------------------------------------------- #
    def process_frame(self, frame):
        """
        Returns:
            confidence : float 0.0–1.0
            action     : 'continue' | 'attempt_decode' | 'halt'
            best_decode: str or None
        """
        self._frame_count += 1
        gray       = preprocess_small(frame)
        frame_area = 320 * 240   # matches preprocess_small output

        # FIP — strided to save CPU
        if self._frame_count % FIP_STRIDE == 0:
            self._last_fips = find_fip_candidates(gray)
        self._fip_history.append(self._last_fips)

        # Decode attempt only when FIPs are promising
        decoded = None
        if len(self._last_fips) >= 2:
            decoded = self._attempt_decode(gray)
        self._decode_history.append(decoded)
        if decoded:
            self.best_decode = decoded

        # Confidence
        fip_s       = self._fip_score(self._last_fips, frame_area)
        stability_s = self._stability_score()
        decode_s    = self._decode_score()
        self.confidence = (0.40 * fip_s +
                           0.30 * stability_s +
                           0.30 * decode_s)

        if self.confidence >= CONFIDENCE_HALT:
            action = 'halt'
        elif self.confidence >= CONFIDENCE_ATTEMPT:
            action = 'attempt_decode'
        else:
            action = 'continue'

        return self.confidence, action, self.best_decode

    def draw_overlay(self, frame):
        h, w = frame.shape[:2]
        bar_w  = int(w * 0.4)
        filled = int(bar_w * self.confidence)
        color  = ((0,255,0)   if self.confidence >= CONFIDENCE_HALT   else
                  (0,165,255) if self.confidence >= CONFIDENCE_ATTEMPT else
                  (0,0,255))
        cv2.rectangle(frame, (20, h-50), (20+bar_w, h-25), (50,50,50), -1)
        cv2.rectangle(frame, (20, h-50), (20+filled, h-25), color, -1)
        cv2.putText(frame, f"Confidence: {self.confidence:.2f}",
                    (20, h-55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,255,255), 2)
        if self.best_decode:
            cv2.putText(frame, f"READING: {self.best_decode}",
                        (20, h-10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,255), 2)