#!/usr/bin/env python3
"""
qr_pipeline_demo_v3.py
───────────────────────
AeroTHON 2026 — QR Detection Pipeline Demo

No pyzbar. No DLL issues. Just OpenCV.

How the mission QR logic works:
  Phase 2 (start zone):    Scan ONE QR code. Store its decoded string verbatim
                           as the Delivery ID — whatever the string happens to be.
  Phase 5 (delivery zone): Scan ALL visible QR codes. Find the one whose decoded
                           string exactly matches the stored Delivery ID.
                           That QR code marks the payload drop location.

  No lookup table. No encoding scheme. Just store the string, match the string.
  The actual QR content is defined by the competition organisers at the event.

Dependencies:
    pip install opencv-contrib-python numpy loguru

Usage:
    python qr_pipeline_demo_v3.py                 # default webcam (index 0)
    python qr_pipeline_demo_v3.py --camera 1      # different camera
    python qr_pipeline_demo_v3.py --image qr.png  # static image

Generate test QR codes (open on phone, hold to webcam):
    https://api.qrserver.com/v1/create-qr-code/?size=400x400&data=ALPHA
    https://api.qrserver.com/v1/create-qr-code/?size=400x400&data=BRAVO
    Scan one to store the Delivery ID, press R, scan the same one again
    to simulate the delivery zone match.

Controls (live camera):
    Q / ESC  — quit
    R        — reset gate + delivery ID (test a new QR)
    S        — save annotated frame as PNG
"""

import argparse
import sys
import time
from collections import deque

import cv2
import numpy as np
from loguru import logger

# ── Logging setup ─────────────────────────────────────────────────────────────
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:<8}</level> | {message}"
)

# ── OpenCV QR detector ────────────────────────────────────────────────────────
# QRCodeDetectorAruco (available in opencv-contrib 4.8+) is more robust
# to perspective distortion and partial occlusion than the base detector.
# Both are pure OpenCV — no external DLLs needed on any platform.
try:
    _detector = cv2.QRCodeDetectorAruco()
    _DETECTOR_NAME = "QRCodeDetectorAruco"
except AttributeError:
    _detector = cv2.QRCodeDetector()
    _DETECTOR_NAME = "QRCodeDetector"

logger.info(f"Detector: cv2.{_DETECTOR_NAME}")
logger.info(f"OpenCV version: {cv2.__version__}")


# ═════════════════════════════════════════════════════════════════════════════
# Pipeline functions
# ═════════════════════════════════════════════════════════════════════════════

def preprocess(frame: np.ndarray) -> np.ndarray:
    """
    Stage 1 — image pre-processing pipeline.

    Exactly mirrors the flight system:
      1. Greyscale conversion
      2. CLAHE equalisation  — improves contrast under variable outdoor lighting
      3. Gaussian blur (5×5) — reduces sensor noise before QR detection

    Returns a single-channel (greyscale) processed image.
    """
    grey    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq      = clahe.apply(grey)
    blurred = cv2.GaussianBlur(eq, (5, 5), 0)
    return blurred


def detect_qr(processed: np.ndarray) -> list[dict]:
    """
    Stage 2 — QR code detection and decoding via OpenCV.

    detectAndDecodeMulti() finds all QR codes in the frame simultaneously.
    Each result contains:
        data   — decoded string (e.g. "03")
        points — 4 corner points of the QR bounding quad (for drawing)

    Falls back to single-QR detectAndDecode() on older OpenCV builds.
    Returns empty list if nothing found.
    """
    results = []

    try:
        # Multi-QR decode (needed for delivery zone scan — multiple codes on ground)
        ret, decoded_list, points_list, _ = _detector.detectAndDecodeMulti(processed)
        if ret and decoded_list:
            for data, pts in zip(decoded_list, points_list):
                if data:
                    results.append({
                        "data":   data.strip(),
                        "points": pts.reshape(-1, 2).astype(int)
                    })

    except (cv2.error, AttributeError):
        # Older OpenCV — single QR only
        try:
            data, pts, _ = _detector.detectAndDecode(processed)
            if data:
                results.append({
                    "data":   data.strip(),
                    "points": pts.reshape(-1, 2).astype(int) if pts is not None else None
                })
        except cv2.error:
            pass

    return results


def store_delivery_id(decoded_string: str) -> str:
    """
    Stage 3 — store the decoded QR string as the Delivery ID.

    The string is stored verbatim exactly as decoded — no mapping,
    no lookup table, no encoding assumption. Whatever the start zone
    QR code contains becomes the Delivery ID. In Phase 5, the delivery
    zone QR whose decoded content exactly matches this string is the target.
    """
    return decoded_string.strip()


# ═════════════════════════════════════════════════════════════════════════════
# Confirmation gate
# ═════════════════════════════════════════════════════════════════════════════

class ConfirmationGate:
    """
    Stage 4 — 3-frame confirmation gate.

    Accepts a detection only when the same decoded string appears in
    `required` consecutive frames without interruption. Any mismatch
    (including a frame with no detection) resets the buffer.

    This rejects single-frame false positives caused by motion blur,
    partial occlusion, or lighting flicker — exactly as in the flight system.
    """

    def __init__(self, required: int = 3):
        self.required  = required
        self._buffer   = deque(maxlen=required)
        self.confirmed: str | None = None
        self._fired    = False             # True returns exactly once

    def update(self, value: str | None) -> bool:
        """
        Feed one frame's decoded string (or None if nothing detected).
        Returns True exactly once — on the frame the gate first confirms.
        Subsequent frames with the same value return False.
        Any mismatch or None resets the buffer and allows re-confirmation.
        """
        if value is None:
            self._buffer.clear()
            return False

        self._buffer.append(value)

        if self._fired:
            return False

        if (len(self._buffer) == self.required
                and len(set(self._buffer)) == 1):
            self.confirmed = value
            self._fired    = True
            return True

        return False

    def reset(self):
        self._buffer.clear()
        self.confirmed = None
        self._fired    = False

    @property
    def progress(self) -> str:
        """Human-readable gate fill status, e.g. '2/3'."""
        if not self._buffer:
            return f"0/{self.required}"
        latest = self._buffer[-1]
        count  = sum(1 for x in self._buffer if x == latest)
        return f"{count}/{self.required}"


# ═════════════════════════════════════════════════════════════════════════════
# Display overlay
# ═════════════════════════════════════════════════════════════════════════════

FONT   = cv2.FONT_HERSHEY_SIMPLEX
GREEN  = (55,  200,  55)
ORANGE = (0,   165, 255)
GREY   = (150, 150, 150)
WHITE  = (220, 220, 220)
BLACK  = (15,   15,  15)
TEAL   = (180, 200,  80)


def _text_bg(img, text, origin, scale, color, thickness=1):
    """Draw text with a dark filled background rectangle."""
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = origin
    cv2.rectangle(img, (x - 2, y - th - 4), (x + tw + 4, y + 4), BLACK, -1)
    cv2.putText(img, text, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray,
                 detections: list[dict],
                 gate: ConfirmationGate,
                 delivery_id: str | None,
                 fps: float,
                 frame_idx: int) -> np.ndarray:

    out = frame.copy()
    h, w = out.shape[:2]

    # ── QR bounding quads and labels ──────────────────────────────────────────
    for det in detections:
        pts  = det["points"]
        data = det["data"]

        # Green once delivery ID is confirmed, orange while still detecting
        color = GREEN if delivery_id else ORANGE

        if pts is not None and len(pts) >= 4:
            cv2.polylines(out, [pts.reshape(-1, 1, 2)], True, color, 2)
            label_pos = (int(pts[0][0]), max(int(pts[0][1]) - 10, 20))
        else:
            label_pos = (10, 50)

        # Show the raw decoded string — no interpretation
        match_marker = "  [MATCH]" if (delivery_id and data.strip() == delivery_id) else ""
        label = f'"{data}"{match_marker}'
        _text_bg(out, label, label_pos, 0.55, color, thickness=1)

    # ── Pipeline stage checklist (top-right) ──────────────────────────────────
    stages = [
        ("Greyscale",       True),
        ("CLAHE eq.",       True),
        ("Gaussian blur",   True),
        ("cv2 detect",      len(detections) > 0),
        ("Gate 3-frame",    gate.confirmed is not None),
        ("Store ID",        delivery_id is not None),
        ("-> /delivery_id", delivery_id is not None),
    ]
    for i, (name, done) in enumerate(stages):
        sym   = "[OK]" if done else "[  ]"
        color = GREEN if done else GREY
        cv2.putText(out, f"{sym} {name}",
                    (w - 215, 26 + i * 22),
                    FONT, 0.46, color, 1, cv2.LINE_AA)

    # ── Status panel (bottom-left) ────────────────────────────────────────────
    lines = [
        (f"FPS {fps:.1f}   frame {frame_idx}",    WHITE),
        (f"Detections:  {len(detections)}",        WHITE),
        (f"Gate:        {gate.progress}",
         GREEN if gate.confirmed else ORANGE),
        (f"Delivery ID: {delivery_id or 'pending'}",
         GREEN if delivery_id else WHITE),
    ]
    lh   = 22
    ph   = len(lines) * lh + 14
    cv2.rectangle(out, (4, h - ph - 4), (320, h - 4), (18, 18, 18), -1)
    cv2.rectangle(out, (4, h - ph - 4), (320, h - 4), (65, 65, 65),  1)
    for i, (text, color) in enumerate(lines):
        cv2.putText(out, text, (10, h - ph + i * lh + lh),
                    FONT, 0.5, color, 1, cv2.LINE_AA)

    # ── Confirmed delivery ID banner (top-centre) ─────────────────────────────
    if delivery_id:
        msg = f"  CONFIRMED: {delivery_id}  "
        (mw, mh), _ = cv2.getTextSize(msg, FONT, 0.8, 2)
        mx = (w - mw) // 2
        cv2.rectangle(out, (mx - 6, 6), (mx + mw + 6, mh + 20), (20, 130, 45), -1)
        cv2.rectangle(out, (mx - 6, 6), (mx + mw + 6, mh + 20), GREEN, 1)
        cv2.putText(out, msg, (mx, mh + 14),
                    FONT, 0.8, (235, 255, 235), 2, cv2.LINE_AA)

    return out


# ═════════════════════════════════════════════════════════════════════════════
# Modes
# ═════════════════════════════════════════════════════════════════════════════

def run_image(path: str):
    """Run the full pipeline on a static image file."""
    logger.info(f"Image mode: {path}")

    frame = cv2.imread(path)
    if frame is None:
        logger.error(f"Cannot load image: {path}")
        sys.exit(1)

    processed  = preprocess(frame)
    detections = detect_qr(processed)
    gate       = ConfirmationGate(required=1)  # single frame — no gate needed

    logger.info(f"Detections: {len(detections)}")

    delivery_id = None
    for det in detections:
        data = det["data"]
        gate.update(data)
        delivery_id = store_delivery_id(data)
        logger.success(f"Decoded: '{data}'")
        logger.info(f"Delivery ID stored: '{delivery_id}'")
        logger.info(f"Published to /delivery_id: '{delivery_id}'")

    if not detections:
        logger.warning("No QR codes detected.")
        logger.info("Check: good lighting, QR is flat and in focus, not too small.")

    display = draw_overlay(frame, detections, gate, delivery_id, fps=0, frame_idx=1)

    # Side-by-side: raw left, processed right
    proc_bgr   = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    h, w       = display.shape[:2]
    proc_bgr   = cv2.resize(proc_bgr, (w // 3, h // 3))
    ph, pw     = proc_bgr.shape[:2]
    display[h - ph : h, 0 : pw] = proc_bgr
    cv2.putText(display, "Processed", (4, h - ph + 14),
                FONT, 0.38, (140, 140, 140), 1)

    cv2.imshow("QR Pipeline — press any key to close", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera(camera_index: int = 0):
    """Live camera loop."""
    logger.info(f"Opening camera {camera_index}")
    logger.info("Hold any QR code in front of the camera to store it as the Delivery ID")
    logger.info("Then press R and scan it again to simulate a delivery zone match")
    logger.info("Q/ESC = quit | R = reset | S = save frame")
    logger.info("─" * 50)

    # CAP_DSHOW is faster on Windows; silently ignored on other platforms
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Cannot open camera {camera_index}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    gate        = ConfirmationGate(required=3)
    delivery_id = None
    fps         = 0.0
    fps_frames  = 0
    fps_t       = time.time()
    frame_idx   = 0
    save_n      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        frame_idx += 1
        fps_frames += 1

        t = time.time()
        if t - fps_t >= 1.0:
            fps       = fps_frames / (t - fps_t)
            fps_frames = 0
            fps_t     = t

        # ── Pipeline ──────────────────────────────────────────────────────────
        processed  = preprocess(frame)
        detections = detect_qr(processed)
        best       = detections[0]["data"] if detections else None

        if gate.update(best) and delivery_id is None:
            delivery_id = store_delivery_id(gate.confirmed)
            logger.success(f"DELIVERY ID CONFIRMED: '{delivery_id}'")
            logger.info(f"Gate buffer: {list(gate._buffer)}")
            logger.info(f"Published to /delivery_id: '{delivery_id}'")

        # ── Display ───────────────────────────────────────────────────────────
        display = draw_overlay(frame, detections, gate, delivery_id, fps, frame_idx)

        # Processed inset (bottom-left)
        proc_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        h, w     = display.shape[:2]
        ph, pw   = h // 4, w // 4
        inset    = cv2.resize(proc_bgr, (pw, ph))
        display[h - ph : h, 0 : pw] = inset
        cv2.putText(display, "Processed (CLAHE+blur)",
                    (4, h - ph + 14), FONT, 0.38, (140, 140, 140), 1)

        cv2.imshow("QR Pipeline Demo — AeroTHON 2026", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('r'):
            gate.reset()
            delivery_id = None
            logger.info("Reset — press R to test a new QR code")
        elif key == ord('s'):
            fname = f"qr_capture_{save_n:03d}.png"
            cv2.imwrite(fname, display)
            logger.info(f"Saved: {fname}")
            save_n += 1

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Done.")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="AeroTHON 2026 QR Detection Pipeline Demo")
    p.add_argument("--camera", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--image",  type=str, default=None, help="Path to a static image")
    args = p.parse_args()

    if args.image:
        run_image(args.image)
    else:
        run_camera(args.camera)


if __name__ == "__main__":
    main()