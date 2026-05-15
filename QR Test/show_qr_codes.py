#!/usr/bin/env python3
"""
show_test_qrs.py
─────────────────
Generates and displays all the QR codes used in the test suite.
Useful for visually verifying what the pipeline is actually seeing.

Run:
    python show_test_qrs.py

Controls:
    Any key — next image
    Q / ESC — quit
"""

import sys
import numpy as np
import cv2

try:
    import qrcode
    from PIL import Image as PILImage
except ImportError:
    print("Run: pip install qrcode[pil]")
    sys.exit(1)

# Try importing pre-processing from the pipeline
try:
    from qr_pipeline_demo_v3 import preprocess, detect_qr
    HAS_PIPELINE = True
except ImportError:
    HAS_PIPELINE = False
    print("Note: qr_pipeline_demo_v3.py not found — will show raw QRs only.")


# ── QR generator (same as test suite) ────────────────────────────────────────
def make_qr_image(data: str, size: int = 300) -> np.ndarray:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    pil_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    pil_img = pil_img.resize((size, size), PILImage.LANCZOS)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def add_noise(frame: np.ndarray, intensity: float = 0.05) -> np.ndarray:
    noise = np.random.normal(0, intensity * 255, frame.shape).astype(np.int16)
    return np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def darken(frame: np.ndarray, factor: float = 0.4) -> np.ndarray:
    return np.clip(frame * factor, 0, 255).astype(np.uint8)


def label(img: np.ndarray, text: str, color=(50, 50, 50)) -> np.ndarray:
    """Add a label bar at the bottom of an image."""
    bar = np.ones((36, img.shape[1], 3), dtype=np.uint8) * 245
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def show(title: str, panels: list[tuple[str, np.ndarray]]):
    """
    Show a row of labelled image panels in one window.
    panels = list of (label_text, bgr_image)
    """
    target_h = 336   # QR 300px + 36px label bar
    labelled  = []
    for lbl, img in panels:
        # Resize to target height maintaining aspect ratio
        h, w  = img.shape[:2]
        scale = target_h / (h + 36)
        new_w = max(int(w * scale), 1)
        img_r = cv2.resize(img, (new_w, int(h * scale)))
        labelled.append(label(img_r, lbl))

    # Pad all to same height
    max_h = max(p.shape[0] for p in labelled)
    padded = []
    for p in labelled:
        diff = max_h - p.shape[0]
        if diff > 0:
            pad = np.ones((diff, p.shape[1], 3), dtype=np.uint8) * 245
            p   = np.vstack([p, pad])
        padded.append(p)

    # Add thin vertical dividers
    divider = np.ones((max_h, 2, 3), dtype=np.uint8) * 180
    row = padded[0]
    for p in padded[1:]:
        row = np.hstack([row, divider, p])

    cv2.imshow(title, row)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    return key


# ── Screens ───────────────────────────────────────────────────────────────────

def screen_phase2_qrs():
    """The single start-zone QR codes used in Phase 2 tests."""
    payloads = ["ALPHA", "BRAVO", "CHARLIE", "DELTA",
                "ECHO",  "FOXTROT", "GOLF",  "HOTEL"]
    panels = []
    for p in payloads:
        qr = make_qr_image(p, size=300)
        if HAS_PIPELINE:
            processed  = preprocess(qr)
            detections = detect_qr(processed)
            detected   = detections[0]["data"] if detections else "NOT DETECTED"
            ok = "OK" if detected == p else f"FAIL ({detected})"
            color = (0, 140, 0) if detected == p else (0, 0, 200)
        else:
            ok, color = "?", (100, 100, 100)
        panels.append((f"{p}  [{ok}]", qr))

    return show("Phase 2 — start zone QR codes (any key = next, Q = quit)", panels)


def screen_delivery_zone():
    """Simulated delivery zone — multiple QR codes, one matches the Delivery ID."""
    delivery_id   = "CHARLIE"
    zone_payloads = ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]

    panels = []
    for p in zone_payloads:
        qr      = make_qr_image(p, size=300)
        is_match = (p == delivery_id)
        lbl     = f"{p}  {'<-- TARGET' if is_match else ''}"
        color   = (0, 140, 0) if is_match else (80, 80, 80)

        if HAS_PIPELINE and is_match:
            processed  = preprocess(qr)
            detections = detect_qr(processed)
            detected   = detections[0]["data"] if detections else "NOT DETECTED"
            ok = "DETECTED OK" if detected == p else f"FAIL ({detected})"
            lbl += f"  [{ok}]"

        panels.append((lbl, qr))

    return show(f"Phase 5 — delivery zone (Delivery ID = '{delivery_id}')", panels)


def screen_pipeline_stages():
    """Show one QR through each pre-processing stage side by side."""
    payload = "GOLF"
    raw = make_qr_image(payload, size=300)

    if not HAS_PIPELINE:
        panels = [("Raw", raw)]
        return show("Pipeline stages (pipeline not available)", panels)

    grey    = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq      = clahe.apply(grey)
    blurred = cv2.GaussianBlur(eq, (5, 5), 0)

    # Run detection on final processed frame
    detections = detect_qr(blurred)
    detected   = detections[0]["data"] if detections else "NOT DETECTED"
    ok_str     = "DETECTED" if detected == payload else f"FAIL ({detected})"

    panels = [
        ("1. Raw (colour)",        raw),
        ("2. Greyscale",           cv2.cvtColor(grey,    cv2.COLOR_GRAY2BGR)),
        ("3. CLAHE eq.",           cv2.cvtColor(eq,      cv2.COLOR_GRAY2BGR)),
        ("4. Gaussian blur",       cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)),
    ]
    # Draw detection box on a copy of blurred
    annotated = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    if detections and detections[0]["points"] is not None:
        pts = detections[0]["points"].reshape(-1, 1, 2)
        cv2.polylines(annotated, [pts], True, (0, 200, 0), 2)
        cv2.putText(annotated, detected, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    panels.append((f"5. Detected: {ok_str}", annotated))

    return show(f"Pipeline stages for '{payload}'", panels)


def screen_noise_variants():
    """Show the same QR under different noise/lighting conditions."""
    payload = "INDIA"
    raw     = make_qr_image(payload, size=300)

    variants = [
        ("Clean",           raw),
        ("Noise 5%",        add_noise(raw, 0.05)),
        ("Noise 15%",       add_noise(raw, 0.15)),
        ("Dark (40%)",      darken(raw, 0.4)),
        ("Dark + noise",    add_noise(darken(raw, 0.4), 0.05)),
    ]

    panels = []
    for name, img in variants:
        if HAS_PIPELINE:
            processed  = preprocess(img)
            detections = detect_qr(processed)
            detected   = detections[0]["data"] if detections else "FAIL"
            ok    = "OK" if detected == payload else "FAIL"
            color = (0, 140, 0) if ok == "OK" else (0, 0, 200)
            lbl   = f"{name}  [{ok}]"
        else:
            lbl, color = name, (80, 80, 80)
        panels.append((lbl, img))

    return show(f"Robustness variants for '{payload}'", panels)


def screen_confirmation_gate():
    """Visualise the 3-frame confirmation gate in action."""
    if not HAS_PIPELINE:
        print("Pipeline not available — skipping gate visualisation.")
        return ord('q')

    from qr_pipeline_demo_v3 import ConfirmationGate

    payload = "JULIET"
    qr      = make_qr_image(payload, size=300)
    noisy1  = add_noise(qr, 0.25)   # might fail to decode
    noisy2  = add_noise(qr, 0.05)

    frames = [
        ("Frame 1 (noisy)",  noisy1),
        ("Frame 2 (clean)",  qr),
        ("Frame 3 (clean)",  qr),
        ("Frame 4 (clean)",  qr),
    ]

    gate   = ConfirmationGate(required=3)
    panels = []

    for name, img in frames:
        processed  = preprocess(img)
        detections = detect_qr(processed)
        decoded    = detections[0]["data"] if detections else None
        fired      = gate.update(decoded)

        status = f"decoded='{decoded or 'NONE'}' gate={gate.progress}"
        if fired:
            status += " --> CONFIRMED"
        color  = (0, 140, 0) if fired else (80, 80, 80)

        annotated = img.copy()
        cv2.putText(annotated, gate.progress,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 180, 0) if fired else (0, 130, 255), 2)
        panels.append((f"{name}  {status}", annotated))

    return show(f"Gate demo for '{payload}' (requires 3 consecutive frames)", panels)


# ── Main ──────────────────────────────────────────────────────────────────────

SCREENS = [
    ("Phase 2 — start zone QR codes",      screen_phase2_qrs),
    ("Phase 5 — delivery zone match",      screen_delivery_zone),
    ("Pipeline stages",                    screen_pipeline_stages),
    ("Robustness: noise and lighting",     screen_noise_variants),
    ("3-frame confirmation gate",          screen_confirmation_gate),
]

def main():
    print("=" * 55)
    print("AeroTHON 2026 — Test QR Visualiser")
    print("=" * 55)
    print("Any key = next screen | Q / ESC = quit")
    print()

    for i, (name, fn) in enumerate(SCREENS):
        print(f"[{i+1}/{len(SCREENS)}] {name}")
        key = fn()
        if key in (ord('q'), 27):
            print("Quit.")
            break

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()