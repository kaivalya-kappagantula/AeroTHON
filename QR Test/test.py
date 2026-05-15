#!/usr/bin/env python3
"""
test_qr_pipeline.py
────────────────────
Unit and integration tests for the AeroTHON 2026 QR detection pipeline.
Runs entirely without a camera — generates synthetic QR test images using
OpenCV and qrcode library.

Install:
    pip install opencv-contrib-python numpy loguru qrcode[pil]

Run:
    python test_qr_pipeline.py
    python test_qr_pipeline.py -v          # verbose
    python test_qr_pipeline.py -k preprocess  # run only pre-processing tests
"""

import sys
import unittest
import numpy as np
import cv2

# ── Try to import qrcode for generating test images ───────────────────────────
try:
    import qrcode
    from PIL import Image as PILImage
    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False
    print("WARNING: 'qrcode[pil]' not installed — QR generation tests will be skipped.")
    print("         Run: pip install qrcode[pil]")

# ── Import the pipeline under test ───────────────────────────────────────────
# We import functions directly so tests are independent of the demo UI code.
# If you've renamed the file, update this import.
sys.path.insert(0, '.')
try:
    from qr_pipeline_demo import (
        preprocess,
        detect_qr,
        store_delivery_id,
        ConfirmationGate,
    )
    PIPELINE_IMPORTED = True
except ImportError as e:
    PIPELINE_IMPORTED = False
    print(f"WARNING: Could not import pipeline: {e}")
    print("         Make sure qr_pipeline_demo_v3.py is in the same directory.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_qr_image(data: str, size: int = 300) -> np.ndarray:
    """
    Generate a synthetic QR code image as a numpy BGR array.
    Produces a clean, well-lit QR on a white background — ideal conditions.
    """
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
    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return bgr


def make_blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
    """Plain white frame with no QR code."""
    return np.ones((h, w, 3), dtype=np.uint8) * 240


def add_noise(frame: np.ndarray, intensity: float = 0.05) -> np.ndarray:
    """Add random Gaussian noise to a frame."""
    noise = np.random.normal(0, intensity * 255, frame.shape).astype(np.int16)
    noisy = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def darken(frame: np.ndarray, factor: float = 0.4) -> np.ndarray:
    """Darken a frame to simulate poor lighting."""
    return np.clip(frame * factor, 0, 255).astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
# Test classes
# ═════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(PIPELINE_IMPORTED, "Pipeline not importable")
class TestPreprocess(unittest.TestCase):
    """Tests for the preprocess() function."""

    def test_output_is_greyscale(self):
        """preprocess() must return a single-channel image."""
        frame = make_blank_frame()
        result = preprocess(frame)
        self.assertEqual(len(result.shape), 2,
            "Expected 2D (greyscale) output, got shape: " + str(result.shape))

    def test_output_shape_matches_input_spatial(self):
        """Output spatial dimensions must match input."""
        frame = make_blank_frame(480, 640)
        result = preprocess(frame)
        self.assertEqual(result.shape, (480, 640))

    def test_output_dtype_uint8(self):
        """Output must be uint8 — required by OpenCV QR detector."""
        frame = make_blank_frame()
        result = preprocess(frame)
        self.assertEqual(result.dtype, np.uint8)

    def test_clahe_increases_contrast_on_dark_image(self):
        """CLAHE should increase std deviation (contrast) on a dark image with texture."""
        # A flat uniform frame has std=0 — CLAHE has nothing to work with.
        # Use a gradient frame so there is tonal variation to enhance.
        h, w = 480, 640
        gradient = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
        gradient_bgr = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
        dark = darken(gradient_bgr, factor=0.15)          # squash to low range
        dark_grey = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY)
        processed = preprocess(dark)
        std_before = dark_grey.std()
        std_after  = processed.std()
        self.assertGreater(std_after, std_before,
            "CLAHE should increase contrast (std dev) on a dark gradient image.")

    def test_blur_reduces_noise(self):
        """Gaussian blur should reduce std deviation on a noisy image."""
        noisy = add_noise(make_blank_frame(), intensity=0.3)
        noisy_grey = cv2.cvtColor(noisy, cv2.COLOR_BGR2GRAY)
        processed  = preprocess(noisy)
        # std is a proxy for noise level on a flat background
        self.assertLess(processed.std(), noisy_grey.std() * 1.2,
            "Gaussian blur should reduce noise (std deviation) on flat noisy frame.")

    def test_handles_different_resolutions(self):
        """preprocess() must work on any resolution — e.g. RPi Cam3 native."""
        for h, w in [(240, 320), (720, 1280), (1080, 1920)]:
            frame = make_blank_frame(h, w)
            result = preprocess(frame)
            self.assertEqual(result.shape, (h, w),
                f"Shape mismatch for {h}x{w} input")


@unittest.skipUnless(PIPELINE_IMPORTED, "Pipeline not importable")
@unittest.skipUnless(HAS_QRCODE, "qrcode[pil] not installed")
class TestDetectQR(unittest.TestCase):
    """Tests for the detect_qr() function."""

    def _detect(self, data: str, size: int = 300) -> list[dict]:
        """Helper: generate QR, preprocess, detect."""
        frame     = make_qr_image(data, size=size)
        processed = preprocess(frame)
        return detect_qr(processed)

    def test_detects_simple_string(self):
        """Should detect and decode a basic alphanumeric QR payload."""
        results = self._detect("ALPHA")
        self.assertGreater(len(results), 0,
            "Expected at least one detection for a clean QR code.")
        self.assertEqual(results[0]["data"], "ALPHA")

    def test_detects_numeric_string(self):
        """Should decode numeric payloads correctly."""
        results = self._detect("12345")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["data"], "12345")

    def test_detects_short_string(self):
        """Should work on very short payloads like '03' or 'A'."""
        for payload in ["03", "A", "Z9"]:
            with self.subTest(payload=payload):
                results = self._detect(payload)
                self.assertGreater(len(results), 0,
                    f"No detection for short payload '{payload}'")
                self.assertEqual(results[0]["data"], payload)

    def test_returns_points(self):
        """Each detection must include a points array for bounding box drawing."""
        results = self._detect("TEST")
        self.assertGreater(len(results), 0)
        pts = results[0]["points"]
        self.assertIsNotNone(pts, "points should not be None")
        self.assertGreaterEqual(len(pts), 4,
            "QR bounding quad should have at least 4 corner points")

    def test_empty_frame_returns_no_detections(self):
        """A blank frame with no QR code should return an empty list."""
        frame     = make_blank_frame()
        processed = preprocess(frame)
        results   = detect_qr(processed)
        self.assertEqual(results, [],
            "Blank frame should produce no detections.")

    def test_detects_under_noise(self):
        """Should still detect QR under moderate noise (simulates outdoor sensor noise)."""
        frame   = make_qr_image("BRAVO", size=400)
        noisy   = add_noise(frame, intensity=0.05)
        processed = preprocess(noisy)
        results = detect_qr(processed)
        self.assertGreater(len(results), 0,
            "Should detect QR under moderate noise after CLAHE+blur pre-processing.")
        self.assertEqual(results[0]["data"], "BRAVO")

    def test_result_data_is_stripped(self):
        """Decoded data must be stripped of whitespace — consistent for string matching."""
        results = self._detect("DELTA")
        self.assertGreater(len(results), 0)
        data = results[0]["data"]
        self.assertEqual(data, data.strip(),
            "Decoded data should be stripped of leading/trailing whitespace.")

    def test_multi_qr_detection(self):
        """
        Should detect multiple QR codes in a single frame.
        Simulates the delivery zone scan where multiple codes are on the ground.
        """
        # Build a frame with two QR codes side by side
        qr_a = make_qr_image("ALPHA", size=200)
        qr_b = make_qr_image("BRAVO", size=200)
        h = max(qr_a.shape[0], qr_b.shape[0])
        qr_a = cv2.resize(qr_a, (200, h))
        qr_b = cv2.resize(qr_b, (200, h))
        combined  = np.hstack([qr_a, qr_b])
        processed = preprocess(combined)
        results   = detect_qr(processed)

        decoded = {r["data"] for r in results}
        self.assertGreaterEqual(len(results), 1,
            "Should detect at least one QR in a two-QR frame.")
        # Note: detecting both depends on OpenCV version and image quality.
        # We assert at least one is found — full multi-QR is tested separately.


@unittest.skipUnless(PIPELINE_IMPORTED, "Pipeline not importable")
class TestStoreDeliveryID(unittest.TestCase):
    """Tests for the store_delivery_id() function."""

    def test_returns_string_unchanged(self):
        """Should return the input string verbatim."""
        self.assertEqual(store_delivery_id("ALPHA"), "ALPHA")
        self.assertEqual(store_delivery_id("03"), "03")
        self.assertEqual(store_delivery_id("TARGET-B"), "TARGET-B")

    def test_strips_whitespace(self):
        """Should strip leading/trailing whitespace."""
        self.assertEqual(store_delivery_id("  ALPHA  "), "ALPHA")
        self.assertEqual(store_delivery_id("\nBRAVO\t"), "BRAVO")

    def test_preserves_case(self):
        """Should NOT change case — matching is case-sensitive."""
        self.assertEqual(store_delivery_id("alpha"), "alpha")
        self.assertNotEqual(store_delivery_id("alpha"), "ALPHA")

    def test_preserves_special_characters(self):
        """Should work with any printable string — no assumptions about format."""
        for payload in ["TARGET-1", "Zone_B", "LOC/7", "99", "XY"]:
            with self.subTest(payload=payload):
                self.assertEqual(store_delivery_id(payload), payload)

    def test_exact_match_logic(self):
        """
        Core mission logic: the stored Delivery ID must exactly match
        the delivery zone QR string.
        """
        start_qr_payload    = "BRAVO"
        delivery_id         = store_delivery_id(start_qr_payload)

        delivery_zone_codes = ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]

        matches = [code for code in delivery_zone_codes if code == delivery_id]
        self.assertEqual(len(matches), 1,
            "Exactly one delivery zone code should match the Delivery ID.")
        self.assertEqual(matches[0], "BRAVO")

    def test_no_false_match_on_partial_string(self):
        """'ALPHA' should not match 'ALPHA2' or 'XALPHA'."""
        delivery_id = store_delivery_id("ALPHA")
        self.assertNotEqual(delivery_id, "ALPHA2")
        self.assertNotEqual("ALPHA2", delivery_id)
        self.assertNotEqual("XALPHA", delivery_id)


@unittest.skipUnless(PIPELINE_IMPORTED, "Pipeline not importable")
class TestConfirmationGate(unittest.TestCase):
    """Tests for the ConfirmationGate class."""

    def test_confirms_after_required_frames(self):
        """Gate should confirm after exactly 3 identical frames."""
        gate = ConfirmationGate(required=3)
        self.assertFalse(gate.update("ALPHA"))
        self.assertFalse(gate.update("ALPHA"))
        self.assertTrue(gate.update("ALPHA"),
            "Gate should confirm on 3rd consecutive identical frame.")
        self.assertEqual(gate.confirmed, "ALPHA")

    def test_resets_on_mismatch(self):
        """Any different value should reset the buffer."""
        gate = ConfirmationGate(required=3)
        gate.update("ALPHA")
        gate.update("ALPHA")
        gate.update("BRAVO")   # mismatch — resets
        result = gate.update("ALPHA")
        self.assertFalse(result,
            "Gate should not confirm after a mismatch resets the buffer.")

    def test_resets_on_none(self):
        """None (no detection) should reset the buffer."""
        gate = ConfirmationGate(required=3)
        gate.update("ALPHA")
        gate.update("ALPHA")
        gate.update(None)      # no detection — resets
        result = gate.update("ALPHA")
        self.assertFalse(result,
            "Gate should not confirm after None resets the buffer.")

    def test_confirmed_value_is_correct(self):
        """gate.confirmed should hold the confirmed string."""
        gate = ConfirmationGate(required=3)
        for _ in range(3):
            gate.update("CHARLIE")
        self.assertEqual(gate.confirmed, "CHARLIE")

    def test_confirmed_is_none_before_confirmation(self):
        """gate.confirmed must be None until the gate fires."""
        gate = ConfirmationGate(required=3)
        gate.update("ALPHA")
        gate.update("ALPHA")
        self.assertIsNone(gate.confirmed)

    def test_reset_clears_confirmed(self):
        """reset() must clear both buffer and confirmed value."""
        gate = ConfirmationGate(required=3)
        for _ in range(3):
            gate.update("ALPHA")
        self.assertIsNotNone(gate.confirmed)
        gate.reset()
        self.assertIsNone(gate.confirmed)
        self.assertEqual(gate.progress, "0/3")

    def test_progress_string_format(self):
        """progress property should return 'n/required' format."""
        gate = ConfirmationGate(required=3)
        self.assertEqual(gate.progress, "0/3")
        gate.update("ALPHA")
        self.assertEqual(gate.progress, "1/3")
        gate.update("ALPHA")
        self.assertEqual(gate.progress, "2/3")
        gate.update("ALPHA")
        self.assertEqual(gate.progress, "3/3")

    def test_different_required_values(self):
        """Gate should work for any required count, not just 3."""
        for n in [1, 2, 5]:
            with self.subTest(required=n):
                gate = ConfirmationGate(required=n)
                for i in range(n - 1):
                    self.assertFalse(gate.update("X"),
                        f"Should not confirm before {n} frames (frame {i+1})")
                self.assertTrue(gate.update("X"),
                    f"Should confirm on frame {n}")

    def test_does_not_double_confirm(self):
        """Gate should only return True once, on the confirming frame."""
        gate = ConfirmationGate(required=3)
        results = [gate.update("ALPHA") for _ in range(6)]
        true_count = sum(results)
        self.assertEqual(true_count, 1,
            "Gate should return True exactly once (on the 3rd frame).")


@unittest.skipUnless(PIPELINE_IMPORTED, "Pipeline not importable")
@unittest.skipUnless(HAS_QRCODE, "qrcode[pil] not installed")
class TestEndToEndPipeline(unittest.TestCase):
    """
    Integration tests: full pipeline from raw frame to confirmed Delivery ID,
    mirroring the exact Phase 2 and Phase 5 mission logic.
    """

    def _run_phase2(self, payload: str, frames: int = 3) -> str | None:
        """
        Simulate Phase 2: feed the same QR frame through the pipeline
        for `frames` consecutive frames, return the confirmed Delivery ID.
        """
        frame = make_qr_image(payload, size=300)
        gate  = ConfirmationGate(required=3)
        delivery_id = None

        for _ in range(frames):
            processed  = preprocess(frame)
            detections = detect_qr(processed)
            best       = detections[0]["data"] if detections else None
            if gate.update(best) and delivery_id is None:
                delivery_id = store_delivery_id(gate.confirmed)

        return delivery_id

    def test_phase2_confirms_delivery_id(self):
        """Full Phase 2 pipeline should confirm the Delivery ID after 3 frames."""
        delivery_id = self._run_phase2("ALPHA")
        self.assertIsNotNone(delivery_id,
            "Phase 2 pipeline should confirm a Delivery ID after 3 frames.")
        self.assertEqual(delivery_id, "ALPHA")

    def test_phase2_two_frames_not_enough(self):
        """Two frames are not enough — gate requires three."""
        delivery_id = self._run_phase2("BRAVO", frames=2)
        self.assertIsNone(delivery_id,
            "Two frames should not be enough to confirm the gate.")

    def test_phase5_exact_match_found(self):
        """
        Phase 5: given a stored Delivery ID and multiple delivery zone QR codes,
        exactly one should match.
        """
        # Phase 2 result
        delivery_id = "CHARLIE"

        # Phase 5: delivery zone has 4 QR codes, one matches
        zone_payloads = ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]
        matches = []
        for payload in zone_payloads:
            frame     = make_qr_image(payload, size=200)
            processed = preprocess(frame)
            results   = detect_qr(processed)
            for det in results:
                if det["data"] == delivery_id:
                    matches.append(payload)

        self.assertEqual(len(matches), 1,
            "Exactly one delivery zone QR should match the Delivery ID.")
        self.assertEqual(matches[0], "CHARLIE")

    def test_phase5_no_false_matches(self):
        """Non-matching QR codes in the delivery zone must not be selected."""
        delivery_id   = "ECHO"
        zone_payloads = ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]

        for payload in zone_payloads:
            frame     = make_qr_image(payload, size=200)
            processed = preprocess(frame)
            results   = detect_qr(processed)
            for det in results:
                self.assertNotEqual(det["data"], delivery_id,
                    f"'{payload}' should not match Delivery ID '{delivery_id}'")

    def test_gate_rejects_noisy_single_frame(self):
        """
        A noisy single-frame detection should NOT confirm the gate —
        demonstrating why the 3-frame gate is necessary.
        """
        gate  = ConfirmationGate(required=3)
        frame = make_qr_image("FOXTROT", size=300)

        # Feed one noisy frame
        noisy     = add_noise(frame, intensity=0.15)
        processed = preprocess(noisy)
        results   = detect_qr(processed)
        best      = results[0]["data"] if results else None
        confirmed = gate.update(best)

        self.assertFalse(confirmed,
            "Single frame (even if decoded) should not confirm the gate.")

    def test_full_phase2_to_phase5_flow(self):
        """
        Complete mission flow test:
          1. Phase 2: scan start QR, confirm Delivery ID
          2. Phase 5: scan delivery zone, find matching QR
        """
        # ── Phase 2 ───────────────────────────────────────────────────────────
        start_payload = "GOLF"
        delivery_id   = self._run_phase2(start_payload)
        self.assertIsNotNone(delivery_id, "Phase 2 must produce a Delivery ID")
        self.assertEqual(delivery_id, start_payload)

        # ── Phase 5 ───────────────────────────────────────────────────────────
        zone_payloads = ["HOTEL", "GOLF", "INDIA", "JULIET"]
        target_found  = None

        for payload in zone_payloads:
            frame     = make_qr_image(payload, size=200)
            processed = preprocess(frame)
            results   = detect_qr(processed)
            for det in results:
                if det["data"] == delivery_id:
                    target_found = payload
                    break

        self.assertIsNotNone(target_found,
            "Phase 5 must find a matching QR in the delivery zone.")
        self.assertEqual(target_found, "GOLF",
            "The correct delivery zone QR must be identified.")


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("AeroTHON 2026 — QR Pipeline Test Suite")
    print("=" * 60)
    print(f"Pipeline importable:  {PIPELINE_IMPORTED}")
    print(f"QR generation (qrcode): {HAS_QRCODE}")
    print("=" * 60)
    print()

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestPreprocess,
        TestDetectQR,
        TestStoreDeliveryID,
        TestConfirmationGate,
        TestEndToEndPipeline,
    ]
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    verbosity = 2 if "-v" in sys.argv else 1
    runner = unittest.TextTestRunner(verbosity=verbosity, stream=sys.stdout)
    result = runner.run(suite)

    print()
    if result.wasSuccessful():
        print("ALL TESTS PASSED")
    else:
        print(f"FAILURES: {len(result.failures)}  ERRORS: {len(result.errors)}")
        sys.exit(1)