import cv2
import numpy as np
from pyzbar.pyzbar import decode as pyzbar_decode
from qr_preprocess import preprocess
from qr_fip import find_fip_candidates, find_qr_regions

def _warp_and_decode(gray, fips):
    """Perspective-warp the FIP region flat, then attempt pyzbar decode."""
    centers = np.array([f['center'] for f in fips], dtype=np.float32)
    size = 300
    dst = np.array([[0,0],[size,0],[size,size]], dtype=np.float32)
    if len(centers) >= 3:
        M = cv2.getAffineTransform(centers[:3], dst)
        warped = cv2.warpAffine(gray, M, (size, size))
        results = pyzbar_decode(warped)
        if results:
            return results[0].data.decode('utf-8').strip()
    return None


def detect_qr_codes(frame, altitude_sim="low"):
    """
    Primary:  pyzbar on full preprocessed frame (fast path)
    Fallback: FIP pipeline → warp → pyzbar (robust path)

    Returns list of:
        {
            'data':   decoded string,
            'center': (cx, cy) in original frame coords,
            'points': corner points (empty if FIP path used),
            'method': 'pyzbar' | 'fip'
        }
    """
    gray = preprocess(frame, altitude_sim)
    results = []

    # --- PRIMARY: pyzbar ---
    for obj in pyzbar_decode(gray):
        data = obj.data.decode('utf-8').strip()
        pts = [(p.x, p.y) for p in obj.polygon]
        if altitude_sim == "high":
            pts = [(x // 2, y // 2) for x, y in pts]
        cx = sum(p[0] for p in pts) // len(pts)
        cy = sum(p[1] for p in pts) // len(pts)
        results.append({
            'data': data,
            'center': (cx, cy),
            'points': pts,
            'method': 'pyzbar'
        })

    # --- FALLBACK: FIP pipeline ---
    if not results:
        fips = find_fip_candidates(gray)
        if len(fips) >= 3:
            for region in find_qr_regions(fips):
                data = _warp_and_decode(gray, list(region['fips']))
                if data:
                    results.append({
                        'data': data,
                        'center': region['center'],
                        'points': [],
                        'method': 'fip'
                    })

    return results