import cv2
import numpy as np
import itertools

FIP_DEPTH_THRESHOLD = 4

def find_fip_candidates(gray):
    """
    Stage 1 of the paper pipeline:
    Canny → contour hierarchy → Douglas-Peucker → FIP candidates
    """
    edges = cv2.Canny(gray, 50, 150)
    contours, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    candidates = []

    for i, _ in enumerate(hierarchy):
        # Count nesting depth
        depth, current = 0, i
        while hierarchy[current][2] != -1:
            depth += 1
            current = hierarchy[current][2]

        if depth < FIP_DEPTH_THRESHOLD:
            continue

        # Douglas-Peucker — must be quadrilateral
        epsilon = 0.02 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)
        if len(approx) != 4:
            continue

        M = cv2.moments(contours[i])
        if M['m00'] == 0:
            continue

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        area = cv2.contourArea(contours[i])
        candidates.append({
            'center': (cx, cy),
            'area': area,
            'contour': contours[i]
        })

    return candidates


def validate_fip_triplet(f1, f2, f3):
    """
    Stage 2 of the paper pipeline:
    Size + distance geometric constraints
    """
    areas = [f1['area'], f2['area'], f3['area']]
    if max(areas) > 2.5 * min(areas):
        return False

    T = max(areas) ** 0.5
    for fa, fb in [(f1,f2), (f1,f3), (f2,f3)]:
        d = np.linalg.norm(
            np.array(fa['center']) - np.array(fb['center'])
        )
        if not (1.6 * T <= d <= 19 * 1.414 * T):
            return False
    return True


def find_qr_regions(candidates):
    """Returns validated FIP triplets with estimated QR center."""
    regions = []
    for triplet in itertools.combinations(candidates, 3):
        if validate_fip_triplet(*triplet):
            cx = sum(f['center'][0] for f in triplet) // 3
            cy = sum(f['center'][1] for f in triplet) // 3
            regions.append({'center': (cx, cy), 'fips': triplet})
    return regions


def draw_fips(frame, fips, regions):
    for f in fips:
        cv2.drawContours(frame, [f['contour']], -1, (255,165,0), 2)
        cv2.circle(frame, f['center'], 4, (255,165,0), -1)
    for r in regions:
        cv2.circle(frame, r['center'], 8, (0,165,255), -1)
        cv2.putText(frame, "FIP GROUP",
                    (r['center'][0]+10, r['center'][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)