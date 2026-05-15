from collections import defaultdict
from qr_validator import make_history, confirm

class QRManager:
    """
    Manages Phase 1 (start QR) and Phase 2 (delivery zone matching).
    Completely decoupled from camera and display logic.

    Output used by mission logic:
        .delivery_target  — the string to match in Phase 2
        process_start_scan()    → returns confirmed string or None
        process_delivery_scan() → returns matched QR dict or None
    """

    def __init__(self):
        self.start_history       = make_history()
        self.delivery_histories  = defaultdict(make_history)
        self.delivery_target     = None

    def process_start_scan(self, detections):
        """Feed Phase 1 detections. Returns confirmed target string or None."""
        self.start_history.append(
            detections[0]['data'] if detections else None
        )
        return confirm(self.start_history)

    def set_delivery_target(self, target):
        self.delivery_target = target
        print(f"[QRManager] Delivery target set: {target}")

    def process_delivery_scan(self, detections):
        """
        Feed Phase 2 detections.
        Returns the matched QR dict (with center) or None.
        """
        if self.delivery_target is None:
            return None
        for qr in detections:
            self.delivery_histories[qr['data']].append(qr['data'])
            if confirm(self.delivery_histories[qr['data']]) == self.delivery_target:
                return qr
        return None

    def reset(self):
        self.__init__()