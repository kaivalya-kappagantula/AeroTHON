from collections import defaultdict
from qr_validator import make_history, confirm

class QRManager:
    def __init__(self):
        self.start_history = make_history()
        self.delivery_histories = defaultdict(make_history)
        self.delivery_target = None

    # --- Phase 1 ---
    def process_start_scan(self, detections):
        if detections:
            self.start_history.append(detections[0]['data'])
        else:
            self.start_history.append(None)
        return confirm(self.start_history)

    def set_delivery_target(self, target):
        self.delivery_target = target
        print(f"[QRManager] Delivery target set: {target}")

    # --- Phase 2 ---
    def process_delivery_scan(self, detections):
        if self.delivery_target is None:
            return None
        for qr in detections:
            code = qr['data']
            self.delivery_histories[code].append(code)
            if confirm(self.delivery_histories[code]) == self.delivery_target:
                return qr   # returns full dict with data, center, points
        return None

    def reset(self):
        self.__init__()