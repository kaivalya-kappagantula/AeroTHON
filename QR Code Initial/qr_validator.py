from collections import deque, Counter

CONFIRM_THRESHOLD = 3
HISTORY_SIZE = 5

def make_history():
    return deque(maxlen=HISTORY_SIZE)

def confirm(history):
    filtered = [x for x in history if x is not None]
    if not filtered:
        return None
    text, count = Counter(filtered).most_common(1)[0]
    return text if count >= CONFIRM_THRESHOLD else None