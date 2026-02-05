import time

MAX_REQUEST_AGE = 30  # seconds

def is_fresh_request(timestamp: int) -> bool:
    current = int(time.time())
    return abs(current - timestamp) <= MAX_REQUEST_AGE
