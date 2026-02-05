import time

_recent_sessions = {}

def replay_protection(user_id, session_id, ttl=10):
    """
    Prevent rapid repeated submissions
    """
    key = f"{user_id}:{session_id}"
    now = time.time()

    if key in _recent_sessions:
        if now - _recent_sessions[key] < ttl:
            return False

    _recent_sessions[key] = now
    return True
