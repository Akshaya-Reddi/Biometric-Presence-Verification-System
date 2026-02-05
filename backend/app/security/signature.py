import hmac
import hashlib

SECRET = b"super-secret-key"

def generate_signature(payload: str) -> str:
    return hmac.new(SECRET, payload.encode(), hashlib.sha256).hexdigest()

def verify_signature(payload: str, signature: str) -> bool:
    expected = generate_signature(payload)
    return hmac.compare_digest(expected, signature)
