import numpy as np
from app.security.crypto import encrypt_embedding, decrypt_embedding


def vector_to_bytes(vector) -> bytes:
    """
    Convert embedding vector → encrypted bytes (DB safe)
    """
    vector = np.asarray(vector, dtype="float32")
    raw_bytes = vector.tobytes()
    return encrypt_embedding(raw_bytes)


def bytes_to_vector(blob: bytes) -> np.ndarray:
    """
    Convert encrypted bytes → embedding vector (RAM only)
    """
    raw_bytes = decrypt_embedding(blob)
    return np.frombuffer(raw_bytes, dtype="float32")
