from cryptography.fernet import Fernet
import os

KEY_PATH = "secret.key"

def generate_key():
    key = Fernet.generate_key()
    with open(KEY_PATH, "wb") as f:
        f.write(key)
    return key

def load_key():
    if not os.path.exists(KEY_PATH):
        return generate_key()
    with open(KEY_PATH, "rb") as f:
        return f.read()

KEY = load_key()
fernet = Fernet(KEY)

def encrypt_embedding(vector_bytes: bytes) -> bytes:
    return fernet.encrypt(vector_bytes)

def decrypt_embedding(encrypted_bytes: bytes) -> bytes:
    return fernet.decrypt(encrypted_bytes)
