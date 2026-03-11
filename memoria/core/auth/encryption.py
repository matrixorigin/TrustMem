"""Token encryption using Fernet symmetric encryption."""

import base64
import hashlib
import os

from cryptography.fernet import Fernet


class TokenEncryption:
    """Encrypt/decrypt tokens using Fernet (AES-128-CBC)."""

    def __init__(self, secret_key: str | None = None):
        """Initialize with secret key from env or parameter.
        
        Raises:
            RuntimeError: If TOKEN_ENCRYPTION_KEY is not set in production.
        """
        key_source = secret_key or os.getenv("TOKEN_ENCRYPTION_KEY")
        
        if not key_source:
            raise RuntimeError(
                "TOKEN_ENCRYPTION_KEY environment variable must be set. "
                "Generate a key with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
            )
        
        # Derive a 32-byte key from arbitrary-length secret
        key = hashlib.sha256(key_source.encode()).digest()
        self._fernet = Fernet(base64.urlsafe_b64encode(key))

    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext token."""
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt encrypted token."""
        return self._fernet.decrypt(ciphertext.encode()).decode()


# Global instance - lazy initialization
_encryptor = None


def _get_encryptor() -> TokenEncryption:
    """Get or create global encryptor instance."""
    global _encryptor
    if _encryptor is None:
        _encryptor = TokenEncryption()
    return _encryptor


def encrypt_token(token: str) -> str:
    """Encrypt a token value."""
    return _get_encryptor().encrypt(token)


def decrypt_token(encrypted: str) -> str:
    """Decrypt a token value."""
    return _get_encryptor().decrypt(encrypted)
