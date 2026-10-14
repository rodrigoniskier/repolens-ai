"""Signed session tokens."""

import base64
import hashlib
import hmac
import time


class TokenExpiredError(Exception):
    """Raised when a session token is past its expiry time."""


def issue_token(user_id: str, secret: bytes, ttl_seconds: int = 3600) -> str:
    """Create an HMAC-signed session token that embeds the user id and expiry."""
    expires = int(time.time()) + ttl_seconds
    payload = f"{user_id}:{expires}".encode()
    signature = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    return base64.urlsafe_b64encode(payload).decode() + "." + signature


def validate_token(token: str, secret: bytes) -> str:
    """Verify the signature and expiry of a session token and return the user id."""
    encoded, _, signature = token.partition(".")
    payload = base64.urlsafe_b64decode(encoded.encode())
    expected = hmac.new(secret, payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("invalid token signature")
    user_id, _, expires = payload.decode().partition(":")
    if int(expires) < time.time():
        raise TokenExpiredError(user_id)
    return user_id
