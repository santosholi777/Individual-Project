"""Password hashing, JWT signing and reset-token generation.

The cryptographic decisions live here and nowhere else, so they can be reviewed
in one place:

* **Passwords** are hashed with **bcrypt** (per-password random salt, adaptive
  cost). Plaintext passwords are never stored, logged or returned.
* **Access tokens** are signed JWTs (HS256). They are *signed*, not encrypted —
  anyone can read the claims, so no secret ever goes in them.
* **Reset tokens** are 256 bits from a CSPRNG. Only their SHA-256 hash is
  stored, so a leaked database still cannot be used to reset anyone's password.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import bcrypt
import jwt

from exceptions import DeepVisionAttendError
from logging_config import get_logger

logger = get_logger(__name__)

#: bcrypt hashes at most 72 bytes and raises beyond that, so signup rejects
#: longer passwords rather than silently ignoring the tail.
MAX_PASSWORD_BYTES = 72


class InvalidTokenError(DeepVisionAttendError):
    """A token was missing, malformed, expired or signed with another key."""

    status_code = 401
    error_code = "invalid_token"


class WeakPasswordError(DeepVisionAttendError):
    """The supplied password does not meet the policy."""

    status_code = 422
    error_code = "weak_password"


# ----------------------------------------------------------------------
# Passwords
# ----------------------------------------------------------------------
def validate_password(password: str, minimum_length: int) -> None:
    """Check a password against the policy.

    Args:
        password: The plaintext password.
        minimum_length: Shortest acceptable length.

    Raises:
        WeakPasswordError: If the password is too short, too long for bcrypt,
            or entirely whitespace.
    """
    if len(password) < minimum_length:
        raise WeakPasswordError(
            f"Password must be at least {minimum_length} characters long.",
            details={"minimum_length": minimum_length},
        )
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise WeakPasswordError(
            f"Password must be at most {MAX_PASSWORD_BYTES} bytes long.",
            details={"maximum_bytes": MAX_PASSWORD_BYTES},
        )
    if not password.strip():
        raise WeakPasswordError("Password must not be only whitespace.")


def hash_password(password: str) -> str:
    """Hash a password with bcrypt.

    Args:
        password: The plaintext password.

    Returns:
        The bcrypt hash, including its salt and cost, as text.
    """
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    """Check a password against a stored bcrypt hash.

    Args:
        password: The plaintext candidate.
        password_hash: The stored hash.

    Returns:
        True when the password matches. A malformed stored hash returns False
        rather than raising — a corrupt row must not become a 500.
    """
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except (ValueError, TypeError) as exc:
        logger.warning("Rejecting a malformed password hash: %s", exc)
        return False


# ----------------------------------------------------------------------
# Access tokens
# ----------------------------------------------------------------------
def create_access_token(
    subject: str,
    secret: str,
    algorithm: str = "HS256",
    expires_minutes: int = 720,
    claims: dict[str, Any] | None = None,
) -> tuple[str, datetime]:
    """Sign a JWT access token.

    Args:
        subject: The user id the token identifies (``sub``).
        secret: Signing secret.
        algorithm: JWS algorithm.
        expires_minutes: Token lifetime.
        claims: Extra public claims, e.g. role and email. Never put a secret
            here — JWT payloads are readable by anyone holding the token.

    Returns:
        ``(token, expires_at)``.
    """
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(minutes=expires_minutes)
    payload: dict[str, Any] = {
        "sub": subject,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        **(claims or {}),
    }
    return jwt.encode(payload, secret, algorithm=algorithm), expires_at


def decode_access_token(token: str, secret: str, algorithm: str = "HS256") -> dict[str, Any]:
    """Verify a JWT's signature and expiry, and return its claims.

    Args:
        token: The encoded token.
        secret: Signing secret.
        algorithm: Expected algorithm. Passing this explicitly is what prevents
            the classic "alg: none" forgery.

    Returns:
        The decoded claims.

    Raises:
        InvalidTokenError: If the token is expired, forged or malformed. The
            reason is deliberately vague to the client and specific in the log.
    """
    try:
        return jwt.decode(token, secret, algorithms=[algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise InvalidTokenError("Your session has expired. Please sign in again.") from exc
    except jwt.InvalidTokenError as exc:
        logger.warning("Rejected an invalid token: %s", exc)
        raise InvalidTokenError("Invalid authentication token.") from exc


# ----------------------------------------------------------------------
# Reset tokens
# ----------------------------------------------------------------------
def generate_reset_token() -> tuple[str, str]:
    """Create a password-reset token.

    Returns:
        ``(token, token_hash)``. Send the token to the user; store only the
        hash. A read-only leak of the database then reveals nothing usable.
    """
    token = secrets.token_urlsafe(32)
    return token, hash_reset_token(token)


def hash_reset_token(token: str) -> str:
    """Hash a reset token for storage and lookup.

    SHA-256 rather than bcrypt: the token is already 256 bits of entropy from a
    CSPRNG, so there is nothing to brute-force and no need for a slow hash.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
