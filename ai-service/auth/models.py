"""Domain types for accounts.

Framework-free, like the rest of ``domain.py`` — the repository maps these to
and from MongoDB documents, and nothing above the repository knows Mongo exists.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from domain import utc_now


class UserRole(str, enum.Enum):
    """What a signed-in account is allowed to do.

    Two roles is deliberate: the project calls for role-based access control,
    and inventing more of them than the system actually distinguishes would be
    decoration.
    """

    #: Full access, including deleting students and their biometric data.
    ADMIN = "admin"
    #: Can take attendance and read the register.
    LECTURER = "lecturer"


@dataclass(slots=True)
class User:
    """A person who can sign in to the system.

    Distinct from :class:`domain.Student`: a student is *recognised* by the
    system and never signs in; a user *operates* the system and has no face
    enrolled. Conflating them would let anyone with a face log in.
    """

    email: str
    name: str
    password_hash: str
    role: UserRole = UserRole.LECTURER
    user_id: str = ""
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    last_login_at: datetime | None = None

    @property
    def is_admin(self) -> bool:
        """Whether this account holds the admin role."""
        return self.role is UserRole.ADMIN

    def to_public_dict(self) -> dict[str, Any]:
        """Serialise for API responses.

        Deliberately omits ``password_hash``. This is the only representation
        that ever leaves the service, so the hash cannot leak by accident.
        """
        return {
            "user_id": self.user_id,
            "email": self.email,
            "name": self.name,
            "role": self.role.value,
            "created_at": self.created_at.isoformat(),
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }


@dataclass(slots=True)
class PasswordResetRequest:
    """A pending password reset.

    Stores the token's **hash**, never the token itself.
    """

    user_id: str
    token_hash: str
    expires_at: datetime
    created_at: datetime = field(default_factory=utc_now)
    used_at: datetime | None = None

    @property
    def is_used(self) -> bool:
        """Whether this token has already been redeemed."""
        return self.used_at is not None

    def is_expired(self, now: datetime | None = None) -> bool:
        """Whether the token is past its expiry."""
        return (now or utc_now()) >= self.expires_at

    @property
    def is_valid(self) -> bool:
        """Whether the token can still be redeemed."""
        return not self.is_used and not self.is_expired()


def normalise_email(email: str) -> str:
    """Canonicalise an email for storage and lookup.

    Addresses are case-insensitive in practice, so they are lowercased — this
    is what stops ``Bob@x.com`` and ``bob@x.com`` becoming two accounts.
    """
    return email.strip().lower()
