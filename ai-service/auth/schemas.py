"""Request/response models for the ``/auth`` endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class SignupRequest(BaseModel):
    """Create an account."""

    email: EmailStr = Field(description="Email address; this is the login name")
    name: str = Field(min_length=1, max_length=128, description="Display name")
    password: str = Field(
        min_length=1,
        max_length=72,
        description="Plaintext password. bcrypt caps at 72 bytes.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "lecturer@college.edu",
                "name": "Dr. Meera Krishnan",
                "password": "correct-horse-battery",
            }
        }
    )


class LoginRequest(BaseModel):
    """Sign in."""

    email: EmailStr
    password: str = Field(min_length=1, max_length=72)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "lecturer@college.edu",
                "password": "correct-horse-battery",
            }
        }
    )


class UserSchema(BaseModel):
    """A signed-in account. Never includes the password hash."""

    user_id: str
    email: str
    name: str
    role: Literal["admin", "lecturer"]
    created_at: datetime
    last_login_at: datetime | None = None


class TokenResponse(BaseModel):
    """A successful sign-in or signup."""

    access_token: str = Field(description="JWT — send as: Authorization: Bearer <token>")
    token_type: Literal["bearer"] = "bearer"
    expires_in: int = Field(description="Token lifetime in seconds")
    user: UserSchema


class ForgotPasswordRequest(BaseModel):
    """Ask for a password-reset link."""

    email: EmailStr

    model_config = ConfigDict(
        json_schema_extra={"example": {"email": "lecturer@college.edu"}}
    )


class ForgotPasswordResponse(BaseModel):
    """The reply to a reset request.

    ``message`` is identical whether or not the address is registered — the
    endpoint must not reveal who has an account.
    """

    success: bool
    message: str
    reset_token: str | None = Field(
        default=None,
        description="Development only (DVA_EXPOSE_RESET_LINK). Null in production.",
    )
    reset_link: str | None = Field(
        default=None,
        description="Development only. In production this is emailed instead.",
    )


class ResetPasswordRequest(BaseModel):
    """Set a new password using a reset token."""

    token: str = Field(min_length=1, description="The token from the reset link")
    password: str = Field(min_length=1, max_length=72, description="The new password")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {"token": "a3f9…", "password": "my-new-password"}
        }
    )


class VerifyResetTokenResponse(BaseModel):
    """Whether a reset link is still usable."""

    valid: bool


class MessageResponse(BaseModel):
    """A simple acknowledgement."""

    success: bool
    message: str


class UserListResponse(BaseModel):
    """Every account (admin only)."""

    count: int
    users: list[UserSchema]
