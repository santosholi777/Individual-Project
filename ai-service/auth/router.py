"""The ``/auth`` endpoints.

Kept in an APIRouter rather than ``app.py`` so the auth surface can be read —
and reviewed — in one file.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status

from auth.dependencies import AdminUser, CurrentUser
from auth.schemas import (
    ForgotPasswordRequest,
    ForgotPasswordResponse,
    LoginRequest,
    MessageResponse,
    ResetPasswordRequest,
    SignupRequest,
    TokenResponse,
    UserListResponse,
    UserSchema,
    VerifyResetTokenResponse,
)
from auth.service import AuthResult, AuthService
from config import Settings, get_settings
from logging_config import get_logger
from schemas import ErrorResponse

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


def get_auth_service() -> AuthService:
    """FastAPI dependency: the auth service.

    Imported lazily to avoid a circular import with the composition root.
    """
    from dependencies import get_container

    return get_container().auth_service


AuthDep = Annotated[AuthService, Depends(get_auth_service)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def _to_token_response(result: AuthResult) -> TokenResponse:
    """Map a sign-in result onto the wire format."""
    return TokenResponse(
        access_token=result.access_token,
        token_type=result.token_type,
        expires_in=result.expires_in,
        user=UserSchema(**result.user.to_public_dict()),
    )


@router.post(
    "/signup",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account",
    responses={
        409: {"model": ErrorResponse, "description": "Email already registered"},
        422: {"model": ErrorResponse, "description": "Password too weak"},
        503: {"model": ErrorResponse, "description": "Database unavailable"},
    },
)
def signup(payload: SignupRequest, service: AuthDep) -> TokenResponse:
    """Create an account and sign in immediately.

    The **first** account created becomes an administrator; every account after
    that is a lecturer.
    """
    return _to_token_response(
        service.signup(
            email=payload.email, name=payload.name, password=payload.password
        )
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Sign in",
    responses={
        401: {"model": ErrorResponse, "description": "Incorrect email or password"},
        503: {"model": ErrorResponse, "description": "Database unavailable"},
    },
)
def login(payload: LoginRequest, service: AuthDep) -> TokenResponse:
    """Exchange an email and password for an access token.

    A wrong address and a wrong password return the same error on purpose: any
    difference would let this endpoint be used to discover who has an account.
    """
    return _to_token_response(service.login(email=payload.email, password=payload.password))


@router.get(
    "/me",
    response_model=UserSchema,
    summary="The signed-in account",
    responses={401: {"model": ErrorResponse, "description": "Not signed in"}},
)
def me(user: CurrentUser) -> UserSchema:
    """Return the account the supplied token belongs to.

    The frontend calls this on start-up to restore a session from a stored
    token, which is also what detects a token that has been revoked.
    """
    return UserSchema(**user.to_public_dict())


@router.post(
    "/forgot-password",
    response_model=ForgotPasswordResponse,
    summary="Request a password reset link",
)
def forgot_password(
    payload: ForgotPasswordRequest, service: AuthDep, settings: SettingsDep
) -> ForgotPasswordResponse:
    """Start a password reset.

    Always returns success, whether or not the address is registered — the same
    reason as the login error. In development (``DVA_EXPOSE_RESET_LINK=true``)
    the link comes back in the response and is written to the log; in production
    that field is null and the link would be emailed.
    """
    result = service.request_password_reset(payload.email)
    return ForgotPasswordResponse(
        success=True,
        message=(
            "If an account exists for that address, a password reset link has "
            "been sent to it."
        ),
        reset_token=result.token,
        reset_link=result.link,
    )


@router.get(
    "/reset-password/verify",
    response_model=VerifyResetTokenResponse,
    summary="Check whether a reset link is still valid",
)
def verify_reset_token(token: str, service: AuthDep) -> VerifyResetTokenResponse:
    """Report whether a reset token can still be redeemed.

    Lets the reset page say "this link has expired" up front, rather than after
    the user has typed a new password twice.
    """
    return VerifyResetTokenResponse(valid=service.verify_reset_token(token))


@router.post(
    "/reset-password",
    response_model=MessageResponse,
    summary="Set a new password using a reset link",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or expired token"},
        422: {"model": ErrorResponse, "description": "Password too weak"},
    },
)
def reset_password(payload: ResetPasswordRequest, service: AuthDep) -> MessageResponse:
    """Complete a password reset.

    The token is single-use, and completing a reset also retires every other
    outstanding link for that account.
    """
    user = service.reset_password(token=payload.token, new_password=payload.password)
    return MessageResponse(
        success=True,
        message=f"Password updated for {user.email}. You can now sign in.",
    )


@router.get(
    "/users",
    response_model=UserListResponse,
    summary="List all accounts (admin only)",
    responses={403: {"model": ErrorResponse, "description": "Admin role required"}},
)
def list_users(admin: AdminUser, service: AuthDep) -> UserListResponse:
    """Return every account. Requires the admin role."""
    users = service.list_users()
    return UserListResponse(
        count=len(users),
        users=[UserSchema(**user.to_public_dict()) for user in users],
    )
