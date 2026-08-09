"""FastAPI guards for protected routes.

Three levels, so a route says exactly what it needs:

* :func:`get_current_user` — a valid token is required.
* :func:`get_optional_user` — a token is used if present; anonymous otherwise.
* :func:`require_admin` — a valid token **and** the admin role.

When ``settings.require_auth`` is false, :func:`get_current_user` returns a
synthetic development user instead of rejecting. That switch exists so the CLI
demos and the evaluation harness keep working without accounts; it must be true
for anything resembling a deployment, and ``/health`` reports which it is.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from auth.models import User, UserRole
from auth.security import InvalidTokenError, decode_access_token
from auth.service import AuthService
from config import Settings, get_settings
from exceptions import DeepVisionAttendError
from logging_config import get_logger

logger = get_logger(__name__)


class NotAuthenticatedError(DeepVisionAttendError):
    """No credentials were supplied."""

    status_code = 401
    error_code = "not_authenticated"


class ForbiddenError(DeepVisionAttendError):
    """The account is signed in but lacks the required role."""

    status_code = 403
    error_code = "forbidden"


#: Stand-in identity used only when authentication is switched off.
ANONYMOUS_DEV_USER = User(
    user_id="dev",
    email="dev@localhost",
    name="Development User",
    password_hash="",
    role=UserRole.ADMIN,
)


def _get_auth_service() -> AuthService:
    """Resolve the auth service from the container.

    Imported lazily to avoid a circular import: ``dependencies`` builds the
    container, and the container builds the auth service.
    """
    from dependencies import get_container

    return get_container().auth_service


def _extract_bearer_token(request: Request) -> str | None:
    """Pull the token out of the Authorization header, if present."""
    header = request.headers.get("Authorization")
    if not header:
        return None

    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def get_current_user(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> User:
    """Resolve the signed-in user, or reject the request.

    Returns:
        The authenticated user.

    Raises:
        NotAuthenticatedError: If no usable token was supplied.
        InvalidTokenError: If the token is expired or forged.
    """
    if not settings.require_auth:
        return ANONYMOUS_DEV_USER

    token = _extract_bearer_token(request)
    if token is None:
        raise NotAuthenticatedError("Sign in to continue.")

    claims = decode_access_token(token, settings.jwt_secret, settings.jwt_algorithm)
    user_id = claims.get("sub")
    if not user_id:
        raise InvalidTokenError("Invalid authentication token.")

    # The user is re-read rather than trusted from the claims: a deleted or
    # demoted account must stop working immediately, not when its token expires.
    user = _get_auth_service().get_user(str(user_id))
    if user is None:
        raise InvalidTokenError("This account no longer exists.")
    return user


def get_optional_user(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> User | None:
    """Resolve the user if a valid token is present, else ``None``.

    Never raises for a bad token — for routes where being signed in changes the
    response but is not required.
    """
    try:
        return get_current_user(request, settings)
    except DeepVisionAttendError:
        return None


def require_admin(
    user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Require the admin role.

    Returns:
        The authenticated admin.

    Raises:
        ForbiddenError: If the account is not an admin.
    """
    if not user.is_admin:
        logger.warning(
            "Blocked a non-admin action by %s (%s)", user.email, user.role.value
        )
        raise ForbiddenError(
            "This action requires an administrator account.",
            details={"required_role": "admin", "your_role": user.role.value},
        )
    return user


#: A signed-in user of any role.
CurrentUser = Annotated[User, Depends(get_current_user)]
#: A signed-in administrator.
AdminUser = Annotated[User, Depends(require_admin)]
