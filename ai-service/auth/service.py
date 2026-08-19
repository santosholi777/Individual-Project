"""Signup, login and password-reset rules.

Two security decisions worth reading before changing anything here:

**Login never says which half was wrong.** "No such account" and "wrong
password" return the identical error, because distinguishing them turns the
login form into a tool for discovering who has an account.

**Forgot-password never says whether the address exists.** The response is the
same either way, for the same reason. (The dev-only ``expose_reset_link``
setting deliberately breaks this, which is exactly why it must be off in
production.)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from auth.models import PasswordResetRequest, User, UserRole, normalise_email
from auth.repository import ResetTokenRepository, UserRepository
from auth.security import (
    create_access_token,
    generate_reset_token,
    hash_password,
    hash_reset_token,
    validate_password,
    verify_password,
)
from config import Settings
from domain import utc_now
from exceptions import DeepVisionAttendError
from logging_config import get_logger

logger = get_logger(__name__)


class InvalidCredentialsError(DeepVisionAttendError):
    """The email or the password was wrong — deliberately not saying which."""

    status_code = 401
    error_code = "invalid_credentials"


class InvalidResetTokenError(DeepVisionAttendError):
    """The reset link is wrong, already used, or expired."""

    status_code = 400
    error_code = "invalid_reset_token"


@dataclass(slots=True)
class AuthResult:
    """A successful sign-in."""

    user: User
    access_token: str
    expires_in: int
    token_type: str = "bearer"


@dataclass(slots=True)
class ResetLink:
    """The outcome of a forgot-password request.

    ``token`` and ``link`` are populated only when ``expose_reset_link`` is on
    (development). In production they stay ``None`` and the link is emailed.
    """

    token: str | None = None
    link: str | None = None


class AuthService:
    """Account creation, sign-in and password reset.

    Args:
        settings: Password policy, token lifetimes and JWT configuration.
        users: Account storage.
        reset_tokens: Reset-token storage.
    """

    def __init__(
        self,
        settings: Settings,
        users: UserRepository,
        reset_tokens: ResetTokenRepository,
    ) -> None:
        self._settings = settings
        self._users = users
        self._reset_tokens = reset_tokens

    # ------------------------------------------------------------------
    # Signup
    # ------------------------------------------------------------------
    def signup(self, email: str, name: str, password: str) -> AuthResult:
        """Create an account and sign the new user straight in.

        The first account created becomes an admin (there has to be one, and
        there is nobody to grant it); everyone after is a lecturer.

        Args:
            email: Email address; used as the login identifier.
            name: Display name.
            password: Plaintext password, checked against the policy here.

        Returns:
            The new user and a fresh access token.

        Raises:
            WeakPasswordError: If the password fails the policy.
            EmailAlreadyRegisteredError: If the email is taken.
            DatabaseUnavailableError: If MongoDB is unreachable.
        """
        validate_password(password, self._settings.password_min_length)

        display_name = name.strip()
        if not display_name:
            raise DeepVisionAttendError("Name must not be empty.", details={"field": "name"})

        role = UserRole.LECTURER
        if self._settings.first_user_is_admin and self._users.count() == 0:
            role = UserRole.ADMIN
            logger.info("First account — granting the admin role")

        user = self._users.create(
            User(
                email=normalise_email(email),
                name=display_name,
                password_hash=hash_password(password),
                role=role,
            )
        )
        return self._issue_token(user)

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------
    def login(self, email: str, password: str) -> AuthResult:
        """Verify credentials and issue an access token.

        Args:
            email: The account's email.
            password: The plaintext password.

        Returns:
            The user and a fresh access token.

        Raises:
            InvalidCredentialsError: If either the email or the password is
                wrong. The error does not distinguish them on purpose.
        """
        user = self._users.get_by_email(email)

        if user is None:
            # Hash anyway so a missing account takes the same time as a wrong
            # password: the difference is otherwise measurable, and it leaks
            # which addresses are registered.
            hash_password(password)
            logger.info("Failed sign-in for an unknown address: %s", normalise_email(email))
            raise InvalidCredentialsError("Incorrect email or password.")

        if not verify_password(password, user.password_hash):
            logger.info("Failed sign-in for %s: wrong password", user.email)
            raise InvalidCredentialsError("Incorrect email or password.")

        self._users.record_login(user.user_id)
        logger.info("Sign-in: %s (%s)", user.email, user.role.value)
        return self._issue_token(user)

    # ------------------------------------------------------------------
    # Password reset
    # ------------------------------------------------------------------
    def request_password_reset(self, email: str) -> ResetLink:
        """Start a password reset.

        Always succeeds from the caller's point of view, whether or not the
        address is registered — see the module docstring.

        Args:
            email: The address the user typed.

        Returns:
            The link, when ``expose_reset_link`` is enabled; otherwise empty.
        """
        user = self._users.get_by_email(email)
        if user is None:
            logger.info(
                "Password reset requested for an unknown address: %s", normalise_email(email)
            )
            return ResetLink()

        # One live link at a time: requesting a new one must retire the old.
        self._reset_tokens.invalidate_for_user(user.user_id)

        token, token_hash = generate_reset_token()
        self._reset_tokens.create(
            PasswordResetRequest(
                user_id=user.user_id,
                token_hash=token_hash,
                expires_at=utc_now()
                + timedelta(minutes=self._settings.reset_token_expire_minutes),
            )
        )

        link = f"{self._settings.frontend_url.rstrip('/')}/reset-password?token={token}"
        logger.info(
            "Password reset issued for %s (valid %s minutes)",
            user.email,
            self._settings.reset_token_expire_minutes,
        )

        if self._settings.expose_reset_link:
            # Development only. In production this is where an email would be
            # sent instead, and the link would never reach the response.
            logger.warning("DEV MODE — reset link for %s: %s", user.email, link)
            return ResetLink(token=token, link=link)

        return ResetLink()

    def reset_password(self, token: str, new_password: str) -> User:
        """Complete a password reset.

        Args:
            token: The token from the reset link.
            new_password: The new plaintext password.

        Returns:
            The updated user.

        Raises:
            InvalidResetTokenError: If the token is unknown, used or expired.
            WeakPasswordError: If the new password fails the policy.
        """
        validate_password(new_password, self._settings.password_min_length)

        request = self._reset_tokens.find_valid(hash_reset_token(token))
        if request is None:
            raise InvalidResetTokenError(
                "This password reset link is invalid or has expired. "
                "Please request a new one."
            )

        user = self._users.get_by_id(request.user_id)
        if user is None:
            # The account was deleted between issuing and redeeming the token.
            raise InvalidResetTokenError("This password reset link is no longer valid.")

        self._users.update_password(user.user_id, hash_password(new_password))
        self._reset_tokens.mark_used(request.token_hash)
        # Burn any other live links: a reset means the old ones are stale.
        self._reset_tokens.invalidate_for_user(user.user_id)

        logger.info("Password reset completed for %s", user.email)
        return user

    def verify_reset_token(self, token: str) -> bool:
        """Whether a reset token is currently redeemable.

        Lets the reset page show "this link has expired" before the user types
        a new password twice.
        """
        return self._reset_tokens.find_valid(hash_reset_token(token)) is not None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------
    def get_user(self, user_id: str) -> User | None:
        """Fetch a user by id, for the token guard."""
        return self._users.get_by_id(user_id)

    def list_users(self) -> list[User]:
        """Every account (admin only, enforced at the route)."""
        return self._users.list_all()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _issue_token(self, user: User) -> AuthResult:
        """Sign an access token for a user."""
        token, _ = create_access_token(
            subject=user.user_id,
            secret=self._settings.jwt_secret,
            algorithm=self._settings.jwt_algorithm,
            expires_minutes=self._settings.jwt_expire_minutes,
            claims={"email": user.email, "name": user.name, "role": user.role.value},
        )
        return AuthResult(
            user=user,
            access_token=token,
            expires_in=self._settings.jwt_expire_minutes * 60,
        )
