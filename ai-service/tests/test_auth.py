"""Tests for accounts, sign-in and password reset.

MongoDB is replaced with in-memory repositories, so the suite stays offline and
fast. The security logic — hashing, JWT, token single-use, expiry, and the
deliberate refusal to reveal which accounts exist — is the real code.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from auth.models import PasswordResetRequest, User, UserRole, normalise_email
from auth.repository import (
    EmailAlreadyRegisteredError,
    ResetTokenRepository,
    UserRepository,
)
from auth.security import (
    InvalidTokenError,
    WeakPasswordError,
    create_access_token,
    decode_access_token,
    generate_reset_token,
    hash_password,
    hash_reset_token,
    validate_password,
    verify_password,
)
from auth.service import AuthService, InvalidCredentialsError, InvalidResetTokenError
from config import Settings
from domain import utc_now

#: HS256 keys must be at least 32 bytes (RFC 7518 §3.2); PyJWT warns otherwise.
TEST_SECRET = "test-signing-secret-at-least-32-bytes-long!"
OTHER_SECRET = "a-completely-different-secret-also-32-bytes+"


class InMemoryUserRepository(UserRepository):
    """A dict-backed stand-in for MongoUserRepository."""

    def __init__(self) -> None:
        self._users: dict[str, User] = {}
        self._next_id = 1

    def create(self, user: User) -> User:
        """Insert, enforcing the unique-email rule the Mongo index provides."""
        email = normalise_email(user.email)
        if any(existing.email == email for existing in self._users.values()):
            raise EmailAlreadyRegisteredError("An account with this email already exists.")
        user.user_id = str(self._next_id)
        user.email = email
        self._next_id += 1
        self._users[user.user_id] = user
        return user

    def get_by_email(self, email: str) -> User | None:
        """Find by normalised email."""
        target = normalise_email(email)
        return next((u for u in self._users.values() if u.email == target), None)

    def get_by_id(self, user_id: str) -> User | None:
        """Find by id."""
        return self._users.get(user_id)

    def update_password(self, user_id: str, password_hash: str) -> None:
        """Replace the stored hash."""
        if user_id in self._users:
            self._users[user_id].password_hash = password_hash

    def record_login(self, user_id: str) -> None:
        """Stamp the sign-in time."""
        if user_id in self._users:
            self._users[user_id].last_login_at = utc_now()

    def count(self) -> int:
        """Number of accounts."""
        return len(self._users)

    def list_all(self) -> list[User]:
        """Every account."""
        return list(self._users.values())


class InMemoryResetTokenRepository(ResetTokenRepository):
    """A list-backed stand-in for MongoResetTokenRepository."""

    def __init__(self) -> None:
        self.requests: list[PasswordResetRequest] = []

    def create(self, request: PasswordResetRequest) -> None:
        """Store a pending reset."""
        self.requests.append(request)

    def find_valid(self, token_hash: str) -> PasswordResetRequest | None:
        """Return an unused, unexpired reset."""
        return next(
            (r for r in self.requests if r.token_hash == token_hash and r.is_valid),
            None,
        )

    def mark_used(self, token_hash: str) -> None:
        """Burn one token."""
        for request in self.requests:
            if request.token_hash == token_hash:
                request.used_at = utc_now()

    def invalidate_for_user(self, user_id: str) -> None:
        """Burn every live token for a user."""
        for request in self.requests:
            if request.user_id == user_id and not request.is_used:
                request.used_at = utc_now()


@pytest.fixture()
def users() -> InMemoryUserRepository:
    """A fresh account store."""
    return InMemoryUserRepository()


@pytest.fixture()
def reset_tokens() -> InMemoryResetTokenRepository:
    """A fresh reset-token store."""
    return InMemoryResetTokenRepository()


@pytest.fixture()
def auth(
    settings: Settings,
    users: InMemoryUserRepository,
    reset_tokens: InMemoryResetTokenRepository,
) -> AuthService:
    """An auth service over the in-memory stores."""
    return AuthService(settings=settings, users=users, reset_tokens=reset_tokens)


class TestPasswordHashing:
    """bcrypt handling."""

    def test_hash_does_not_contain_the_password(self) -> None:
        """The stored value must not reveal the secret."""
        assert "hunter2000" not in hash_password("hunter2000")

    def test_correct_password_verifies(self) -> None:
        """The happy path."""
        assert verify_password("hunter2000", hash_password("hunter2000")) is True

    def test_wrong_password_fails(self) -> None:
        """The whole point."""
        assert verify_password("wrong-password", hash_password("hunter2000")) is False

    def test_same_password_hashes_differently_each_time(self) -> None:
        """Per-password salt: identical passwords must not collide in the store."""
        assert hash_password("hunter2000") != hash_password("hunter2000")

    def test_malformed_hash_returns_false_rather_than_raising(self) -> None:
        """A corrupt row must not become a 500 on the login path."""
        assert verify_password("hunter2000", "not-a-bcrypt-hash") is False


class TestPasswordPolicy:
    """Signup password rules."""

    def test_short_password_is_rejected(self) -> None:
        """Below the minimum length."""
        with pytest.raises(WeakPasswordError, match="at least 8"):
            validate_password("short", 8)

    def test_password_over_the_bcrypt_limit_is_rejected(self) -> None:
        """bcrypt silently ignores anything past 72 bytes, so say so instead."""
        with pytest.raises(WeakPasswordError, match="at most 72 bytes"):
            validate_password("x" * 73, 8)

    def test_whitespace_only_password_is_rejected(self) -> None:
        """Long enough, but not a password."""
        with pytest.raises(WeakPasswordError, match="whitespace"):
            validate_password(" " * 10, 8)

    def test_acceptable_password_passes(self) -> None:
        """A normal password is fine."""
        validate_password("correct-horse", 8)


class TestAccessTokens:
    """JWT signing and verification."""

    def test_round_trip_preserves_the_claims(self) -> None:
        """What is signed is what comes back."""
        token, _ = create_access_token("user-1", TEST_SECRET, claims={"role": "admin"})
        claims = decode_access_token(token, TEST_SECRET)

        assert claims["sub"] == "user-1"
        assert claims["role"] == "admin"

    def test_token_signed_with_another_secret_is_rejected(self) -> None:
        """A forged token must not pass."""
        token, _ = create_access_token("user-1", TEST_SECRET)

        with pytest.raises(InvalidTokenError):
            decode_access_token(token, OTHER_SECRET)

    def test_expired_token_is_rejected(self) -> None:
        """Sessions must actually end."""
        token, _ = create_access_token("user-1", TEST_SECRET, expires_minutes=-1)

        with pytest.raises(InvalidTokenError, match="expired"):
            decode_access_token(token, TEST_SECRET)

    def test_garbage_is_rejected(self) -> None:
        """Not a token at all."""
        with pytest.raises(InvalidTokenError):
            decode_access_token("nonsense", TEST_SECRET)

    def test_expiry_is_reported(self) -> None:
        """The caller needs to know how long the token lasts."""
        _, expires_at = create_access_token("user-1", TEST_SECRET, expires_minutes=60)
        assert expires_at > utc_now() + timedelta(minutes=59)


class TestSignup:
    """Account creation."""

    def test_creates_an_account_and_signs_in(self, auth: AuthService) -> None:
        """Signup returns a usable token immediately."""
        result = auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse")

        assert result.user.email == "lecturer@college.edu"
        assert result.user.name == "Dr. Meera"
        assert result.access_token

    def test_first_account_becomes_admin(self, auth: AuthService) -> None:
        """Somebody has to be the admin, and there is nobody to grant it."""
        assert auth.signup("first@college.edu", "First", "correct-horse").user.role is UserRole.ADMIN

    def test_later_accounts_are_lecturers(self, auth: AuthService) -> None:
        """Only the first account is privileged."""
        auth.signup("first@college.edu", "First", "correct-horse")
        second = auth.signup("second@college.edu", "Second", "correct-horse")

        assert second.user.role is UserRole.LECTURER

    def test_email_is_normalised(self, auth: AuthService) -> None:
        """Addresses are case-insensitive in practice."""
        result = auth.signup("  Lecturer@College.EDU  ", "Dr. Meera", "correct-horse")
        assert result.user.email == "lecturer@college.edu"

    def test_duplicate_email_is_rejected_case_insensitively(
        self, auth: AuthService
    ) -> None:
        """Bob@x.com and bob@x.com must not become two accounts."""
        auth.signup("lecturer@college.edu", "First", "correct-horse")

        with pytest.raises(EmailAlreadyRegisteredError):
            auth.signup("LECTURER@College.edu", "Impostor", "correct-horse")

    def test_weak_password_is_rejected(self, auth: AuthService) -> None:
        """The policy applies at signup."""
        with pytest.raises(WeakPasswordError):
            auth.signup("lecturer@college.edu", "Dr. Meera", "abc")

    def test_password_is_not_stored_in_plaintext(
        self, auth: AuthService, users: InMemoryUserRepository
    ) -> None:
        """The single most important property of the whole module."""
        auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse")
        stored = users.get_by_email("lecturer@college.edu")

        assert stored is not None
        assert stored.password_hash != "correct-horse"
        assert "correct-horse" not in stored.password_hash

    def test_public_dict_never_exposes_the_hash(self, auth: AuthService) -> None:
        """The API representation must not leak the hash."""
        payload = auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse").user.to_public_dict()
        assert "password_hash" not in payload


class TestLogin:
    """Sign-in."""

    def test_correct_credentials_return_a_token(self, auth: AuthService) -> None:
        """The happy path."""
        auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse")
        result = auth.login("lecturer@college.edu", "correct-horse")

        assert result.access_token
        assert result.user.email == "lecturer@college.edu"

    def test_token_identifies_the_user(self, auth: AuthService, settings: Settings) -> None:
        """The token's subject is what the guard resolves."""
        signup = auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse")
        claims = decode_access_token(
            auth.login("lecturer@college.edu", "correct-horse").access_token,
            settings.jwt_secret,
            settings.jwt_algorithm,
        )
        assert claims["sub"] == signup.user.user_id

    def test_wrong_password_is_rejected(self, auth: AuthService) -> None:
        """The whole point."""
        auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse")

        with pytest.raises(InvalidCredentialsError):
            auth.login("lecturer@college.edu", "wrong-password")

    def test_unknown_email_is_rejected(self, auth: AuthService) -> None:
        """No account, no token."""
        with pytest.raises(InvalidCredentialsError):
            auth.login("nobody@college.edu", "correct-horse")

    def test_unknown_email_and_wrong_password_give_the_same_message(
        self, auth: AuthService
    ) -> None:
        """Distinguishing them would leak which addresses are registered."""
        auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse")

        with pytest.raises(InvalidCredentialsError) as wrong_password:
            auth.login("lecturer@college.edu", "wrong-password")
        with pytest.raises(InvalidCredentialsError) as unknown_email:
            auth.login("nobody@college.edu", "correct-horse")

        assert wrong_password.value.message == unknown_email.value.message

    def test_login_is_case_insensitive_on_email(self, auth: AuthService) -> None:
        """Users do not remember how they capitalised it."""
        auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse")
        assert auth.login("LECTURER@COLLEGE.EDU", "correct-horse").access_token

    def test_login_is_recorded(
        self, auth: AuthService, users: InMemoryUserRepository
    ) -> None:
        """The last sign-in time is stamped."""
        auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse")
        auth.login("lecturer@college.edu", "correct-horse")

        stored = users.get_by_email("lecturer@college.edu")
        assert stored is not None and stored.last_login_at is not None


class TestPasswordReset:
    """Forgot-password and reset."""

    def test_reset_token_is_stored_hashed(
        self, auth: AuthService, reset_tokens: InMemoryResetTokenRepository
    ) -> None:
        """A leaked database must not be usable to reset anyone's password."""
        auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse")
        result = auth.request_password_reset("lecturer@college.edu")

        assert result.token is not None
        stored = reset_tokens.requests[0].token_hash
        assert stored != result.token
        assert stored == hash_reset_token(result.token)

    def test_unknown_email_does_not_reveal_itself(self, auth: AuthService) -> None:
        """No token, no error — the caller cannot tell the address is unknown."""
        result = auth.request_password_reset("nobody@college.edu")
        assert result.token is None

    def test_reset_changes_the_password(self, auth: AuthService) -> None:
        """The new password works and the old one stops working."""
        auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse")
        token = auth.request_password_reset("lecturer@college.edu").token
        assert token is not None

        auth.reset_password(token, "a-brand-new-password")

        assert auth.login("lecturer@college.edu", "a-brand-new-password").access_token
        with pytest.raises(InvalidCredentialsError):
            auth.login("lecturer@college.edu", "correct-horse")

    def test_token_is_single_use(self, auth: AuthService) -> None:
        """A reset link must not be replayable."""
        auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse")
        token = auth.request_password_reset("lecturer@college.edu").token
        assert token is not None
        auth.reset_password(token, "first-new-password")

        with pytest.raises(InvalidResetTokenError):
            auth.reset_password(token, "second-new-password")

    def test_requesting_a_new_link_retires_the_old_one(self, auth: AuthService) -> None:
        """Only the newest link may work."""
        auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse")
        first = auth.request_password_reset("lecturer@college.edu").token
        second = auth.request_password_reset("lecturer@college.edu").token
        assert first is not None and second is not None

        with pytest.raises(InvalidResetTokenError):
            auth.reset_password(first, "new-password")
        auth.reset_password(second, "new-password")

    def test_expired_token_is_rejected(
        self,
        settings: Settings,
        users: InMemoryUserRepository,
        reset_tokens: InMemoryResetTokenRepository,
    ) -> None:
        """Reset links must not live forever."""
        settings.reset_token_expire_minutes = 30
        auth = AuthService(settings=settings, users=users, reset_tokens=reset_tokens)
        auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse")
        token = auth.request_password_reset("lecturer@college.edu").token
        assert token is not None

        # Wind the clock past the expiry.
        reset_tokens.requests[0].expires_at = utc_now() - timedelta(minutes=1)

        with pytest.raises(InvalidResetTokenError, match="expired"):
            auth.reset_password(token, "new-password")

    def test_unknown_token_is_rejected(self, auth: AuthService) -> None:
        """A made-up link is not a reset."""
        with pytest.raises(InvalidResetTokenError):
            auth.reset_password("not-a-real-token", "new-password")

    def test_weak_new_password_is_rejected(self, auth: AuthService) -> None:
        """The policy applies to resets too, not just signup."""
        auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse")
        token = auth.request_password_reset("lecturer@college.edu").token
        assert token is not None

        with pytest.raises(WeakPasswordError):
            auth.reset_password(token, "abc")

    def test_verify_reports_validity(self, auth: AuthService) -> None:
        """The reset page checks the link before showing the form."""
        auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse")
        token = auth.request_password_reset("lecturer@college.edu").token
        assert token is not None

        assert auth.verify_reset_token(token) is True
        auth.reset_password(token, "new-password")
        assert auth.verify_reset_token(token) is False

    def test_link_points_at_the_frontend(self, auth: AuthService, settings: Settings) -> None:
        """The dev link must be clickable, not just a token."""
        auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse")
        result = auth.request_password_reset("lecturer@college.edu")

        assert result.link is not None
        assert result.link.startswith(settings.frontend_url)
        assert "reset-password?token=" in result.link

    def test_link_is_withheld_when_not_in_dev_mode(
        self,
        settings: Settings,
        users: InMemoryUserRepository,
        reset_tokens: InMemoryResetTokenRepository,
    ) -> None:
        """In production the link is emailed, never returned by the API."""
        settings.expose_reset_link = False
        auth = AuthService(settings=settings, users=users, reset_tokens=reset_tokens)
        auth.signup("lecturer@college.edu", "Dr. Meera", "correct-horse")

        result = auth.request_password_reset("lecturer@college.edu")
        assert result.token is None and result.link is None
        # The token still exists server-side — only the response withholds it.
        assert len(reset_tokens.requests) == 1


class TestResetTokenGeneration:
    """The tokens themselves."""

    def test_tokens_are_unique(self) -> None:
        """A predictable token would be a way in."""
        tokens = {generate_reset_token()[0] for _ in range(100)}
        assert len(tokens) == 100

    def test_token_has_enough_entropy(self) -> None:
        """32 urlsafe bytes is ~43 characters."""
        token, _ = generate_reset_token()
        assert len(token) >= 40

    def test_hash_is_deterministic(self) -> None:
        """Lookup depends on the same token hashing the same way."""
        token, token_hash = generate_reset_token()
        assert hash_reset_token(token) == token_hash
