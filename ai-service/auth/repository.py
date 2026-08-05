"""MongoDB storage for accounts and password-reset tokens.

The abstract repositories mirror the pattern in ``database.py``: services depend
on the interface, so the backend could be swapped without touching them.

The connection is **lazy**. MongoDB being down must not stop the service from
booting — the face recognition pipeline has nothing to do with accounts, and a
kiosk that already holds a token should keep working. Auth endpoints then fail
with a clear 503 instead of the whole service refusing to start.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError, PyMongoError

from auth.models import PasswordResetRequest, User, UserRole, normalise_email
from config import Settings
from exceptions import DeepVisionAttendError
from logging_config import get_logger

logger = get_logger(__name__)


class DatabaseUnavailableError(DeepVisionAttendError):
    """MongoDB could not be reached."""

    status_code = 503
    error_code = "database_unavailable"


class EmailAlreadyRegisteredError(DeepVisionAttendError):
    """An account already exists with this email address."""

    status_code = 409
    error_code = "email_already_registered"


# ======================================================================
# Interfaces
# ======================================================================
class UserRepository(ABC):
    """Stores accounts."""

    @abstractmethod
    def create(self, user: User) -> User:
        """Insert a new user and return it with its assigned id."""

    @abstractmethod
    def get_by_email(self, email: str) -> User | None:
        """Find a user by email, or ``None``."""

    @abstractmethod
    def get_by_id(self, user_id: str) -> User | None:
        """Find a user by id, or ``None``."""

    @abstractmethod
    def update_password(self, user_id: str, password_hash: str) -> None:
        """Replace a user's password hash."""

    @abstractmethod
    def record_login(self, user_id: str) -> None:
        """Stamp the user's last successful sign-in."""

    @abstractmethod
    def count(self) -> int:
        """Number of registered accounts."""

    @abstractmethod
    def list_all(self) -> list[User]:
        """Every account, oldest first."""


class ResetTokenRepository(ABC):
    """Stores password-reset tokens."""

    @abstractmethod
    def create(self, request: PasswordResetRequest) -> None:
        """Store a pending reset."""

    @abstractmethod
    def find_valid(self, token_hash: str) -> PasswordResetRequest | None:
        """Return an unused, unexpired reset for this token hash."""

    @abstractmethod
    def mark_used(self, token_hash: str) -> None:
        """Burn a token so it cannot be redeemed twice."""

    @abstractmethod
    def invalidate_for_user(self, user_id: str) -> None:
        """Burn every outstanding token for a user."""


# ======================================================================
# MongoDB connection
# ======================================================================
class MongoConnection:
    """Lazily-created MongoDB client shared by the repositories.

    Args:
        settings: Supplies the URI, database name and timeout.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: MongoClient | None = None
        self._lock = threading.Lock()
        self._indexes_ready = False

    def client(self) -> MongoClient:
        """Return the client, connecting on first use.

        Raises:
            DatabaseUnavailableError: If MongoDB cannot be reached.
        """
        if self._client is not None:
            return self._client

        with self._lock:
            if self._client is not None:
                return self._client

            logger.info("Connecting to MongoDB at %s", self._settings.mongodb_uri)
            client: MongoClient = MongoClient(
                self._settings.mongodb_uri,
                serverSelectionTimeoutMS=self._settings.mongodb_timeout_ms,
                connectTimeoutMS=self._settings.mongodb_timeout_ms,
                tz_aware=True,
            )
            try:
                # The constructor is lazy; this is what actually proves the
                # server is reachable.
                client.admin.command("ping")
            except PyMongoError as exc:
                client.close()
                raise DatabaseUnavailableError(
                    "Cannot reach MongoDB. Start it with: "
                    "brew services start mongodb-community@7.0",
                    details={"uri": self._settings.mongodb_uri, "cause": str(exc)},
                ) from exc

            self._client = client
            logger.info("MongoDB connected (database '%s')", self._settings.mongodb_db)
            return client

    def database(self):
        """Return the configured database."""
        return self.client()[self._settings.mongodb_db]

    def ensure_indexes(self) -> None:
        """Create the indexes the auth collections rely on. Idempotent.

        The unique index on ``email`` is what actually enforces one account per
        address — a check-then-insert in application code would race.
        """
        if self._indexes_ready:
            return

        database = self.database()
        database["users"].create_index([("email", ASCENDING)], unique=True, name="uniq_email")
        database["password_resets"].create_index(
            [("token_hash", ASCENDING)], unique=True, name="uniq_token_hash"
        )
        # Mongo's TTL monitor deletes expired resets on its own, so old tokens
        # do not pile up forever.
        database["password_resets"].create_index(
            [("expires_at", ASCENDING)], expireAfterSeconds=0, name="ttl_expires_at"
        )
        self._indexes_ready = True
        logger.info("MongoDB indexes ensured")

    def ping(self) -> bool:
        """Whether MongoDB is reachable right now, for ``/health``."""
        try:
            self.client().admin.command("ping")
            return True
        except (DatabaseUnavailableError, PyMongoError):
            return False

    def close(self) -> None:
        """Close the client, if one was opened."""
        if self._client is not None:
            self._client.close()
            self._client = None


def _as_utc(value: Any) -> datetime | None:
    """Normalise a stored datetime to UTC-aware, tolerating naive values."""
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


# ======================================================================
# MongoDB implementations
# ======================================================================
class MongoUserRepository(UserRepository):
    """Accounts stored in the ``users`` collection."""

    def __init__(self, connection: MongoConnection) -> None:
        self._connection = connection

    @property
    def _collection(self) -> Collection:
        self._connection.ensure_indexes()
        return self._connection.database()["users"]

    @staticmethod
    def _to_user(document: dict[str, Any]) -> User:
        """Map a Mongo document to a domain user."""
        return User(
            user_id=str(document["_id"]),
            email=document["email"],
            name=document["name"],
            password_hash=document["password_hash"],
            role=UserRole(document.get("role", UserRole.LECTURER.value)),
            created_at=_as_utc(document.get("created_at")) or datetime.now(timezone.utc),
            updated_at=_as_utc(document.get("updated_at")) or datetime.now(timezone.utc),
            last_login_at=_as_utc(document.get("last_login_at")),
        )

    def create(self, user: User) -> User:
        """Insert a new user.

        Raises:
            EmailAlreadyRegisteredError: If the email is taken.
            DatabaseUnavailableError: If MongoDB is unreachable.
        """
        document = {
            "email": normalise_email(user.email),
            "name": user.name,
            "password_hash": user.password_hash,
            "role": user.role.value,
            "created_at": user.created_at,
            "updated_at": user.updated_at,
            "last_login_at": None,
        }
        try:
            result = self._collection.insert_one(document)
        except DuplicateKeyError as exc:
            raise EmailAlreadyRegisteredError(
                "An account with this email address already exists.",
                details={"email": normalise_email(user.email)},
            ) from exc
        except PyMongoError as exc:
            raise DatabaseUnavailableError(f"Could not create the account: {exc}") from exc

        user.user_id = str(result.inserted_id)
        logger.info("Created account %s (%s)", user.email, user.role.value)
        return user

    def get_by_email(self, email: str) -> User | None:
        """Find a user by email."""
        try:
            document = self._collection.find_one({"email": normalise_email(email)})
        except PyMongoError as exc:
            raise DatabaseUnavailableError(f"Could not read the account: {exc}") from exc
        return self._to_user(document) if document else None

    def get_by_id(self, user_id: str) -> User | None:
        """Find a user by id.

        A malformed id is a miss, not a crash — ids arrive from JWT claims and
        must never be able to 500 the service.
        """
        from bson import ObjectId
        from bson.errors import InvalidId

        try:
            document = self._collection.find_one({"_id": ObjectId(user_id)})
        except (InvalidId, TypeError):
            return None
        except PyMongoError as exc:
            raise DatabaseUnavailableError(f"Could not read the account: {exc}") from exc
        return self._to_user(document) if document else None

    def update_password(self, user_id: str, password_hash: str) -> None:
        """Replace a user's password hash."""
        from bson import ObjectId

        try:
            self._collection.update_one(
                {"_id": ObjectId(user_id)},
                {
                    "$set": {
                        "password_hash": password_hash,
                        "updated_at": datetime.now(timezone.utc),
                    }
                },
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(f"Could not update the password: {exc}") from exc

    def record_login(self, user_id: str) -> None:
        """Stamp the last sign-in time. Best-effort: never fail a valid login."""
        from bson import ObjectId

        try:
            self._collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": {"last_login_at": datetime.now(timezone.utc)}},
            )
        except PyMongoError as exc:
            logger.warning("Could not record the login time: %s", exc)

    def count(self) -> int:
        """Number of accounts."""
        try:
            return int(self._collection.count_documents({}))
        except PyMongoError as exc:
            raise DatabaseUnavailableError(f"Could not count accounts: {exc}") from exc

    def list_all(self) -> list[User]:
        """Every account, oldest first."""
        try:
            return [
                self._to_user(document)
                for document in self._collection.find().sort("created_at", ASCENDING)
            ]
        except PyMongoError as exc:
            raise DatabaseUnavailableError(f"Could not list accounts: {exc}") from exc


class MongoResetTokenRepository(ResetTokenRepository):
    """Password-reset tokens stored in the ``password_resets`` collection."""

    def __init__(self, connection: MongoConnection) -> None:
        self._connection = connection

    @property
    def _collection(self) -> Collection:
        self._connection.ensure_indexes()
        return self._connection.database()["password_resets"]

    def create(self, request: PasswordResetRequest) -> None:
        """Store a pending reset."""
        try:
            self._collection.insert_one(
                {
                    "user_id": request.user_id,
                    "token_hash": request.token_hash,
                    "expires_at": request.expires_at,
                    "created_at": request.created_at,
                    "used_at": None,
                }
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(f"Could not create the reset token: {exc}") from exc

    def find_valid(self, token_hash: str) -> PasswordResetRequest | None:
        """Return an unused, unexpired reset for this token hash.

        Expiry is filtered in the query rather than trusted to the TTL monitor,
        which only sweeps every ~60 seconds.
        """
        try:
            document = self._collection.find_one(
                {
                    "token_hash": token_hash,
                    "used_at": None,
                    "expires_at": {"$gt": datetime.now(timezone.utc)},
                }
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(f"Could not read the reset token: {exc}") from exc

        if not document:
            return None
        return PasswordResetRequest(
            user_id=document["user_id"],
            token_hash=document["token_hash"],
            expires_at=_as_utc(document["expires_at"]) or datetime.now(timezone.utc),
            created_at=_as_utc(document.get("created_at")) or datetime.now(timezone.utc),
            used_at=_as_utc(document.get("used_at")),
        )

    def mark_used(self, token_hash: str) -> None:
        """Burn a token so it cannot be redeemed twice."""
        try:
            self._collection.update_one(
                {"token_hash": token_hash},
                {"$set": {"used_at": datetime.now(timezone.utc)}},
            )
        except PyMongoError as exc:
            raise DatabaseUnavailableError(f"Could not update the reset token: {exc}") from exc

    def invalidate_for_user(self, user_id: str) -> None:
        """Burn every outstanding token for a user.

        Called after a successful reset: requesting several links and then using
        the oldest must not leave the others live.
        """
        try:
            self._collection.update_many(
                {"user_id": user_id, "used_at": None},
                {"$set": {"used_at": datetime.now(timezone.utc)}},
            )
        except PyMongoError as exc:
            logger.warning("Could not invalidate old reset tokens: %s", exc)
