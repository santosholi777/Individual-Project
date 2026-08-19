"""Authentication and user management.

Adds accounts, login and password reset to the AI service, backed by MongoDB.

* :mod:`auth.security` — password hashing, JWT signing, reset tokens.
* :mod:`auth.models` — the User domain type.
* :mod:`auth.repository` — user + reset-token storage (MongoDB).
* :mod:`auth.service` — signup, login and reset rules.
* :mod:`auth.dependencies` — FastAPI guards (``get_current_user``, roles).
* :mod:`auth.router` — the ``/auth`` endpoints.
"""
