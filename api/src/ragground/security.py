"""Password hashing (argon2id) and token primitives."""

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

_hasher = PasswordHasher()
# Verified against when the email doesn't exist, so login takes the same time either way
# (prevents user enumeration by timing).
_DUMMY_HASH = _hasher.hash("timing-equaliser-not-a-real-password")

JWT_ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str | None) -> bool:
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and password_hash is not None
    except (VerificationError, InvalidHashError):
        return False


def create_access_token(user_id: uuid.UUID, org_id: uuid.UUID, secret: str, ttl_s: int) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "org": str(org_id),
        "typ": "access",
        "iat": now,
        "exp": now + timedelta(seconds=ttl_s),
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


class InvalidTokenError(Exception):
    pass


def decode_access_token(token: str, secret: str) -> uuid.UUID:
    """Returns the user id. Raises InvalidTokenError for any bad/expired/wrong-type token."""
    try:
        payload = jwt.decode(
            token, secret, algorithms=[JWT_ALGORITHM], options={"require": ["exp", "sub"]}
        )
        if payload.get("typ") != "access":
            raise InvalidTokenError("not an access token")
        return uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, ValueError) as exc:
        raise InvalidTokenError(str(exc)) from exc


def new_refresh_token() -> tuple[str, str]:
    """Returns (token for the client, sha256 hash to store)."""
    token = secrets.token_urlsafe(32)
    return token, hash_refresh_token(token)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
