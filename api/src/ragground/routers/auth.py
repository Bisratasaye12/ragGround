"""Sign-up, login, refresh-token rotation, logout, and the current user."""

import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ragground.config import Settings
from ragground.deps import CurrentUser, SessionDep, SettingsDep
from ragground.models import Org, RefreshToken, Role, User
from ragground.schemas import LoginIn, RefreshIn, RegisterIn, TokenPair, UserOut
from ragground.security import (
    create_access_token,
    hash_password,
    hash_refresh_token,
    new_refresh_token,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])

_BAD_CREDENTIALS = HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid email or password")
_BAD_REFRESH = HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid refresh token")


async def _issue_tokens(
    session: AsyncSession, settings: Settings, user: User, family_id: uuid.UUID | None = None
) -> TokenPair:
    token, token_hash = new_refresh_token()
    session.add(
        RefreshToken(
            user_id=user.id,
            family_id=family_id or uuid.uuid4(),
            token_hash=token_hash,
            expires_at=datetime.now(UTC) + timedelta(seconds=settings.refresh_token_ttl_s),
        )
    )
    await session.commit()
    return TokenPair(
        access_token=create_access_token(
            user.id, user.org_id, settings.jwt_secret, settings.access_token_ttl_s
        ),
        refresh_token=token,
        expires_in=settings.access_token_ttl_s,
    )


@router.post("/register", response_model=TokenPair, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterIn, session: SessionDep, settings: SettingsDep) -> TokenPair:
    org = Org(name=body.org_name)
    session.add(org)
    await session.flush()
    user = User(
        org_id=org.id, email=body.email, password_hash=hash_password(body.password), role=Role.admin
    )
    session.add(user)
    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered") from None
    return await _issue_tokens(session, settings, user)


@router.post("/login", response_model=TokenPair)
async def login(body: LoginIn, session: SessionDep, settings: SettingsDep) -> TokenPair:
    user = await session.scalar(select(User).where(User.email == body.email))
    # verify_password runs even when the user doesn't exist (constant-ish timing).
    if not verify_password(body.password, user.password_hash if user else None):
        raise _BAD_CREDENTIALS
    assert user is not None
    if not user.is_active:
        raise _BAD_CREDENTIALS
    return await _issue_tokens(session, settings, user)


@router.post("/refresh", response_model=TokenPair)
async def refresh(body: RefreshIn, session: SessionDep, settings: SettingsDep) -> TokenPair:
    now = datetime.now(UTC)
    stored = await session.scalar(
        select(RefreshToken)
        .where(RefreshToken.token_hash == hash_refresh_token(body.refresh_token))
        .with_for_update()
    )
    if stored is None:
        raise _BAD_REFRESH
    if stored.revoked_at is not None:
        # A rotated token was presented again: it was probably stolen. Kill the whole session.
        await session.execute(
            update(RefreshToken)
            .where(RefreshToken.family_id == stored.family_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=now)
        )
        await session.commit()
        raise _BAD_REFRESH
    if stored.expires_at <= now:
        raise _BAD_REFRESH

    user = await session.get(User, stored.user_id)
    if user is None or not user.is_active:
        raise _BAD_REFRESH
    stored.revoked_at = now
    return await _issue_tokens(session, settings, user, family_id=stored.family_id)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(body: RefreshIn, session: SessionDep) -> None:
    """Revokes the session (token family) the refresh token belongs to. Idempotent."""
    stored = await session.scalar(
        select(RefreshToken).where(
            RefreshToken.token_hash == hash_refresh_token(body.refresh_token)
        )
    )
    if stored is not None:
        await session.execute(
            update(RefreshToken)
            .where(RefreshToken.family_id == stored.family_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=datetime.now(UTC))
        )
        await session.commit()


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser) -> User:
    return user
