"""Shared FastAPI dependencies: settings, DB session, current user, admin guard."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ragground.config import Settings
from ragground.db import get_session
from ragground.models import Role, User
from ragground.security import InvalidTokenError, decode_access_token

_bearer = HTTPBearer(auto_error=False)


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


SessionDep = Annotated[AsyncSession, Depends(get_session)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_user(
    session: SessionDep,
    settings: SettingsDep,
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> User:
    if creds is None:
        raise _UNAUTHORIZED
    try:
        user_id = decode_access_token(creds.credentials, settings.jwt_secret)
    except InvalidTokenError:
        raise _UNAUTHORIZED from None
    # Loaded on every request (not trusted from the token) so deactivation and role changes
    # take effect immediately rather than when the access token expires.
    user = await session.scalar(
        select(User).where(User.id == user_id).options(selectinload(User.groups))
    )
    if user is None or not user.is_active:
        raise _UNAUTHORIZED
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def require_admin(user: CurrentUser) -> User:
    if user.role != Role.admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin role required")
    return user


AdminUser = Annotated[User, Depends(require_admin)]
