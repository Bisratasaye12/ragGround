"""Org administration: members and groups. Every query is scoped to the caller's org."""

import uuid

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from ragground.deps import AdminUser, CurrentUser, SessionDep
from ragground.models import Group, User, user_groups
from ragground.schemas import GroupCreateIn, GroupOut, UserCreateIn, UserOut
from ragground.security import hash_password

router = APIRouter(tags=["org"])

_NOT_FOUND = HTTPException(status.HTTP_404_NOT_FOUND, "Not found")


async def _org_user(session: SessionDep, org_id: uuid.UUID, user_id: uuid.UUID) -> User:
    user = await session.scalar(select(User).where(User.id == user_id, User.org_id == org_id))
    if user is None:
        raise _NOT_FOUND  # also returned for other orgs' users: don't reveal they exist
    return user


async def _org_group(session: SessionDep, org_id: uuid.UUID, group_id: uuid.UUID) -> Group:
    group = await session.scalar(select(Group).where(Group.id == group_id, Group.org_id == org_id))
    if group is None:
        raise _NOT_FOUND
    return group


@router.get("/users", response_model=list[UserOut])
async def list_users(admin: AdminUser, session: SessionDep) -> list[User]:
    result = await session.scalars(
        select(User)
        .where(User.org_id == admin.org_id)
        .options(selectinload(User.groups))
        .order_by(User.created_at)
    )
    return list(result)


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(body: UserCreateIn, admin: AdminUser, session: SessionDep) -> User:
    user = User(
        org_id=admin.org_id,
        email=body.email,
        password_hash=hash_password(body.password),
        role=body.role,
        groups=[],
    )
    session.add(user)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Email already registered") from None
    return user


@router.post("/users/{user_id}/deactivate", response_model=UserOut)
async def deactivate_user(user_id: uuid.UUID, admin: AdminUser, session: SessionDep) -> User:
    if user_id == admin.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "You cannot deactivate yourself")
    user = await _org_user(session, admin.org_id, user_id)
    user.is_active = False
    await session.commit()
    await session.refresh(user, ["groups"])
    return user


@router.get("/groups", response_model=list[GroupOut])
async def list_groups(user: CurrentUser, session: SessionDep) -> list[Group]:
    result = await session.scalars(
        select(Group).where(Group.org_id == user.org_id).order_by(Group.name)
    )
    return list(result)


@router.post("/groups", response_model=GroupOut, status_code=status.HTTP_201_CREATED)
async def create_group(body: GroupCreateIn, admin: AdminUser, session: SessionDep) -> Group:
    group = Group(org_id=admin.org_id, name=body.name)
    session.add(group)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, "Group name already exists") from None
    return group


@router.put("/groups/{group_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def add_member(
    group_id: uuid.UUID, user_id: uuid.UUID, admin: AdminUser, session: SessionDep
) -> None:
    await _org_group(session, admin.org_id, group_id)
    await _org_user(session, admin.org_id, user_id)
    await session.execute(
        pg_insert(user_groups)
        .values(user_id=user_id, group_id=group_id)
        .on_conflict_do_nothing()  # PUT is idempotent
    )
    await session.commit()


@router.delete("/groups/{group_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(
    group_id: uuid.UUID, user_id: uuid.UUID, admin: AdminUser, session: SessionDep
) -> None:
    await _org_group(session, admin.org_id, group_id)
    await session.execute(
        delete(user_groups).where(
            user_groups.c.group_id == group_id, user_groups.c.user_id == user_id
        )
    )
    await session.commit()
