"""Document access control, expressed as SQL so it can be applied *inside* retrieval queries.

Retrieval (M3) joins chunks to documents and adds `readable_by(user)` to the WHERE clause, so
ranking only ever sees chunks the user may read. Nothing is filtered after the fact in Python.
"""

from sqlalchemy import ColumnElement, and_, exists, or_, select, true

from ragground.models import Document, Role, User, document_groups, user_groups


def readable_by(user: User) -> ColumnElement[bool]:
    """WHERE-clause for documents `user` may read."""
    same_org = Document.org_id == user.org_id
    if not user.is_active:
        return and_(same_org, ~true())
    if user.role == Role.admin:
        return same_org

    has_no_groups = ~exists().where(document_groups.c.document_id == Document.id)
    shares_a_group = exists(
        select(1)
        .select_from(
            document_groups.join(user_groups, document_groups.c.group_id == user_groups.c.group_id)
        )
        .where(document_groups.c.document_id == Document.id, user_groups.c.user_id == user.id)
    )
    return and_(same_org, or_(has_no_groups, shares_a_group))
