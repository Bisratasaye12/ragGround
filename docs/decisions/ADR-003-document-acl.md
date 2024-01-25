# ADR-003: Document ACLs as a SQL predicate applied before ranking

**Status:** accepted (2026-11-06)

## Context
Users must never receive an answer built from a document they cannot read. Filtering retrieved
chunks in application code *after* ranking leaks information (the top-k is shaped by forbidden
documents, and a bug in the filter leaks text directly).

## Decision
Access rule:
- A user reads a document if it is in **their org** and either the document has **no groups**
  (org-wide) or the user belongs to **at least one** of the document's groups.
- Org **admins** read every document in their org (they manage groups, so hiding documents from
  them gives no real protection).
- **Inactive** users read nothing.

`ragground.acl.readable_by(user)` returns this rule as a SQLAlchemy boolean expression. Retrieval
adds it to the WHERE clause of the vector and full-text queries, so ranking only ever sees
permitted chunks.

## Consequences
- One implementation of the rule, tested directly against Postgres (`tests/test_acl.py`), with an
  eval metric planned on top: **ACL leak rate must be 0** in the golden set (M5) and CI gate (M6).
- `group_id` columns on both join tables are indexed so the EXISTS subqueries stay cheap.
- Cross-org access returns 404 rather than 403 in the admin API, so other orgs' object IDs can't be
  probed.
