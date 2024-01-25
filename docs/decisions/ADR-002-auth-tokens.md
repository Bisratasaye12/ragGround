# ADR-002: Short-lived JWT access tokens + rotating opaque refresh tokens

**Status:** accepted (2026-11-06)

## Context
The API serves a browser app and later CLI/CI clients. It owns the authorization logic (orgs,
roles, document ACLs), so it should also own authentication rather than delegate to a hosted IdP
for this project.

## Decision
- **Access token:** HS256 JWT, 15 min TTL, claims `sub`, `org`, `typ=access`. The user row is
  still loaded on every request, so deactivation and role changes apply immediately instead of
  waiting for expiry. (Cost: one indexed primary-key lookup per request.)
- **Refresh token:** 256-bit random opaque string; only its SHA-256 is stored. 7-day TTL.
  **Rotated on every use.** Tokens from one login share a `family_id`; presenting an
  already-rotated token is treated as theft and revokes the whole family (reuse detection, as in
  OAuth 2.0 Security BCP). Logout revokes the family.
- **Passwords:** argon2id (argon2-cffi defaults). Login runs a hash verification even for unknown
  emails, and returns an identical 401 for wrong password vs unknown user (no user enumeration).
- The app refuses to start outside development with the default or a short JWT secret.

## Consequences
- Stolen refresh tokens have a short useful life and trigger revocation when the real client
  refreshes.
- Stateless access-token verification is traded for an immediate-revocation guarantee via the
  per-request user lookup.
- Not yet done: rate limiting on `/auth/login` (planned with Redis), email verification, SSO.
