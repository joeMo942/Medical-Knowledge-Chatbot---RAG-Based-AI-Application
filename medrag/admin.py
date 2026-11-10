"""Access to the /admin endpoints: `Authorization: Bearer <token>` matching MEDRAG_ADMIN_TOKEN.

Fail closed: with no token configured every /admin endpoint answers 503, so an unset variable never leaves one open.
`admin_guard(token)` returns the FastAPI dependency; add it to each admin route with `dependencies=[Depends(guard)]`."""
from __future__ import annotations

import hmac

from fastapi import HTTPException, Request

DISABLED = "admin disabled: set MEDRAG_ADMIN_TOKEN"


def admin_guard(token: str):
    async def require_admin(request: Request) -> None:
        if not token:
            raise HTTPException(503, DISABLED)
        scheme, _, given = request.headers.get("authorization", "").partition(" ")
        # Compared as bytes: hmac.compare_digest raises on a non-ASCII str.
        if scheme.lower() != "bearer" or not hmac.compare_digest(given.strip().encode(), token.encode()):
            raise HTTPException(401, "missing or wrong admin token")

    return require_admin
