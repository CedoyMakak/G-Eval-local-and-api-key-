from __future__ import annotations

import re

from fastapi import HTTPException, Request

LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost", "testclient"}
_SECRET_RE = re.compile(
    r"(?i)(sk-[a-z0-9_-]{6,}|Bearer\s+\S+|api[_-]?key\s*[=:]\s*\S+)"
)


def require_local_client(request: Request) -> None:
    host = (request.client.host if request.client else "") or ""
    if host not in LOCAL_HOSTS:
        raise HTTPException(
            status_code=403,
            detail="Настройки судьи доступны только с этого компьютера.",
        )


def public_error_message(exc: Exception) -> str:
    return _SECRET_RE.sub("***", str(exc))
