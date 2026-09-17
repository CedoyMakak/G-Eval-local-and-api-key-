from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def mask_secret(value: str) -> str:
    secret = (value or "").strip()
    if not secret:
        return ""
    if len(secret) <= 8:
        return "••••"
    return f"{secret[:4]}••••{secret[-4:]}"


def update_env(values: dict[str, str]) -> None:
    current: list[str] = []
    if ENV_PATH.exists():
        current = ENV_PATH.read_text(encoding="utf-8").splitlines()

    seen: set[str] = set()
    rewritten: list[str] = []
    for line in current:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in line:
            key = line.split("=", 1)[0].strip()
            if key in values:
                rewritten.append(f"{key}={values[key]}")
                seen.add(key)
                continue
        rewritten.append(line)

    missing = [key for key in values if key not in seen]
    if missing:
        if rewritten and rewritten[-1].strip():
            rewritten.append("")
        rewritten.extend(f"{key}={values[key]}" for key in missing)

    ENV_PATH.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    for key, value in values.items():
        os.environ[key] = value
