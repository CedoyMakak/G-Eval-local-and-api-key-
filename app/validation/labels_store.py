from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import LabeledItem

LABELS_PATH = Path(__file__).resolve().parents[2] / "data" / "human_labels.json"


def load_labels() -> list[dict]:
    if not LABELS_PATH.exists():
        return []
    raw = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        return []
    return raw


def save_label(item: LabeledItem) -> tuple[LabeledItem, int]:
    rows = load_labels()
    if not item.id:
        item = item.model_copy(update={"id": _next_id(rows)})
    payload = item.model_dump()
    for index, row in enumerate(rows):
        if row.get("id") == item.id:
            rows[index] = payload
            break
    else:
        rows.append(payload)
    LABELS_PATH.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return item, len(rows)


def _next_id(rows: list[dict]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"ui-{stamp}-{len(rows) + 1:03d}"
