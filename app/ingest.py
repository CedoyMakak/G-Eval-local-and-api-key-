from __future__ import annotations

import csv
import io
import json
from typing import Any

MAX_BATCH_ITEMS = 100

_FIELD_ALIASES = {
    "question": "question",
    "вопрос": "question",
    "prompt": "question",
    "query": "question",
    "answer": "answer",
    "ответ": "answer",
    "candidate": "answer",
    "output": "answer",
    "completion": "answer",
    "reference": "reference",
    "эталон": "reference",
    "ref": "reference",
    "gold": "reference",
    "target": "reference",
    "context": "context",
    "контекст": "context",
    "id": "id",
}


class IngestError(ValueError):
    pass


def parse_eval_documents(text: str, filename: str = "") -> list[dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        raise IngestError("Файл пустой.")
    name = (filename or "").lower()
    if name.endswith(".csv") or name.endswith(".tsv"):
        rows = _parse_table(raw, "\t" if name.endswith(".tsv") else None)
    elif name.endswith(".jsonl") or name.endswith(".ndjson"):
        rows = _parse_jsonl(raw)
    else:
        rows = _parse_auto(raw)
    items = [_normalize_item(row, index) for index, row in enumerate(rows, start=1)]
    if not items:
        raise IngestError("Не нашёл ни одной пары question + answer.")
    if len(items) > MAX_BATCH_ITEMS:
        raise IngestError(f"Слишком много строк: {len(items)}. Максимум {MAX_BATCH_ITEMS}.")
    return items


def _parse_auto(raw: str) -> list[dict[str, Any]]:
    if raw[0] in "[{":
        data = json.loads(raw)
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            if isinstance(data.get("items"), list):
                return [row for row in data["items"] if isinstance(row, dict)]
            return [data]
        raise IngestError("JSON должен быть массивом объектов или {items: [...]}.")
    if "\n" in raw and raw.lstrip()[:1] == "{":
        try:
            return _parse_jsonl(raw)
        except IngestError:
            pass
    return _parse_table(raw, None)


def _parse_jsonl(raw: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.splitlines(), start=1):
        piece = line.strip()
        if not piece:
            continue
        try:
            row = json.loads(piece)
        except json.JSONDecodeError as exc:
            raise IngestError(f"JSONL, строка {line_no}: {exc.msg}") from exc
        if isinstance(row, dict):
            rows.append(row)
    if not rows:
        raise IngestError("JSONL не содержит объектов.")
    return rows


def _parse_table(raw: str, delimiter: str | None) -> list[dict[str, Any]]:
    sample = raw[:4000]
    if delimiter is None:
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = "," if sample.count(",") >= sample.count(";") else ";"
    reader = csv.DictReader(io.StringIO(raw), delimiter=delimiter)
    if not reader.fieldnames:
        raise IngestError("У таблицы нет заголовка. Нужны колонки question,answer.")
    rows = [dict(row) for row in reader if any((value or "").strip() for value in row.values())]
    if not rows:
        raise IngestError("CSV без строк данных.")
    return rows


def _normalize_item(row: dict[str, Any], index: int) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for key, value in row.items():
        canon = _FIELD_ALIASES.get(str(key).strip().lstrip("\ufeff").lower())
        if canon and value is not None and str(value).strip():
            mapped[canon] = str(value).strip()
    question = mapped.get("question")
    answer = mapped.get("answer")
    if not question or not answer:
        raise IngestError(f"Строка {index}: нужны поля question и answer.")
    item = {"question": question, "answer": answer}
    if mapped.get("reference"):
        item["reference"] = mapped["reference"]
    if mapped.get("context"):
        item["context"] = mapped["context"]
    if mapped.get("id"):
        item["id"] = mapped["id"]
    return item
