from __future__ import annotations

import re

from app.evaluators.lexical import tokenize
from app.schemas import FactCheckMetrics

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")


def compute_factcheck(answer: str, context: str | None) -> FactCheckMetrics | None:
    if not context or not context.strip():
        return None

    answer_tokens = set(tokenize(answer))
    context_tokens = set(tokenize(context))
    if not answer_tokens:
        return FactCheckMetrics(score=0.0, context_overlap=0.0, unsupported_numbers=0, method="context-overlap")

    overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
    answer_nums = set(_normalize_num(value) for value in _NUM_RE.findall(answer))
    context_nums = set(_normalize_num(value) for value in _NUM_RE.findall(context))
    unsupported = sorted(answer_nums - context_nums)
    penalty = 0.2 * min(len(unsupported), 3)
    score = max(0.0, min(1.0, overlap - penalty))
    if unsupported:
        score = min(score, 0.45)
    return FactCheckMetrics(
        score=round(score, 4),
        context_overlap=round(overlap, 4),
        unsupported_numbers=len(unsupported),
        method="context-overlap",
    )


def _normalize_num(value: str) -> str:
    return value.replace(",", ".")
