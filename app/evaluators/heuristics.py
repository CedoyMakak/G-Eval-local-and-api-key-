from __future__ import annotations

import re

_WORD_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def compute_heuristics(answer: str, question: str, reference: str | None) -> dict[str, float]:
    words = _WORD_RE.findall(answer)
    q_words = set(_WORD_RE.findall(question.lower()))
    a_words = set(token.lower() for token in words)
    overlap = len(q_words & a_words) / len(q_words) if q_words else 0.0

    length = len(words)
    # короткие фактические ответы («Париж», «323») не штрафуем
    if length == 0:
        length_score = 0.0
    elif length < 3:
        length_score = 0.7 if reference else 0.45
    else:
        length_score = min(1.0, 0.45 + length / 50.0)
    if reference:
        ref_len = max(1, len(_WORD_RE.findall(reference)))
        ratio = length / ref_len
        length_ratio = 1.0 - min(1.0, abs(np_log_ratio(ratio)))
        if ratio <= 6:
            length_ratio = max(length_ratio, 0.55)
    else:
        length_ratio = length_score

    return {
        "length_words": float(length),
        "question_overlap": round(overlap, 4),
        "length_score": round(length_score, 4),
        "length_ratio_score": round(max(0.0, length_ratio), 4),
    }


def np_log_ratio(ratio: float) -> float:
    if ratio <= 0:
        return 1.0
    import math

    return abs(math.log(ratio)) / math.log(4)
