from __future__ import annotations

import re

from app.schemas import LexicalMetrics

_TOKEN_RE = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def compute_lexical(answer: str, reference: str | None) -> LexicalMetrics:
    if not reference or not reference.strip():
        return LexicalMetrics(rouge_l=None, bleu=None)

    return LexicalMetrics(
        rouge_l=_rouge_l(answer, reference),
        bleu=_bleu(answer, reference),
    )


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text)]


def _rouge_l(answer: str, reference: str) -> float:
    """ROUGE-L по Unicode-токенам: стандартный rouge-score выкидывает кириллицу."""
    pred = tokenize(answer)
    ref = tokenize(reference)
    if not pred or not ref:
        return 0.0

    lcs = _lcs_length(pred, ref)
    precision = lcs / len(pred)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    beta2 = 1.2**2
    return float((1 + beta2) * precision * recall / (recall + beta2 * precision))


def _lcs_length(left: list[str], right: list[str]) -> int:
    n, m = len(left), len(right)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        for j in range(1, m + 1):
            if left[i - 1] == right[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[m]


def _bleu(answer: str, reference: str) -> float:
    import sacrebleu

    score = sacrebleu.sentence_bleu(
        answer.strip(),
        [reference.strip()],
        tokenize="intl",
    ).score
    return max(0.0, min(1.0, score / 100.0))
